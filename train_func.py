import os
import torch
import logging
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from tools.utils import set_seed
from eval_func import evaluate
from models import BioModelClassify

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
    

logger = logging.getLogger("BIOMODEL")

# Main training function for the model
def train(args, train_dataset, model, tokenizer):
    USE_CLASSIFY = isinstance(model, BioModelClassify)
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.output_dir)

    # Calculate effective training batch size
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)

    # Determine total training steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]

    # Use AdamW optimizer for improved weight decay handling
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Mixed precision training if specified
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Multi-GPU training setup
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training setup
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Start training
    logger.info("***** Running training *****")
    logger.info("  Num Features = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # Set global_step to the global step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning from path %s.", args.model_name_or_path)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Set random seed
    set_seed(args.seed, args.n_gpu)

    for epoch_no in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_len = len(train_dataloader)
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()  # Set model to training mode
            batch = tuple(t.to(args.device) for t in batch)  # Move batch to device
            outputs = model(batch, is_training=True)  # Forward pass
            
            # Model outputs are always a tuple in transformers
            loss = outputs[0]  # Extract the loss

            # Average loss across multiple GPUs
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # Backward pass with mixed precision if enabled
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()  # Accumulate loss
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Clip gradients
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()  # Update parameters
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()  # Reset gradients
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.local_rank == -1 and args.eval_every_x_step:
                        results = evaluate(args, model, tokenizer, prefix=str(epoch_no))
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                            logger.info(f"Step-{global_step} {key}: {value}")
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info(f"Step-{global_step} lr: {scheduler.get_lr()[0]}")
                    logger.info(f"Step-{global_step} loss: {(tr_loss - logging_loss) / args.logging_steps}")
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    saveModel(output_dir, model, tokenizer, optimizer, scheduler, args)
                    if USE_CLASSIFY:
                        model.saveBCHeader(output_dir)

            # Stop training if maximum steps reached
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
        
        # Log metrics at the end of each epoch
        if args.local_rank in [-1, 0] and args.logging_every_epoch:
            if args.local_rank == -1 and args.eval_every_epoch:
                results = evaluate(args, model, tokenizer, prefix=str(epoch_no))
                for key, value in results.items():
                    tb_writer.add_scalar("eval_{}".format(key), value, epoch_no)
                    logger.info(f"Epoch-{epoch_no} {key}: {value}")
            tb_writer.add_scalar("loss", (tr_loss - logging_loss) / epoch_len, epoch_no)
            logger.info(f"Epoch-{epoch_no} average loss: {(tr_loss - logging_loss) / epoch_len}")
            logging_loss = tr_loss

        # Save model checkpoint at the end of each epoch
        if args.local_rank in [-1, 0] and args.save_every_epoch:
            output_dir = os.path.join(args.output_dir, "checkpoint-epoch{}".format(epoch_no))
            saveModel(output_dir, model, tokenizer, optimizer, scheduler, args)
            if USE_CLASSIFY:
                model.saveBCHeader(output_dir)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step  # Return training step and average loss


# Save the model weights, tokenizer, training arguments, and the optimizer and scheduler states
def saveModel(output_dir, model, tokenizer, optimizer, scheduler, args):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))  # Save model weights
    tokenizer.save_pretrained(output_dir)  # Save tokenizer
    torch.save(args, os.path.join(output_dir, "training_args.bin"))  # Save training arguments
    logger.info("Saving model checkpoint to %s", output_dir)

    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))  # Save optimizer state
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))  # Save scheduler state
    logger.info("Saving optimizer and scheduler states to %s", output_dir)
