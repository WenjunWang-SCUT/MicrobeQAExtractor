import os
import torch
import logging
import datetime
from transformers import AutoConfig, AutoTokenizer

from config.args_init import get_parser,MODEL_CLASS_TABLE
from tools.utils import set_seed
from tools.load_examples import load_and_cache_examples
from train_func import train
from eval_func import evaluate
from models import BioModelClassify

# Initialize a logger for tracking and recording messages during program execution
logger = logging.getLogger("BIOMODEL")

# Disable parallelism in the tokenizer to avoid potential resource conflicts during concurrent processing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    """
    Main function that sets up the configuration for training and evaluation,
    initializes model parameters, and executes the training and evaluation flow.
    """
    # Parse command-line arguments
    args = get_parser()
    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args.output_dir = os.path.join(args.output_dir, current_time)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Check if the document stride setting is valid
    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    # Check if the output directory exists and is not empty
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Set up CUDA, GPU
    if args.local_rank == -1 or args.no_cuda:
        if args.single_gpu:
            if torch.cuda.is_available():
                torch.cuda.set_device(args.gpu)
                device = torch.device('cuda')
                args.n_gpu = 1
            else:
                device = torch.device('cpu')
                args.n_gpu = 0
        else:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    # Set up file handler for logging to a file
    fh_file = logging.FileHandler(os.path.join(args.output_dir, current_time + '.log'), encoding='utf8')
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s -  %(message)s")
    fh_file.setLevel(logging.DEBUG)
    fh_file.setFormatter(formatter)
    logger.addHandler(fh_file)

    # Set random seed
    set_seed(args.seed, args.n_gpu)

    # Load pre-trained model configuration and tokenizer
    if args.local_rank not in [-1, 0]:
        # Ensure that only the first process downloads the model and vocab
        torch.distributed.barrier()

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        use_fast=False,
    )

    # Load model from model_name_or_path
    target_model = MODEL_CLASS_TABLE[args.model_class]
    model = target_model(args.model_name_or_path, config, args)

    if args.local_rank == 0:
        # Process synchronization barrier
        torch.distributed.barrier()

    model.to(args.device)  # Move the model to the designated device

    logger.info("Training/evaluation parameters %s", args)

    # If fp16 (mixed precision) training is set, register half precision operations for einsum
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Perform model training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info("Global step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save the model only for the main process
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, 'pytorch_model.bin'))
        tokenizer.save_pretrained(args.output_dir)

        # Save training arguments for future reference
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        if isinstance(model, BioModelClassify):
            model_to_save.saveBCHeader(dir=args.output_dir)

        # Load the saved model and tokenizer
        model = target_model(args.model_name_or_path, config, args)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'pytorch_model.bin')))
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case, use_fast=False)
        model.to(args.device)

    # Perform model evaluation
    if ((args.do_eval and not args.do_train) or (args.do_train and args.num_train_epochs == 0)) and args.local_rank in [-1, 0]:
        logger.info("Loading weights from path %s for evaluation", args.model_name_or_path)
        checkpoint = args.model_name_or_path
        if isinstance(model, BioModelClassify):
            model.loadBCHeader(os.path.join(checkpoint, "biclassify_header.pt"))
        model.to(args.device)

        # Evaluate the model performance
        evaluate(args, model, tokenizer, prefix="")

# Main entry point
if __name__ == "__main__":
    main()
