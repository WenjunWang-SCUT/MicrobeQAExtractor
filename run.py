import glob
import os
import torch
import logging
import datetime

from transformers import (
    WEIGHTS_NAME,
    AutoConfig,
    AutoTokenizer,
)

from arguments import get_parser
from tools.utils import set_seed
from tools.load_examples import load_and_cache_examples
from train_func import train
from eval_func import evaluate
from arguments import MODEL_CLASS_TABLE
from models.derived_models import BioModelClassify

logger = logging.getLogger("BIOBERT")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    args = get_parser()
    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args.output_dir = os.path.join(args.output_dir, current_time)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

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
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda: # local_rank -1
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
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    fh_file = logging.FileHandler(os.path.join(args.output_dir, current_time + '.log'), encoding='utf8')  # handler
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s -  %(message)s")
    fh_file.setLevel(logging.DEBUG)
    fh_file.setFormatter(formatter)
    logger.addHandler(fh_file)

    # Set seed
    set_seed(args.seed, args.n_gpu)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower() #
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None, # None
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None, # None
        use_fast=False, # False
    )

    # load model from model_name_or_path
    target_model = MODEL_CLASS_TABLE[args.model_class]
    model = target_model(args.model_name_or_path, config, args)
    if args.use_exist_model:
        model.load_state_dict(torch.load(args.exist_model_path))

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False) 
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, 'pytorch_model.bin'))
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        if isinstance(model, BioModelClassify):
            model_to_save.saveBCHeader(dir=args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = target_model(args.model_name_or_path, config, args)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'pytorch_model.bin')))
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case, use_fast=False)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    if ((args.do_eval and not args.do_train) or (args.do_train and args.num_train_epochs == 0)) and args.local_rank in [-1, 0]:

        logger.info("Loading weights from path %s for evaluation", args.model_name_or_path)
        checkpoint = args.model_name_or_path

        if isinstance(model, BioModelClassify):
            model.loadBCHeader(os.path.join(checkpoint, "biclassify_header.pt"))
        model.to(args.device)

        # Evaluate
        evaluate(args, model, tokenizer, prefix="")


if __name__ == "__main__":
    main()
