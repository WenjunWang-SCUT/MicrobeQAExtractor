import os
import argparse

from models import (
    BioModel,
    BioModelClassifyOne, 
    BioModelClassifyTwo, 
    BioModelClassifyCNN,
)

MODEL_CLASS_TABLE = {
    "BioModel": BioModel,
    "BioModelClassifyOne": BioModelClassifyOne,
    "BioModelClassifyTwo": BioModelClassifyTwo,
    "BioModelClassifyCNN": BioModelClassifyCNN,
}

# Function to define and return the argument parser
def get_parser():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add arguments for model path, output directory, and other settings
    parser.add_argument(
        "--model_name_or_path",
        default="biobert_v1.1_pubmed_squad_v2",
        type=str,
        help="Specifies the path to a pre-trained model for fine-tuning or an already trained model for evaluation.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.getcwd(), "output"),
        type=str,
        help="Sets the output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="BioModel",
        choices=MODEL_CLASS_TABLE.keys(),
        help="Indicates the model class to use. The BioModel class serves as a unified interface or wrapper for both DeBERTaV3 and BioBERT models.",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input test file. If a data dir is specified, will look for the file there.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name."
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name.",
    )
    parser.add_argument(
        "--cache_dir",
        default="data-cache",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3.",
    )
    parser.add_argument(
        "--with_neg",
        action="store_true",
        help="If true, the examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="Sets the maximum length of each tokenized input. Longer texts will be chunked into segments of this length.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument( 
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.") 
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation on the test set.")
    parser.add_argument(
        "--eval_every_x_step", action="store_true", default=False, help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument( 
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=8e-6, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. A number of warnings are expected for a normal evaluation.",
    )
    parser.add_argument("--logging_steps", type=int, default=0, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available.")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory."
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and test sets."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local_rank for distributed training on gpus.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit.",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--threads", type=int, default=1, help="Multiple threads for converting example to features.")
    parser.add_argument("--single_gpu", action="store_true", help="Just use one gpu.")
    parser.add_argument("--gpu", type=int, default=0, help="Which gpu to use.")
    parser.add_argument("--eval_every_epoch", action="store_true", default=True, help="Evaluate every epoch.")
    parser.add_argument("--logging_every_epoch", action="store_true", default=True, help="Log every epoch.")
    parser.add_argument("--save_every_epoch", action="store_true", default=True, help="Save model every epoch.")
    parser.add_argument("--data_augment", action="store_true", default=False, help="Augment the training set.")

    # Return the parsed arguments
    return  parser.parse_args()
