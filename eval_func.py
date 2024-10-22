import os
import timeit
import torch
import json
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from transformers.data.processors.squad import SquadResult

from tools.load_examples import load_and_cache_examples
from tools.eval_index import eval_metrics
from processors.postprocess import compute_predictions_logits
from tools.utils import to_list, getQuesType, SquadImpossible
from models import BioModelClassify

# Get the logger "BIOMODEL"
logger = logging.getLogger("BIOMODEL")

# Evaluates a single batch of data
def eval_one_batch(model, batch, features):
    batch_result = []
    batch_pred_unsolvable = []

    # Disable gradient computation and pass the batch to the model in evaluation mode
    with torch.no_grad():
        feature_indices = batch[3]
        outputs = model(batch, is_training=False)

    # Iterate over feature indices to collect predictions
    for i, feature_index in enumerate(feature_indices):
        eval_feature = features[feature_index.item()]
        unique_id = int(eval_feature.unique_id)
        if isinstance(model, BioModelClassify):
            batch_pred_unsolvable.append(SquadImpossible(unique_id, outputs['pred_impossible'][i]))

        # Convert output logits to lists
        output_list = [to_list(outputs['start_logits'][i]), to_list(outputs['end_logits'][i])]

        # Some models use 5 arguments for their predictions, while the other models only use two.
        if len(output_list) >= 5:
            start_logits = output_list[0]
            start_top_index = output_list[1]
            end_logits = output_list[2]
            end_top_index = output_list[3]
            cls_logits = output_list[4]

            # Create a SquadResult object with detailed prediction info
            result = SquadResult(
                unique_id,
                start_logits,
                end_logits,
                start_top_index=start_top_index,
                end_top_index=end_top_index,
                cls_logits=cls_logits,
            )
        else:
            start_logits, end_logits = output_list
            result = SquadResult(unique_id, start_logits, end_logits)

        batch_result.append(result)
    return batch_result, batch_pred_unsolvable

# Evaluates the model on the test dataset
def eval_by_model(args, model, eval_dataloader, features, tqdm_enabled=True):
    all_results = []
    all_pred_unsolvable = []

    # Iterate over the evaluation DataLoader with a progress bar
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not tqdm_enabled):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        # Evaluate the current batch and collect results
        batch_result, batch_pred_unsolvable = eval_one_batch(model, batch, features)
        all_results += batch_result
        all_pred_unsolvable += batch_pred_unsolvable

    return all_results, all_pred_unsolvable

# Main evaluation function to assess the model's performance
def evaluate(args, model, tokenizer, prefix=""):
    USE_CLASSIFY = isinstance(model, BioModelClassify)

    # Load data features from cache or dataset file
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # Calculate effective evaluation batch size
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Setup DataLoader for evaluation
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Multi-GPU evaluation setup
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Logging evaluation start
    logger.info("***** Running evaluation *****")
    logger.info("  Num features = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # Evaluate the model, record start evaluation time and total evaluation time
    start_time = timeit.default_timer()
    all_results, all_pred_unsolvable = eval_by_model(args, model, eval_dataloader, features)
    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per feature)", evalTime, evalTime / len(dataset))

    # Prepare output files
    output_prediction_file = os.path.join(args.output_dir, "predictions.json")
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
    if args.with_neg:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds.json")
    else:
        output_null_log_odds_file = None
    
    # Prepare question type and answers
    ques_type = {example.qas_id: getQuesType(example.question_text) for example in examples}
    answers = {example.answers['id']: example.answers['answers'] for example in examples}

    # Generate predictions based on model outputs
    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.with_neg,
        args.null_score_diff_threshold,
        tokenizer,
        all_pred_unsolvable if USE_CLASSIFY else None,
    )

    # Evaluate the predictions against the answers
    eval_rst = eval_metrics(answers, predictions, ques_type, args.output_dir, prefix, str(args.null_score_diff_threshold))

    # Logging evaluation results
    logger.info("** Evaluation Info *********************************")
    logger.info(f"   Test samples num: {len(examples)}")
    logger.info("** Evaluation Results *********************************")
    logger.info(f"   EM = {eval_rst['exact_match']:.2f}")
    logger.info(f"   F1 score = {eval_rst['f1']:.2f}")

    # Read predictions from the output file
    with open(output_prediction_file, 'r') as reader:
        pre_json = json.load(reader)
        f1_scores = eval_rst['all_f1']

    # Prepare a summary table for different question types
    table = {
        'Gram': [],
        'Locations': [],
        'Diseases': [],
        'Pathogenicity': [],
        'Sensitivity': [],
        'Resistance': [],
        'Oxygen': [],
        'Morphology': []
    }

    # Categorizes the evaluation results by question type and stores them in the summary table
    for idx, example in enumerate(examples):
        ques_name = getQuesType(example.question_text)
        tmp = {'qas_id': example.qas_id, 'prediction': pre_json[example.qas_id], 'f1_score': f1_scores[idx]}
        table[ques_name].append(tmp)

    # Save the summary table to a JSON file
    with open(os.path.join(args.output_dir, 'bucket.json'), 'w') as fh:
        json.dump(table, fh)

    return {
        "EM": eval_rst["exact_match"], 
        "F1_score": eval_rst["f1"]
    }
