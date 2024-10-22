from __future__ import print_function
from collections import Counter
import string
import os
import re
import json


# Convert to lowercase and remove punctuation, articles ('a', 'an', 'the') and extra whitespace
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# Compute the F1 score between the predicted and ground truth text
def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# Check if the normalized prediction exactly matches the normalized ground truth
def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

# Calculate the maximum metric score
def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

# Evaluate the prediction metrics, including F1 score and Exact match (EM) score
def eval_metrics(answers, predictions, ques_type, output_dir="", prefix="", threshold=""):
    total_f1 = total_em = solvable_f1 = solvable_em = 0.0
    total_num = solvable_total_num = tp_num = fp_num = fn_num = tn_num = 0
    correct_qid = []
    all_f1 = []
    err_qid = []

    table = {
        'Gram': {'total_num': 0, 'total_f1': 0, 'total_em': 0, 'solvable_num': 0, 'solvable_f1': 0, 'solvable_em': 0, 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        'Locations': {'total_num': 0, 'total_f1': 0, 'total_em': 0, 'solvable_num': 0, 'solvable_f1': 0, 'solvable_em': 0, 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        'Diseases': {'total_num': 0, 'total_f1': 0, 'total_em': 0, 'solvable_num': 0, 'solvable_f1': 0, 'solvable_em': 0, 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        'Pathogenicity': {'total_num': 0, 'total_f1': 0, 'total_em': 0, 'solvable_num': 0, 'solvable_f1': 0, 'solvable_em': 0, 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        'Sensitivity': {'total_num': 0, 'total_f1': 0, 'total_em': 0, 'solvable_num': 0, 'solvable_f1': 0, 'solvable_em': 0, 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        'Resistance': {'total_num': 0, 'total_f1': 0, 'total_em': 0, 'solvable_num': 0, 'solvable_f1': 0, 'solvable_em': 0, 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        'Oxygen': {'total_num': 0, 'total_f1': 0, 'total_em': 0, 'solvable_num': 0, 'solvable_f1': 0, 'solvable_em': 0, 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        'Morphology': {'total_num': 0, 'total_f1': 0, 'total_em': 0, 'solvable_num': 0, 'solvable_f1': 0, 'solvable_em': 0, 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
    }

    for qid in answers:
        if qid not in predictions:
            message = "Unanswered question {} will receive score 0.".format(qid)
            print(message)
            continue

        total_num += 1
        q_type = ques_type[qid]
        table[q_type]['total_num'] += 1
        ground_truths = list(map(lambda x: x['text'], answers[qid]))
        prediction = predictions[qid]['text']

        if ground_truths:
            if prediction:
                cur_em = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
                cur_f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
                # tp_qid.append(qid)
                tp_num += 1
                table[q_type]['tp'] += 1
            else:
                cur_em = 0
                cur_f1 = 0
                # fn_qid.append(qid)
                fn_num += 1
                table[q_type]['fn'] += 1
                err_qid.append(qid)

            solvable_total_num += 1
            table[q_type]['solvable_num'] += 1
            solvable_em += cur_em
            table[q_type]['solvable_em'] += cur_em
            solvable_f1 += cur_f1
            table[q_type]['solvable_f1'] += cur_f1
        else:
            if prediction:
                cur_em = 0
                cur_f1 = 0
                # fp_qid.append(qid)
                fp_num += 1
                table[q_type]['fp'] += 1
                err_qid.append(qid)

            else:
                cur_em = 1
                cur_f1 = 1
                # tn_qid.append(qid)
                tn_num += 1
                table[q_type]['tn'] += 1
        
        if int(cur_em) == 1:
            correct_qid.append(qid)
        total_f1 += cur_f1
        table[q_type]['total_f1'] += cur_f1
        total_em += cur_em
        table[q_type]['total_em'] += cur_em
        all_f1.append(cur_f1)

    total_em = 100.0 * total_em / total_num
    total_f1 = 100.0 * total_f1 / total_num
    solvable_em = 100.0 * solvable_em / solvable_total_num
    solvable_f1 = 100.0 * solvable_f1 / solvable_total_num

    for key in table:
        if table[key]['total_num'] > 0:
            table[key]['total_em'] = 100.0 * table[key]['total_em'] / table[key]['total_num']
            table[key]['total_f1'] = 100.0 * table[key]['total_f1'] / table[key]['total_num']

        if table[key]['solvable_num'] > 0:
            table[key]['solvable_em'] = 100.0 * table[key]['solvable_em'] / table[key]['solvable_num']
            table[key]['solvable_f1'] = 100.0 * table[key]['solvable_f1'] / table[key]['solvable_num']

    save_dir = os.path.join(os.path.join(output_dir, 'checkpoint-{}'.format(prefix))) if prefix else output_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "err_qid_{}.json".format(threshold)), "w") as writer:
        writer.write(json.dumps(err_qid, indent=4) + "\n")

    with open(os.path.join(save_dir, 'bucket_{}.json'.format(threshold)), 'w') as writer:
        writer.write(json.dumps(table, indent=4) + "\n")

    return {
        'f1': total_f1, 
        'exact_match': total_em, 
        'all_f1': all_f1, 
        'solvable_f1': solvable_f1, 
        'solvable_em': solvable_em,
        'tp': tp_num,
        'fp': fp_num,
        'tn': tn_num,
        'fn': fn_num,
    }
