import torch
import numpy as np
import random
import json
from typing import Dict, List, Any


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def to_list(tensor1d):
    return tensor1d.detach().cpu().tolist() 

def cntSampleNum(dataset_path):
    cnt = 0
    with open(dataset_path, 'r') as fh:
        dataset = json.load(fh)
        for species in dataset['data']:
            for article in species['docs']:
                for paragraph in article['context']:
                    cnt += len(paragraph['qas'])
    return cnt

def cntTotalNum(bucket_path: str)->int:
    total_num = 0
    with open(bucket_path, 'r') as fh:
        datas: Dict[str, List] = json.load(fh)
        for _, val in datas.items():
            total_num += len(val)
    return total_num

def cntNumByQues(bucket_path: str)->Dict[str, int]:
    total_num = {}
    with open(bucket_path, 'r') as fh:
        datas: Dict[str, List] = json.load(fh)
        for key, val in datas.items():
            total_num[key] = len(val)
    return total_num

def cntFailedNum(bucket_datas: List[Dict[str, Any]], threshold: float)->int:
    cnt = 0
    for data in bucket_datas:
        if data['f1_score'] < threshold:
            cnt += 1
    return cnt

def cntPassNum(bucket_datas: List[Dict[str, Any]], threshold: float)->int:
    cnt = 0
    for data in bucket_datas:
        if data['f1_score'] >= threshold:
            cnt += 1
    return cnt

def cntFailedRate(bucket_datas: List[Dict[str, Any]], threshold: float)->float:
    cnt = 0
    for data in bucket_datas:
        if data['f1_score'] < threshold:
            cnt += 1
    return cnt / len(bucket_datas)

def cntPassRate(bucket_datas: List[Dict[str, Any]], threshold: float)->float:
    cnt = 0
    for data in bucket_datas:
        if data['f1_score'] >= threshold:
            cnt += 1
    return cnt / len(bucket_datas)

def cntEveryFailedRate(bucket_path: str, threshold: float=0.8)->Dict[str, int]:
    total_rate = {}
    with open(bucket_path, 'r') as fh:
        datas: Dict[str, List] = json.load(fh)
        for key, val in datas.items():
            total_rate[key] = cntFailedRate(val, threshold)
    return total_rate

def cntEveryPassRate(bucket_path: str, threshold: float=0.8)->Dict[str, int]:
    total_rate = {}
    with open(bucket_path, 'r') as fh:
        datas: Dict[str, List] = json.load(fh)
        for key, val in datas.items():
            total_rate[key] = cntPassRate(val, threshold)
    return total_rate

def cntTotalPassRate(bucket_path: str, threshold: float=0.8)->int:
    total_num = 0
    pass_num = 0
    with open(bucket_path, 'r') as fh:
        datas: Dict[str, List] = json.load(fh)
        for _, val in datas.items():
            total_num += len(val)
            pass_num += cntPassNum(val, threshold)
    return pass_num / total_num

def cntTotalF1Score(bucket_path: str)->float:
    cnt = 0
    f1_score = 0
    with open(bucket_path, 'r') as fh:
        datas: Dict[str, List] = json.load(fh)
        for _, val in datas.items():
            for item in val:
                f1_score += item['f1_score']
                cnt += 1
    return f1_score / cnt

def cntTotalExactlyMatch(bucket_path: str)->float:
    total_num = 0
    pass_num = 0
    with open(bucket_path, 'r') as fh:
        datas: Dict[str, List] = json.load(fh)
        for _, val in datas.items():
            total_num += len(val)
            pass_num += cntPassNum(val, 1.0)
    return pass_num / total_num
