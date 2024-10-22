import torch
import numpy as np
import random
import re

def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def to_list(tensor1d):
    return tensor1d.detach().cpu().tolist() 

# Count the number of failed samples based on F1 score for a given bucket of data
def cntFailedNumByQuesType(bucket_data, threshold = 1):
    count = 0
    for data in bucket_data:
        if data["f1_score"] < threshold:
            count += 1
    return count

# List of regex patterns to identify different question types
ques_patterns = [
    "Whether [\s\S]* is gram-positive or gram-negative?",
    "Where does [\s\S]* normally exist?",
    "What kinds of diseases can [\s\S]* cause?",
    "What about the pathogenicity of [\s\S]*?",
    "What kinds of drugs are [\s\S]* sensitive to?",
    "What kinds of drugs are [\s\S]* resistant to?",
    "How about [\s\S]*'s requirement for oxygen?",
    "What is the shape of [\s\S]*?",

    # Augmented patterns for Locations
    "What are the typical habitats of [\s\S]*?",
    "In what environments can [\s\S]* be found?",
    "Where is [\s\S]* commonly present?",
    "In what locations can [\s\S]* usually be found?",
    "What are the common sites where [\s\S]* is known to inhabit?",

    # Augmented patterns for Diseases
    "What are the diseases that can be caused by [\s\S]*?",
    "Which diseases are associated with [\s\S]* infection?",
    "What health problems can result from [\s\S]* colonization?",
    "What is the disease spectrum of [\s\S]*?",
    "What types of illnesses can [\s\S]* contribute to?",

    # Augmented patterns for Sensitivity
    "Which drugs are effective against [\s\S]*?",
    "What medications can be used to treat [\s\S]* infections?",
    "Which antibiotics are recommended for treating [\s\S]* infections?",
    "What drugs have been shown to be active against [\s\S]*?",
    "What are the drugs that [\s\S]* is vulnerable to?",
]

# List of corresponding question types for each pattern in ques_patterns
ques_types = [
    'Gram',
    'Locations',
    'Diseases',
    'Pathogenicity',
    'Sensitivity',
    'Resistance',
    'Oxygen',
    'Morphology',

    'Locations',
    'Locations',
    'Locations',
    'Locations',
    'Locations',

    'Diseases',
    'Diseases',
    'Diseases',
    'Diseases',
    'Diseases',

    'Sensitivity',
    'Sensitivity',
    'Sensitivity',
    'Sensitivity',
    'Sensitivity',
]

# Determine the type of question by matching the given question against predefined patterns
def getQuesType(question):
    for idx, ques_pattern in enumerate(ques_patterns):
        res = re.match(ques_pattern, question)
        if res != None:
            return ques_types[idx]
    print(question)
    return None

class SquadImpossible:
    def __init__(self, unique_id, impossible):
        self.impossible = impossible
        self.unique_id = unique_id
