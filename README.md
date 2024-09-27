# MicrobeQAExtractor

This repository provides the MicrobeDB dataset, the pathogenic microorganism knowledge extraction code, and the trained model weights.

## Models
- [`deberta-v3-base-microbedb-v1`](https://drive.google.com/drive/folders/1xuML3xoTqkQZAlKNoiUy2bFZ3laDXXuL?usp=drive_link)
- [`biobert_v1.1_microbedb_v1`](https://drive.google.com/drive/folders/1dcClcx9_vZcLblzhi8yPNNXG_GJ_jPaj?usp=drive_link)

## Configuration

### Environment Setup
```bash
conda env create -f config/env_requirements.yaml
conda activate qa
```

<!-- ### Additional Requirements
- Transforms
- pandas : Transforms the SQuAD prediction file into the BioASQ format (`pip install pandas`)
- tensorboardX : SummaryWriter module (`pip install tensorboardX`) -->

### Dataset
Dataset directory: `./MicrobeDB`. Please place `train_set.json` and `test_set.json` in this directory.

*Tips: You can download the MicrobeDB dataset from this GitHub repository in two ways:*

*1. Click the **“<> Code ▾”** button located below the repository name and navigation tabs, then select **“Download ZIP”** to download the entire repository, which includes the MicrobeDB dataset.*

*2. Alternatively, you can download the training or test sets individually. For the training set (`train_set.json`) or the test set (`test_set.json`), navigate to the respective file’s page and click the **“Download raw file”** icon to download the specific file.*

### Download Pre-trained model
- Download the pre-trained model
- Place the model files (such as weights, tokenizer, and configuration) in project's main directory under `<model_name>/`

### Run Commands
`--data_dir` defines which directory the dataset is in
```bash
python run.py \
    --model_name_or_path $MODEL_PATH \
    --model_class BioModel \
    --data_dir $DATASET \
    --max_seq_length 384 \
    --seed 0 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --single_gpu \
    --gpu $GPU \
# Train
    --do_train \
    --num_train_epochs $EPOCH \
    --logging_every_epoch \
    --learning_rate 8e-6 \
    --per_gpu_train_batch_size 24 \
    --train_file train_set.json \
    --data_augment \

# Evaluation
    --do_eval \
    --predict_file test_set.json
```
Or just run the preject by shell script `run.sh`
```bash
./run.sh [GPU] [EPOCH] [MODEL_PATH] [DATASET] [OUTPUT_DIR]
# like
./run.sh           # default: Use No.0 GPU, run 0 epoch, employ the trained deberta-v3-base-microbedb-v1 model to test, the dataset is MicrobeDB, the output directory is ./output
./run.sh 1         # Use No.1 GPU
./run.sh 1 3      # Use No.1 GPU, run 3 epochs
...
``` 

## Directory Structure

- `/MicrobeDB`: Dataset files
  - `test_set.json`: Test dataset for evaluating the model.
  - `train_set.json`: Training dataset for training the model.
- `/config`: Configuration files
  - `args_init.py`: Parses command-line arguments for configuring model training and evaluation.
  - `env_requirements.yaml`: Specifies the dependencies and environment settings required for model training and evaluation.
- `/processors`: Processing scripts.
  - `postprocess.py`: Generates final predictions based on the model's logits output. 
  - `preprocess.py`: Preprocesses the MicrobeDB dataset, including ChatGPT augmentation, and converts a list of examples into a list of features that can be directly given as input to a QA model.
- `/tools`: Utility scripts.
  - `eval_index.py`: Assesses the accuracy of text-based answers, including calculating exact match scores and F1 scores and generates statistical results. 
  - `load_examples.py`: Loads and caches data examples along with their converted features for a QA model.
  - `utils.py`: Contains utility functions such as setting a random seed, counting prediction failures based on question types, and other utilities.
- `README.md`: Provides an overview of the repository.
- `eval_func.py`: Implements the evaluation workflow, including data loading, batch processing, model inference, and result computation.
- `models.py`: Encapsulates the existing QA models (e.g., DeBERTaV3 and BioBERT) from the Transformers library.
- `run.py`: The main script that handles the entire training and evaluation process for the QA model.
- `run.sh`: Shell script to automate the execution of `run.py` with configurable parameters.
- `train_func.py`: Implements the training workflow, including model training, logging and monitoring, and model saving.

## Related work
This code comes from related work: **Interpretation knowledge extraction for genetic testing via question-answer model**

Authors: Wenjun Wang, Huanxin Chen, Hui Wang, Lin Fang, Huan Wang, Yi Ding, Yao Lu* and Qingyao Wu

*: Correspondent author

## Contact
For help or issues using MicrobeQAExtractor, please create an issue.
