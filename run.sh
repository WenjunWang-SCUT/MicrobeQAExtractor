GPU=${1:-0}
EPOCH=${2:-0}
MODEL_PATH=${3:-"./models/deberta-v3-base-microbedb-v1"}
DATASET=${4:-"./MicrobeDB"}
OUTPUT_DIR=${5:-"./output"}
python run.py \
    --model_type bert \
    --model_name_or_path $MODEL_PATH \
    --load_remote_model \
    --model_class BioModel\
    --data_dir $DATASET \
    --max_seq_length 384 \
    --seed 0 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --single_gpu \
    --gpu $GPU \
    --do_train \
    --num_train_epochs $EPOCH \
    --learning_rate 8e-6 \
    --per_gpu_train_batch_size 24 \
    --train_file train_set.json \
    --data_augment \
    --do_eval \
    --predict_file test_set.json
