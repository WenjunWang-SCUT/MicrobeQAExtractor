GPU=${1:-0}
EPOCH=${2:-5}
ROOT=${3:-"./MicrobeDB"}
MODEL_CLASS=${4:-"BioModel"}
OUTPUT_DIR=${5:-"./output"}
python run.py \
    --model_type bert \
    --model_name_or_path models\
    --load_remote_model \
    --model_class $MODEL_CLASS\
    --data_dir $ROOT \
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
    --eval_every_epoch \
    --logging_every_epoch \
    --save_every_epoch \
    --do_eval \
    --predict_file test_set.json
