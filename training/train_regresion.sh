#!/usr/bin/env bash

LOG_ALIAS="3ddfa"
LOG_DIR="logs/regresion_train"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/${LOG_ALIAS}_`date +'%Y-%m-%d_%H:%M.%S'`.log"
#echo $LOG_FILE

./main.py --data_type="regresion" \
    --mode="train" \
    --num-classes=62 \
    --gray='false' \
    --arch="mobilenet_05" \
    --binary_model='normal' \
    --start-epoch=0 \
    --snapshot="snapshot/mb05_mask" \
    --resume="model_wpdc_0.0351/mb05_mask_checkpoint_epoch_925.pth.tar" \
    --warmup=-1 \
    --devices-id=7 \
    --workers=8 \
    --epochs=600 \
    --save_epochs=5 \
    --milestones=100,400,550 \
    --batch-size=32 \
    --base-lr=0.0005 \
    --adjust_lr="normal" \
    --loss_type="mseLoss" \
    --optimizer_type="sgd" \
    --resample-num=132 \
    --print-freq=50 \
    --train_file="/ssd/chenyuyang/Code/3DDFA/dataset/3ddfa_datasets/mask_list/train_mask.txt" \
    --test_file="/ssd/chenyuyang/Code/3DDFA/dataset/3ddfa_datasets/mask_list/val_mask.txt" \
    --root="/ssd/chenyuyang/Code/3DDFA/dataset/3ddfa_datasets" \
    --log-file="${LOG_FILE}"