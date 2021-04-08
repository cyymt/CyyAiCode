#!/usr/bin/env bash

LOG_ALIAS="mask"
LOG_DIR="logs/recognation_train"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/${LOG_ALIAS}_`date +'%Y-%m-%d_%H:%M.%S'`.log"
#echo $LOG_FILE
# --loss_type="focalLoss" \
./main.py --data_type="recognation" \
    --mode="test" \
    --prune_percent=0.85 \
    --prune_mode="constant" \
    --prune_s=0.001 \
    --num-classes=2 \
    --gray='false' \
    --cutmix='false' \
    --teacher_arch="" \
    --teacher_resume="" \
    --teacher_T=3 \
    --teacher_alpha=0.8 \
    --arch="mobilenet_2" \
    --resume="snapshot/prune/mb2_prune_checkpoint_epoch_60.pth.tar" \
    --binary_model='normal' \
    --start-epoch=0 \
    --snapshot="snapshot/prune/mb2_prune" \
    --warmup=5 \
    --devices-id="0,1" \
    --workers=8 \
    --epochs=250 \
    --save_epochs=20 \
    --milestones=60,120,200 \
    --batch-size=512 \
    --val-batch-size=512 \
    --base-lr=0.01 \
    --adjust_lr="cosine" \
    --loss_type="labelSmooth" \
    --optimizer_type="sgd" \
    --resample-num=132 \
    --print-freq=20 \
    --train_file="../cat_dog/train.txt" \
    --test_file="../cat_dog/val.txt" \
    --root="../cat_dog" \
    --log-file="${LOG_FILE}"
