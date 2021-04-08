#!/usr/bin/env bash

LOG_ALIAS="retinaFace"
LOG_DIR="logs/detector_train"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/${LOG_ALIAS}_`date +'%Y-%m-%d_%H:%M.%S'`.log"
#echo $LOG_FILE

./main.py --data_type="detector" \
    --mode="train" \
    --num-classes=20 \
    --arch="mobilenet_025" \
    --binary_model='normal' \
    --start-epoch=0 \
    --snapshot="snapshot/retinaFace" \
    --resume="weights/retinaFace/mobilenet0.25_Final.pth" \
    --warmup=-1 \
    --devices-id=7 \
    --workers=4 \
    --epochs=600 \
    --save_epochs=5 \
    --milestones=100,400,550 \
    --batch-size=4 \
    --base-lr=0.0005 \
    --adjust_lr="normal" \
    --loss_type="multiboxLoss" \
    --optimizer_type="sgd" \
    --resample-num=132 \
    --print-freq=50 \
    --train_file="/ssd/chenyuyang/Code/datasets/wider_face_landmark5/train/label.txt" \
    --test_file="" \
    --root="/ssd/chenyuyang/Code/datasets/wider_face_landmark5/train/images" \
    --log-file="${LOG_FILE}"

# --dataset_type="COCO" \
# --dataset_root="/ssd/chenyuyang/Code/datasets/coco_2017" \