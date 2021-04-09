#!/bin/bash

python simple_online_demo.py --root_path ./ml_real_time_gesrec/ \
    --resume_path ./weights/jester_resnext_101_RGB_32.pth \
    --train_crop random \
    --modality RGB \
    --n_classes 27 \
    --n_finetune_classes 27 \
    --model resnext \
    --model_depth 101 \
    --resnet_shortcut B \
    --resnext_cardinality 32 \
    --groups 3 \
    --sample_duration 32 \
    --downsample 2 \
    --batch_size 32 \
    --n_threads 0 \
    --clf_threshold_final 0.15\
    --no_cuda \
    --no_fc