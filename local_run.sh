#!/bin/bash

# # show ${RUNTIME_SCRIPT_DIR}
# echo ${RUNTIME_SCRIPT_DIR}
# # enter train workspace
# cd ${RUNTIME_SCRIPT_DIR}
#  torch.multinomial 动态负采样——需要纠偏
# embedding 初始化 + 去掉 triple 是真有用
export WORKSPACE_DIR='/mnt/d/life/learn/Tencent_AAC'
export TRAIN_LOG_PATH=${WORKSPACE_DIR}/train_log
export TRAIN_TF_EVENTS_PATH=${WORKSPACE_DIR}/train_tf_events
export TRAIN_DATA_PATH=${WORKSPACE_DIR}/TencentGR_1k
export USER_CACHE_PATH=${WORKSPACE_DIR}/user_cache
export TRAIN_CKPT_PATH=${WORKSPACE_DIR}/train_checkpoint


# 🔧 修复训练不稳定性的优化参数
python -u main.py --batch_size 64 \
    --lr 0.005 \
    --hidden_units 32 \
    --num_blocks 4 \
    --num_heads 4 \
    --num_epochs 1 \
    --temperature 0.05 \
    --infonce_weight 1 \
    --weight_decay 0.001 \
    --dropout_rate 0.15 \

