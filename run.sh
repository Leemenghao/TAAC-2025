#!/bin/bash

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}
#  torch.multinomial 动态负采样——需要纠偏
# embedding 初始化 + 去掉 triple 是真有用
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 🔧 修复训练不稳定性的优化参数
python -u main.py --batch_size 180 \
    --lr 0.001 \
    --hidden_units 256 \
    --num_blocks 8 \
    --num_heads 8 \
    --num_epochs 10 \
    --temperature 0.03 \
    --infonce_weight 1 \
    --weight_decay 0.0005 \
    --dropout_rate 0.15 \
    --use_flash_infonce \
    --flash_block_size 256 \

