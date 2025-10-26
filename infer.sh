export WORKSPACE_DIR='/mnt/d/life/learn/Tencent_AAC'
export TRAIN_LOG_PATH=${WORKSPACE_DIR}/train_log
export TRAIN_TF_EVENTS_PATH=${WORKSPACE_DIR}/train_tf_events
export TRAIN_DATA_PATH=${WORKSPACE_DIR}/TencentGR_1k
export USER_CACHE_PATH=${WORKSPACE_DIR}/user_cache
export TRAIN_CKPT_PATH=${WORKSPACE_DIR}/train_checkpoint

export MODEL_OUTPUT_PATH=${WORKSPACE_DIR}/train_checkpoint/global_step14.epoch1.valid_loss=7.9424
export EVAL_DATA_PATH=${WORKSPACE_DIR}/eval_data

export EVAL_RESULT_PATH=${WORKSPACE_DIR}/eval_result


python -u infer.py --batch_size 64 \
    --lr 0.005 \
    --hidden_units 32 \
    --num_blocks 4 \
    --num_heads 4 \
    --num_epochs 1 \
    --temperature 0.05 \
    --infonce_weight 1 \
    --weight_decay 0.001 \
    --dropout_rate 0.15 \
    --use_flash_infonce \
    --flash_block_size 256 \