import argparse
import json
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from dataset import MyDataset
from utils import set_seed
from torch.amp import autocast, GradScaler
from model import BaselineModel
from utils import optimized_embedding_init, validate_initialization, split_dataset
from torch.utils.data import Subset
import yaml


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=256, type=int)
    parser.add_argument('--num_blocks', default=8, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dropout_rate', default=0.15, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', default=False, type=bool)
    parser.add_argument('--temperature', default=0.05, type=float)
    parser.add_argument('--infonce_weight', default=1, type=float)
    parser.add_argument('--weight_decay', default=0.001, type=float)
    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    parser.add_argument('--use_flash_infonce', action='store_true', default=True,
                       help='使用Flash InfoNCE优化显存使用，基于Flash Attention思想实现分块计算')
    parser.add_argument('--flash_block_size', default=1024, type=int,
                       help='Flash InfoNCE的分块大小，越小显存占用越少但计算略慢，推荐256-1024')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    user_cache = os.environ.get('USER_CACHE_PATH')  # 添加user_cache路径

    set_seed(3407)

    dataset = MyDataset(data_path, user_cache, args)
    
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.99, 0.01])

    # 创建collate_fn包装函数，因为现在collate_fn是实例方法
    def collate_wrapper(batch):
        return dataset.collate_fn(batch)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=12,  # 降低worker数量避免文件冲突
        collate_fn=collate_wrapper,
        pin_memory=True,  # 启用内存锁定
        persistent_workers=True,  # 保持worker进程
        prefetch_factor=4  # 降低预取因子
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,  # 验证时使用更少worker
        collate_fn=collate_wrapper,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args, writer).to(args.device)

    scaler = GradScaler() 

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    optimized_embedding_init(model, args)

    validate_initialization(model)

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    num_training_steps = len(train_loader) * args.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=int(0.01 * num_training_steps),  # 1%预热
                    num_training_steps=num_training_steps
                )

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break

        for step, batch in enumerate(train_loader):
            (seq, pos, neg, pop_neg, token_type, next_token_type, next_action_type, 
             seq_feat_tensors, pos_feat_tensors, neg_feat_tensors, pop_neg_feat_tensors) = batch
            
            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=torch.bfloat16):
            # with autocast(device_type="cuda"):
            # 1. infonce loss
                infonce_loss = model.forward_infonce(
                    seq, pos, neg, pop_neg, token_type, next_token_type, next_action_type, 
                    seq_feat_tensors, pos_feat_tensors, neg_feat_tensors, pop_neg_feat_tensors, global_step
                )
            writer.add_scalar('Loss/infonce_loss', infonce_loss.item(), global_step)

            loss = infonce_loss

            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)

            global_step += 1

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            writer.add_scalar('Model/grad_norm', grad_norm.item(), global_step)

            optimizer.step()
            scheduler.step()

        model.eval()
        valid_loss_sum = 0
        start_time = time.time()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for step, batch in enumerate(valid_loader):
                (seq, pos, neg, pop_neg, token_type, next_token_type, next_action_type, 
                 seq_feat_tensors, pos_feat_tensors, neg_feat_tensors, pop_neg_feat_tensors) = batch

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    # 1. infonce loss
                    infonce_loss = model.forward_infonce(
                        seq, pos, neg, pop_neg, token_type, next_token_type, next_action_type, 
                        seq_feat_tensors, pos_feat_tensors, neg_feat_tensors, pop_neg_feat_tensors, global_step
                    )

                loss = infonce_loss
                valid_loss_sum += loss.item()
                if step % 100 == 0:
                    print(f"Valid step {step} time: {time.time() - start_time:.2f}s")
                    start_time = time.time()
        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)
        torch.cuda.empty_cache()
        
        print(f"Epoch {epoch} - Valid Loss: {valid_loss_sum:.4f}")

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.epoch{epoch}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
