import argparse
import json
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import struct
from pathlib import Path
import hashlib
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyTestDataset, save_emb, OptimizedTestDataset
from model import BaselineModel

import yaml


def get_ckpt_path():
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
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
    parser.add_argument('--temperature', default=0.03, type=float)
    parser.add_argument('--infonce_weight', default=1, type=float)
    parser.add_argument('--weight_decay', default=0.005, type=float)

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    parser.add_argument('--use_flash_infonce', action='store_true', default=True,
                       help='使用Flash InfoNCE优化显存使用，基于Flash Attention思想实现分块计算')
    parser.add_argument('--flash_block_size', default=1024, type=int,
                       help='Flash InfoNCE的分块大小，越小显存占用越少但计算略慢，推荐256-1024')

    args = parser.parse_args()

    return args


def read_result_ids(file_path):
    with open(file_path, 'rb') as f:
        # Read the header (num_points_query and FLAGS_query_ann_top_k)
        num_points_query = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes
        query_ann_top_k = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes

        print(f"num_points_query: {num_points_query}, query_ann_top_k: {query_ann_top_k}")

        # Calculate how many result_ids there are (num_points_query * query_ann_top_k)
        num_result_ids = num_points_query * query_ann_top_k

        # Read result_ids (uint64_t, 8 bytes per value)
        result_ids = np.fromfile(f, dtype=np.uint64, count=num_result_ids)

        return result_ids.reshape((num_points_query, query_ann_top_k))


def _stable_hash_to_int(text: str) -> int:
    """将字符串稳定映射为非负整数（与Python进程无关）。"""
    if not isinstance(text, str):
        text = str(text)
    h = hashlib.md5(text.encode('utf-8')).digest()
    return int.from_bytes(h[:8], 'little', signed=False)


def process_cold_start_feat(feat, feat_types, feat_statistics):
    """
    冷启动特征优化：
    - 稀疏/多值稀疏特征：未见过的字符串使用哈希分桶映射到合法ID范围（避开padding=0）
    - 连续特征：无法解析时回退为0.0
    """
    processed_feat = {}
    if feat is None:
        return processed_feat

    sparse_keys = set(feat_types.get('item_sparse', [])) | set(feat_types.get('user_sparse', []))
    array_keys = set(feat_types.get('item_array', [])) | set(feat_types.get('user_array', []))
    continual_keys = set(feat_types.get('item_continual', [])) | set(feat_types.get('user_continual', []))

    for feat_id, feat_value in feat.items():
        if feat_id in sparse_keys:
            if isinstance(feat_value, str):
                vocab = max(1, int(feat_statistics.get(feat_id, 1)))
                mapped = 1 + (_stable_hash_to_int(feat_value) % vocab)
                processed_feat[feat_id] = mapped
            else:
                processed_feat[feat_id] = int(feat_value)
        elif feat_id in array_keys:
            if isinstance(feat_value, list):
                vocab = max(1, int(feat_statistics.get(feat_id, 1)))
                mapped_list = []
                for v in feat_value:
                    if isinstance(v, str):
                        mapped_list.append(1 + (_stable_hash_to_int(v) % vocab))
                    else:
                        mapped_list.append(int(v))
                processed_feat[feat_id] = mapped_list
            elif isinstance(feat_value, str):
                vocab = max(1, int(feat_statistics.get(feat_id, 1)))
                processed_feat[feat_id] = [1 + (_stable_hash_to_int(feat_value) % vocab)]
            else:
                processed_feat[feat_id] = [int(feat_value)]
        elif feat_id in continual_keys:
            if isinstance(feat_value, str):
                processed_feat[feat_id] = 0.0
            else:
                try:
                    processed_feat[feat_id] = float(feat_value)
                except Exception:
                    processed_feat[feat_id] = 0.0
        else:
            # 其它类型（例如多模态emb由外层填充）保持原样
            processed_feat[feat_id] = feat_value
    return processed_feat


def get_candidate_emb(indexer, feat_types, feat_default_value, mm_emb_dict, feat_statistics, model):
    """
    生产候选库item的id和embedding

    Args:
        indexer: 索引字典
        feat_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feature_default_value: 特征缺省值
        mm_emb_dict: 多模态特征字典
        model: 模型
    Returns:
        retrieve_id2creative_id: 索引id->creative_id的dict
    """
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    item_ids, creative_ids, retrieval_ids, features = [], [], [], []
    retrieve_id2creative_id = {}

    # 预计算各多模态特征的均值向量，作为冷启动回退（比全零更稳）
    EMB_DEFAULTS = {}
    for emb_k, emb_map in mm_emb_dict.items():
        try:
            if len(emb_map) > 0:
                EMB_DEFAULTS[emb_k] = np.mean(np.stack(list(emb_map.values()), axis=0), axis=0).astype(np.float32)
        except Exception:
            pass

    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            # 读取item特征，并补充缺失值
            feature = line['features']
            creative_id = line['creative_id']
            retrieval_id = line['retrieval_id']
            item_id = indexer[creative_id] if creative_id in indexer else 0
            missing_fields = set(
                feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']
            ) - set(feature.keys())
            feature = process_cold_start_feat(feature, feat_types, feat_statistics)
            for feat_id in missing_fields:
                feature[feat_id] = feat_default_value[feat_id]
            for feat_id in feat_types['item_emb']:
                if creative_id in mm_emb_dict[feat_id]:
                    feature[feat_id] = mm_emb_dict[feat_id][creative_id]
                else:
                    if feat_id in EMB_DEFAULTS:
                        feature[feat_id] = EMB_DEFAULTS[feat_id]
                    else:
                        feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)

            item_ids.append(item_id)
            creative_ids.append(creative_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = creative_id

    # 保存候选库的embedding和sid
    model.save_item_emb(item_ids, retrieval_ids, features, os.environ.get('EVAL_RESULT_PATH'))
    with open(Path(os.environ.get('EVAL_RESULT_PATH'), "retrive_id2creative_id.json"), "w") as f:
        json.dump(retrieve_id2creative_id, f)
    return retrieve_id2creative_id


def infer():
    args = get_args()
    data_path = os.environ.get('EVAL_DATA_PATH')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    user_cache = os.environ.get('USER_CACHE_PATH')
    
    # 使用优化的数据集和数据加载器
    print("🚀 Using OptimizedTestDataset for faster inference...")
    test_dataset = OptimizedTestDataset(data_path, user_cache, args)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=8,  # 启用多进程
        collate_fn=test_dataset.collate_fn,
        pin_memory=torch.cuda.is_available(),  # GPU内存固定
        persistent_workers=True,  # 保持worker进程
        prefetch_factor=2  # 预读取因子
    )
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args, writer).to(args.device)
    model.eval()

    ckpt_path = get_ckpt_path()
    ckpt = torch.load(ckpt_path, map_location=torch.device(args.device))


    model.load_state_dict(ckpt)  # ✅ 现在肯定能加载！

    all_embs = []
    user_list = []

    # 记录推理开始时间
    import time
    start_time = time.time()
    temp_time = start_time
    print(f"Inference started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    for step, batch in enumerate(test_loader):
        seq, token_type, seq_feat, user_id = batch
        seq = seq.to(args.device)
        with torch.no_grad():
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.predict(seq, seq_feat, token_type)
        for i in range(logits.shape[0]):
            emb = logits[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
            all_embs.append(emb)
        user_list += user_id
        if step % 1000 == 0:
            print(f"Inference step {step} time: {time.time() - temp_time:.2f}s")
            temp_time = time.time()

    # 输出性能统计
    inference_time = time.time() - start_time
    total_samples = len(user_list)
    samples_per_second = total_samples / inference_time if inference_time > 0 else 0

    # 获取缓存统计
    cache_stats = test_dataset.get_cache_stats()
    
    print(f"\n📊 Inference Performance Stats:")
    print(f"   • Total samples: {total_samples}")
    print(f"   • Inference time: {inference_time:.2f}s") 
    print(f"   • Throughput: {samples_per_second:.1f} samples/sec")
    print(f"   • Cache hit rate: {cache_stats['hit_rate']}")
    print(f"   • Cache size: {cache_stats['cache_size']}")
    print(f"   • DataLoader workers: {test_loader.num_workers}")
    print(f"   • Batch size: {args.batch_size}")

    # 生成候选库的embedding 以及 id文件
    retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        test_dataset.feat_statistics,
        model,
    )
    all_embs = np.concatenate(all_embs, axis=0)
    # 保存query文件
    save_emb(all_embs, Path(os.environ.get('EVAL_RESULT_PATH'), 'query.fbin'))

    del model, test_loader, test_dataset, all_embs
    writer.close()
    del writer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # ANN 检索 - 使用PyTorch CUDA实现替代C++版本
    ann_cmd = [
        "python", str(Path(__file__).parent / "torch_ann.py"),
        "--dataset_vector_file_path", str(Path(os.environ.get("EVAL_RESULT_PATH"), "embedding.fbin")),
        "--dataset_id_file_path", str(Path(os.environ.get("EVAL_RESULT_PATH"), "id.u64bin")),
        "--query_vector_file_path", str(Path(os.environ.get("EVAL_RESULT_PATH"), "query.fbin")),
        "--result_id_file_path", str(Path(os.environ.get("EVAL_RESULT_PATH"), "id100.u64bin")),
        "--query_ann_top_k", "10",
        "--faiss_metric_type", "0",  # 使用内积/余弦相似度
        "--batch_size", "1000"  # 批处理大小，可根据GPU内存调整
    ]
    import subprocess
    result = subprocess.run(ann_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ANN search failed with error: {result.stderr}")
        raise RuntimeError(f"ANN search failed: {result.stderr}")
    else:
        print(f"ANN search completed successfully: {result.stdout}")

    # 取出top-k
    top10s_retrieved = read_result_ids(Path(os.environ.get("EVAL_RESULT_PATH"), "id100.u64bin"))
    top10s_untrimmed = []
    for top10 in tqdm(top10s_retrieved):
        for item in top10:
            top10s_untrimmed.append(retrieve_id2creative_id.get(int(item), 0))

    top10s = [top10s_untrimmed[i : i + 10] for i in range(0, len(top10s_untrimmed), 10)]

    return top10s, user_list


# if __name__ == "__main__":
#     infer()