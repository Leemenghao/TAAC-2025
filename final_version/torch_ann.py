#!/usr/bin/env python3
"""
PyTorch CUDA implementation of ANN search to replace the C++ faiss_demo.
"""

import argparse
import struct
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path


def parse_binary_file(filename, dtype):
    """
    解析二进制文件，支持向量文件(.fbin)和ID文件(.u64bin)
    
    Args:
        filename: 文件路径
        dtype: 数据类型，'float32' 或 'uint64'
        
    Returns:
        data: 解析的数据
        num_points: 点数
        num_dimensions: 维度
    """
    with open(filename, 'rb') as f:
        # 读取文件头
        num_points = struct.unpack('I', f.read(4))[0]  # uint32_t
        num_dimensions = struct.unpack('I', f.read(4))[0]  # uint32_t
        
        print(f"File {filename}: num_points={num_points}, num_dimensions={num_dimensions}")
        
        if dtype == 'float32':
            # 向量文件
            data = np.fromfile(f, dtype=np.float32, count=num_points * num_dimensions)
            data = data.reshape((num_points, num_dimensions))
        elif dtype == 'uint64':
            # ID文件，维度应该为1
            assert num_dimensions == 1, f"ID file should have dimension 1, got {num_dimensions}"
            data = np.fromfile(f, dtype=np.uint64, count=num_points)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
            
        return data, num_points, num_dimensions


def write_result_file(filename, num_queries, top_k, result_ids):
    """
    写入结果文件，格式与C++版本一致
    
    Args:
        filename: 结果文件路径
        num_queries: 查询数量
        top_k: topk数量
        result_ids: 结果ID数组，shape为(num_queries, top_k)
    """
    with open(filename, 'wb') as f:
        # 写入文件头
        f.write(struct.pack('I', num_queries))  # uint32_t
        f.write(struct.pack('I', top_k))       # uint32_t
        
        # 写入结果数据
        result_ids_flat = result_ids.flatten().astype(np.uint64)
        f.write(result_ids_flat.tobytes())


def cosine_similarity_search(query_vectors, dataset_vectors, top_k, device):
    """
    使用余弦相似度进行搜索（对应faiss_metric_type=0的内积）
    """
    # 为避免一次性计算占用过多显存，这里对 dataset_vectors 按块计算
    num_queries = query_vectors.size(0)
    num_dataset = dataset_vectors.size(0)

    chunk_size = 20480

    best_vals = None
    best_indices = None

    for start in range(0, num_dataset, chunk_size):
        end = min(start + chunk_size, num_dataset)
        dataset_chunk = dataset_vectors[start:end]

        # 确保分块与查询在同一设备上
        if dataset_chunk.device != query_vectors.device:
            dataset_chunk = dataset_chunk.to(query_vectors.device, non_blocking=True)

        # 计算当前分块的相似度（向量已归一化，此处即内积）
        sims = torch.mm(query_vectors, dataset_chunk.t())  # [num_queries, chunk]

        # 先在分块内取 top-k，减少后续合并开销
        k_in_chunk = min(top_k, sims.size(1))
        chunk_vals, chunk_idx = torch.topk(sims, k=k_in_chunk, dim=1, largest=True)
        chunk_idx = chunk_idx + start  # 转为全局索引

        if best_vals is None:
            best_vals = chunk_vals
            best_indices = chunk_idx
        else:
            # 合并已有全局 top-k 与当前分块 top-k，再取全局 top-k
            merged_vals = torch.cat([best_vals, chunk_vals], dim=1)
            merged_idx = torch.cat([best_indices, chunk_idx], dim=1)
            best_vals, selector = torch.topk(merged_vals, k=top_k, dim=1, largest=True)
            best_indices = torch.gather(merged_idx, 1, selector)

        # 释放中间结果引用以便显存回收
        del sims

    return best_indices


def l2_distance_search(query_vectors, dataset_vectors, top_k, device):
    """
    使用L2距离进行搜索（对应faiss_metric_type=1）
    """
    # 计算L2距离：||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    query_sq = torch.sum(query_vectors**2, dim=1, keepdim=True)
    dataset_sq = torch.sum(dataset_vectors**2, dim=1, keepdim=True)
    
    # 使用批量矩阵乘法计算距离
    distances = query_sq + dataset_sq.t() - 2 * torch.mm(query_vectors, dataset_vectors.t())
    
    # 找到top-k最小距离的
    _, indices = torch.topk(distances, k=top_k, dim=1, largest=False)
    
    return indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_vector_file_path', required=True, help='数据集向量文件路径')
    parser.add_argument('--dataset_id_file_path', required=True, help='数据集ID文件路径')
    parser.add_argument('--query_vector_file_path', required=True, help='查询向量文件路径')
    parser.add_argument('--result_id_file_path', required=True, help='结果ID文件路径')
    parser.add_argument('--query_ann_top_k', type=int, default=10, help='ANN top-k')
    parser.add_argument('--faiss_metric_type', type=int, default=0, help='距离度量类型，0=内积，1=L2')
    parser.add_argument('--batch_size', type=int, default=2048, help='批处理大小')
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 读取数据集向量
    print("Loading dataset vectors...")
    dataset_vectors, num_dataset, dataset_dim = parse_binary_file(args.dataset_vector_file_path, 'float32')
    
    # 读取数据集ID
    print("Loading dataset IDs...")
    dataset_ids, num_dataset_ids, _ = parse_binary_file(args.dataset_id_file_path, 'uint64')
    
    # 验证数据一致性
    assert num_dataset == num_dataset_ids, f"Dataset size mismatch: vectors={num_dataset}, ids={num_dataset_ids}"
    
    # 读取查询向量
    print("Loading query vectors...")
    query_vectors, num_queries, query_dim = parse_binary_file(args.query_vector_file_path, 'float32')
    
    # 验证维度一致性
    assert dataset_dim == query_dim, f"Dimension mismatch: dataset={dataset_dim}, query={query_dim}"
    
    # 转换为PyTorch张量并移到GPU
    dataset_vectors = torch.from_numpy(dataset_vectors).to('cpu')
    query_vectors = torch.from_numpy(query_vectors).to('cpu')
    
    print(f"Starting ANN search: {num_queries} queries, top_k={args.query_ann_top_k}")
    
    # 创建结果存储
    all_indices = []
    
    # 分批处理查询以节省GPU内存
    batch_size = args.batch_size
    for i in range(0, num_queries, batch_size):
        end_idx = min(i + batch_size, num_queries)
        query_batch = query_vectors[i:end_idx]
        query_batch = query_batch.to(device)
        
        # print(f"Processing queries {i+1}-{end_idx}/{num_queries}")
        
        # 根据距离度量类型选择搜索方法
        if args.faiss_metric_type == 0:  # 内积/余弦相似度
            indices = cosine_similarity_search(query_batch, dataset_vectors, args.query_ann_top_k, device)
        elif args.faiss_metric_type == 1:  # L2距离
            indices = l2_distance_search(query_batch, dataset_vectors, args.query_ann_top_k, device)
        else:
            raise ValueError(f"Unsupported metric type: {args.faiss_metric_type}")
        
        all_indices.append(indices.cpu())
        query_batch.to('cpu')
    
    # 合并所有结果
    all_indices = torch.cat(all_indices, dim=0)
    
    # 将向量索引转换为实际ID
    result_ids = dataset_ids[all_indices.numpy()]
    
    # 创建输出目录
    result_dir = Path(args.result_id_file_path).parent
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 写入结果文件
    print(f"Writing results to {args.result_id_file_path}")
    write_result_file(args.result_id_file_path, num_queries, args.query_ann_top_k, result_ids)
    
    print("ANN search completed successfully!")


if __name__ == "__main__":
    main()
