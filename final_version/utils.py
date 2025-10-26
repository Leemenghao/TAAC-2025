import torch
import torch.nn.functional as F
import json
import pickle
import numpy as np
from pathlib import Path
import os
import logging
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Subset
import random
from torch.amp import autocast
import traceback
from typing import Dict

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class SwiGLU(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # SwiGLU激活函数：使用两个线性变换和SiLU门控
        self.w1 = torch.nn.Linear(in_features, out_features, bias=True)
        self.w2 = torch.nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        # SwiGLU(x) = W1(x) * SiLU(W2(x))
        return self.w1(x) * F.silu(self.w2(x))

class SENet(nn.Module):
    def __init__(self, num_features, reducation=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(num_features, num_features // reducation),
            nn.ReLU(),
            nn.Linear(num_features // reducation, num_features),
            nn.Sigmoid()
        )
    
    def forward(self, x):  # [B, L, N, H]
        # 只在H维度上池化，保持更多信息
        pooled = x.mean(dim=-1)  # [B, L, N]
        
        # 通过SE模块计算特征权重
        weights = self.se(pooled)  # [B, L, N]
        weights = weights.unsqueeze(-1)  # [B, L, N, 1]
        
        return x * weights

class BilinearInteraction(nn.Module):
    def __init__(self, num_fields, embedding_dim, selected_pairs=None):
        super().__init__()        
        self.W = nn.Parameter(torch.empty(embedding_dim, embedding_dim))
        nn.init.xavier_uniform_(self.W)
        
        if selected_pairs is None:
            # 默认：全部交叉
            i_idx, j_idx = torch.triu_indices(num_fields, num_fields, offset=1)
        else:
            # 只取用户指定的交叉
            i_idx, j_idx = zip(*selected_pairs)
            i_idx = torch.tensor(i_idx, dtype=torch.long)
            j_idx = torch.tensor(j_idx, dtype=torch.long)
        # 预计算索引（避免每次 forward 都计算）
        self.register_buffer('i_idx', i_idx)
        self.register_buffer('j_idx', j_idx)

    def forward(self, x):  # x: [B, L, N, H]
        B, L, N, H = x.shape

        x_ = x.view(B*L, N, H)             # [B*L, N, H]
        Wx = x_ @ self.W                        # [B*L, N, H]

        vi = Wx[:, self.i_idx, :]               # [B*L, num_pairs, H]
        vj = x_[:, self.j_idx, :]               # [B*L, num_pairs, H]

        interactions = torch.einsum("bnh,bnh->bn", vi, vj).view(B, L, -1)

        return interactions

class SENetBilinearDNN(nn.Module):
    def __init__(self, input_dim, num_fields, hidden_size, dnn_hidden_units, 
                bilinear_weight=0.1, dropout_rate=0.1, selected_pairs=None):
        super().__init__()

        # 计算总输入维度
        self.input_dim = input_dim
        if selected_pairs is None:
            self.bilinear_dim = num_fields * (num_fields - 1) // 2
        else:
            self.bilinear_dim = len(selected_pairs)
        self.dropout_rate = dropout_rate

        self.se = SENet(num_fields)
        self.bilinear = BilinearInteraction(num_fields, hidden_size, selected_pairs)
        self.dnn = nn.Sequential(
            nn.Linear(self.input_dim + self.bilinear_dim, dnn_hidden_units),
            SwiGLU(dnn_hidden_units, dnn_hidden_units),
        )

        self.bilinear_weight = bilinear_weight
        # self.residual_proj = nn.Linear(self.input_dim, dnn_hidden_units)
        # self.norm = nn.RMSNorm(dnn_hidden_units)

    def forward(self, x):  # [B, L, N, H]
        # Step 1: SENet 重标定
        x = self.se(x)  # [B, L, N, H]
        
        # Step 2: Bilinear 交叉
        cross_feats = self.bilinear(x)  # [B, L, N*(N-1)/2]
        if self.dropout_rate > 0:
            cross_feats = F.dropout(cross_feats, self.dropout_rate, training=self.training)
        cross_feats = cross_feats * self.bilinear_weight  # 减弱Bilinear影响
        
        # Step 3: 拼接 + DNN
        x_flat = x.view(x.size(0), x.size(1), -1).contiguous()  # [B, L, N*H]
        combined = torch.cat([x_flat, cross_feats], dim=-1)  # [B, L, N*H + N*(N-1)/2]
        output = self.dnn(combined)  # [B, L, D]

        # Step 4: 残差连接
        # output = self.norm(output + self.residual_proj(x_flat))  # [B, L, D]

        return output


def optimized_embedding_init(model, args):
    """优化的embedding初始化策略"""
    
    # 1. 核心ID embedding
    item_std_core = np.sqrt(2.0 / args.hidden_units)
    user_std_core = np.sqrt(2.0 / args.hidden_units)
    torch.nn.init.normal_(model.item_emb.weight.data, std=item_std_core)
    torch.nn.init.normal_(model.user_emb.weight.data, std=user_std_core)
    
    # 2. 位置embedding - 使用较小的初始化
    # torch.nn.init.normal_(model.pos_emb.weight.data, std=0.02)
    
    # 3. 稀疏特征embedding - 改进的频率感知初始化
    for k, emb_layer in model.sparse_emb.items():
        vocab_size = emb_layer.num_embeddings
        
        # 🔧 改进的初始化策略：避免小词汇表标准差过小
        if vocab_size <= 10:  # 小词汇表特征
            std = 0.05  # 固定使用较大的标准差，保证表达能力
        elif vocab_size <= 100:  # 中等词汇表特征
            std = 0.08
        else:  # 大词汇表特征
            std = min(0.1, np.sqrt(2.0 / (vocab_size + args.hidden_units)))
            
        # 确保最小标准差不小于0.02
        std = max(std, 0.02)
        
        torch.nn.init.normal_(emb_layer.weight.data, std=std)
    
    # 4. 多模态特征变换层
    for transform_layer in model.emb_transform.values():
        torch.nn.init.xavier_uniform_(transform_layer.weight.data, gain=0.1)
        if transform_layer.bias is not None:
            torch.nn.init.zeros_(transform_layer.bias.data)
    
    # 5. Padding token处理
    # model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0  
    model.user_emb.weight.data[0, :] = 0
    for emb_layer in model.sparse_emb.values():
        emb_layer.weight.data[0, :] = 0


def validate_initialization(model):
    """验证初始化是否合理"""
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            # 检查是否有异常值
            std = param.std().item()
            mean = param.mean().item()
            print(f"{name}: mean={mean:.6f}, std={std:.6f}")
            
            # 警告异常情况
            if std > 1.0 or std < 1e-4:
                print(f"Warning: {name} has unusual std: {std}")


def split_dataset(dataset, train_ratio=0.9, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    split = int(len(indices) * train_ratio)
    train_idx, valid_idx = indices[:split], indices[split:]
    return train_idx, valid_idx

class CachedComplexRoPE(nn.Module):
    """高精度缓存复数旋转因子的RoPE实现"""
    
    def __init__(self, head_dim, max_seq_len=8192, base=10000.0):
        """
        Args:
            head_dim: 头维度
            max_seq_len: 最大序列长度
            base: 位置编码基数
        """
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 根据精度等级设置计算精度
        self.compute_dtype = torch.float64
        
        # 预计算并缓存高精度复数旋转因子
        self._precompute_complex_rotations()
    
    def _precompute_complex_rotations(self):
        """预计算高精度复数旋转因子"""
        assert self.head_dim % 2 == 0
        
        # 使用高精度计算
        device = torch.device('cpu')  # 在CPU上预计算以避免GPU内存问题
        
        # 计算频率 (使用更高精度)
        freq_indices = torch.arange(0, self.head_dim, 2, dtype=self.compute_dtype)
        freqs = 1.0 / (self.base ** (freq_indices / self.head_dim))
        
        positions = torch.arange(self.max_seq_len, dtype=self.compute_dtype)
        angles = torch.outer(positions, freqs)  # [max_seq_len, head_dim//2]
        
        # 预计算复数旋转因子 e^(i*angles) - 使用更高精度
        ones = torch.ones_like(angles)
        rotate_complex = torch.polar(ones, angles)
        
        # 注册为buffer，但保持高精度
        self.register_buffer('rotate_complex_cached_high', rotate_complex, persistent=False)
    
    def forward(self, x, seq_dim=-2):
        """高精度复数RoPE前向传播"""
        seq_len = x.size(seq_dim)
        device = x.device
        x = x.to(torch.float32)
        original_dtype = x.dtype
        
        # 获取缓存的高精度旋转因子并转移到输入设备
        rotate_complex = self.rotate_complex_cached_high[:seq_len].to(device)  # [seq_len, head_dim//2]
        
        rotate_complex = rotate_complex.to(torch.complex64)
        
        # 调整维度以匹配输入张量
        while rotate_complex.dim() < x.dim():
            rotate_complex = rotate_complex.unsqueeze(0)
        
        # 将输入转为复数形式
        x_reshaped = x.view(*x.shape[:-1], self.head_dim // 2, 2)
        x_complex = torch.view_as_complex(x_reshaped)
        
        # 复数旋转（保持精度）
        with autocast(device_type="cuda", enabled=False):  # ← 禁用 autocast，避免 complex32
            rotated_complex = x_complex * rotate_complex  # 安全地用 complex64 计算
        
        # 转回实数
        result = torch.view_as_real(rotated_complex).view_as(x)
        
        return result.to(original_dtype)