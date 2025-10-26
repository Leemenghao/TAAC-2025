from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import SwiGLU, SENet, CachedComplexRoPE, SENetBilinearDNN


from dataset import save_emb


class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(FlashMultiHeadAttention, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        # self.dropout_rate = dropout_rate
        self.dropout_rate = 0.0

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        self.qkv_linear = torch.nn.Linear(hidden_units, hidden_units * 3)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)

        self.rope = CachedComplexRoPE(head_dim=self.head_dim, max_seq_len=102, base=10000.0)
    
        
    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V
        qkv = self.qkv_linear(query)
        Q, K, V = qkv.chunk(3, dim=-1)


        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE
        Q = self.rope(Q, seq_dim=-2)
        K = self.rope(K, seq_dim=-2)

        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 使用内置的Flash Attention
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=attn_mask.unsqueeze(1)
            )
        else:
            # 降级到标准注意力机制
            scale = (self.head_dim) ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)

        # reshape回原来的格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)
        # 最终的线性变换并添加LayerNorm，在外部已经做了，所以后续这里可以去掉
        output = self.out_linear(attn_output)

        return output, None


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.liner1 = torch.nn.Linear(hidden_units, hidden_units * 4)
        self.activation = SwiGLU(hidden_units * 4, hidden_units * 4)
        # self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.dropout1 = torch.nn.Dropout(p=0.0)

        self.liner2 = torch.nn.Linear(hidden_units * 4, hidden_units)
        # self.dropout2 = torch.nn.Dropout(p=dropout_rate)
        self.dropout2 = torch.nn.Dropout(p=0.0)
        self.norm = torch.nn.RMSNorm(hidden_units, eps=1e-8)

    def forward(self, inputs):
        residual = inputs
        outputs = self.dropout2(self.liner2(self.dropout1(self.activation(self.liner1(inputs)))))
        outputs = self.norm(outputs + residual)
        return outputs


class BaselineModel(torch.nn.Module):
    """
    Args:
        user_num: 用户数量
        item_num: 物品数量
        feat_statistics: 特征统计信息，key为特征ID，value为特征数量
        feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        args: 全局参数

    Attributes:
        user_num: 用户数量
        item_num: 物品数量
        dev: 设备
        norm_first: 是否先归一化
        maxlen: 序列最大长度
        item_emb: Item Embedding Table
        user_emb: User Embedding Table
        sparse_emb: 稀疏特征Embedding Table
        emb_transform: 多模态特征的线性变换
        userdnn: 用户特征拼接后经过的全连接层
        itemdnn: 物品特征拼接后经过的全连接层
    """

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args, writer):  #
        super(BaselineModel, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen
        self.temperature = args.temperature
        self.writer = writer
        
        # Flash InfoNCE配置 - 保持原有
        self.use_flash_infonce = getattr(args, 'use_flash_infonce', True)
        self.flash_block_size = getattr(args, 'flash_block_size', 1024)
        
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        # self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)

        # self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()

        # SE注意力机制
        self.continual_transform = torch.nn.ModuleDict()

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self._init_feat_info(feat_statistics, feat_types)

        self.user_feat_num = len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT) + len(self.USER_CONTINUAL_FEAT)
        self.item_feat_num = len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT) + len(self.ITEM_CONTINUAL_FEAT) + len(self.ITEM_EMB_FEAT)

        # 特征加权 + 特征交叉 + DNN
        self.userSeBilinearDNN = SENetBilinearDNN(
            self.user_feat_num * args.hidden_units,
            self.user_feat_num,
            args.hidden_units,
            args.hidden_units,
            bilinear_weight=0.1,
            # dropout_rate=args.dropout_rate
            dropout_rate=0.0,
            selected_pairs=((0, 1), (0, 6), (0, 7), (6, 7))
        )
        self.itemSeBilinearDNN = SENetBilinearDNN(
            self.item_feat_num * args.hidden_units,
            self.item_feat_num,
            args.hidden_units,
            args.hidden_units,
            bilinear_weight=0.1,
            # dropout_rate=args.dropout_rate
            dropout_rate=0.0,
            selected_pairs=((0, 3), (0, 11), (0, 15), (3, 11))
        )
        
        self.last_layernorm = torch.nn.RMSNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.RMSNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = FlashMultiHeadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate
            )  # 优化：用FlashAttention替代标准Attention
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.RMSNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        for k in self.USER_SPARSE_FEAT:
            if k in ['301', '302']:
                self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k], args.hidden_units)
            else:
                self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        # 为多模态特征变换添加LayerNorm,嵌入层后不应该立即跟layernorm
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)

        # 为连续特征添加变换层，将1维映射到hidden_units维
        for k in self.ITEM_CONTINUAL_FEAT:
            self.continual_transform[k] = torch.nn.Sequential(
                torch.nn.Linear(1, args.hidden_units),
                torch.nn.RMSNorm(args.hidden_units, eps=1e-8),
                SwiGLU(args.hidden_units, args.hidden_units)
            )
        for k in self.USER_CONTINUAL_FEAT:
            self.continual_transform[k] = torch.nn.Sequential(
                torch.nn.Linear(1, args.hidden_units),
                torch.nn.RMSNorm(args.hidden_units, eps=1e-8),
                SwiGLU(args.hidden_units, args.hidden_units)
            )

    def _init_feat_info(self, feat_statistics, feat_types):
        """
        将特征统计信息（特征数量）按特征类型分组产生不同的字典，方便声明稀疏特征的Embedding Table

        Args:
            feat_statistics: 特征统计信息，key为特征ID，value为特征数量
            feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        """
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}  # 记录的是不同多模态特征的维度



    def feat2emb(self, seq, feature_tensors, mask=None, include_user=False):
        """
        优化版本：直接使用预处理好的特征tensor，避免重复转换
        
        Args:
            seq: 序列ID
            feature_tensors: 预处理好的特征tensor字典，key为特征ID，value为tensor
            mask: 掩码，1表示item，2表示user
            include_user: 是否处理用户特征

        Returns:
            seqs_emb: 序列特征的Embedding
        """
        seq = seq.to(self.dev)
        

        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_embedding = self.user_emb(user_mask * seq)
            item_embedding = self.item_emb(item_mask * seq)
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_emb(seq)
            item_feat_list = [item_embedding]


        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list),
        ]

        if include_user:
            all_feat_types.extend([
                (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
            ])

        # 直接使用预处理好的tensor进行embedding
        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue

            for k in feat_dict:
                if k not in feature_tensors:
                    continue
                    
                tensor_feature = feature_tensors[k].to(self.dev)

                if feat_type.endswith('sparse'):
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                elif feat_type.endswith('array'):
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                elif feat_type.endswith('continual'):
                    # feat_list.append(tensor_feature)
                    feat_list.append(self.continual_transform[k](tensor_feature))

                del tensor_feature


        # 分块处理多模态嵌入特征 - 更激进的内存管理
        for k in self.ITEM_EMB_FEAT:
            if k in feature_tensors:  
                # 获取CPU上的tensor信息，但不立即转移到GPU
                tensor_feature_cpu = feature_tensors[k]
                batch_size, seq_len, emb_dim = tensor_feature_cpu.shape
                # print(f"    🔧 Processing multimodal {k}: {tensor_feature_cpu.shape} = {tensor_feature_cpu.numel()*4/(1024**3):.3f}GB")
                
                # 极小的chunk_size，逐块转移和处理
                chunk_size = 8  # 一次只处理一个位置
                transformed_chunks = []
                
                for start_idx in range(0, seq_len, chunk_size):
                    end_idx = min(start_idx + chunk_size, seq_len)
                    
                    # 只转移当前需要的小块到GPU
                    chunk_cpu = tensor_feature_cpu[:, start_idx:end_idx, :]
                    chunk_gpu = chunk_cpu.to(self.dev)
                    
                    try:
                        with torch.no_grad():
                            transformed_chunk = self.emb_transform[k](chunk_gpu)
                        transformed_chunks.append(transformed_chunk)
                    except RuntimeError as e:
                        print(f"    ❌ Chunk {start_idx}-{end_idx} failed: {e}")
                        # 失败时创建零张量
                        zero_chunk = torch.zeros(
                            batch_size, end_idx - start_idx, self.emb_transform[k].out_features,
                            device=self.dev, dtype=chunk_gpu.dtype
                        )
                        transformed_chunks.append(zero_chunk)
                    
                    # 立即释放chunk
                    del chunk_cpu, chunk_gpu
                    
                    # 每个chunk都清理显存
                    torch.cuda.empty_cache()
                
                if transformed_chunks:
                    transformed_emb = torch.cat(transformed_chunks, dim=1)
                    item_feat_list.append(transformed_emb)
                    del transformed_chunks, transformed_emb
                
                torch.cuda.empty_cache()

        # for k in self.ITEM_EMB_FEAT:
        #     if k in feature_tensors:
        #         tensor_feature = feature_tensors[k].to(self.dev)
        #         transformed_emb = self.emb_transform[k](tensor_feature)
        #         item_feat_list.append(transformed_emb)

        all_item_emb = self.itemSeBilinearDNN(torch.stack(item_feat_list, dim=2))
        # all_item_emb = 0
        #small_chunk=8
        #for i in range(0, len(item_feat_list), small_chunk):
        #    chunk_emb = torch.stack(item_feat_list[i:i+small_chunk], dim=2).to(self.dev)
        #    all_item_emb += self.itemSeBilinearDNN(chunk_emb)
        #    del chunk_emb
        #    torch.cuda.empty_cache()
        # all_item_emb = transformed_sum
        
        if include_user:
            # SE注意力
            all_user_emb = self.userSeBilinearDNN(torch.stack(user_feat_list, dim=2))
            #all_user_emb = 0
            #small_chunk=8
            #for i in range(0, len(item_feat_list), small_chunk):
            #    chunk_emb = torch.stack(item_feat_list[i:i+small_chunk], dim=2).to(self.dev)
            #    all_user_emb += self.itemSeBilinearDNN(chunk_emb)
            #    del chunk_emb
            #    torch.cuda.empty_cache()
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb
            
        return seqs_emb

    def log2feats(self, log_seqs, mask, seq_feature_tensors):
        """
        Args:
            log_seqs: 序列ID
            mask: token类型掩码，1表示item token，2表示user token
            seq_feature_tensors: 预处理好的序列特征tensor字典

        Returns:
            seqs_emb: 序列的Embedding，形状为 [batch_size, maxlen, hidden_units]
        """
        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]
        seqs = self.feat2emb(log_seqs, seq_feature_tensors, mask=mask, include_user=True)
        seqs *= self.item_emb.embedding_dim**0.5

        maxlen = seqs.shape[1]
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        for i in range(len(self.attention_layers)):
            if self.norm_first:
                # Pre-Norm架构：先归一化再处理
                # 1. 自注意力
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                # Post-Norm架构：先处理再归一化
                # 1. 自注意力
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)

        return log_feats
    
    def predict(self, log_seqs, seq_feature_tensors, mask):
        """
        计算用户序列的表征
        Args:
            log_seqs: 用户序列ID
            seq_feature_tensors: 预处理好的序列特征tensor字典
            mask: token类型掩码，1表示item token，2表示user token
        Returns:
            final_feat: 用户序列的表征，形状为 [batch_size, hidden_units]
        """
        log_feats = self.log2feats(log_seqs, mask, seq_feature_tensors)

        final_feat = log_feats[:, -1, :]
        
        # 推理时也进行L2归一化，与训练保持一致
        final_feat = F.normalize(final_feat, dim=-1)

        return final_feat

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        """
        生成候选库item embedding，用于检索

        Args:
            item_ids: 候选item ID（re-id形式）
            retrieval_ids: 候选item ID（检索ID，从0开始编号，检索脚本使用）
            feat_dict: 训练集所有item特征字典，key为特征ID，value为特征值
            save_path: 保存路径
            batch_size: 批次大小
        """
        all_embs = []

        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))

            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            batch_feat = []
            for i in range(start_idx, end_idx):
                batch_feat.append(feat_dict[i])
            # print(f"batch_feat :{batch_feat}")

            # 需要创建一个临时的特征处理器来转换特征
            # 这里我们需要模拟dataset的特征预处理过程
            feature_batch = [batch_feat]  # 包装成batch格式
            # print(f"feature_batch : {len(feature_batch[0])}")
            feature_tensors = self._preprocess_item_features_for_inference(feature_batch, len(item_ids[start_idx:end_idx]))


            batch_emb = self.feat2emb(item_seq, feature_tensors, include_user=False).squeeze(0)
            # batch_emb = batch_emb / batch_emb.norm(dim=-1, keepdim=True)
            batch_emb = F.normalize(batch_emb, dim=-1)

            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        # 合并所有批次的结果并保存
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))

    def _preprocess_item_features_for_inference(self, feature_batch, seq_len):
        """
        为推理阶段预处理物品特征，简化版本的特征转换
        
        Args:
            feature_batch: 特征batch，形状为 [1, seq_len]
            seq_len: 序列长度
            
        Returns:
            feature_tensors: 预处理好的特征tensor字典
        """
        batch_size = 1  # 推理时batch_size固定为1
        feature_tensors = {}
        
        # 处理物品特征
        for feat_id in self.ITEM_SPARSE_FEAT:
            batch_data = np.zeros((batch_size, seq_len), dtype=np.int64)
            for j, item_feat in enumerate(feature_batch[0]):
                batch_data[0, j] = item_feat.get(feat_id, 0)
            feature_tensors[feat_id] = torch.from_numpy(batch_data)

        # print(feature_tensors)
        
        for feat_id in self.ITEM_ARRAY_FEAT:
            max_array_len = 1
            # 计算最大数组长度
            for item_feat in feature_batch[0]:
                if feat_id in item_feat and isinstance(item_feat[feat_id], list):
                    max_array_len = max(max_array_len, len(item_feat[feat_id]))
            
            batch_data = np.zeros((batch_size, seq_len, max_array_len), dtype=np.int64)
            for j, item_feat in enumerate(feature_batch[0]):
                if feat_id in item_feat:
                    item_data = item_feat[feat_id]
                    if isinstance(item_data, list):
                        actual_len = min(len(item_data), max_array_len)
                        batch_data[0, j, :actual_len] = item_data[:actual_len]
                    else:
                        batch_data[0, j, 0] = item_data
            feature_tensors[feat_id] = torch.from_numpy(batch_data)
        
        for feat_id in self.ITEM_CONTINUAL_FEAT:
            batch_data = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
            for j, item_feat in enumerate(feature_batch[0]):
                batch_data[0, j, 0] = float(item_feat.get(feat_id, 0))
            feature_tensors[feat_id] = torch.from_numpy(batch_data)
        
        for feat_id in self.ITEM_EMB_FEAT:
            emb_dim = self.ITEM_EMB_FEAT[feat_id]
            batch_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)
            for j, item_feat in enumerate(feature_batch[0]):
                if feat_id in item_feat and isinstance(item_feat[feat_id], np.ndarray):
                    batch_data[0, j] = item_feat[feat_id]
            feature_tensors[feat_id] = torch.from_numpy(batch_data)
        
        return feature_tensors

    def compute_flash_infonce_loss(self, seq_embs, pos_embs, neg_embs, pop_neg_embs, loss_mask, global_step, block_size=512):
        """
        Flash InfoNCE: 基于Flash Attention思想的内存高效InfoNCE实现
        """
        
        # 1. 归一化嵌入
        seq_embs = F.normalize(seq_embs, dim=-1)
        pos_embs = F.normalize(pos_embs, dim=-1) 
        neg_embs = F.normalize(neg_embs, dim=-1)
        pop_neg_embs = F.normalize(pop_neg_embs, dim=-1)
        # 2. 筛选有效位置
        valid_mask = loss_mask.bool()
        seq_embs_valid = seq_embs[valid_mask]  # [N, H]
        pos_embs_valid = pos_embs[valid_mask]  # [N, H] 
        
        if seq_embs_valid.size(0) == 0:
            return torch.tensor(0.0, device=seq_embs.device, requires_grad=True)
        
        # 3. 计算正样本相似度（温度缩放前，用于统计）
        pos_similarities = F.cosine_similarity(seq_embs_valid, pos_embs_valid, dim=-1)  # [N] 范围[-1,1]
        pos_logits = pos_similarities / self.temperature  # 温度缩放后用于损失计算
        
        # 记录原始相似度（范围[-1,1]）
        self.writer.add_scalar('Model/flash_nce_pos_logits', pos_similarities.mean().item(), global_step)
        
        # 4. 收集负样本相似度用于统计
        _, _, hidden_size = neg_embs.shape
        neg_embs_flat = neg_embs.view(-1, hidden_size)  # [B*L, H] 已经归一化过了
        neg_embs_flat = torch.cat([neg_embs_flat, pop_neg_embs.view(-1, hidden_size)], dim=0)
        total_neg_samples = neg_embs_flat.size(0)

        
        # 初始化在线Softmax统计量
        max_logits = pos_logits.clone()  # [N] 
        sum_exp = torch.exp(pos_logits - max_logits)  # [N]
        
        # 🆕 用于统计的负样本相似度收集
        neg_similarities_for_stats = []
        
        # 5. 分块处理所有负样本
        for start_idx in range(0, total_neg_samples, block_size):
            end_idx = min(start_idx + block_size, total_neg_samples)
            
            # 当前块的负样本嵌入（已经归一化，不要重复归一化）
            neg_block = neg_embs_flat[start_idx:end_idx]  # [block_size, H]
            
            # 计算当前块的相似度 [N, block_size] 范围[-1,1]
            block_similarities = torch.matmul(seq_embs_valid, neg_block.transpose(-1, -2))
            block_logits = block_similarities / self.temperature  # 温度缩放后
            
            # 🆕 收集少量负样本相似度用于统计（节省内存）
            if start_idx == 0:  # 只收集第一个块的数据用于统计
                neg_similarities_for_stats = block_similarities.detach()
            
            # 在线更新Softmax统计量
            block_max = torch.max(block_logits, dim=-1)[0]  # [N]
            new_max = torch.max(max_logits, block_max)  # [N]
            
            # 重新缩放并更新
            sum_exp = sum_exp * torch.exp(max_logits - new_max)
            block_exp_sum = torch.sum(torch.exp(block_logits - new_max.unsqueeze(-1)), dim=-1)  # [N]
            sum_exp = sum_exp + block_exp_sum
            max_logits = new_max
        
        # 6. 计算最终的InfoNCE损失
        log_prob = pos_logits - max_logits - torch.log(sum_exp)
        loss = -log_prob.mean()
        
        # 7. 正确记录统计信息
        with torch.no_grad():
            # 记录负样本的真实相似度平均值（范围[-1,1]）
            avg_neg_similarities = neg_similarities_for_stats.mean().item()
            self.writer.add_scalar('Model/flash_nce_neg_logits', avg_neg_similarities, global_step)
            self.writer.add_scalar('Model/flash_nce_max_logits', max_logits.mean().item(), global_step)
        
        return loss

    def compute_infonce_loss(self, seq_embs, pos_embs, neg_embs, pop_neg_embs, loss_mask, global_step):
        """
        InfoNCE损失计算，支持Flash和原始两种实现
        """
        # 检查是否使用Flash版本
        if self.use_flash_infonce:
            return self.compute_flash_infonce_loss(
                seq_embs, pos_embs, neg_embs, pop_neg_embs, loss_mask, global_step, self.flash_block_size
            )
        
        # 原始实现保持不变
        hidden_size = neg_embs.size(-1)
        seq_embs = F.normalize(seq_embs, dim=-1)
        pos_embs = F.normalize(pos_embs, dim=-1)
        pos_logits = F.cosine_similarity(seq_embs, pos_embs, dim=-1).unsqueeze(-1)
        self.writer.add_scalar('Model/nce_pos_logits', pos_logits.mean().item(), global_step)

        neg_embs = F.normalize(neg_embs, dim=-1)
        pop_neg_embs = F.normalize(pop_neg_embs, dim=-1)
        neg_embedding_all = neg_embs.reshape(-1, hidden_size)
        neg_embedding_all = torch.cat([neg_embedding_all, pop_neg_embs.reshape(-1, hidden_size)], dim=0)
        neg_logits = torch.matmul(seq_embs, neg_embedding_all.transpose(-1, -2))
        self.writer.add_scalar('Model/nce_neg_logits', neg_logits.mean().item(), global_step)

        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits = logits[loss_mask.bool()] / self.temperature
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.int64)
        loss = F.cross_entropy(logits, labels)

        return loss.mean()
    
    def forward_infonce(
        self, user_item, pos_seqs, neg_seqs, pop_neg_seqs, mask, next_mask, next_action_type, 
        seq_feature_tensors, pos_feature_tensors, neg_feature_tensors, pop_neg_feature_tensors, global_step
    ):
        """
        使用InfoNCE损失的前向传播函数

        Args:
            user_item: 用户序列ID
            pos_seqs: 正样本序列ID  
            neg_seqs: 负样本序列ID
            pop_neg_seqs: 流行度负样本序列ID
            mask: token类型掩码，1表示item token，2表示user token
            next_mask: 下一个token类型掩码，1表示item token，2表示user token
            next_action_type: 下一个token动作类型，0表示曝光，1表示点击
            seq_feature_tensors: 预处理好的序列特征tensor字典
            pos_feature_tensors: 预处理好的正样本特征tensor字典
            neg_feature_tensors: 预处理好的负样本特征tensor字典
            pop_neg_feature_tensors: 预处理好的流行度负样本特征tensor字典
        Returns:
            infonce_loss: InfoNCE损失值
        """
        # 1. 获取序列表示
        seq_embs = self.log2feats(user_item, mask, seq_feature_tensors)
        loss_mask = (next_mask == 1).to(self.dev)
        
        # 2. 获取正负样本嵌入
        pos_embs = self.feat2emb(pos_seqs, pos_feature_tensors, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature_tensors, include_user=False)
        pop_neg_embs = self.feat2emb(pop_neg_seqs, pop_neg_feature_tensors, include_user=False)
        # 3. 计算InfoNCE损失
        infonce_loss = self.compute_infonce_loss(seq_embs, pos_embs, neg_embs, pop_neg_embs, loss_mask, global_step)
        
        return infonce_loss


   