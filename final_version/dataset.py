import json
import pickle
import struct
from pathlib import Path
import copy
import random
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading
import multiprocessing
import os
import time
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Subset
import hashlib


class UserFeatureManager:
    """用户特征管理器 - 专门处理用户相关特征"""
    
    def __init__(self, indexer, feature_types):
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.user_feature_types = {
            'sparse': feature_types.get('user_sparse', []),
            'array': feature_types.get('user_array', []),
            'continual': feature_types.get('user_continual', [])
        }
        self._user_cache = {}
        self._cache_lock = threading.RLock()
        
    def get_user_features(self, user_id, user_feat_dict):
        """获取用户特征，带缓存"""
        if user_id == 0 or user_feat_dict is None:
            return {}
            
        # 检查缓存
        cache_key = user_id
        with self._cache_lock:
            if cache_key in self._user_cache:
                return self._user_cache[cache_key]
        
        # 处理用户特征
        processed_feat = {}
        for feat_id, feat_value in user_feat_dict.items():
            if feat_id in self.user_feature_types['sparse']:
                processed_feat[feat_id] = feat_value
            elif feat_id in self.user_feature_types['array']:
                processed_feat[feat_id] = feat_value if isinstance(feat_value, list) else [feat_value]
            elif feat_id in self.user_feature_types['continual']:
                processed_feat[feat_id] = float(feat_value)
        
        # 缓存结果
        with self._cache_lock:
            if len(self._user_cache) < 5000:  # 限制缓存大小
                self._user_cache[cache_key] = processed_feat
        
        return processed_feat


class ItemFeatureManager:
    """物品特征管理器 - 专门处理物品相关特征"""
    
    def __init__(self, data_dir, user_cache_dir, indexer, feature_types, mm_emb_ids):
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.item_feature_types = {
            'sparse': feature_types.get('item_sparse', []),
            'array': feature_types.get('item_array', []),
            'continual': feature_types.get('item_continual', []),
            'emb': feature_types.get('item_emb', [])
        }
        
        # 尝试加载高效特征加载器
        self._load_item_features(data_dir, user_cache_dir)
        
        # 加载多模态嵌入
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), mm_emb_ids)
        
        # 特征缓存
        self._item_cache = {}
        self._cache_lock = threading.RLock()
        
    def _load_item_features(self, data_dir, user_cache_dir):
        """加载物品特征数据"""
        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        self.use_efficient_loader = False
        print("ItemFeatureManager: Using original feature loader")

    def get_item_features(self, item_id, item_feat_dict=None):
        """获取物品特征，带缓存和多模态嵌入"""
        if item_id == 0:
            return {}
        
        # 检查缓存
        cache_key = item_id
        with self._cache_lock:
            if cache_key in self._item_cache:
                return self._item_cache[cache_key]
        
        # 获取基础特征
        if item_feat_dict is not None:
            # 使用传入的特征字典
            base_features = item_feat_dict
        elif self.use_efficient_loader:
            base_features = self.efficient_loader.get_item_features(item_id)
        else:
            base_features = self.item_feat_dict.get(str(item_id), {})
        
        # 处理物品特征
        processed_feat = {}
        for feat_id, feat_value in base_features.items():
            if feat_id in self.item_feature_types['sparse']:
                processed_feat[feat_id] = feat_value
            elif feat_id in self.item_feature_types['array']:
                processed_feat[feat_id] = feat_value if isinstance(feat_value, list) else [feat_value]
            elif feat_id in self.item_feature_types['continual']:
                processed_feat[feat_id] = float(feat_value)
        
        # 添加多模态嵌入
        for feat_id in self.item_feature_types['emb']:
            if item_id in self.indexer_i_rev:
                item_key = self.indexer_i_rev[item_id]
                if item_key in self.mm_emb_dict[feat_id]:
                    emb_value = self.mm_emb_dict[feat_id][item_key]
                    if isinstance(emb_value, np.ndarray):
                        processed_feat[feat_id] = emb_value
        
        # 缓存结果
        with self._cache_lock:
            if len(self._item_cache) < 10000:  # 限制缓存大小
                self._item_cache[cache_key] = processed_feat
        
        return processed_feat
    
    def get_valid_item_ids(self):
        """获取所有有效的物品ID"""
        if self.use_efficient_loader:
            # 从efficient loader获取
            valid_items = []
            for item_id in range(1, len(self.indexer_i_rev) + 1):
                if item_id in self.efficient_loader:
                    valid_items.append(item_id)
            return valid_items
        else:
            # 从原始字典获取
            return [int(item_id_str) for item_id_str in self.item_feat_dict.keys() 
                   if item_id_str.isdigit()]


class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

    def __init__(self, data_dir, user_cache_dir, args):
        """
        初始化数据集
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.user_cache_dir = Path(user_cache_dir)
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id
        
        # 初始化缓存和锁
        self._user_data_cache = {}
        self._cache_lock = threading.RLock()
        self._max_cache_size = getattr(args, 'user_cache_size', 10000)
        
        # 加载基础数据
        self._load_data_and_offsets()
        self._load_indexer()
        
        # 先初始化基础特征信息（不包含嵌入维度）
        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_basic_feat_info()
        
        # 初始化分离的特征管理器
        self.user_manager = UserFeatureManager(self.indexer, self.feature_types)
        self.item_manager = ItemFeatureManager(
            data_dir, user_cache_dir, self.indexer, self.feature_types, self.mm_emb_ids
        )
        
        # 完成特征信息初始化（包含嵌入维度）
        self._complete_feat_info()
        
        # 预计算负采样候选集
        self._precompute_negative_candidates()
        
        # 预分配数组模板
        self._init_array_templates()
        
        # 添加worker初始化标志
        self._worker_initialized = False

        # 初始化时间差分桶（可选，也可以在首次使用时懒加载）
        if getattr(args, 'preprocess_time_diff', True):
            num_bins = getattr(args, 'time_diff_bins', 19)
            self._cached_bin_edges = self._preprocess_time_diff_bins(num_bins)
        
        # 构建物品池
        self.all_items = np.arange(self.itemnum)

        # 计算流行度
        self.item_popularity = self._compute_item_popularity()
        self.popularity_items = np.array(list(self.item_popularity.keys()), dtype=np.int32)

        self.popularity_weights = np.array(list(self.item_popularity.values()))

    # 流行度采样
    def _compute_item_popularity(self, alpha=0.5):
        """
        第一个函数：从数据集中获取item曝光数据，计算每个item的流行度，并持久化到user_cache中
        
        Returns:
            dict: {item_id: popularity_score} 流行度字典
        """
        popularity_file = self.user_cache_dir / 'item_popularity.pkl'
        
        if popularity_file.exists():
            with open(popularity_file, 'rb') as f:
                item_popularity = pickle.load(f)
        else:
            print("开始计算item流行度...")
            all_item_ids = []
            
            # 遍历所有用户数据，统计item曝光次数
            for uid in range(len(self.seq_offsets)):
                user_sequence = self._load_user_data(uid)
                
                for record_tuple in user_sequence:
                    _, i, _, _, _, _ = record_tuple
                    if i and isinstance(i, int) and i > 0:
                        all_item_ids.append(i)
            
            all_item_ids = np.array(all_item_ids, dtype=np.int64)
            counts = np.bincount(all_item_ids, minlength=self.itemnum+1)

            # 幂次缩放 (alpha=0.5 类似 word2vec 负采样)
            counts = counts ** alpha
            total = counts.sum()
            probs = counts / total

            # 计算流行度概率
            item_popularity = {item_id: p for item_id, p in enumerate(probs) if p > 0}
            
            # 持久化到user_cache
            with open(popularity_file, 'wb') as f:
                pickle.dump(item_popularity, f)
            print(f"流行度数据已保存到: {popularity_file}")
        
        return item_popularity
    
    def _popularity_sampling(self, num_negatives=40 * 102):
        """
        第二个函数：基于流行度采样负样本
        
        Args:
            num_negatives: 采样的负样本数量，默认5000
            
        Returns:
            tuple: (negative_item_ids, negative_features) 负样本ID列表和特征列表
        """
        popularity_file = self.user_cache_dir / 'item_popularity.pkl'
        
        if self.item_popularity is None:
            # 检查是否存在持久化文件
            if popularity_file.exists():
                with open(popularity_file, 'rb') as f:
                    self.item_popularity = pickle.load(f)
            else:
                self.item_popularity = self._compute_item_popularity()
        
            # 基于流行度进行加权采样
            self.popularity_items = np.array(list(self.item_popularity.keys()), dtype=np.int32)

            self.popularity_weights = np.array(list(self.item_popularity.values()))

        
        # 基于流行度加权采样
        sampled_items = np.random.choice(
            self.popularity_items, 
            size=num_negatives, 
            replace=True, 
            p=self.popularity_weights
        )
        
        # 获取采样item的特征
        negative_features = []
        temp_feat = []
        ii = 0
        for item_id in sampled_items:
            item_feat = self.item_manager.get_item_features(item_id)
            item_feat = self.fill_missing_feat(item_feat, item_id, is_user=False)
            temp_feat.append(item_feat)
            ii += 1
            if ii % 102 == 0:
                negative_features.append(temp_feat)
                temp_feat = []
        negative_features.append(temp_feat)

        # print(f"成功采样{len(sampled_items)}个负样本")
        
        return sampled_items.reshape(-1, 102), negative_features

    def _set_train_indices(self, train_indices):
        """
        设置训练集索引
        Args:
            train_indices: 训练集索引
        """
        self.train_indices = train_indices

    def _preprocess_time_diff_bins(self, num_bins=19, force_rebuild=False):
        """
        预处理所有数据集中的时间差，计算并缓存分界点
        
        Args:
            num_bins: 分桶数量，默认19
            force_rebuild: 是否强制重新计算，默认False
            
        Returns:
            bin_edges: 分桶边界数组
        """
        cache_file = self.user_cache_dir / f"time_diff_bins_{num_bins}.pkl"
        
        # 检查缓存是否存在且有效
        if not force_rebuild and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    if cached_data.get('num_bins') == num_bins:
                        print(f"Loaded cached time diff bins from {cache_file}")
                        return cached_data['bin_edges']
            except Exception as e:
                print(f"Failed to load cached bins: {e}, rebuilding...")
        
        print(f"Preprocessing time differences for {num_bins} bins...")
        all_time_diffs = []
        
        # 遍历所有用户数据收集时间差（使用seq_offsets的索引范围，避免越界）
        for uid in range(len(self.seq_offsets)):
            try:
                user_sequence = self._load_user_data(uid)
                if len(user_sequence) < 2:
                    continue
                    
                # 按时间戳排序
                user_sequence.sort(key=lambda x: x[5])
                timestamps = [record[5] for record in user_sequence]
                
                if len(timestamps) >= 2:
                    ts_arr = np.array(timestamps, dtype=np.int64)
                    time_diff = ts_arr[1:] - ts_arr[:-1]
                    # 过滤掉非正数的时间差
                    valid_diffs = time_diff[time_diff > 0]
                    if len(valid_diffs) > 0:
                        all_time_diffs.extend(valid_diffs)
                        
            except Exception as e:
                print(f"Error processing user {uid}: {e}")
                continue
        
        if len(all_time_diffs) == 0:
            print("No valid time differences found, using default bins")
            bin_edges = np.linspace(0, 10, num_bins + 1)
        else:
            # 对所有时间差进行log变换
            all_time_diffs = np.array(all_time_diffs)
            time_diff_log = np.log(all_time_diffs + 1)
            
            try:
                # 使用分位数创建等频分桶
                quantiles = np.linspace(0, 1, num_bins + 1)
                bin_edges = np.quantile(time_diff_log, quantiles)
                bin_edges = np.unique(bin_edges)
                
                # 如果unique后桶数不足，用等距分桶
                if len(bin_edges) < num_bins + 1:
                    bin_edges = np.linspace(time_diff_log.min(), time_diff_log.max(), num_bins + 1)
                    print(f"Using equal distance binning due to insufficient unique quantiles")
                    
            except Exception as e:
                print(f"Binning failed: {e}, using equal distance")
                bin_edges = np.linspace(time_diff_log.min(), time_diff_log.max(), num_bins + 1)
        
        # 缓存结果
        try:
            self.user_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_data = {
                'num_bins': num_bins,
                'bin_edges': bin_edges,
                'total_samples': len(all_time_diffs)
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Cached time diff bins to {cache_file}")
        except Exception as e:
            print(f"Failed to cache bins: {e}")
        
        return bin_edges
    
    def bucket_hour_8(self, hour: np.ndarray) -> np.ndarray:
        """
        将 0~23 小时划分为 8 个桶
        返回值: 0~7
        """
        #        0     1     2     3     4     5     6     7
        # 桶名: 深夜  清晨  上午  中午  下午  傍晚  晚上  深夜2
        # 小时: 0-2  3-5   6-8   9-11  12-14 15-17 18-20 21-23

        buckets = np.zeros_like(hour, dtype=np.int32)
        buckets = np.where((hour >= 3) & (hour <= 5), 1, buckets)
        buckets = np.where((hour >= 6) & (hour <= 8), 2, buckets)
        buckets = np.where((hour >= 9) & (hour <= 11), 3, buckets)
        buckets = np.where((hour >= 12) & (hour <= 14), 4, buckets)
        buckets = np.where((hour >= 15) & (hour <= 17), 5, buckets)
        buckets = np.where((hour >= 18) & (hour <= 20), 6, buckets)
        buckets = np.where(hour >= 21, 7, buckets)  # 21-23
        # 0-2 默认为 0
        return buckets

    def _insert_time_diff(self, ext_user_sequence, temp_timestamps):
        """
        将时间差插入到ext_user_sequence中，使用预处理的分界点
        """
        # 使用numpy进行差分，列表不支持直接相减
        if temp_timestamps is None or len(temp_timestamps) < 2:
            return ext_user_sequence
            
        # 获取缓存的分界点
        if not hasattr(self, '_cached_bin_edges'):
            self._cached_bin_edges = self._preprocess_time_diff_bins()
        
        ts_arr = np.array(temp_timestamps, dtype=np.int64)
        prev_ts_arr = np.roll(ts_arr, 1)
        prev_ts_arr[0] = ts_arr[0]
        time_diff = ts_arr - prev_ts_arr
        time_diff[0] = 0
        time_diff[1] = abs(time_diff[1])
        # print(f"time_diff {time_diff} len {len(time_diff)} \n seq-len {len(ext_user_sequence)}\n ts_arr {ts_arr}")
        # 对时间差进行log等频分桶，转化为特征'300'
        # 处理时间差为0或负数的情况，加1避免log(0)
        time_diff_log = np.log(time_diff + 1)
        
        # hour, weekday, month
        ts_utc8 = ts_arr + 8 * 3600
        hours = (ts_utc8 % 86400) // 3600
        hours = self.bucket_hour_8(hours)
        # print(f"hours {hours} len {len(hours)}")
        
        weekdays = ((ts_utc8 // 86400 + 4) % 7).astype(np.int32)
        # months = pd.to_datetime(ts_utc8, unit='s').month.to_numpy()
        # print(f"hours {hours} len {len(hours)} \n weekdays {weekdays} len {len(weekdays)} \n months {months} len {len(months)}")

        # time decay / up
        first_ts = ts_arr[1]
        delta_ts = ts_arr - first_ts
        delta_ts[0] = abs(delta_ts[0])
        # print(f"delta_ts {delta_ts}")
        delta_scaled = np.log1p(delta_ts / 86400)
        

        try:
            # 使用预处理的分界点进行分桶
            bin_edges = self._cached_bin_edges
            num_bins = len(bin_edges) - 1
            
            # 将时间差分配到桶中，桶编号从1开始
            time_diff_bins = np.digitize(time_diff_log, bin_edges[1:-1]) + 1
            # 确保桶编号在1-num_bins范围内
            time_diff_bins = np.clip(time_diff_bins, 1, num_bins)
            
        except Exception:
            # 如果分桶失败，使用默认值1
            time_diff_bins = np.ones(len(time_diff), dtype=int)
        time_diff_bins[0] = 0 # 第一位会分到桶1
        # print(f"time_diff_bins {time_diff_bins}")
        # exit()
        for idx, record in enumerate(ext_user_sequence):
            i, user_feat, item_feat, record_type, action_type = record

            if user_feat is None:
                user_feat = {}
            
            user_feat['300'] = int(time_diff_bins[idx])
            user_feat['301'] = int(hours[idx])
            user_feat['302'] = int(weekdays[idx])
            # user_feat['303'] = int(months[idx])
            user_feat['304'] = float(delta_scaled[idx])
            ext_user_sequence[idx] = (i, user_feat, item_feat, record_type, action_type)
        return ext_user_sequence

    def _transfer_context_feat(self, item_feat, user_feat, feat_list):
            """
            将时间特征合并到item_feat中
            """
            if item_feat is None:
                item_feat = {}
            if user_feat is None:
                user_feat = {}
            
            # 仅合并时间相关的四个特征，避免引入其它用户特征
            for k in feat_list:
                if k in user_feat:
                    item_feat[k] = user_feat[k]
            return item_feat

    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        # 不在初始化时打开文件，而是在需要时打开
        self.data_file_path = self.data_dir / "seq.jsonl"
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)
        
        # 为每个进程创建独立的文件句柄
        self._local_data = threading.local()
    
    def _load_indexer(self):
        """加载索引器"""
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer
        self.indexer['f']['300'] = [i for i in range(1, 21)] # 时间差（log等频分）
        # self.indexer['f']['300'] = [i for i in range(1, 41)] # 时间差（log等频分）
        self.indexer['f']['301'] = [i for i in range(0, 8)] # 分成8个桶
        self.indexer['f']['302'] = [i for i in range(0, 7)] # 星期
        # self.indexer['f']['303'] = [i for i in range(1, 13)] # 月份
    
    def _precompute_negative_candidates(self):
        """预计算负采样候选集"""
        valid_items = self.item_manager.get_valid_item_ids()
        valid_items = [item_id for item_id in valid_items if 1 <= item_id <= self.itemnum]
        self.negative_candidates = np.array(valid_items, dtype=np.int32)
        print(f"Precomputed {len(self.negative_candidates)} negative candidates")
    
    def _init_array_templates(self):
        """初始化数组模板，避免重复分配"""
        self.seq_template = np.zeros([self.maxlen + 1], dtype=np.int32)
        self.pos_template = np.zeros([self.maxlen + 1], dtype=np.int32)
        self.neg_template = np.zeros([self.maxlen + 1], dtype=np.int32)
        self.token_type_template = np.zeros([self.maxlen + 1], dtype=np.int32)
        self.next_token_type_template = np.zeros([self.maxlen + 1], dtype=np.int32)
        self.next_action_type_template = np.zeros([self.maxlen + 1], dtype=np.int32)

    def _get_data_file(self):
        """
        获取当前线程/进程的数据文件句柄
        """
        if not hasattr(self._local_data, 'data_file'):
            self._local_data.data_file = open(self.data_file_path, 'rb')
        return self._local_data.data_file

    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据，带缓存优化和多进程安全

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        # 检查缓存
        with self._cache_lock:
            if uid in self._user_data_cache:
                return self._user_data_cache[uid]
        
        # 从文件加载 - 使用线程本地的文件句柄
        try:
            data_file = self._get_data_file()
            data_file.seek(self.seq_offsets[uid])
            line = data_file.readline()
            
            # 解码并解析JSON
            line_str = line.decode('utf-8').strip()
            if not line_str:
                raise ValueError(f"Empty line for uid {uid}")
            
            data = json.loads(line_str)
            
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            # 如果JSON解析失败，尝试重新打开文件
            print(f"JSON decode error for uid {uid}, retrying... Error: {e}")
            try:
                # 关闭并重新打开文件句柄
                if hasattr(self._local_data, 'data_file'):
                    self._local_data.data_file.close()
                    delattr(self._local_data, 'data_file')
                
                data_file = self._get_data_file()
                data_file.seek(self.seq_offsets[uid])
                line = data_file.readline()
                line_str = line.decode('utf-8').strip()
                data = json.loads(line_str)
                
            except Exception as retry_e:
                print(f"Retry failed for uid {uid}: {retry_e}")
                # 返回空数据作为fallback
                return []
        
        except Exception as e:
            print(f"Unexpected error loading uid {uid}: {e}")
            return []
        
        # 更新缓存
        with self._cache_lock:
            if len(self._user_data_cache) >= self._max_cache_size:
                # 简单的FIFO策略，移除最早的条目
                oldest_key = next(iter(self._user_data_cache))
                del self._user_data_cache[oldest_key]
            
            self._user_data_cache[uid] = data
        
        return data

    def _random_neq(self, s):
        """
        优化的负采样：从预计算的候选集中采样

        Args:
            s: 序列中已有的item集合

        Returns:
            t: 不在序列s中的随机整数
        """
        # 从预计算的候选集中随机选择
        max_attempts = 10
        for _ in range(max_attempts):
            idx = np.random.randint(0, len(self.negative_candidates))
            t = self.negative_candidates[idx]
            if t not in s:
                return t
        
        # 如果多次尝试失败，使用更保守的方法
        available_candidates = self.negative_candidates[~np.isin(self.negative_candidates, list(s))]
        if len(available_candidates) > 0:
            return np.random.choice(available_candidates)
        else:
            # 最后的回退方案
            return np.random.choice(self.negative_candidates)

    def __getitem__(self, uid):
        """
        优化的数据获取函数

        Args:
            uid: 用户ID(reid)

        Returns:
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, aug_neg_feat1, aug_neg_feat2
        """
        # 确保worker进程正确初始化
        if not self._worker_initialized:
            self._worker_initialized = True
        user_sequence = self._load_user_data(uid)
        user_sequence.sort(key=lambda x: x[5]) # 按照时间戳排序

        # 预处理用户序列
        ext_user_sequence = []
        temp_timestamps = []

        user_tuple = None  # 保存用户特征元组

        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, timestamp = record_tuple
            # user_feat_temp = self._transfer_time_feat((user_feat or {}).copy(), timestamp) # 将时间戳转换为时间特征
            # if len(user_feat_temp.items()) > 3: print(f"user_feat_temp {[item for item in user_feat_temp.items()]}")
            if u and user_feat:
                user_tuple = (u, user_feat, item_feat, 2, action_type)
                # temp_timestamps.insert(0, timestamp)
            if i and item_feat:
                ext_user_sequence.append((i, user_feat, item_feat, 1, action_type))
                temp_timestamps.append(timestamp)
        
        ext_user_sequence = self._insert_time_diff(ext_user_sequence, temp_timestamps)

        # 使用模板复制而不是重新分配
        seq = self.seq_template.copy()
        pos = self.pos_template.copy()
        neg = self.neg_template.copy()
        token_type = self.token_type_template.copy()
        next_token_type = self.next_token_type_template.copy()
        next_action_type = self.next_action_type_template.copy()

        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)

        # 预计算item集合用于负采样
        ts = set()
        exclude_items = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[3] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])
            if record_tuple[3] == 1 and record_tuple[0] and record_tuple[4] == 1:
                exclude_items.add(record_tuple[0])
            

        nxt = ext_user_sequence[-1]
        idx = self.maxlen

        if user_tuple is not None:
            first_user_id, first_user_feat, first_item_feat, first_user_type_, first_user_act_type = user_tuple
        else:
            first_user_feat = None
        # 使用分离的特征管理器批量处理
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, user_feat, item_feat, type_, act_type = record_tuple
            next_i, next_user_feat, next_item_feat, next_type, next_act_type = nxt

            item_feat_temp = self._transfer_context_feat((item_feat or {}).copy(), user_feat, ['300', '301', '302', '303'])
            
            item_feat_temp = self._transfer_context_feat(item_feat_temp, first_user_feat, ['103', '104', '105', '109', '106', '107', '108', '110'])

            # 分离处理用户和物品特征
            if type_ == 2:  # 用户特征
                processed_feat = self.user_manager.get_user_features(i, user_feat)
                feat = self.fill_missing_feat(processed_feat, i, is_user=True)
            else:  # 物品特征
                # 这里不能进去，因为item_feat_temp被修改了，如果缓存就会出现问题
                # processed_feat = self.item_manager.get_item_features(i, item_feat_temp)
                feat = self.fill_missing_feat(item_feat_temp, i, is_user=False)
            
            if next_type == 2:  # 下一个是用户特征
                # 进不来
                processed_next_feat = self.user_manager.get_user_features(next_i, next_user_feat)
                next_feat = self.fill_missing_feat(processed_next_feat, next_i, is_user=True)
            else:  # 下一个是物品特征
                processed_next_feat = self.item_manager.get_item_features(next_i, next_item_feat)
                next_feat = self.fill_missing_feat(processed_next_feat, next_i, is_user=False)
            # print(f"next_feat {[item for item in next_feat.items()]}")
            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat
            
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                
                # 优化负采样
                neg_id = self._random_neq(ts)
                neg[idx] = neg_id
                
                # 使用物品管理器获取负样本特征
                neg_feat[idx] = self.item_manager.get_item_features(neg_id)
                neg_feat[idx] = self.fill_missing_feat(neg_feat[idx], neg_id, is_user=False)
            
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        # 将用户特征放在序列的第一位
        if user_tuple is not None:
            u, user_feat, item_feat, type_, act_type = user_tuple
            # 处理用户特征
            processed_user_feat = self.user_manager.get_user_features(u, user_feat)
            processed_user_feat = self.fill_missing_feat(processed_user_feat, u, is_user=True)
            
            # 放在序列的第一位（索引0）
            seq[0] = u
            token_type[0] = type_
            seq_feat[0] = processed_user_feat
            # 用户特征不需要pos和neg，保持默认值

        # 批量处理None值
        default_feat = self.feature_default_value
        seq_feat = np.where(seq_feat == None, default_feat, seq_feat)
        pos_feat = np.where(pos_feat == None, default_feat, pos_feat)
        neg_feat = np.where(neg_feat == None, default_feat, neg_feat)

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, ts, exclude_items

    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)

    def _init_basic_feat_info(self):
        """
        初始化基础特征信息（不包含嵌入维度）

        Returns:
            feat_default_value: 特征缺省值
            feat_types: 特征类型
            feat_statistics: 特征统计信息
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109'
                                    ,'300', '301', '302'
                                    # , '301', '302', '303' 
        ]
        feat_types['item_sparse'] = [
            '100', '117', '111', '118', '101', '102', '119', '120',
            '114', '112', '121', '115', '122', '116',
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = ['304']
        feat_types['item_continual'] = []

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        
        # 嵌入特征的默认值稍后设置
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = None  # 占位符，稍后设置

        return feat_default_value, feat_types, feat_statistics
    
    def _complete_feat_info(self):
        """
        完成特征信息初始化，设置嵌入特征的默认值
        """
        for feat_id in self.feature_types['item_emb']:
            if feat_id in self.item_manager.mm_emb_dict:
                emb_dict = self.item_manager.mm_emb_dict[feat_id]
                if emb_dict:
                    # 获取第一个嵌入的维度
                    first_emb = next(iter(emb_dict.values()))
                    if isinstance(first_emb, np.ndarray):
                        self.feature_default_value[feat_id] = np.zeros(
                            first_emb.shape[0], dtype=np.float32
                        )
                    else:
                        # 如果不是numpy数组，设置一个默认维度
                        self.feature_default_value[feat_id] = np.zeros(64, dtype=np.float32)
                else:
                    # 如果嵌入字典为空，设置默认维度
                    self.feature_default_value[feat_id] = np.zeros(64, dtype=np.float32)

    @lru_cache(maxsize=10000)
    def _get_all_feat_ids(self):
        """缓存所有特征ID列表"""
        all_feat_ids = []
        for feat_type in self.feature_types.values(): 
            all_feat_ids.extend(feat_type)
        return set(all_feat_ids)

    def fill_missing_feat(self, feat, item_id, is_user=False):
        """
        优化的特征填充函数，支持用户和物品特征分离处理

        Args:
            feat: 特征字典
            item_id: 物品ID（如果是用户特征则为用户ID）
            is_user: 是否为用户特征

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat is None:
            feat = {}
        
        # 使用字典更新而不是逐个复制
        filled_feat = dict(self.feature_default_value)  # 先填充默认值
        filled_feat.update(feat)  # 再更新实际值
        
        # 如果是物品特征，处理多模态嵌入
        if not is_user:
            for feat_id in self.feature_types['item_emb']:
                if item_id != 0 and item_id in self.item_manager.indexer_i_rev:
                    item_key = self.item_manager.indexer_i_rev[item_id]
                    if item_key in self.item_manager.mm_emb_dict[feat_id]:
                        emb_value = self.item_manager.mm_emb_dict[feat_id][item_key]
                        if isinstance(emb_value, np.ndarray):
                            filled_feat[feat_id] = emb_value

        return filled_feat


    def collate_fn(self, batch):
        """
        优化版本：预处理特征tensor转换，避免在模型中重复转换

        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            pos: 正样本ID, torch.Tensor形式
            neg: 负样本ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            next_token_type: 下一个token类型, torch.Tensor形式
            seq_feat_tensors: 预处理好的序列特征tensor字典
            pos_feat_tensors: 预处理好的正样本特征tensor字典
            neg_feat_tensors: 预处理好的负样本特征tensor字典
            aug_neg_feat1_tensors: 预处理好的增强负样本1特征tensor字典
            aug_neg_feat2_tensors: 预处理好的增强负样本2特征tensor字典
        """
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, ts, exclude_items = zip(*batch)
        
        # 去除假负样本
        neg, neg_feat = self.correct_false_negatives(neg, neg_feat, exclude_items)
        # neg, neg_feat = self._dedup_batch_negatives_with_resample(neg, neg_feat)

        # 使用流行度采样负样本
        pop_neg, pop_neg_feat = self._popularity_sampling()
        pop_neg, pop_neg_feat = self.correct_false_negatives(pop_neg, pop_neg_feat, ts)
        
        # 转换基础tensor
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        pop_neg = torch.from_numpy(np.array(pop_neg))

        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        
        
        # 预处理特征tensor - 批量转换所有特征类型
        seq_feat_tensors = self._preprocess_feature_tensors(seq_feat)
        pos_feat_tensors = self._preprocess_feature_tensors(pos_feat)
        neg_feat_tensors = self._preprocess_feature_tensors(neg_feat)
        pop_neg_feat_tensors = self._preprocess_feature_tensors(pop_neg_feat)

        return (seq, pos, neg, pop_neg, token_type, next_token_type, next_action_type, 
                seq_feat_tensors, pos_feat_tensors, neg_feat_tensors, pop_neg_feat_tensors)

    def correct_false_negatives(self, neg, neg_feat, exclude_items):
        """
        去除假负样本
        """
        # 1. 合并所有正样本去重
        all_ts = np.concatenate([np.array(list(sample), dtype=np.int32) for sample in exclude_items])
        batch_pos_set = set(all_ts)  # 更快去重
        batch_pos_arr = np.fromiter(batch_pos_set, dtype=np.int32)

        # 2. 构建安全候选池（使用预计算的 all_items）
        safe_candidates = np.setdiff1d(self.all_items, batch_pos_arr, assume_unique=True)

        # 3. 批量替换假负样本
        new_neg = []
        new_neg_feat = []
        for (neg, neg_feat) in zip(neg, neg_feat):
            mask = np.isin(neg, batch_pos_arr)
            if np.any(mask):
                num_replace = mask.sum()
                new_negs = np.random.choice(safe_candidates, size=num_replace, replace=True)
                neg[mask] = new_negs

                # 更新 neg_feat
                for i in range(len(mask)):
                    if mask[i]:
                        nf = self.item_manager.get_item_features(int(neg[i]))
                        neg_feat[i] = self.fill_missing_feat(nf, int(neg[i]), is_user=False)

            new_neg.append(neg)
            new_neg_feat.append(neg_feat)
        # print(f"len {len(new_neg)}")
        return new_neg, new_neg_feat

    def _dedup_batch_negatives_with_resample(self, neg, neg_feat):
        """
        对 batch 内负样本去重（全局唯一），并通过从全局池采样替换重复项，
        保证每个序列的负样本数量不变（形状不变）。

        ✅ 使用动态安全池 set（高效删除）
        ✅ 用 NumPy 向量化判断“是否首次出现”
        ✅ 一次性采样当前序列所有重复项所需 ID
        ✅ 减少 Python 循环，提升局部性
        """
        seen = set()
        safe_pool = set(self.all_items)  # 动态维护
        new_neg = []
        new_neg_feat = []

        for neg_list, feat_list in zip(neg, neg_feat):
            neg_arr = np.array(neg_list) if isinstance(neg_list, list) else neg_list
            feat_list = list(feat_list)  # 确保可修改

            # 向量化：找出哪些是首次出现
            mask_not_seen = ~np.isin(neg_arr, list(seen))  # 注意：这里把 seen 转成 list
            mask_seen = ~mask_not_seen

            # 初始化输出
            out_neg = neg_arr.copy()
            out_feat = feat_list.copy()

            # 处理首次出现的样本：加入 seen，从 safe_pool 移除
            first_occur_ids = neg_arr[mask_not_seen]
            for fid in first_occur_ids:
                seen.add(fid)
                if fid in safe_pool:
                    safe_pool.remove(fid)

            # 处理重复样本：一次性采样所需数量
            num_repeat = mask_seen.sum()
            if num_repeat > 0:
                if len(safe_pool) < num_repeat:
                    raise ValueError(f"Safe pool exhausted. Need {num_repeat}, only {len(safe_pool)} left.")

                # 一次性采样多个（避免多次调用 random.sample）
                sampled_ids = random.sample(safe_pool, num_repeat)
                for sid in sampled_ids:
                    safe_pool.remove(sid)
                    seen.add(sid)

                # 替换重复位置
                repeat_indices = np.where(mask_seen)[0]
                for i, idx in enumerate(repeat_indices):
                    new_id = sampled_ids[i]
                    out_neg[idx] = new_id
                    # 更新特征
                    new_feat = self.item_manager.get_item_features(new_id)
                    out_feat[idx] = self.fill_missing_feat(new_feat, new_id, is_user=False)

            new_neg.append(out_neg.tolist())
            new_neg_feat.append(out_feat)

        return new_neg, new_neg_feat    

    def _preprocess_feature_tensors(self, feature_batch):
        """
        将特征batch预处理为tensor字典，避免在模型中重复转换
        
        Args:
            feature_batch: 特征batch，形状为 [batch_size, maxlen]，每个元素为特征字典
            
        Returns:
            feature_tensors: 预处理好的特征tensor字典，key为特征ID，value为tensor
        """
        batch_size = len(feature_batch)
        feature_tensors = {}
        
        # 获取所有需要处理的特征类型
        all_feat_types = [
            (self.feature_types['item_sparse'], 'sparse'),
            (self.feature_types['item_array'], 'array'),
            (self.feature_types['item_continual'], 'continual'),
            (self.feature_types['user_sparse'], 'sparse'),
            (self.feature_types['user_array'], 'array'),
            (self.feature_types['user_continual'], 'continual'),
            (self.feature_types['item_emb'], 'emb')
        ]
        
        # 批量处理每种特征类型
        for feat_list, feat_type in all_feat_types:
            for feat_id in feat_list:
                if feat_type == 'array':
                    # Array类型特征需要特殊处理
                    feature_tensors[feat_id] = self._convert_array_feature_to_tensor(feature_batch, feat_id, batch_size)
                elif feat_type == 'emb':
                    # 多模态嵌入特征
                    feature_tensors[feat_id] = self._convert_emb_feature_to_tensor(feature_batch, feat_id, batch_size)
                elif feat_type == 'continual':
                    # 连续特征
                    feature_tensors[feat_id] = self._convert_continual_feature_to_tensor(feature_batch, feat_id, batch_size)
                else:
                    # Sparse类型特征
                    feature_tensors[feat_id] = self._convert_sparse_feature_to_tensor(feature_batch, feat_id, batch_size)
        
        return feature_tensors

    def _convert_array_feature_to_tensor(self, feature_batch, feat_id, batch_size):
        """转换Array类型特征为tensor"""
        max_array_len = 0
        max_seq_len = 0
        
        # 计算最大长度
        for i in range(batch_size):
            seq_data = [item.get(feat_id, self.feature_default_value[feat_id]) for item in feature_batch[i]]
            max_seq_len = max(max_seq_len, len(seq_data))
            for item_data in seq_data:
                if isinstance(item_data, list):
                    max_array_len = max(max_array_len, len(item_data))
                else:
                    max_array_len = max(max_array_len, 1)
        
        # 创建tensor
        batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
        for i in range(batch_size):
            seq_data = [item.get(feat_id, self.feature_default_value[feat_id]) for item in feature_batch[i]]
            for j, item_data in enumerate(seq_data):
                if isinstance(item_data, list):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]
                else:
                    batch_data[i, j, 0] = item_data
        
        return torch.from_numpy(batch_data)

    def _convert_emb_feature_to_tensor(self, feature_batch, feat_id, batch_size):
        """转换多模态嵌入特征为tensor"""
        emb_dim = len(self.feature_default_value[feat_id])
        max_seq_len = max(len(feature_batch[i]) for i in range(batch_size))
        
        batch_emb_data = np.zeros((batch_size, max_seq_len, emb_dim), dtype=np.float32)
        
        for i, seq in enumerate(feature_batch):
            for j, item in enumerate(seq):
                if feat_id in item and isinstance(item[feat_id], np.ndarray):
                    batch_emb_data[i, j] = item[feat_id]
                else:
                    batch_emb_data[i, j] = self.feature_default_value[feat_id]
        
        return torch.from_numpy(batch_emb_data)

    def _convert_continual_feature_to_tensor(self, feature_batch, feat_id, batch_size):
        """转换连续特征为tensor"""
        max_seq_len = max(len(feature_batch[i]) for i in range(batch_size))
        batch_data = np.zeros((batch_size, max_seq_len, 1), dtype=np.float32)
        
        for i in range(batch_size):
            seq_data = [item.get(feat_id, self.feature_default_value[feat_id]) for item in feature_batch[i]]
            for j, item_data in enumerate(seq_data):
                batch_data[i, j, 0] = float(item_data)
        
        return torch.from_numpy(batch_data)

    def _convert_sparse_feature_to_tensor(self, feature_batch, feat_id, batch_size):
        """转换稀疏特征为tensor"""
        max_seq_len = max(len(feature_batch[i]) for i in range(batch_size))
        batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        
        for i in range(batch_size):
            seq_data = [item.get(feat_id, self.feature_default_value[feat_id]) for item in feature_batch[i]]
            for j, item_data in enumerate(seq_data):
                batch_data[i, j] = item_data
        
        return torch.from_numpy(batch_data)
    
    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, '_local_data') and hasattr(self._local_data, 'data_file'):
                self._local_data.data_file.close()
        except:
            pass

class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, user_cache_dir, args):
        super().__init__(data_dir, user_cache_dir, args)
        self.mm_emb_dict = self.item_manager.mm_emb_dict

    def _load_data_and_offsets(self):
        # 使用父类的多进程安全方法
        self.data_file_path = self.data_dir / "predict_seq.jsonl"
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)
        
        # 为每个进程创建独立的文件句柄
        self._local_data = threading.local()

    def _process_cold_start_feat(self, feat):
        """
        冷启动特征优化：
        - 稀疏/多值稀疏特征：未见过的字符串使用哈希分桶映射到合法ID范围（避开padding=0）
        - 连续特征：无法解析时回退为0.0
        """
        if feat is None:
            return {}

        def _stable_hash_to_int(text: str) -> int:
            if not isinstance(text, str):
                text = str(text)
            h = hashlib.md5(text.encode('utf-8')).digest()
            return int.from_bytes(h[:8], 'little', signed=False)

        processed_feat = {}

        sparse_keys = set(self.feature_types.get('item_sparse', [])) | set(self.feature_types.get('user_sparse', []))
        array_keys = set(self.feature_types.get('item_array', [])) | set(self.feature_types.get('user_array', []))
        continual_keys = set(self.feature_types.get('item_continual', [])) | set(self.feature_types.get('user_continual', []))

        for feat_id, feat_value in feat.items():
            if feat_id in sparse_keys:
                if isinstance(feat_value, str):
                    vocab = max(1, int(self.feat_statistics.get(feat_id, 1)))
                    mapped = 1 + (_stable_hash_to_int(feat_value) % vocab)
                    processed_feat[feat_id] = mapped
                else:
                    processed_feat[feat_id] = int(feat_value)
            elif feat_id in array_keys:
                if isinstance(feat_value, list):
                    vocab = max(1, int(self.feat_statistics.get(feat_id, 1)))
                    mapped_list = []
                    for v in feat_value:
                        if isinstance(v, str):
                            mapped_list.append(1 + (_stable_hash_to_int(v) % vocab))
                        else:
                            mapped_list.append(int(v))
                    processed_feat[feat_id] = mapped_list
                elif isinstance(feat_value, str):
                    vocab = max(1, int(self.feat_statistics.get(feat_id, 1)))
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
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户在self.data_file中储存的行号
        Returns:
            seq: 用户序列ID
            token_type: 用户序列类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            user_id: user_id eg. user_xxxxxx ,便于后面对照答案
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据
        user_sequence.sort(key=lambda x: x[5]) # 按照时间戳排序

        ext_user_sequence = []
        temp_timestamps = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, timestamp = record_tuple
            user_feat_temp = self._transfer_time_feat((user_feat or {}).copy(), timestamp)
            if u:
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u]
            if u and user_feat:
                if type(u) == str:
                    u = 0
                if user_feat:
                    user_feat_temp = self._process_cold_start_feat(user_feat_temp)
                ext_user_sequence.insert(0, (u, user_feat_temp, item_feat, 2, action_type))

            if i and item_feat:
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, user_feat_temp, item_feat, 1, action_type))
                temp_timestamps.append(timestamp)
        
        ext_user_sequence = self._insert_time_diff(ext_user_sequence, temp_timestamps)

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)

        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[3] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        for record_tuple in reversed(ext_user_sequence):
            i, user_feat, item_feat, type_, action_type = record_tuple
            item_feat_temp = self._transfer_context_feat((item_feat or {}).copy(), user_feat, ['300', '301', '302', '303'])
            
            if type_ == 2:
                feat = self.fill_missing_feat(user_feat, i, is_user=True)
            else:
                feat = self.fill_missing_feat(item_feat_temp, i, is_user=False)
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = feat
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

        return seq, token_type, seq_feat, user_id

    def __len__(self):
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)

    def collate_fn(self, batch):
        """
        测试数据集的collate_fn，也需要预处理特征tensor

        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            seq_feat_tensors: 预处理好的序列特征tensor字典
            user_id: user_id, str
        """
        seq, token_type, seq_feat, user_id = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        
        # 预处理特征tensor
        seq_feat_tensors = self._preprocess_feature_tensors(seq_feat)

        return seq, token_type, seq_feat_tensors, user_id


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding，支持多种文件格式

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 4096, "86": 3584}
    mm_emb_dict = {}
    
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT.get(feat_id, 64)  # 默认维度64
        emb_dict = {}
        
        # 尝试多种加载方式
        loaded = False
        
        # 方式1: 尝试加载pkl文件
        pkl_path = Path(mm_path, f'emb_{feat_id}_{shape}.pkl')
        if pkl_path.exists():
            try:
                with open(pkl_path, 'rb') as f:
                    emb_dict = pickle.load(f)
                loaded = True
                print(f'Loaded #{feat_id} mm_emb from PKL file')
            except Exception as e:
                print(f"Error loading PKL file for {feat_id}: {e}")
        
        # 方式2: 尝试从目录中加载JSON文件
        if not loaded:
            base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
            if base_path.exists():
                try:
                    # 加载JSON文件
                    json_files = list(base_path.glob('*.json'))
                    if json_files:
                        for json_file in json_files:
                            with open(json_file, 'r', encoding='utf-8') as file:
                                for line in file:
                                    if line.strip():
                                        data_dict_origin = json.loads(line.strip())
                                        if 'emb' not in data_dict_origin:
                                            continue
                                        insert_emb = data_dict_origin['emb']
                                        if isinstance(insert_emb, list):
                                            insert_emb = np.array(insert_emb, dtype=np.float32)
                                        data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                                        emb_dict.update(data_dict)
                        loaded = True
                        print(f'Loaded #{feat_id} mm_emb from JSON files')
                    
                    # 如果没有JSON文件，尝试加载part文件
                    if not loaded:
                        part_files = list(base_path.glob('part-*'))
                        if part_files:
                            for part_file in part_files:
                                try:
                                    with open(part_file, 'r', encoding='utf-8') as file:
                                        for line in file:
                                            if line.strip():
                                                data_dict_origin = json.loads(line.strip())
                                                if 'emb' not in data_dict_origin:
                                                    continue
                                                insert_emb = data_dict_origin['emb']
                                                if isinstance(insert_emb, list):
                                                    insert_emb = np.array(insert_emb, dtype=np.float32)
                                                data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                                                emb_dict.update(data_dict)
                                except Exception as e:
                                    print(f"Error loading part file {part_file}: {e}")
                                    continue
                            loaded = True
                            print(f'Loaded #{feat_id} mm_emb from part files')
                        
                except Exception as e:
                    print(f"Error loading from directory for {feat_id}: {e}")
        
        # 如果都没有加载成功，创建空字典
        if not loaded:
            print(f"Warning: Could not load mm_emb for feat_id {feat_id}, using empty dict")
        
        mm_emb_dict[feat_id] = emb_dict
        print(f'Completed loading #{feat_id} mm_emb, got {len(emb_dict)} embeddings')
    
    return mm_emb_dict



class OptimizedTestDataset(MyTestDataset):
    """
    推理优化的测试数据集 - 基于MyTestDataset，使用多进程和预读取优化
    """
    
    def __init__(self, data_dir, user_cache_dir, args):
        super().__init__(data_dir, user_cache_dir, args)
        
        # 多进程优化参数
        self.num_workers = min(multiprocessing.cpu_count(), 16)  # 限制进程数
        self.prefetch_factor = 4  # 预读取倍数
        self.chunk_size = 1000  # 块大小
        
        # 初始化数据块分配
        self._init_data_chunks()
        
        # 预处理缓存
        self._preprocess_cache = {}
        self._cache_hits = 0
        self._total_requests = 0
        
        # 添加mm_emb_dict属性，指向item_manager中的mm_emb_dict
        self.mm_emb_dict = self.item_manager.mm_emb_dict
        
        print(f"OptimizedTestDataset initialized with {self.num_workers} workers, chunk_size={self.chunk_size}")

    def _init_data_chunks(self):
        """将数据分块以支持多进程处理"""
        total_samples = len(self.seq_offsets)
        self.chunks = []
        
        for i in range(0, total_samples, self.chunk_size):
            end_idx = min(i + self.chunk_size, total_samples)
            self.chunks.append((i, end_idx))
        
        print(f"Data divided into {len(self.chunks)} chunks")

    def _process_cold_start_feat(self, feat):
        """
        冷启动特征处理，与父类保持一致（哈希分桶OOV），并加入结果缓存。
        """
        if feat is None:
            return {}

        self._total_requests += 1

        feat_key = str(sorted(feat.items())) if feat else ""
        if feat_key in self._preprocess_cache:
            self._cache_hits += 1
            return self._preprocess_cache[feat_key]

        processed_feat = super()._process_cold_start_feat(feat)

        if len(self._preprocess_cache) < 10000:
            self._preprocess_cache[feat_key] = processed_feat

        return processed_feat

    def __getitem__(self, uid):
        """
        优化的数据获取函数，基于MyTestDataset但添加缓存和预处理优化
        """
        user_sequence = self._load_user_data(uid)  # 使用父类的加载方法
        user_sequence.sort(key=lambda x: x[5]) # 按照时间戳排序

        ext_user_sequence = []
        user_id = None
        temp_timestamps = []
        user_tuple = None
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, timestamp = record_tuple
            # user_feat_temp = self._transfer_time_feat((user_feat or {}).copy(), timestamp)
            
            # 处理用户信息（与原始MyTestDataset保持一致）
            if u:
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u]
            if u and user_feat:
                if type(u) == str:
                    u = 0
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)  # 使用优化版本
                user_tuple = (u, user_feat, item_feat, 2, action_type)
                # ext_user_sequence.insert(0, (u, user_feat_temp, item_feat, 2, action_type))
                # temp_timestamps.append(timestamp)

            # 处理物品信息（与原始MyTestDataset保持一致）
            if i and item_feat:
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)  # 使用优化版本
                ext_user_sequence.append((i, user_feat, item_feat, 1, action_type))
                temp_timestamps.append(timestamp)

        ext_user_sequence = self._insert_time_diff(ext_user_sequence, temp_timestamps)

        # 快速数组分配（优化版本）
        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)

        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[3] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        if user_tuple is not None:
            first_user_id, first_user_feat, first_item_feat, first_user_type_, first_user_act_type = user_tuple
        else:
            first_user_feat = None
        for record_tuple in reversed(ext_user_sequence):
            i, user_feat, item_feat, type_, action_type = record_tuple
            item_feat_temp = self._transfer_context_feat((item_feat or {}).copy(), user_feat, ['300', '301', '302', '303'])
            
            item_feat_temp = self._transfer_context_feat(item_feat_temp, first_user_feat, ['103', '104', '105', '109', '106', '107', '108', '110'])
            
            if type_ == 2:
                feat = self.fill_missing_feat(user_feat, i, is_user=True)
            else:
                feat = self.fill_missing_feat(item_feat_temp, i, is_user=False)
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = feat
            idx -= 1
            if idx == -1:
                break

        # 将用户特征放在序列的第一位
        if user_tuple is not None:
            u, user_feat, item_feat, type_, act_type = user_tuple
            # 处理用户特征
            processed_user_feat = self.user_manager.get_user_features(u, user_feat)
            processed_user_feat = self.fill_missing_feat(processed_user_feat, u, is_user=True)
            
            # 放在序列的第一位（索引0）
            seq[0] = u
            token_type[0] = type_
            seq_feat[0] = processed_user_feat

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

        return seq, token_type, seq_feat, user_id

    def collate_fn(self, batch):
        """
        优化的测试数据集collate_fn，使用原有逻辑但添加性能优化
        """
        seq, token_type, seq_feat, user_id = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        
        # 预处理特征tensor（使用父类方法）
        seq_feat_tensors = self._preprocess_feature_tensors(seq_feat)

        return seq, token_type, seq_feat_tensors, user_id

    def get_cache_stats(self):
        """获取缓存统计信息"""
        hit_rate = self._cache_hits / max(1, self._total_requests) * 100
        return {
            'cache_hits': self._cache_hits,
            'total_requests': self._total_requests,
            'hit_rate': f"{hit_rate:.2f}%",
            'cache_size': len(self._user_data_cache)
        }

