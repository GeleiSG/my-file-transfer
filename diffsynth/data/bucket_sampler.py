# data/samplers.py
import os
from collections import defaultdict
from typing import Dict, List, Sized, Optional, Iterator, Any
from torch.utils.data.dataset import Dataset
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, Sampler
# 从我们创建的常量文件中导入预设的宽高比字典
from diffsynth.data.utils_data import get_logger,get_closest_ratio_key,get_duration_bin_index

import time

#获取一个日志记录器实例
logger = get_logger(__name__)



# =====================================================================================
# 简单的采样器 (用于 enable_bucket=False)
# =====================================================================================

class ImageVideoSampler(BatchSampler):
    """
    一个简单的批次采样器，其主要功能是将数据按类型（图像或视频）分组。
    它确保每个生成的批次中只包含一种类型的数据。
    它不按宽高比分桶。
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 drop_last: bool = False
                ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # 为图像和视频分别创建一个桶
        self.bucket = {'image':[], 'video':[]}

    def __iter__(self):
        """迭代器，用于生成批次。"""
        # 从上游采样器（如 RandomSampler）中获取索引
        for idx in self.sampler:
            # 从数据集中查询该索引对应的数据类型,并将索引放入对应的bucket中
            content_type = self.dataset.dataset_list[idx].get('type', 'image')
            self.bucket[content_type].append(idx)

            # 当视频桶满了，就 yield 这个批次，并清空桶
            if len(self.bucket['video']) == self.batch_size:
                bucket = self.bucket['video']
                yield bucket.copy()
                bucket.clear()
             #当图像桶满了，也做同样的操作
            elif len(self.bucket['image']) == self.batch_size:
                bucket = self.bucket['image']
                yield bucket.copy()
                bucket.clear()
        '''
        原版代码没有这个逻辑
        # 在迭代结束时，处理剩余的不完整批次（如果 drop_last=False）
        if not self.drop_last:
            if self.bucket['video']:
                yield self.bucket['video'].copy()
            if self.bucket['image']:
                yield self.bucket['image'].copy()
        '''

class RandomSampler(Sampler[int]):
    """
    一个可复现的随机采样器。
    它接受一个 `generator` 对象，使得在设置相同种子时，每次运行的随机顺序都一样。
    这是保证实验可复现性的重要一环。
    """
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        aspect_ratios_dict (dict): The predefined aspect ratios.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
    """
    data_source: Dataset
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        # for resume
        self._pos_start = 0 # 用于断点续训时记录上一次的起始位置

        if not isinstance(self.replacement, bool):
            raise TypeError(f"replacement should be a boolean value, but got replacement={self.replacement}")

        if self.num_samples is not None and not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self._num_samples}")

        if self.generator is None:
            # 如果未提供生成器，则创建一个默认的
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = self.generator
    
    @property
    def num_samples(self) -> int:
        """返回总采样数。"""
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)

        if self.replacement:
            # 有放回抽样：分批次生成随机索引，每批 32 个，加速小批量生成
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=self.generator).tolist()
            # 生成剩余的 num_samples % 32 个
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=self.generator).tolist()
        else:
            # # 无放回抽样：num_samples // n 次完整遍历，每次都打乱一次
            # yield from torch.randperm(n, generator=generator).tolist()
            # 这是原始代码，比较复杂：无放回抽样：num_samples // n 次完整遍历，每次都打乱一次
            for _ in range(self.num_samples // n):
                xx = torch.randperm(n, generator=self.generator).tolist()  # 打乱索引列表
                # 从 _pos_start 开始依次 yield，循环结束后重置 _pos_start
                if self._pos_start >= n:
                    self._pos_start = 0
                logger.info(f"xx top 10 {xx[:10]}, pos_start={self._pos_start}")  # 调试信息：显示打乱后的前 10
                for idx in range(self._pos_start, n):
                    yield xx[idx]
                    # 更新下次迭代开始的位置（环回）
                    self._pos_start = (self._pos_start + 1) % n
                self._pos_start = 0
            # 最后 yield 剩余的 num_samples % n 个
            rem = torch.randperm(n, generator=self.generator).tolist()[:self.num_samples % n]
            yield from rem

    def __len__(self) -> int:
        return self.num_samples


import torch
import torch.distributed as dist
import math
from torch.utils.data.distributed import DistributedSampler
from typing import Sized, Iterator, Optional
import torch
import math
from torch.utils.data.distributed import DistributedSampler
from typing import Sized, Iterator, Optional

class DistributedRandomBatchSampler(DistributedSampler):
    """
    一个可复现的、支持分布式训练的随机批次采样器。
    它结合了 DistributedSampler 的数据分发功能和 RandomSampler 的可复现随机性。
    """
    
    def __init__(self, dataset: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None):
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        # 调用父类 DistributedSampler 的构造函数
        # 父类会自动设置 self.dataset, self.num_replicas, self.rank 等属性
        super().__init__(dataset, shuffle=True)
        
        self.replacement = replacement
        
        # 处理随机性：确保每个进程使用不同的随机种子
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self.generator = torch.Generator()
            # 使用 rank 偏移种子，确保每个进程有不同的随机序列
            self.generator.manual_seed(seed + self.rank)
        else:
            self.generator = generator
            
        # num_samples 是每个进程的样本数
        if num_samples is None:
            # 确保总样本数能被整除，以避免数据不均衡
            # ⚠️ 注意：这里使用 self.dataset
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        else:
            self.num_samples = num_samples

    def __iter__(self) -> Iterator[int]:
        # ⚠️ 注意：这里使用 self.dataset
        total_samples = len(self.dataset)
        
        # 1. 生成所有数据的全局随机索引序列
        if self.replacement:
            # 有放回抽样：从所有数据中随机抽样
            indices = torch.randint(
                high=total_samples,
                size=(total_samples,),
                dtype=torch.int64,
                generator=self.generator
            ).tolist()
        else:
            # 无放回抽样：生成一个全局的随机排列
            indices = torch.randperm(total_samples, generator=self.generator).tolist()

        # 2. 根据进程 rank 和 world_size 对索引进行切片
        start_idx = self.rank * self.num_samples
        end_idx = start_idx + self.num_samples
        subset_indices = indices[start_idx:end_idx]
        
        # 3. 将切片后的索引返回给 DataLoader
        yield from subset_indices

    def __len__(self) -> int:
        return self.num_samples

# =====================================================================================
#   高级采样器 (用于 enable_bucket=True 时)
# =====================================================================================

class AspectRatioImageVideoSampler(BatchSampler):
    """
    同时分图像和视频的高宽比批采样器。
    """
    def __init__(
        self,
        sampler: Sampler,
        dataset: Dataset,
        batch_size: int,
        aspect_ratios_dict: Dict[float, Any],
        drop_last: bool = False,
        video_duration_bins: Optional[List[float]] = None
    ) -> None:
        # --- 参数校验 ---
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        # --- 核心属性初始化 ---
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.aspect_ratios_dict = aspect_ratios_dict
        self.drop_last = drop_last
        # video duration bins
        self.video_duration_bins = video_duration_bins
        # image/video 各自 buckets
        ## 这是一个两层的字典，第一层按'image'/'video'分类，第二层按宽高比分类
        self.buckets = {
            'image': {r: [] for r in aspect_ratios_dict},
            'video': {r: [] for r in aspect_ratios_dict}
        }

        # 如果定义了视频时长分桶规则，则为每个宽高比桶创建时长子桶
        if self.video_duration_bins is not None:
            # 时长桶的数量是分割点数量 + 1
            num_duration_bins = len(self.video_duration_bins) + 1
            for aspect_ratio_key in self.buckets['video']:
                # 每个宽高比下，都有一套完整的时长桶
                self.buckets['video'][aspect_ratio_key] = {
                    i: [] for i in range(num_duration_bins)
                }
        else:
            # 如果不按时长分桶，则视频桶结构与图像桶相同（为了代码兼容性）
            for aspect_ratio_key in self.buckets['video']:
                self.buckets['video'][aspect_ratio_key] = []

        self.current_available_bucket_keys = list(aspect_ratios_dict.keys())

    def __iter__(self):
        # 从上游采样器（如 RandomSampler）中逐个获取数据索引
        for idx in self.sampler:
            content_type = self.dataset[idx].get('type', 'image')
            height, width, duration = self.dataset.get_dimensions_and_duration(idx)
            # 跳过无效数据（例如，无法读取尺寸的文件）
            if height == 0 or width == 0:
                logger.info(f"Warning: Skipping item {idx} with invalid dimensions ({height}x{width}).")
                continue 

            closest_ratio_key = get_closest_ratio_key(height, width, self.aspect_ratios_dict)
            if closest_ratio_key not in self.current_available_bucket_keys:
                continue
            
            #    将索引放入正确的桶中
            #    首先根据 content_type 选择 'image' 或 'video' 的桶集合，
            #    然后根据 closest_ratio_key 选择最终的那个桶（列表）。
            if content_type == 'image':
                bucket = self.buckets[content_type][closest_ratio_key]
            elif content_type == 'video':
                # 对于视频，进一步根据时长分桶
                if self.video_duration_bins is not None:
                    # 计算该时长对应的桶索引
                    duration_bin = get_duration_bin_index(duration, self.video_duration_bins)
                    # 获取对应的宽高比和时长桶
                    bucket = self.buckets[content_type][closest_ratio_key][duration_bin]
                else:
                    # 如果没有时长分桶，则直接使用宽高比桶
                    bucket = self.buckets[content_type][closest_ratio_key]
            else:
                logger.warning(f"Unknown content type '{content_type}' for index {idx}. Skipping.")
                continue
            # 将索引添加到对应的桶中
            bucket.append(idx)
            # 如果桶已满，就把它作为一个完整的批次 yield 出去
            if len(bucket) == self.batch_size:
                yield bucket.copy()
                bucket.clear()
        # --- 循环结束后，处理所有桶中剩余的、不完整的批次 ---
        if not self.drop_last:
            for type_buckets  in self.buckets.values():
                for bucket in type_buckets .values():
                    if bucket:
                        yield bucket.copy()

import random

class OptimizedAspectRatioSampler(BatchSampler):
    """
    优化版的高宽比批采样器。

    通过在初始化时预计算所有样本的分桶信息，极大地加速了每个 epoch 的启动速度。
    它首先遍历一次整个数据集，将每个样本的索引根据其类型、宽高比和（可选的）时长
    分类到不同的桶中。在每个 epoch 开始时（调用 __iter__），它仅处理由上游采样器
    （如 DistributedSampler）提供的当前 epoch 的索引，将它们快速分组并生成批次。
    """
    def __init__(
        self,
        sampler: Sampler,
        dataset: Dataset,
        batch_size: int,
        aspect_ratios_dict: Dict[float, Any],
        drop_last: bool = False,
        video_duration_bins: Optional[List[float]] = None
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')

        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.aspect_ratios_dict = aspect_ratios_dict
        self.drop_last = drop_last
        self.video_duration_bins = video_duration_bins

        # 核心优化：在初始化时预计算所有数据项的分桶信息
        # self.idx_to_bucket_key 是一个映射，可以快速查询任意索引属于哪个桶
        self.idx_to_bucket_key = {}
        
        print("Pre-computing aspect ratio buckets for all samples. This may take a while on the first run...")
        # (可选) 可以加入 tqdm 来显示进度
        # from tqdm import tqdm
        # for idx in tqdm(range(len(dataset)), desc="Pre-computing buckets"):
        for idx in range(len(dataset)):
            try:
                content_type = self.dataset[idx].get('type', 'image')
                height, width, duration = self.dataset.get_dimensions_and_duration(idx)
                
                if height == 0 or width == 0:
                    logger.warning(f"Skipping item {idx} with invalid dimensions ({height}x{width}).")
                    continue

                closest_ratio_key = get_closest_ratio_key(height, width, self.aspect_ratios_dict)

                if content_type == 'image':
                    bucket_key = ('image', closest_ratio_key)
                elif content_type == 'video':
                    if self.video_duration_bins is not None:
                        duration_bin = get_duration_bin_index(duration, self.video_duration_bins)
                        bucket_key = ('video', closest_ratio_key, duration_bin)
                    else:
                        bucket_key = ('video', closest_ratio_key)
                else:
                    logger.warning(f"Unknown content type '{content_type}' for index {idx}. Skipping.")
                    continue
                
                self.idx_to_bucket_key[idx] = bucket_key
            except Exception as e:
                logger.error(f"Error processing index {idx}: {e}")

        print("Bucket pre-computation finished.")


    def __iter__(self):
        # 1. 为当前 epoch 创建桶
        #    这些桶只包含由 self.sampler (DistributedSampler) 为当前进程和 epoch 选择的索引
        epoch_buckets: Dict[Any, List[int]] = {}
        
        # 2. 快速将当前 epoch 的索引分配到桶中
        #    这个循环非常快，因为它只涉及字典查询和列表追加，没有磁盘 I/O
        for idx in self.sampler:
            bucket_key = self.idx_to_bucket_key.get(idx)
            if bucket_key:
                if bucket_key not in epoch_buckets:
                    epoch_buckets[bucket_key] = []
                epoch_buckets[bucket_key].append(idx)

        # 3. 从每个桶中创建批次
        all_batches = []
        for bucket_key in epoch_buckets:
            # 在桶内打乱顺序，增加随机性
            random.shuffle(epoch_buckets[bucket_key])
            
            # 将桶内的索引切分成批次
            bucket_indices = epoch_buckets[bucket_key]
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i:i + self.batch_size]
                
                # 根据 drop_last 处理最后一个不完整的批次
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                
                all_batches.append(batch)
        
        # 4. 全局打乱所有批次的顺序
        #    这可以防止模型在训练初期连续看到相同宽高比的数据
        random.shuffle(all_batches)

        # 使用 `yield from` 高效地交出所有批次
        yield from all_batches
        
    def __len__(self) -> int:
        # 这个长度计算也需要基于新的逻辑
        if self.drop_last:
            # 计算每个进程会产生的批次数
            num_samples_for_process = len(self.sampler)
            # 这是一个近似值，因为分桶会导致一些样本无法凑成完整批次
            return num_samples_for_process // self.batch_size
        else:
            # 如果不丢弃，长度会更接近
            num_samples_for_process = len(self.sampler)
            return (num_samples_for_process + self.batch_size - 1) // self.batch_size
        
import json        
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
# class AllInOneAspectRatioSampler(BatchSampler):
#     """
#     一个功能完备的宽高比批采样器，内部集成了元数据缓存和读取重试逻辑。

#     这个采样器被设计为与任何标准的、简单的 Dataset 对象协同工作，
#     无需对 Dataset 类进行任何修改。

#     功能特性:
#     1. 预计算: 在初始化时一次性获取所有数据的元信息。
#     2. 缓存: 将计算出的元信息保存到磁盘文件。下次启动时直接加载，实现秒级启动。
#     3. 重试: 当从数据集中获取尺寸或时长失败时，会自动进行多次重试。
#     """
#     def __init__(
#         self,
#         sampler: Sampler,
#         dataset: Dataset,
#         batch_size: int,
#         aspect_ratios_dict: Dict[float, Any],
#         drop_last: bool = False,
#         video_duration_bins: Optional[List[float]] = None,
#         cache_path: str = "/mnt/workspace/sampler_metadata_cache.json",
#         n_retries: int = 3,
#         retry_delay: int = 1,
#         force_recompute: bool = False
#     ) -> None:
#         # --- 基础参数校验 ---
#         if not isinstance(sampler, Sampler):
#             raise TypeError('sampler should be an instance of ``Sampler``, but got {sampler}')
#         if not isinstance(batch_size, int) or batch_size <= 0:
#             raise ValueError('batch_size should be a positive integer value, but got batch_size={batch_size}')

#         # --- 核心属性初始化 ---
#         self.sampler = sampler
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.aspect_ratios_dict = aspect_ratios_dict
#         self.drop_last = drop_last
#         self.video_duration_bins = video_duration_bins
        
#         # --- 新增功能参数 ---
#         self.cache_path = cache_path
#         self.n_retries = n_retries
#         self.retry_delay = retry_delay

#         # --- 核心逻辑 ---
#         metadata_list = self._load_or_create_metadata(force_recompute)
#         self.idx_to_bucket_key = self._build_bucket_map(metadata_list)

#     def _load_or_create_metadata(self, force_recompute: bool) -> List[Dict]:
#         """[内部方法] 加载或创建元数据缓存。"""
#         if os.path.exists(self.cache_path) and not force_recompute:
#             logger.info(f"Loading metadata from cache: {self.cache_path}")
#             try:
#                 with open(self.cache_path, 'r') as f:
#                     loaded_metadata = json.load(f)
#                 if len(loaded_metadata) == len(self.dataset):
#                     return loaded_metadata
#                 else:
#                     logger.warning("Cache size mismatch with dataset. Recomputing...")
#             except (json.JSONDecodeError, IOError) as e:
#                 logger.error(f"Failed to load cache file: {e}. Recomputing...")
        
#         return self._compute_and_save_metadata()

#     def _compute_and_save_metadata(self) -> List[Dict]:
#         """[内部方法] 遍历数据集计算元数据，并在需要时保存。"""
#         logger.info("Metadata cache not found or invalid. Computing now. This may take a while...")
        
#         temp_metadata = []
#         for idx in tqdm(range(len(self.dataset)), desc="Computing Metadata"):
#             meta = self._get_metadata_with_retry(idx)
#             temp_metadata.append(meta)
            
#         if self.cache_path:
#             try:
#                 logger.info(f"Saving metadata cache to: {self.cache_path}")
#                 with open(self.cache_path, 'w') as f:
#                     json.dump(temp_metadata, f)
#             except IOError as e:
#                 logger.error(f"Could not save metadata cache: {e}")
                
#         return temp_metadata

#     def _get_metadata_with_retry(self, idx: int) -> Dict:
#         """[内部方法] 带有重试逻辑的元数据获取。直接调用 dataset 的方法。"""
#         for attempt in range(self.n_retries):
#             try:
#                 # 假设 dataset[idx] 返回一个可以 .get('type') 的字典
#                 # 并且 dataset 有 get_dimensions_and_duration 方法
#                 content_type = self.dataset[idx].get('type', 'image')
#                 height, width, duration = self.dataset.get_dimensions_and_duration(idx)
                
#                 if height == 0 or width == 0:
#                     raise ValueError(f"Invalid dimensions received: {height}x{width}")
                
#                 return {
#                     'height': height, 'width': width, 'duration': duration,
#                     'type': content_type, 'valid': True
#                 }
#             except Exception as e:
#                 logger.warning(f"Attempt {attempt + 1}/{self.n_retries} failed for item {idx}: {e}")
#                 if attempt < self.n_retries - 1:
#                     time.sleep(self.retry_delay)
#                 else:
#                     logger.error(f"All {self.n_retries} retries failed for item {idx}. Marking as invalid.")
        
#         # 所有重试都失败后
#         return {'height': 0, 'width': 0, 'duration': 0.0, 'type': 'unknown', 'valid': False}
    
#     def _build_bucket_map(self, metadata_list: List[Dict]) -> Dict:
#         """[内部方法] 根据元数据列表构建分桶映射。"""
#         logger.info("Building bucket map from metadata...")
#         idx_to_bucket_key = {}
#         for idx, meta in enumerate(tqdm(metadata_list, desc="Building Bucket Map")):
#             if not meta.get('valid', False):
#                 continue
            
#             content_type = meta['type']
#             height, width, duration = meta['height'], meta['width'], meta['duration']

#             closest_ratio_key = get_closest_ratio_key(height, width, self.aspect_ratios_dict)

#             if content_type == 'image':
#                 bucket_key = ('image', closest_ratio_key)
#             elif content_type == 'video':
#                 if self.video_duration_bins is not None:
#                     duration_bin = get_duration_bin_index(duration, self.video_duration_bins)
#                     bucket_key = ('video', closest_ratio_key, duration_bin)
#                 else:
#                     bucket_key = ('video', closest_ratio_key)
#             else:
#                 continue
            
#             idx_to_bucket_key[idx] = bucket_key
#         logger.info("Bucket map built successfully.")
#         return idx_to_bucket_key

#     def __iter__(self):
#         epoch_buckets: Dict[Any, List[int]] = {}
        
#         for idx in self.sampler:
#             bucket_key = self.idx_to_bucket_key.get(idx)
#             if bucket_key:
#                 if bucket_key not in epoch_buckets:
#                     epoch_buckets[bucket_key] = []
#                 epoch_buckets[bucket_key].append(idx)

#         all_batches = []
#         for bucket_indices in epoch_buckets.values():
#             random.shuffle(bucket_indices)
            
#             for i in range(0, len(bucket_indices), self.batch_size):
#                 batch = bucket_indices[i:i + self.batch_size]
#                 if len(batch) < self.batch_size and self.drop_last:
#                     continue
#                 all_batches.append(batch)
        
#         random.shuffle(all_batches)
#         yield from all_batches
        
#     def __len__(self) -> int:
#         if self.drop_last:
#             num_samples_for_process = sum(1 for idx in self.sampler if idx in self.idx_to_bucket_key)
#             return num_samples_for_process // self.batch_size
#         else:
#             num_samples_for_process = sum(1 for idx in self.sampler if idx in self.idx_to_bucket_key)
#             return (num_samples_for_process + self.batch_size - 1) // self.batch_size

class AllInOneAspectRatioSampler(BatchSampler):
    """
    一个功能完备的宽高比批采样器，内部集成了元数据缓存和读取重试逻辑。
    此版本增强了缓存机制，支持断点续算和定期保存。
    """
    def __init__(
        self,
        sampler: Sampler,
        dataset: Dataset,
        batch_size: int,
        aspect_ratios_dict: Dict[float, Any],
        drop_last: bool = False,
        video_duration_bins: Optional[List[float]] = None,
        cache_path: str = "/mnt/workspace/sampler_metadata_cache.json",
        n_retries: int = 3,
        retry_delay: int = 1,
        force_recompute: bool = False,
        save_interval: int = 100  # 新增参数：每处理100条数据就保存一次
    ) -> None:
        # --- 基础参数校验 ---
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, but got batch_size={batch_size}')

        # --- 核心属性初始化 ---
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.aspect_ratios_dict = aspect_ratios_dict
        self.drop_last = drop_last
        self.video_duration_bins = video_duration_bins
        
        # --- 新增功能参数 ---
        self.cache_path = cache_path
        self.n_retries = n_retries
        self.retry_delay = retry_delay
        self.save_interval = save_interval

        # --- 核心逻辑 ---
        # 旧的 _compute_and_save_metadata 已被合并到 _load_or_create_metadata 中
        metadata_list = self._load_or_create_metadata(force_recompute)
        self.idx_to_bucket_key = self._build_bucket_map(metadata_list)

    def _save_cache(self, metadata_list: List[Dict]):
        """[内部辅助方法] 将元数据列表安全地写入缓存文件。"""
        if self.cache_path:
            try:
                # 为了防止写入过程中断导致文件损坏，采用先写临时文件再重命名的原子操作
                temp_path = self.cache_path + ".tmp"
                with open(temp_path, 'w') as f:
                    json.dump(metadata_list, f)
                os.replace(temp_path, self.cache_path)
            except IOError as e:
                logger.error(f"Could not save metadata cache to {self.cache_path}: {e}")

    def _load_or_create_metadata(self, force_recompute: bool) -> List[Dict]:
        """
        [内部核心方法] 加载元数据，如果缓存不完整则从断点处继续计算，并定期保存。
        """
        if force_recompute and os.path.exists(self.cache_path):
            logger.info(f"Force recompute enabled. Deleting existing cache: {self.cache_path}")
            os.remove(self.cache_path)

        metadata_list = []
        if os.path.exists(self.cache_path):
            logger.info(f"Loading metadata from cache: {self.cache_path}")
            try:
                with open(self.cache_path, 'r') as f:
                    metadata_list = json.load(f)
                logger.info(f"Loaded {len(metadata_list)} items from cache.")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load cache file: {e}. Starting from scratch.")
                metadata_list = []

        start_idx = len(metadata_list)
        total_items = len(self.dataset)

        if start_idx >= total_items:
            logger.info("Metadata cache is complete and matches dataset size.")
            # 如果数据集缩小了，则裁剪缓存以匹配
            return metadata_list[:total_items]

        logger.info(f"Resuming metadata computation from index {start_idx}...")
        
        # 使用 try...finally 确保即使中断也能保存进度
        try:
            # 仅对需要计算的部分显示进度条
            remaining_indices = range(start_idx, total_items)
            progress = tqdm(remaining_indices, desc="Computing Metadata")
            
            for idx in progress:
                meta = self._get_metadata_with_retry(idx)
                metadata_list.append(meta)
                
                # 每 N 条数据保存一次进度 (Requirement 2)
                # (idx - start_idx + 1) 是当前循环处理的第几条数据
                if (idx - start_idx + 1) % self.save_interval == 0:
                    progress.set_description(f"Saving progress at item {idx+1}...")
                    self._save_cache(metadata_list)
                    progress.set_description("Computing Metadata")
        finally:
            # 无论程序是正常结束还是被中断，都执行最后一次保存
            logger.info("Computation finished or interrupted. Saving final progress...")
            self._save_cache(metadata_list)
            
        return metadata_list

    def _get_metadata_with_retry(self, idx: int) -> Dict:
        # 这个方法保持不变
        for attempt in range(self.n_retries):
            try:
                content_type = self.dataset[idx].get('type', 'image')
                height, width, duration = self.dataset.get_dimensions_and_duration(idx)
                if height == 0 or width == 0:
                    raise ValueError(f"Invalid dimensions received: {height}x{width}")
                return {'height': height, 'width': width, 'duration': duration, 'type': content_type, 'valid': True}
            except Exception as e:
                if attempt >= self.n_retries - 1:
                    logger.error(f"All {self.n_retries} retries failed for item {idx}. Marking as invalid. Error: {e}")
                time.sleep(self.retry_delay)
        return {'height': 0, 'width': 0, 'duration': 0.0, 'type': 'unknown', 'valid': False}
    
    def _build_bucket_map(self, metadata_list: List[Dict]) -> Dict:
        # 这个方法保持不变
        logger.info("Building bucket map from metadata...")
        idx_to_bucket_key = {}
        for idx, meta in enumerate(tqdm(metadata_list, desc="Building Bucket Map")):
            if not meta.get('valid', False):
                continue
            content_type = meta['type']
            height, width, duration = meta['height'], meta['width'], meta['duration']
            closest_ratio_key = get_closest_ratio_key(height, width, self.aspect_ratios_dict)
            if content_type == 'image':
                bucket_key = ('image', closest_ratio_key)
            elif content_type == 'video':
                if self.video_duration_bins is not None:
                    duration_bin = get_duration_bin_index(duration, self.video_duration_bins)
                    bucket_key = ('video', closest_ratio_key, duration_bin)
                else:
                    bucket_key = ('video', closest_ratio_key)
            else:
                continue
            idx_to_bucket_key[idx] = bucket_key
        logger.info("Bucket map built successfully.")
        return idx_to_bucket_key

    def __iter__(self):
        # 这个方法保持不变
        epoch_buckets: Dict[Any, List[int]] = {}
        for idx in self.sampler:
            bucket_key = self.idx_to_bucket_key.get(idx)
            if bucket_key:
                if bucket_key not in epoch_buckets: epoch_buckets[bucket_key] = []
                epoch_buckets[bucket_key].append(idx)
        all_batches = []
        for bucket_indices in epoch_buckets.values():
            random.shuffle(bucket_indices)
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last: continue
                all_batches.append(batch)
        random.shuffle(all_batches)
        yield from all_batches
        
    def __len__(self) -> int:
        # 这个方法保持不变
        if self.drop_last:
            num_samples_for_process = sum(1 for idx in self.sampler if idx in self.idx_to_bucket_key)
            return num_samples_for_process // self.batch_size
        else:
            num_samples_for_process = sum(1 for idx in self.sampler if idx in self.idx_to_bucket_key)
            return (num_samples_for_process + self.batch_size - 1) // self.batch_size
        
worker_dataset = None
worker_n_retries = 3
worker_retry_delay = 2

def _worker_init(dataset, n_retries, retry_delay):
    """每个工作进程的初始化函数"""
    global worker_dataset, worker_n_retries, worker_retry_delay
    worker_dataset = dataset
    worker_n_retries = n_retries
    worker_retry_delay = retry_delay

def _get_metadata_for_worker(idx: int) -> Dict:
    """由每个工作进程执行的实际任务函数"""
    global worker_dataset, worker_n_retries, worker_retry_delay
    for attempt in range(worker_n_retries):
        try:
            content_type = worker_dataset[idx].get('type', 'image')
            height, width, duration = worker_dataset.get_dimensions_and_duration(idx)
            if height == 0 or width == 0:
                raise ValueError(f"Invalid dimensions: {height}x{width}")
            return {'idx': idx, 'height': height, 'width': width, 'duration': duration, 'type': content_type, 'valid': True}
        except Exception:
            if attempt < worker_n_retries - 1:
                time.sleep(worker_retry_delay)
    return {'idx': idx, 'height': 0, 'width': 0, 'duration': 0.0, 'type': 'unknown', 'valid': False}

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RobustParallelSampler(BatchSampler):
    """
    一个极其健壮和高效的宽高比批采样器。

    功能特性:
    1. 并行计算: 使用多进程并行获取元数据，极大缩短首次运行时间。
    2. 断点续算: 定期保存进度。如果中途停止，下次运行时会自动从断点处继续。
    3. 内部缓存: 最终结果缓存到磁盘，实现后续秒级启动。
    4. 重试机制: 文件读取失败时自动重试。
    """
    def __init__(
        self,
        sampler: Sampler,
        dataset: Dataset,
        batch_size: int,
        aspect_ratios_dict: Dict[float, Any],
        drop_last: bool = False,
        video_duration_bins: Optional[List[float]] = None,
        num_workers: int = 8,
        cache_path: str = "sampler_metadata_cache.json",
        save_interval: int = 1000,
        n_retries: int = 3,
        retry_delay: int = 2,
        force_recompute: bool = False
    ) -> None:
        # ... (基础参数校验与之前相同) ...
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.aspect_ratios_dict = aspect_ratios_dict
        self.drop_last = drop_last
        self.video_duration_bins = video_duration_bins
        
        # --- 新增功能参数 ---
        self.num_workers = max(1, num_workers)
        self.cache_path = cache_path
        self.save_interval = save_interval
        self.n_retries = n_retries
        self.retry_delay = retry_delay

        # --- 核心逻辑 ---
        metadata_list = self._load_or_create_metadata(force_recompute)
        self.idx_to_bucket_key = self._build_bucket_map(metadata_list)

    def _save_cache(self, metadata_list: List[Dict]):
        """[内部方法] 安全地保存缓存文件"""
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(metadata_list, f)
        except IOError as e:
            logger.error(f"Could not save metadata cache to {self.cache_path}: {e}")

    def _load_or_create_metadata(self, force_recompute: bool) -> List[Dict]:
        """[内部方法] 实现断点续算和并行计算的核心"""
        if force_recompute and os.path.exists(self.cache_path):
            os.remove(self.cache_path)

        metadata_list = []
        if os.path.exists(self.cache_path):
            logger.info(f"Loading existing metadata from cache: {self.cache_path}")
            try:
                with open(self.cache_path, 'r') as f:
                    metadata_list = json.load(f)
                # 校验缓存是否损坏或过时
                if not isinstance(metadata_list, list) or (metadata_list and not isinstance(metadata_list[0], dict)):
                    raise ValueError("Cache file is corrupted.")
                logger.info(f"Loaded {len(metadata_list)} items from cache.")
            except (json.JSONDecodeError, IOError, ValueError) as e:
                logger.error(f"Cache file is corrupted or invalid: {e}. Starting from scratch.")
                metadata_list = []

        start_idx = len(metadata_list)
        total_items = len(self.dataset)

        if start_idx >= total_items:
            logger.info("Metadata cache is complete. No re-computation needed.")
            if len(metadata_list) > total_items: # 如果数据集缩小了，则裁剪缓存
                return metadata_list[:total_items]
            return metadata_list

        indices_to_process = list(range(start_idx, total_items))
        logger.info(f"Resuming metadata computation from index {start_idx}. Items to process: {len(indices_to_process)}")
        
        # 使用进程池并行处理
        results = [None] * total_items # 预先分配好完整列表
        for i in range(start_idx):
             results[i] = metadata_list[i]
        
        # 使用 try...finally 确保即使中断也能保存进度
        try:
            with ProcessPoolExecutor(max_workers=self.num_workers, initializer=_worker_init, initargs=(self.dataset, self.n_retries, self.retry_delay)) as executor:
                # 提交所有任务
                futures = {executor.submit(_get_metadata_for_worker, i): i for i in indices_to_process}
                
                # 使用 tqdm 显示进度
                progress = tqdm(as_completed(futures), total=len(indices_to_process), desc="Computing Metadata")
                
                for i, future in enumerate(progress):
                    result = future.result()
                    original_idx = result['idx']
                    results[original_idx] = result # 将结果放回正确的位置
                    
                    # 每N条保存一次进度
                    if (i + 1) % self.save_interval == 0:
                        progress.set_description(f"Saving progress to cache...")
                        self._save_cache(results)
                        progress.set_description("Computing Metadata")

        finally:
            # 无论程序是正常结束还是被中断(Ctrl+C)，都尝试保存最后一次的进度
            logger.info("Computation finished or interrupted. Saving final progress...")
            self._save_cache(results)
            
        return results

    def _build_bucket_map(self, metadata_list: List[Dict]) -> Dict:
        # ... (此方法与上一版完全相同，无需改动)
        logger.info("Building bucket map from metadata...")
        idx_to_bucket_key = {}
        for idx, meta in enumerate(tqdm(metadata_list, desc="Building Bucket Map")):
            if not meta.get('valid', False):
                continue
            content_type, height, width, duration = meta['type'], meta['height'], meta['width'], meta['duration']
            closest_ratio_key = get_closest_ratio_key(height, width, self.aspect_ratios_dict)
            if content_type == 'image':
                bucket_key = ('image', closest_ratio_key)
            elif content_type == 'video':
                if self.video_duration_bins is not None:
                    duration_bin = get_duration_bin_index(duration, self.video_duration_bins)
                    bucket_key = ('video', closest_ratio_key, duration_bin)
                else:
                    bucket_key = ('video', closest_ratio_key)
            else:
                continue
            idx_to_bucket_key[idx] = bucket_key
        logger.info("Bucket map built successfully.")
        return idx_to_bucket_key

    def __iter__(self):
        # ... (此方法与上一版完全相同，无需改动)
        epoch_buckets = {}
        for idx in self.sampler:
            bucket_key = self.idx_to_bucket_key.get(idx)
            if bucket_key:
                if bucket_key not in epoch_buckets: epoch_buckets[bucket_key] = []
                epoch_buckets[bucket_key].append(idx)
        all_batches = []
        for bucket_indices in epoch_buckets.values():
            random.shuffle(bucket_indices)
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last: continue
                all_batches.append(batch)
        random.shuffle(all_batches)
        yield from all_batches
        
    def __len__(self) -> int:
        # ... (此方法与上一版完全相同，无需改动)
        if self.drop_last:
            num_samples = sum(1 for idx in self.sampler if idx in self.idx_to_bucket_key)
            return num_samples // self.batch_size
        else:
            num_samples = sum(1 for idx in self.sampler if idx in self.idx_to_bucket_key)
            return (num_samples + self.batch_size - 1) // self.batch_size
