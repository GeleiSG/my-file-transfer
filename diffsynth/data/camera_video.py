import argparse
import json
import math
import os
import einops
import torch
import random
import numpy as np
import torchvision
import torchvision.transforms.v2 as v2
import imageio
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision import transforms
from einops import rearrange
import tqdm

from diffsynth.data.bucket_sampler import (
                                             AspectRatioImageVideoSampler,
                                             DistributedRandomBatchSampler,
                                             RandomSampler)
from diffsynth.data.camera_utils import get_camera_sparse_embedding, get_plucker_embedding, get_plucker_embedding_cpu, get_relative_pose, ray_condition
from diffsynth.data.transforms import get_aspect_ratio
from diffsynth.data.utils_data import get_closest_ratio_key

from decord import VideoReader, cpu
from contextlib import contextmanager
import gc
import subprocess
import json
import logging
import time
from diffsynth.data.dataset_preprocess import preprocess_metadata
import pickle

@contextmanager
def VideoReaderManager(video_path, num_threads=1, **kwargs):
    """
    一个健壮的 VideoReader 上下文管理器。
    - 确保资源被正确释放。
    - 默认设置 num_threads=1，这是在多进程 DataLoader 中最推荐的设置。
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=num_threads, **kwargs)
    try:
        yield vr
    finally:
        # 显式关闭，然后清理
        try:
            vr.close()
        except AttributeError:
            pass # 兼容可能没有 close 方法的旧版本
        del vr
        gc.collect()

class CameraVideoDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            base_path, 
            annotation_paths, 
            steps_per_epoch, 
            max_num_frames=149, 
            frame_interval=1, 
            num_frames=81, 
            video_duration_bins=[2, 3, 4, 5, 6],
            is_i2v=False, 
            is_camera=False
        ):
        t0 = time.time()
        processed_metadata_path="/mnt/workspace/processed_metadata_withoutMiraData9K.pkl"
        print(f"正在从 {processed_metadata_path} 加载预处理好的元数据...")
        try:
            # 1. 检查缓存文件是否存在
            if os.path.exists(processed_metadata_path):
                print(f"发现缓存文件: {processed_metadata_path}，正在尝试加载...")
                with open(processed_metadata_path, 'rb') as f:
                    # 2. 如果文件存在，尝试加载
                    #    这里可以捕获特定的加载错误，比如文件损坏
                    try:
                        self.metadata = pickle.load(f)
                        print(f"元数据加载成功！共计{len(self.metadata)}")
                    except (pickle.UnpicklingError, EOFError) as e:
                        print(f"警告：缓存文件 '{processed_metadata_path}' 已损坏或为空 ({e})。将重新进行预处理...")
                        # 将 metadata 设为 None，触发下面的预处理逻辑
                        self.metadata = None
            else:
                # 如果文件不存在，也标记为 None，触发预处理
                print(f"未找到缓存文件: {processed_metadata_path}")
                self.metadata = None

            # 3. 如果 metadata 未能成功加载，则执行预处理
            if self.metadata is None:
                print("开始重新预处理元数据，这可能需要一些时间...")
                print(f"源标注文件: {annotation_paths[0]}")
                print(f"数据根目录: {base_path}")
                print(f"将保存至: {processed_metadata_path}") # <--- 直接使用原始期望的路径

                # 直接调用预处理函数，并将结果保存在我们期望的路径
                preprocess_metadata(
                    annotation_paths, 
                    base_path, 
                    processed_metadata_path, # <--- 将新文件保存到原始路径
                    num_frames=num_frames
                )
                
                # 4. 从新创建的文件中加载数据
                print("预处理完成，正在加载新生成的元数据...")
                with open(processed_metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                print("新元数据加载成功！")

        except Exception as e:
            # 捕获其他所有意外错误，例如权限问题等
            print(f"在数据加载/预处理过程中发生严重错误: {e}")
            # 在这里可以决定是抛出异常让程序停止，还是进行其他处理
            raise e
        self.metadata = self.metadata[:10000]
        self.text = [entry["text"] if random.random() < 0.9 else "" for entry in self.metadata]
        ASPECT_RATIO = get_aspect_ratio(size=632) # 假设 video_sample_size 是固定的
        self.aspect_ratio_sample_size = {key: [int(x / 16) * 16 for x in ASPECT_RATIO[key]] for key in ASPECT_RATIO.keys()}

        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.video_sample_fps = 16
        self.num_frames = num_frames
        self.is_i2v = is_i2v
        self.is_camera = is_camera
        self.steps_per_epoch = steps_per_epoch
        self.video_duration_bins = video_duration_bins

        print(f"dataset_init_use_time:{time.time()-t0}")
    
    def __len__(self):
        if self.steps_per_epoch > 0:
            return self.steps_per_epoch
        else:
            return len(self.metadata)
    
    def _get_dimensions_and_duration(self, idx):
        """【重大改动】此方法现在是快速的字典查找，不再执行文件I/O"""
        item = self.metadata[idx % len(self.metadata)]
        return item.get('height', 0), item.get('width', 0), item.get('duration', 0.0)

    def _find_replacement(self, a):
        # 找到 <= a 的最大形如 (4k+1) 的数
        while a > 0:
            if (a - 1) % 4 == 0:
                return a
            a -= 1
        return 0

    def _load_video_with_decord(self, video_path, indices):
        """使用 decord 和上下文管理器高效、健壮地加载视频帧"""
        if len(indices)==0: # 对空列表的更Pythonic的检查
            return None
            
        try:
            # 使用 with 语句调用上下文管理器
            with VideoReaderManager(video_path, num_threads=1) as vr:
                # 检查请求的索引是否越界
                max_index = max(indices)
                if max_index >= len(vr):
                    # print(f"警告: 请求的索引 {max_index} 超出视频 '{video_path}' 的总帧数 {len(vr)}")
                    return None
                
                frames = vr.get_batch(indices).asnumpy()
            
            # BGR -> RGB, HWC -> CHW, 归一化
            frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
            frames_tensor = frames_tensor[:, [2, 1, 0], :, :]
            return frames_tensor.float() / 255.0

        except Exception as e:
            # print(f"错误: 使用decord加载 '{video_path}' 失败: {e}")
            return None
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False

    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        # frame = self.crop_and_resize(frame)
        first_frame = frame
        # frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame

    def get_dimensions_and_duration(self, idx):
        """[公共接口] 为 Sampler 提供获取数据尺寸和时长的方法。"""
        return self._get_dimensions_and_duration(idx=idx)
    
    def _get_sample_indices(self, item_meta):
        """辅助函数，用于计算采样帧的索引"""
        if self.video_sample_fps <= 0. or item_meta.get('fps', None) is None:
            video_sample_stride = 1
        else:
            src_fps = item_meta['fps']
            video_sample_stride = src_fps / float(self.video_sample_fps)

        video_length = self._find_replacement(item_meta['nb_frames'])

        if video_length < 13:
            print(f"视频帧数 {video_length} < 13，可能无法采样到足够帧数")
        if "RealEstate10K" in item_meta['path']:
            self.video_length_drop_start = 0.05
            self.video_length_drop_end = 0.95
        else:
            self.video_length_drop_start = 0.0
            self.video_length_drop_end = 1.0 

        min_sample_n_frames = min(
            self.num_frames,
            int(video_length * (self.video_length_drop_end - self.video_length_drop_start) // video_sample_stride)
        )

        if min_sample_n_frames == 0:
            raise ValueError(f"No Frames in video.")
        
        # 为了对齐face处理时加入的4n + 1帧数对齐处理
        video_length = int(self.video_length_drop_end * video_length)

        clip_length = math.ceil(min(video_length, (min_sample_n_frames - 1) * video_sample_stride + 1))

        lower_bound = int(self.video_length_drop_start * video_length)
        upper_bound = video_length - clip_length

        if lower_bound > upper_bound:
            # 如果上界小于下界，说明没有随机空间了，只能从上界开始
            start_frame_id = upper_bound
        else:
            start_frame_id = random.randint(lower_bound, upper_bound)
        # start_frame_id = 0
        indices = np.floor(start_frame_id + np.arange(min_sample_n_frames) * video_sample_stride).astype(int)
        indices = np.clip(indices, 0, video_length - 1)

        return indices

    # def __getitem__(self, index):
    #     try:
    #         data_id = index % len(self.metadata)
    #         item_meta = self.metadata[data_id]
            
    #         # 1. 计算采样索引
    #         indices = self._get_sample_indices(item_meta['nb_frames'])
    #         if len(indices) == 0:
    #             return None

    #         # 2. 加载所有原始视频数据 (此时是未变换的原始尺寸)
    #         video = self._load_video_with_decord(item_meta['path'], indices)
    #         if video is None:
    #             return None

    #         video_id = list(indices)
            
    #         # 3. 构建包含所有原始数据的基础字典
    #         data = {"text": self.text[data_id], "video": video, "path": item_meta['path'], "video_id": video_id, "type": "video"}
            
    #         if self.is_i2v:
    #             data["first_frame"] = video[0].clone()

    #         if self.is_camera:
    #             if item_meta["camera_extrinsics"] is None:
    #                 return None
    #             render_video = self._load_video_with_decord(item_meta['render_video_path'], indices)
    #             render_mask = self._load_video_with_decord(item_meta['render_mask_path'], indices)
    #             if render_video is None or render_mask is None:
    #                 return None
    #             data['render_video'] = render_video
    #             data['render_mask'] = render_mask
    #             # 注意：此时内外参还是 numpy 数组，在计算完 embedding 后再转为 Tensor
    #             data["camera_extrinsics"] = item_meta["camera_extrinsics"]
    #             data["camera_intrinsics"] = item_meta["camera_intrinsics"]

    #         # ---------------------------------------------------------------------------------
    #         # 核心改动：将变换和计算逻辑从 collate_fn 移到这里
    #         # ---------------------------------------------------------------------------------

    #         # 4. 【新增】根据样本自身尺寸，计算其所属分桶的目标尺寸
    #         # video 张量的形状是 [F, C, H, W]
    #         _, _, h, w = data["video"].shape
    #         closest_ratio = get_closest_ratio_key(h, w, ratios_dict=self.aspect_ratio_sample_size)
    #         closest_size = self.aspect_ratio_sample_size[closest_ratio] # 例如: [320, 512]

    #         if closest_size[0] / h > closest_size[1] / w:
    #             resize_size = closest_size[0], int(w * closest_size[0] / h)
    #         else:
    #             resize_size = int(h * closest_size[1] / w), closest_size[1]
                
    #         # 5. 【新增】定义并应用图像/视频变换
    #         # 使用 transforms.v2，它可以直接处理 [..., H, W] 格式的张量，无需循环
    #         transform = transforms.Compose([
    #             transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
    #             transforms.CenterCrop(closest_size),
    #             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #         ])

    #         # 对字典中所有需要变换的张量进行处理
    #         for key in ["video", "first_frame"]:
    #             if key in data:
    #                 data[key] = transform(data[key])
            
    #         # 6. 【新增】计算 Plücker Embedding
    #         if self.is_camera:
    #             # 使用变换后的最终尺寸来计算 Embedding
    #             # t_plucker = time.time()
    #             camera_plucker_embedding = get_plucker_embedding_cpu(
    #                 data["camera_extrinsics"], 
    #                 data["camera_intrinsics"], 
    #                 height=closest_size[0], 
    #                 width=closest_size[1]
    #             )
    #             # print("plucker_consume:", time.time()-t_plucker)
    #             # 根据采样出的 video_id 选择对应的 embedding
    #             data["camera_plucker_embedding"] = camera_plucker_embedding[:, video_id, :, :]
                
    #             # 【改动】将内外参转为 Tensor
    #             data["camera_extrinsics"] = torch.from_numpy(data["camera_extrinsics"])
    #             data["camera_intrinsics"] = torch.from_numpy(data["camera_intrinsics"])

    #         else: # 确保即使不在 is_camera 模式下也有这个字段，避免 collate 出错
    #             data["camera_plucker_embedding"] = torch.zeros(1) # 放一个占位符

    #         # 7. 返回处理完成的、所有张量尺寸都已统一的数据
    #         return data

    #     except Exception as e:
    #         # traceback.print_exc()
    #         # print(f"警告: 处理样本 {index} 时发生未知错误: {e}。将跳过此样本。")
    #         return None

    def __getitem__(self, index):
        try:
            data_id = index % len(self.metadata)
            item_meta = self.metadata[data_id]
            
            # 1. 计算采样索引 (保留)
            indices = self._get_sample_indices(item_meta)
            if len(indices) == 0:
                return None

            # 2. 加载所有原始视频数据，不进行任何变换 (保留)
            video = self._load_video_with_decord(item_meta['path'], indices)
            if video is None:
                return None

            video_id = list(indices)

            # ===== 修改开始 =====
            # 随机选择 1 到 5 帧作为 ID 图像列表
            num_id_images = random.randint(1, 5)
            id_indices = [random.randint(0, video.shape[0] - 1) for _ in range(num_id_images)]
            # 将这些帧收集到一个列表中
            id_images_list = [video[i].clone() for i in id_indices]
            # ===== 修改结束 =====
            
            # 3. 构建包含所有原始数据的基础字典 (保留)
            data = {
                "text": self.text[data_id], 
                "video": video,
                "path": item_meta['path'], 
                "video_id": video_id, 
                "type": "video",
                "id_image": id_images_list # 【重要】现在返回的是一个Tensor列表
            }
            
            if self.is_i2v:
                data["first_frame"] = video[0].clone() # <-- 这是原始尺寸的帧

            if self.is_camera:
                if item_meta.get("camera_extrinsics") is None: # 使用 .get() 更安全
                    return None
                    
                render_video = self._load_video_with_decord(item_meta['render_video_path'], indices)
                render_mask = self._load_video_with_decord(item_meta['render_mask_path'], indices)
                if render_video is None or render_mask is None:
                    return None
                    
                data['render_video'] = render_video
                data['render_mask'] = render_mask
                
                # 【注意】只传递原始的 numpy 数组
                data["camera_extrinsics"] = item_meta["camera_extrinsics"] if not np.any(item_meta["camera_extrinsics"]) else item_meta["camera_extrinsics"][indices]
                data["camera_intrinsics"] = item_meta["camera_intrinsics"] if not np.any(item_meta["camera_intrinsics"]) else item_meta["camera_intrinsics"][indices]

            # 4. 直接返回原始数据，所有变换和计算都已移除
            return data

        except Exception as e:
            # traceback.print_exc()
            # print(f"警告: 处理样本 {index} 时发生未知错误: {e}。将跳过此样本。")
            return None
        


def get_dataloader(args, sample_n_frames_bucket_interval=4):
    # Get the frame length at different resolutions according to token_length
    def get_length_to_frame_num(token_length):
        if args.image_sample_size > args.video_sample_size:
            sample_sizes = list(range(args.video_sample_size, args.image_sample_size + 1, 32))

            if sample_sizes[-1] != args.image_sample_size:
                sample_sizes.append(args.image_sample_size)
        else:
            sample_sizes = [args.image_sample_size]
            
        length_to_frame_num = {
            sample_size: min(token_length / sample_size / sample_size, 81) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1 for sample_size in sample_sizes
        }  # 选择最小的帧数裁剪


        return length_to_frame_num

    def collate_fn(examples):
        # Get token length
        video_sample_frames = 81
        video_sample_size = 632
        target_token_length = video_sample_frames * video_sample_size * video_sample_size
        # length_to_frame_num = get_length_to_frame_num(target_token_length)


        # Create new output
        new_examples = {}
        new_examples["video"] = []
        new_examples["text"] = []
        new_examples["path"] = []
        new_examples["video_id"] = []
        new_examples["type"] = []
        new_examples["render_video"] = []
        new_examples["render_mask"] = []
        # new_examples["camera_extrinsics"] = []
        # new_examples["camera_intrinsics"] = []
        new_examples["camera_plucker_embedding"] = []

        new_examples["first_frame"] = []
        # Get downsample ratio in image and videos
        pixel_value = examples[0]["video"]
        data_type = examples[0]["type"]
        batch_video_length = video_sample_frames

        ASPECT_RATIO = get_aspect_ratio(size=video_sample_size)
        aspect_ratio_sample_size = {key : [x for x in ASPECT_RATIO[key]] for key in ASPECT_RATIO.keys()}

        for example in examples:
            # To 0~1
            pixel_values = example["video"]
            f, c, h, w  = np.shape(pixel_value)
            closest_ratio = get_closest_ratio_key(h, w, ratios_dict=aspect_ratio_sample_size)
            closest_size = aspect_ratio_sample_size[closest_ratio]
            closest_size = [int(x / 16) * 16 for x in closest_size]
            
            if closest_size[0] / h > closest_size[1] / w:
                resize_size = closest_size[0], int(w * closest_size[0] / h)
            else:
                resize_size = int(h * closest_size[1] / w), closest_size[1]
            transform = transforms.Compose([
                transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(closest_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])
            if "first_frame" in example:
                first_frame = example["first_frame"]
                first_frame = transform(first_frame)
                new_examples["first_frame"].append(first_frame)
            new_examples["video"].append(transform(pixel_values))
            if "render_video" in example:
                new_examples["render_video"].append(example["render_video"])
                new_examples["render_mask"].append(example["render_mask"])
            else:
                new_examples["render_video"].append(transform(pixel_values))
                new_examples["render_mask"].append(transform(pixel_values))
            new_examples["text"].append(example["text"])
            new_examples["path"].append(example['path'])
            new_examples["video_id"].append(example["video_id"])
            new_examples["type"].append(example["type"])

            video_id = example["video_id"]

            if torch.any(example["camera_extrinsics"]):
                camera_plucker_embedding = get_plucker_embedding_cpu(example["camera_extrinsics"], example["camera_intrinsics"], height=closest_size[0], width=closest_size[1])
                camera_plucker_embedding = camera_plucker_embedding[:,video_id,:,:]
                new_examples["camera_plucker_embedding"].append(camera_plucker_embedding)
            else:
                zero_camera_plucker_embedding = torch.zeros([6, len(video_id), closest_size[0], closest_size[1]])
                new_examples["camera_plucker_embedding"].append(zero_camera_plucker_embedding)

            # needs the number of frames to be 4n + 1.
            batch_video_length = int(
                min(
                    batch_video_length,
                    (len(pixel_values) - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1,
                )
            )
        if batch_video_length == 0:
            batch_video_length = 1
        # Limit the number of frames to the same
        new_examples["video"] = torch.stack(
            [example[:batch_video_length] for example in new_examples["video"]])
        new_examples["render_video"] = torch.stack(
            [example[:batch_video_length] for example in new_examples["render_video"]])
        new_examples["render_mask"] = torch.stack(
            [example[:batch_video_length] for example in new_examples["render_mask"]])
        if "first_frame" in new_examples:
            new_examples["first_frame"] = torch.stack(
                [example for example in new_examples["first_frame"]])
        if 'camera_plucker_embedding' in new_examples:
            new_examples["camera_plucker_embedding"] = torch.stack(
                [example[:,:batch_video_length,:,:] for example in new_examples["camera_plucker_embedding"]]
            )
        # try:
        #     new_examples["camera_extrinsics"] = torch.stack(
        #         [example[:batch_video_length] for example in new_examples["camera_extrinsics"]])
        #     new_examples["camera_intrinsics"] = torch.stack(
        #         [example[:batch_video_length] for example in new_examples["camera_intrinsics"]])
        # except Exception as e:
        #     print(batch_video_length)
        #     for example in new_examples["camera_extrinsics"]:
        #         print(example.shape)
        #     for example in new_examples["video"]:
        #         print(example.shape)
        return new_examples

    # 加载数据集
    dataset_path = args.dataset_path
    dataset = CameraVideoDataset(
        dataset_path,
        args.dataset_list,
        steps_per_epoch=args.steps_per_epoch,
        max_num_frames=121,
        num_frames=args.num_frames,
        is_i2v=args.is_i2v,     # 根据你的使用情况
        is_camera=args.is_camera,   # 确保启用 camera 相关字段
    )
    batch_size = args.per_device_batch_size
    num_workers= args.dataloader_num_workers
    if args.enable_bucket:
        ASPECT_RATIO = get_aspect_ratio(size=args.video_sample_size)
        aspect_ratio_sample_size = {key : [x for x in ASPECT_RATIO[key]] for key in ASPECT_RATIO.keys()}
        batch_sampler_generator = torch.Generator().manual_seed(42)
        batch_sampler = AspectRatioImageVideoSampler(
            sampler=RandomSampler(dataset, generator=batch_sampler_generator), dataset=dataset,
            batch_size=batch_size, drop_last=True,
            aspect_ratios_dict=aspect_ratio_sample_size,
            video_duration_bins = None,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
    # else:
    #     dataloader = torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         num_workers=num_workers,
    #     )        
    return dataset, dataloader, batch_sampler
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a data script.")
    parser.add_argument(
        "--image_sample_size",
        type=int,
        default=632
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=632
    )
    parser.add_argument(
        "--token_sample_size",
        type=int,
        default=632
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/mnt/data/camera_datasets/MuteApo/RealCam-Vid",
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=0,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=4,
        help="Batch size of each device",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--dataset_list",
        type=parse_comma_separated_list,
        default='/mnt/data/camera_datasets/MuteApo/RealCam-Vid/RealCam-Vid_train_camera.npz',
        help="dataset file list."
    )
    parser.add_argument(
        "--is_i2v",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--is_camera",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--enable_bucket",
        type=bool,
        default=True
    )
    args = parser.parse_args()
    return args
def parse_comma_separated_list(value):
    """解析逗号分隔的字符串为列表"""
    return [item.strip() for item in value.split(',')]
import torch
from torch.utils.data import DataLoader
from collections import defaultdict, Counter

def verify_batches_resolution_consistency(dataloader, num_batches=10):
    """
    验证每个batch内的图像分辨率是否一致
    """
    print("=== Batch Resolution Consistency Check ===")
    
    batch_resolutions = []
    inconsistent_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        videos = batch['video'] 

        # 获取batch中所有图像的分辨率
        batch_shapes = []
        for i in range(videos.shape[0]):
            video = videos[i]
            shape = video.shape  # (H, W)
            batch_shapes.append(shape)
        
        # 检查是否所有图像分辨率相同
        unique_shapes = set(batch_shapes)
        is_consistent = len(unique_shapes) == 1
        
        print(f"Batch {batch_idx}:")
        print(f"  Batch size: {len(batch_shapes)}")
        print(f"  Unique resolutions: {unique_shapes}")
        print(f"  Consistent: {'✅' if is_consistent else '❌'}")
        
        if not is_consistent:
            inconsistent_batches += 1
            shape_counts = Counter(batch_shapes)
            print(f"  Resolution distribution: {dict(shape_counts)}")
        
        batch_resolutions.extend(batch_shapes)
    
    print(f"Summary:")
    print(f"  Total batches checked: {num_batches}")
    print(f"  Inconsistent batches: {inconsistent_batches}")
    print(f"  Consistency rate: {(num_batches - inconsistent_batches) / num_batches * 100:.1f}%")
    
    # 整体分辨率分布
    overall_resolution_counts = Counter(batch_resolutions)
    print(f"  Overall resolution distribution: {dict(overall_resolution_counts)}")
    
    return inconsistent_batches == 0

# 使用示例
# dataloader = DataLoader(dataset, batch_sampler=your_bucket_sampler)
# is_consistent = verify_batches_resolution_consistency(dataloader)
if __name__ == "__main__":
    args = parse_args()
    dataloader = get_dataloader(args)
    is_consistent = verify_batches_resolution_consistency(dataloader)
    print(is_consistent)

