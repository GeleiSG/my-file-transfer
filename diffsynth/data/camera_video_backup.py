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

from decord import VideoReader
from contextlib import contextmanager
import gc
import subprocess
import json
import logging
import time


@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    """
    一个上下文管理器，旨在确保 decord.VideoReader 的资源被正确、及时地释放。
    `decord` 在某些情况下可能存在内存泄漏风险，此管理器通过显式删除和垃圾回收来增强其健壮性。
    """
    # 初始化 VideoReader。设置 num_threads=0 是为了避免与 PyTorch DataLoader 的多进程（num_workers > 0）机制发生冲突或死锁。
    # 这里的 `*args` 和 `**kwargs` 使得这个管理器可以接受 VideoReader 的所有原生参数。
    vr = VideoReader(*args, **kwargs)
    try:
        # yield 将 VideoReader 对象返回给 with 语句块
        yield vr
    finally:
        # with 语句块结束后，无论成功或异常，这里的代码都会被执行
        # decord 的 VideoReader 需要手动关闭，这里通过 del 和强制垃圾回收 (gc.collect) 来确保资源被释放
        # 优先用官方 close 方法
        try:
            vr.close()
        except AttributeError:
            pass
        del vr
        gc.collect()

logger = logging.getLogger(__name__)

def get_video_meta_ffprobe(video_path: str) -> dict | None:
    """
    使用 ffprobe 高效、可靠地获取视频的分辨率和时长。
    此函数不解码任何视频帧，只读取元数据。

    Args:
        video_path (str): 视频文件的路径。

    Returns:
        dict: 包含 'width', 'height', 'duration' 的字典，失败时返回 None。
    """
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,duration,nb_frames',
        '-of', 'json',
        video_path
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        meta_data = json.loads(result.stdout)['streams'][0]

        # --- 分辨率 ---
        width = int(meta_data['width'])
        height = int(meta_data['height'])

        # --- 时长计算 (更稳健的方式) ---
        # 优先使用 'duration' 字段，如果不存在或为0，则通过帧数和帧率计算
        if 'duration' in meta_data and float(meta_data['duration']) > 0:
            duration = float(meta_data['duration'])
        else:
            # r_frame_rate 通常是 '24/1' 或 '30000/1001' 这样的分数
            num, den = map(int, meta_data.get('r_frame_rate', '0/1').split('/'))
            if den == 0: # 避免除零错误
                return None
            fps = num / den
            nb_frames = int(meta_data.get('nb_frames', '0'))
            if fps == 0 or nb_frames == 0:
                return None
            duration = nb_frames / fps
            
        return {'width': width, 'height': height, 'duration': duration}

    except FileNotFoundError:
        logger.critical("命令 'ffprobe' 未找到。请确保 FFmpeg 已安装并在系统 PATH 中。")
        # 如果这是个致命错误，可以考虑直接抛出异常
        raise
    except (subprocess.CalledProcessError, KeyError, IndexError, json.JSONDecodeError):
        # 捕获所有可能的错误：文件损坏、非视频文件、JSON解析失败等
        # logger.warning(f"无法获取视频元信息: {video_path}")
        return None

class CameraVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, annotation_paths, steps_per_epoch, max_num_frames=149, frame_interval=1, num_frames=81, is_i2v=False, is_camera=False):
        
        t0 = time.time()
        valid_metadata = []
        # merge_data
        for meta_file_path in annotation_paths:
            print(f"loading annotations from {meta_file_path} ...")
            if meta_file_path.endswith('.npz'):
                metadata = np.load(meta_file_path, allow_pickle=True)["arr_0"]  # 加载npz，拿arr_0
                valid_metadata.extend(metadata)
            elif meta_file_path.endswith('.json'):
                extend_data1 = json.load(open(meta_file_path))
                for i, data in enumerate(extend_data1):
                    extrinsics = []
                    intrinsics = []
                    update_data = {}
                    if "extrinsic_array" not in data:
                        update_data["camera_extrinsics"] = np.zeros((num_frames,4,4))
                        update_data["camera_intrinsics"] = np.zeros((num_frames,4))
                    else:
                        for i in range(np.array(data["extrinsic_array"]).shape[0]):
                            mat_4x4 = np.eye(4)
                            mat_4x4[:3, :] = np.array(data["extrinsic_array"])[i]
                            fx, fy, cx, cy = np.array(data["intrinsic_array"])[i][0][0], np.array(data["intrinsic_array"])[i][1][1], np.array(data["intrinsic_array"])[i][0][2], np.array(data["intrinsic_array"])[i][1][2] 
                            extrinsics.append(mat_4x4)
                            intrinsics.append(np.array([fx,fy,cx,cy]))
                        camera_extrinsics = np.stack(extrinsics)
                        camera_intrinsics = np.stack(intrinsics)
                        update_data["camera_extrinsics"] = camera_extrinsics
                        update_data["camera_intrinsics"] = camera_intrinsics
                    if 'selected_id' in data:
                        update_data["video_id"] = data['selected_id']
                    update_data["video_path"] = data['video_path']
                    update_data["camera_caption"] = data["caption"] if 'caption' in data else data["text"]
                    valid_metadata.append(update_data)


        print("update total number of videos: ", len(valid_metadata))
        # if camera_caption_path is not None and os.path.exists(camera_caption_path):
        #     camera_captions = json.load(open(camera_caption_path, "r"))
        #     self.camera_captions = {entry["video_path"]: entry["camera_caption"] for entry in camera_captions}

        # self.text = [entry["long_caption"] +  ' ' + self.camera_captions[entry["video_path"]] if entry["video_path"] in self.camera_captions else entry["long_caption"] for entry in valid_metadata]
        # self.text = []
        # for entry in metadata:
        #     if entry["video_path"] in camera_captions:
        #         self.text.append(entry["long_caption"] + ' ' + self.camera_captions[entry["video_path"]] if random.random() < 0.5 else "")
        #     else:
        #         self.text.append(entry["long_caption"] if random.random() < 0.5 else "") 
        # self.text = [entry["camera_caption"] if "camera_caption" in entry else entry["short_caption"] for entry in valid_metadata]
        # self.text = ["" for entry in valid_metadata]
        # self.text = ["" for entry in valid_metadata]
        self.text = [entry["camera_caption"] if random.random() < 0.9 else "" for entry in valid_metadata]

        # if "camera_caption" in valid_metadata[0]:
        #     self.text = [entry["camera_caption"] if random.random() < 0.3 else "" for entry in valid_metadata]
        # else:
        #     self.text = [entry["short_caption"] if random.random() < 0.3 else "" for entry in valid_metadata]
        
        self.path = []
        for entry in valid_metadata:
            if 'MultiCamVideo-Dataset' in entry["video_path"]:
                self.path.append(os.path.join('/mnt/data/camera_datasets/KwaiVGI/MultiCamVideo-Dataset', entry["video_path"]))
            elif 'openhumanvid' in entry["video_path"]:
                self.path.append(os.path.join('/mnt/data/omnihuman', entry["video_path"]))
            elif 'mixkit' in entry["video_path"]:
                self.path.append(os.path.join('/mnt/data/hdd/user_workspace/duanke/video_mixkit_81f_26347/output', entry["video_path"]))
            else:
                self.path.append(os.path.join(base_path, entry["video_path"]))
        render_iterator = map(self.get_point_cloud_video_path, self.path)
        render_paths_iter, mask_paths_iter = zip(*render_iterator)
        self.render_video_path = list(render_paths_iter)
        self.render_mask_path = list(mask_paths_iter)

        # self.render_video_path = self.path
        # self.render_mask_path = ['render_mask.mp4' for entry in valid_metadata]
        # os.path.join('/mnt/data/camera_datasets/KwaiVGI/MultiCamVideo-Dataset', entry["video_path"]) if 'MultiCamVideo-Dataset' in entry["video_path"] else os.path.join(base_path, entry["video_path"]) for entry in valid_metadata
        # self.path = [os.path.join(base_path, entry["video_path"]) if '' for entry in valid_metadata]

        # self.align_factor = [float(entry["align_factor"]) for entry in valid_metadata]
        # self.camera_scale = [float(entry["camera_scale"]) for entry in valid_metadata]
        # self.vtss_score = [float(entry["vtss_score"]) for entry in valid_metadata]
        
        self.camera_extrinsics = [entry["camera_extrinsics"] for entry in valid_metadata]
        self.camera_intrinsics = []
        for entry in valid_metadata:
            if entry["camera_intrinsics"].ndim == 1:
                self.camera_intrinsics.append(np.repeat(entry["camera_intrinsics"][np.newaxis, :], len(entry["camera_extrinsics"]), axis=0))
            else:
                self.camera_intrinsics.append(entry["camera_intrinsics"])
        self.video_id = [entry["video_id"] if "video_id" in entry else [] for entry in valid_metadata]
        # self.camera_intrinsics = [np.repeat(entry["camera_intrinsics"][np.newaxis, :], len(entry["camera_extrinsics"]), axis=0) for entry in metadata] # 这里repeat对每一帧保留一个内参，用于统一不同帧不同内参
        # x = get_plucker_embedding(self.camera_extrinsics[0], self.camera_intrinsics[0], height, width)

        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.video_sample_fps = 16
        self.num_frames = num_frames
        self.is_i2v = is_i2v
        self.is_camera = is_camera
        self.steps_per_epoch = steps_per_epoch

        print(f"dataset_init_use_time:{time.time()-t0}")

    def get_point_cloud_video_path(self, video_path):
        directory_path = os.path.dirname(video_path)
        file_name = os.path.basename(video_path)
        file_name_without_ext , ext = os.path.splitext(file_name)

        render_video_path = os.path.join(directory_path, f"{file_name_without_ext}_render{ext}")
        render_mask_path = os.path.join(directory_path, f"{file_name_without_ext}_render_mask{ext}")

        return render_video_path, render_mask_path
        

    def load_video_frames(self, reader, indices):
        frames = []
        frames_list = []
        first_frame = None
        for frame_id in indices:
            frame = reader.get_data(frame_id)
            frames_list.append(frame_id.item())
            frame = Image.fromarray(frame)
            if first_frame is None:
                first_frame = frame
            frames.append(frame)
        reader.close()

        frames = torch.stack([ToTensor()(frame) for frame in frames], dim=0)  # [f,c,h,w]
        first_frame = ToTensor()(first_frame)
        # frames = torch.stack(frames, dim=0)
        # frames = rearrange(frames, "T C H W -> C T H W")
        
        # first_frame = v2.functional.center_crop(first_frame, output_size=(self.height, self.width))
        # first_frame = np.array(first_frame)
        
        if self.is_i2v:
            return frames_list, frames, first_frame
        else:
            return frames_list, frames

    def load_frames_using_imageio(self, reader, file_path, indices, frame_process, max_num_frames):
        
        frames = []
        frames_list = []
        first_frame = None
        for frame_id in indices:
            frame = reader.get_data(frame_id)
            frames_list.append(frame_id.item())
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = frame
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        
        first_frame = v2.functional.center_crop(first_frame, output_size=(self.height, self.width))
        first_frame = np.array(first_frame)
        
        if self.is_i2v:
            return frames_list, frames, first_frame, max_num_frames
        else:
            return frames_list, frames, max_num_frames
    
    def load_simple_video(self, video_path, indices=None):
        try:
            reader = imageio.get_reader(video_path)
            metainfo = reader.get_meta_data()
        except Exception as e:
            print(e)
            return None
        frames = []
        if indices is None:
            total_frames = reader.count_frames()
            for i in range(total_frames):
                frame = reader.get_data(i)
                frames.append(Image.fromarray(frame))
        else:
            for frame_id in indices:
                frame = reader.get_data(frame_id)
                frames.append(Image.fromarray(frame))
        reader.close()     
        frames = torch.stack([ToTensor()(frame) for frame in frames], dim=0)  # [f,c,h,w]
        return frames

    def find_replacement(self, a):
            # 取 <= a 的最大形如 (4k+1) 的数，用于某些模型对帧数的约束
            while a > 0:
                if (a - 1) % 4 == 0:
                    return a
                a -= 1
            return 0
    def load_video(self, file_path, video_id=None):
        try:
            reader = imageio.get_reader(file_path)
            metainfo = reader.get_meta_data()
        except Exception as e:
            print(e)
            return None, None
        if self.video_sample_fps <= 0. or metainfo.get('fps', None) is None:
            video_sample_stride = 1
        else:
            src_fps = metainfo['fps']
            video_sample_stride = src_fps / float(self.video_sample_fps)
        
        video_length = self.find_replacement(reader.count_frames())

        if video_length < 13:
            print(f"视频帧数 {video_length} < 13")
        self.video_length_drop_start = 0.1
        self.video_length_drop_end = 0.9

        min_sample_n_frames = min(video_length, self.num_frames)
        min_sample_n_frames = min(
            self.num_frames,
            int(video_length) * (self.video_length_drop_end - self.video_length_drop_start) // video_sample_stride)

        if min_sample_n_frames == 0:
            raise ValueError(f"No Frames in video.")
            # video_length = int(self.video_length_drop_end * len(video_reader))
        # 为了对齐face处理时加入的4n + 1帧数对齐处理
        video_length = int(self.video_length_drop_end * video_length)

        clip_length = math.ceil(min(video_length, (min_sample_n_frames - 1) * video_sample_stride + 1))

        start_frame_id = random.randint(int(self.video_length_drop_start * video_length),
                    video_length - clip_length) if video_length != clip_length else 0
        # start_frame_id = 0
        indices = np.floor(start_frame_id + np.arange(min_sample_n_frames) * video_sample_stride).astype(int)
        indices = np.clip(indices, 0, video_length - 1)

        # if len(video_id) != 0:
        #     start_frame_id = torch.tensor(video_id[0])
        #     indices = []
        #     for frame_id in range(self.num_frames):
        #         indices.append(start_frame_id + frame_id * self.frame_interval)
        # else:
        #     if self.num_frames > reader.count_frames():
        #         indices = np.linspace(0, max_num_frames - 1, self.num_frames).astype(int)
        #     else:
        #         start_frame_id = torch.randint(0, max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        #         indices = []
        #         for frame_id in range(self.num_frames):
        #             indices.append(start_frame_id + frame_id * self.frame_interval)

        frames = self.load_video_frames(reader, indices)
            
        return frames, indices
    
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

    def _get_dimensions_and_duration_from_file(self, idx):
        """
        [内部方法] 获取数据的尺寸和时长。
        如果数据类型不是视频，则时长返回 0。
        """
        idx = idx % len(self.path)
        path = self.path[idx]
        data_type = "image" if self.is_image(path) else "video"
        try:
            if data_type == 'video':
                # reader = imageio.get_reader(path)
                # metainfo = reader.get_meta_data()
                # height, width = metainfo['size'][1], metainfo['size'][0]
                # if 'duration' in metainfo and metainfo['duration'] != math.inf:
                #     duration = metainfo['duration']
                # else:
                #     duration = reader.count_frames() / metainfo['fps'] # 返回视频时长（秒）
                # with VideoReader_contextmanager(path) as vr:
                #     height, width = vr[0].shape[0], vr[0].shape[1]
                #     duration = len(vr) / vr.get_avg_fps()  # 返回视频时长（秒）
                meta = get_video_meta_ffprobe(path)
                height, width, duration = meta['height'], meta['width'], meta['duration']
                return height, width, duration
            else:  # image
                with Image.open(path) as img:
                    width, height = img.size
                return height, width, 0.0  # 图像没有时长
        except Exception as e:
            print(f"Warning: Could not get dimensions or duration for item {idx} ({path}): {e}")
            return 0, 0, 0.0

    def get_dimensions_and_duration(self, idx):
        """[公共接口] 为 Sampler 提供获取数据尺寸和时长的方法。"""
        return self._get_dimensions_and_duration_from_file(idx=idx)

    def __getitem__(self, index):
        t1 = time.time()
        video = None
        while True:
            # 1. 获取当前尝试的数据索引和基本信息
            data_id = index % len(self.path)
            path = self.path[data_id]
            text = self.text[data_id]

            # 2. 加载主视频/图片数据
            video_data = None
            indices = None
            type = "image" if self.is_image(path) else "video"

            if type == "image":
                if self.is_i2v:
                    # I2V模式下遇到图片直接跳过，尝试下一个
                    index = random.randint(0, len(self.path) - 1)
                    continue
                video_data = self.load_image(path)
            else:
                # load_video 应该返回 (video, indices) 或者在失败时返回 (None, None)
                video_data, indices = self.load_video(path)

            # 如果主数据加载失败，立即重试下一个随机索引
            if video_data is None:
                index = random.randint(0, len(self.path) - 1)
                continue
                
            # 解包视频数据
            if self.is_i2v:
                video_id, video, first_frame = video_data
            else:
                video_id, video = video_data

            # 3. 构建基础数据字典
            data = {"text": text, "video": video, "path": path, "video_id": video_id, "type": type}
            if self.is_i2v:
                data["first_frame"] = first_frame

            # 4. 如果需要，加载相机和掩码等附加数据
            if self.is_camera:
                # 检查相机外参数据是否有效
                has_valid_extrinsics = np.any(self.camera_extrinsics[data_id])
                if not has_valid_extrinsics:
                    # 如果无效，则填充零值并直接返回
                    data["camera_extrinsics"] = torch.zeros_like(torch.from_numpy(self.camera_extrinsics[data_id]))
                    data["camera_intrinsics"] = torch.zeros_like(torch.from_numpy(self.camera_intrinsics[data_id]))
                    return data # 成功加载，跳出循环并返回
                
                # 检查帧数是否匹配
                if self.camera_extrinsics[data_id].shape[0] < max(video_id):
                    index = random.randint(0, len(self.path) - 1)
                    continue

                # 加载渲染视频和掩码
                render_mask = self.load_simple_video(self.render_mask_path[data_id], indices)
                render_video = self.load_simple_video(self.render_video_path[data_id], indices)

                # 如果附加数据加载失败，重试下一个
                if render_mask is None or render_video is None:
                    index = random.randint(0, len(self.path) - 1)
                    continue

                # 附加数据加载成功，添加到字典中
                data['render_mask'] = render_mask
                data['render_video'] = render_video
                data["camera_extrinsics"] = torch.from_numpy(self.camera_extrinsics[data_id])
                data["camera_intrinsics"] = torch.from_numpy(self.camera_intrinsics[data_id])
            print(f"get_data_use_time:{time.time()-t1}")
            return data
    
    def __len__(self):
        if self.steps_per_epoch > 0:
            return self.steps_per_epoch
        else:
            return len(self.path)


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
        batch_sampler_generator = torch.Generator().manual_seed(3407)
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

