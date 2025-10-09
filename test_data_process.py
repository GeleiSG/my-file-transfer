import os
import torch
from PIL import Image
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from diffsynth.data.camera_video import CameraVideoDataset
import torch.nn as nn
import imageio
from einops import rearrange
import numpy as np
import subprocess


dataset_path = '/mnt/data/hdd/yqf_camera_datasets/MuteApo/RealCam-Vid'
dataset = CameraVideoDataset(
    dataset_path,
    os.path.join(dataset_path, 'RealCam-Vid_test.npz'),
    os.path.join(dataset_path, 'camera_caption_total.json'),
    steps_per_epoch=5000,
    max_num_frames=129,
    frame_interval=1,
    num_frames=81,
    height=480,
    width=832,
    is_i2v=True,     # 根据你的使用情况
    is_camera=True   # 确保启用 camera 相关字段
)
dataloader = torch.utils.data.DataLoader(
    dataset,
    shuffle=False,
    batch_size=1,
    num_workers=4,
)

data = {}

# for batch_idx, batch in enumerate(dataloader):
#     target_text = batch["text"][0]
#     target_camera = batch["camera_embedding"]
#     # print(target_camera)

#     path = batch["path"][0]
#     first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
#     key = path.split('/')[-1]
#     data[key] = {
#         "idx": batch_idx,
#         "camera": target_camera,
#         "text": target_text,
#         "path": path,
#         "first_frame": batch["first_frame"][0].cpu()
#     }
# torch.save(data, 'data/test_total_data.pt')
data = torch.load('data/test_total_data.pt')
print(len(data))
valid_paths = ['28597f24a2e1ef7875adff3aec0b4f161d3063d4187df77d1e74f804b418254b_2.mp4',
'000004173507.0.005.mp4',
'000000000479.4.012.mp4',
'e0ea922c93901290231a221494f63f5361644204e77baeca46986baa5b41b3b6_0.mp4',
'919df00389b110e1.mp4',
'000001203791.13.027.mp4',
'000000726896.0.026.mp4',
'000001887524.63.015.mp4',
'000000312473.40.008.mp4',
'e51df845689c6cb9.mp4']

valid_data = {}
for batch_idx, path in enumerate(valid_paths):
    if path in data:
        valid_data[batch_idx] = data[path]
print(len(valid_data))
torch.save(valid_data, 'data/test_valid_data.pt')
