import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from diffsynth import ModelManager, save_video, VideoData
from diffsynth.data.camera_video import CameraVideoDataset
from diffsynth.models.utils import load_state_dict
import torch.nn as nn
from PIL import Image
from diffsynth.pipelines.wan_camera_video import WanVideoCameraPipeline
import imageio
from einops import rearrange
import numpy as np
import subprocess

# Load the full dit and camera adapter models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    ["/root/hdd/user_workspace/yuqifan/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32, # Image Encoder is loaded with float32
)
model_manager.load_models([
    "/root/hdd/user_workspace/yuqifan/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors",
    "/root/hdd/user_workspace/yuqifan/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth",
    "/root/hdd/user_workspace/yuqifan/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth",
], torch_dtype=torch.bfloat16)
pipe = WanVideoCameraPipeline.from_model_manager(model_manager, device="cuda")
# 2. Initialize additional modules introduced in ReCamMaster
dim=pipe.dit.blocks[0].self_attn.q.weight.shape[0]
for block_i, block in enumerate(pipe.dit.blocks):
    # if block_i < 20:
        block.cam_encoder = nn.Linear(16, dim)
        block.projector = nn.Linear(dim, dim)
        # nn.init.kaiming_normal_(block.cam_encoder.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(block.cam_encoder.bias, 0)
        # block.projector.weight.data.zero_()
        # block.projector.bias.data.zero_()
        block.cam_encoder.weight.data.zero_()
        block.cam_encoder.bias.data.zero_()
        block.projector.weight = nn.Parameter(torch.eye(dim))
        block.projector.bias = nn.Parameter(torch.zeros(dim))

# 3. Load ReCamMaster checkpoint
steps = 750
# state_dict = torch.load(f"models_1.3b_adapter_abs_tensor/lightning_logs/version_1/checkpoints/step{steps}.ckpt", map_location="cpu")
# state_dict = torch.load(f"models_1.3b_adapter_20/lightning_logs/version_0/checkpoints/step{steps}.ckpt", map_location="cpu")
# state_dict = torch.load(f"models_multi/lightning_logs/version_28/checkpoints/step{steps}.ckpt", map_location="cpu")
# state_dict = torch.load(f"models_1.3b_adapter_res_tensor/lightning_logs/version_61/checkpoints/epoch=0-step=3000.ckpt", map_location="cpu")
# state_dict = torch.load(f"/root/hdd/user_workspace/yuqifan/cameractrl/models_1.3b_adapter_res_tensor/lightning_logs/version_63/checkpoints/step20000.ckpt", map_location="cpu")
# state_dict = torch.load(f"models_1.3b_adapter_full_tensor/tensorboardlog/version_0/checkpoints/step{steps}.ckpt", map_location="cpu")
# state_dict = torch.load(f"models_1.3b_adapter_full_tensor/tensorboardlog/version_14/checkpoints/step3000.ckpt", map_location="cpu")
# state_dict = torch.load(f"models_1.3b_adapter_full_tensor/tensorboardlog/version_16/checkpoints/step4000.ckpt", map_location="cpu")
state_dict = torch.load(f"models_1.3b_full_adapter_multicam_tensor/tensorboardlog/version_0/checkpoints/step750.ckpt", map_location="cpu")
pipe.dit.load_state_dict(state_dict, strict=False) # 6000 steps after train (7.9 服务器重启前跑了6000steps)
pipe.to("cuda")
pipe.to(dtype=torch.bfloat16)

pipe.enable_vram_management(num_persistent_param_in_dit=None)


# dataset_path = '/mnt/data/camera_datasets/MuteApo/RealCam-Vid'
dataset_path = '/mnt/data/camera_datasets/KwaiVGI/MultiCamVideo-Dataset'
dataset = CameraVideoDataset(
    dataset_path,
    os.path.join(dataset_path, 'MultiCamVideo_metricscale.npz'),
    os.path.join(dataset_path, 'camera_caption_total.json'),
    steps_per_epoch=10,
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
    shuffle=True,
    batch_size=1,
    num_workers=4,
)

import json
total_data = torch.load('data/test_valid_data.pt')
real_img = json.load(open('data/openhumanvid/caption_part.json'))
for batch_idx, batch in enumerate(dataloader):
# for batch_idx, key in enumerate(total_data):
    # batch = total_data[key]
    # target_camera = batch["camera"]
    # camera_embedding = torch.cat([batch["camera_intrinsics"], rearrange(batch["camera_extrinsics"], 'b c d f -> b c (d f)')], dim=-1)
    camera_embedding = rearrange(batch["camera_extrinsics"], 'b c d f -> b c (d f)')
    path = batch["path"]
    target_text = batch["text"][0]
    first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
    video = pipe(
            prompt=target_text,
            negative_prompt="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            input_image=first_frame,
            camera_pose =camera_embedding,
            num_inference_steps=50,
            height=480, width=832,
            seed=0, tiled=True
        )
    save_video(video, f"data/valid_test/0720/video_adapter_repeat_camera_1projector_64_1e-4_epoch50_steps{steps}_{batch_idx}.mp4", fps=16, quality=5)
    print(path)
    for image_id in real_img:
        first_frame = Image.open(f"data/openhumanvid/{image_id}.png")
        target_text = real_img[image_id]
        video = pipe(
            prompt=target_text,
            negative_prompt="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            input_image=first_frame,
            camera_pose =camera_embedding,
            num_inference_steps=50,
            height=480, width=832,
            seed=0, tiled=True
        )
        save_video(video, f"data/valid_test/0720/video_adapter_repeat_camera_1projector_64_1e-4_epoch50_steps{steps}_{batch_idx}_{image_id}.mp4", fps=16, quality=5)
    # if batch_idx == 3:
    #     break