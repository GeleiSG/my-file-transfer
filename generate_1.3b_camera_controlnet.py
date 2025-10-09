import os

import einops
from omegaconf import OmegaConf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from diffsynth.data.camera_utils import get_plucker_embedding
from diffsynth.models.wan_video_camera_adapter import SimpleAdapter

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
pipe = WanVideoCameraPipeline.from_model_manager(model_manager, device="cpu")
# 2. Initialize additional modules introduced in ReCamMaster
dim=pipe.dit.blocks[0].self_attn.q.weight.shape[0]
# controlnet
cfg = OmegaConf.load("/mnt/data/ssd/user_workspace/yuqifan/unicontrol/diffsynth/configs/config_yqf.json")
pipe.dit.controlnet_cfg = cfg.controlnet_cfg
pipe.dit.build_controlnet()

# 3. Load ReCamMaster checkpoint
steps = 100
state_dict = torch.load(f"/mnt/workspace/checkpoint/wan21_1-3b_uni3c_baseline_nomira_1e-5_bz56_1epochs/tensorboardlog/version_0/checkpoints/step{steps}.ckpt", map_location="cpu")

pipe.dit.load_state_dict(state_dict, strict=True) # 6000 steps after train (7.9 服务器重启前跑了6000steps)
pipe.device = "cuda"
pipe.torch_dtype = torch.bfloat16
pipe = pipe.to("cuda", dtype=pipe.torch_dtype)


dataset_path = '/mnt/data/camera_datasets/MuteApo/RealCam-Vid'
dataset = CameraVideoDataset(
    dataset_path,
    os.path.join(dataset_path, 'RealCam-Vid_train_camera.npz'),
    None,
    steps_per_epoch=5,
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
    num_workers=0,
)

import json
total_data = torch.load('data/test_valid_data.pt')
real_img = json.load(open('data/openhumanvid/caption_part.json'))
height = 480
width = 832
tiled=False
tile_size=(34, 34)
tile_stride=(18, 16)
tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
for batch_idx, batch in enumerate(dataloader):
    # render video
    render_video = batch["render_video"]
    video_tensor = render_video[0] # Shape: [C, T, H, W]
    video_tensor = (video_tensor * 0.5 + 0.5).clamp(0, 1)
    video_tensor_rearranged = einops.rearrange(video_tensor, 'c t h w -> t h w c')
    video_np = (video_tensor_rearranged.cpu().numpy() * 255).astype(np.uint8)
    output_mp4_path = f"data/train_eval/1007/test_render_video_{batch_idx}.mp4"
    print(render_video.shape)
    print(f"正在保存视频到: {output_mp4_path}")
    imageio.mimsave(output_mp4_path, video_np, fps=16) # fps可以根据你的需要调整
    

    render_video = render_video.to(dtype=pipe.torch_dtype, device=pipe.device)
    render_latent = pipe.encode_video(render_video, **tiler_kwargs)

    render_mask = batch['render_mask_video']
    render_mask = render_mask[:, 0:1, :, :, :]
    render_mask[render_mask < 0.5] = 0
    render_mask[render_mask >= 0.5] = 1

    video_tensor = render_mask[0]
    video_tensor_rearranged = einops.rearrange(video_tensor, 'c t h w -> t h w c')
    video_np = (video_tensor_rearranged.cpu().numpy() * 255).astype(np.uint8)
    output_mp4_path = f"data/train_eval/1007/test_render_mask_{batch_idx}.mp4"
    print(render_video.shape)
    print(f"正在保存视频到: {output_mp4_path}")
    imageio.mimsave(output_mp4_path, video_np, fps=16) # fps可以根据你的需要调整

    ccc
    
    camera_embedding = get_plucker_embedding(batch["camera_extrinsics"], batch["camera_intrinsics"], height, width, device=pipe.device)

    path = batch["path"][0]
    video_id = batch["video_id"]
    print(video_id)
    print(path)
    os.system(f"cp {path} data/train_eval/1007/test_video_{batch_idx}.mp4")
    target_text = batch["text"][0]
    print(target_text)
    first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
    video = pipe(
        prompt=target_text,
        negative_prompt="镜头摇晃，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=first_frame,
        render_latent=render_latent,
        render_mask=render_mask,
        camera_pose = camera_embedding,
        num_inference_steps=50,
        height=height, width=width,
        num_frames=len(video_id),
        seed=42, tiled=True
    )
    save_video(video, f"data/train_eval/1007/video_uni3c_camera_56_1e-5_steps{steps}_{batch_idx}.mp4", fps=16, quality=5)
    # for image_id in real_img:
    #     first_frame = Image.open(f"data/openhumanvid/{image_id}.png")
    #     target_text = real_img[image_id]
    #     video = pipe(
    #         prompt=target_text,
    #         negative_prompt="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    #         input_image=first_frame,
    #         camera_pose =camera_embedding,
    #         num_inference_steps=50,
    #         height=480, width=832,
    #         seed=0, tiled=True
    #     )
    #     save_video(video, f"data/train_eval/0805/pretrain100k_video_patchify_plucker_camera_64_5e-5_steps{steps}_{batch_idx}_{image_id}.mp4", fps=16, quality=5)