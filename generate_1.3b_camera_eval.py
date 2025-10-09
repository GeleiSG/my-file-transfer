import os
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
    # "/root/hdd/user_workspace/yuqifan/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors",
    "/root/hdd/user_workspace/yuqifan/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth",
    "/root/hdd/user_workspace/yuqifan/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth",
], torch_dtype=torch.bfloat16)
text_encoder_path = "/root/hdd/user_workspace/yuqifan/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth"
vae_path = "/root/hdd/user_workspace/yuqifan/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth"
image_encoder_path = "/root/hdd/user_workspace/yuqifan/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"

model_path = [text_encoder_path, vae_path]
if image_encoder_path is not None:
    model_path.append(image_encoder_path)
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
model_manager.load_models(model_path)
pipe = WanVideoCameraPipeline.from_model_manager(model_manager)
pipe.device = "cuda"
pipe = pipe.to("cuda")

dataset_path = '/mnt/data/hdd/datasets/camera_datasets/MuteApo/RealCam-Vid'
dataset = CameraVideoDataset(
    dataset_path,
    os.path.join(dataset_path, 'RealCam-Vid_test.npz'),
    os.path.join(dataset_path, 'camera_caption_total.json'),
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
    num_workers=4,
)
root = '/mnt/data/hdd/datasets/camera_datasets/MuteApo/RealCam-Vid'
metadata = np.load(os.path.join(root, 'RealCam-Vid_test.npz'), allow_pickle=True)["arr_0"]

total_data = dict()
for entry in metadata:
    key = os.path.join(root, entry["video_path"])
    total_data[key] = entry
data = {}
tiled=True
tile_size=(34, 34)
tile_stride=(18, 16)
tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
for batch_idx, batch in enumerate(dataloader):
    
    # # 写入视频
    path = batch["path"][0]
    gt_path = f"/root/hdd/user_workspace/yuqifan/cameractrl/data/train_eval/gt_{batch_idx}.mp4"
    
    print(path)
    new_path = f"/root/hdd/user_workspace/yuqifan/cameractrl/data/train_eval/{batch_idx}_" + path.split('/')[-1]
    os.system(f"cp {path} {new_path}")
    first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())

    text, video, path = batch["text"][0], batch["video"], batch["path"][0]

    # prompt
    print(text)
    prompt_emb = pipe.encode_prompt(text)
    # video
    video = video.to(dtype=pipe.torch_dtype, device=pipe.device)
    latents = pipe.encode_video(video, **tiler_kwargs)[0]
    # image
    if "first_frame" in batch:
        first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
        _, _, num_frames, height, width = video.shape
        image_emb = pipe.encode_image(first_frame, None, num_frames, height, width)
    else:
        image_emb = {}
    video_id = torch.cat(batch["video_id"]).cpu().numpy()
    camera_extrinsic = batch["camera_extrinsic"][0]
    camera_intrinsic = batch["camera_intrinsic"][0]
    data[batch_idx] = {
        "gt_camera": total_data[path],
        "latents": latents,
        "camera_extrinsic": camera_extrinsic,
        "camera_intrinsic": camera_intrinsic,
        "text": text,
        "path": path,
        "first_frame": batch["first_frame"][0].cpu().numpy(),
        "video_id": video_id,
        "prompt_emb": prompt_emb,
        "image_emb": image_emb,
    }
torch.save(data, 'data/train_eval/5_data.pt')
exit()