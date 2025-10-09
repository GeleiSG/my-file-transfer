
import os
import einops
import torch
from PIL import Image
import torchvision
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.data.camera_utils import get_plucker_embedding, get_plucker_embedding_cpu
from diffsynth.data.camera_video import CameraVideoDataset
from diffsynth.data.pointcloud import point_rendering_train_stage
from diffusers.utils import export_to_video
from diffsynth.data.transforms import get_aspect_ratio
from diffsynth.data.utils_data import get_closest_ratio_key
from diffsynth.pipelines.wan_camera_video import WanVideoCameraPipeline, ModelManager
from omegaconf import OmegaConf
from pytorch3d.renderer.points.rasterizer import PointsRasterizationSettings
from torchvision.transforms import v2
from torchvision import transforms
from torchvision.utils import save_image
# from modelscope import dataset_snapshot_download
import json
import numpy as np
import imageio

from third_party.depth_pro import depth_pro



model_configs = []
# Load the full dit and camera adapter models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    ["/mnt/data/video_public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32, # Image Encoder is loaded with float32
)
model_manager.load_models([
    "/mnt/data/video_public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors",
    "/mnt/data/video_public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth",
    "/mnt/data/video_public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth",
], torch_dtype=torch.float16)
pipe = WanVideoCameraPipeline.from_model_manager(model_manager, device="cpu")

dim=pipe.dit.blocks[0].self_attn.q.weight.shape[0]
cfg = OmegaConf.load("/mnt/data/ssd/user_workspace/duanke/unicontrol/diffsynth/configs/config_yqf.json")
pipe.dit.controlnet_cfg = cfg.controlnet_cfg
pipe.dit.build_controlnet()

# state_dict = load_state_dict("models/train/Wan2.2-TI2V-5B_full_plucker_camera/epoch-1.safetensors")
# state_dict = load_state_dict("models/train/Wan2.2-TI2V-5B_full_plucker_camera_epoch5/epoch-1.safetensors")
# state_dict = load_state_dict("/mnt/workspace/checkpoint/wan21_1-3b_plucker_pointcloud_epoch_2/tensorboardlog/version_0/checkpoints/step600.ckpt")
# state_dict = load_state_dict("/mnt/workspace/checkpoint/wan21_1-3b_plucker_pointcloud_epoch_2_withoutplucker/tensorboardlog/version_0/checkpoints/step200.ckpt")
# state_dict = load_state_dict("/mnt/workspace/checkpoint/wan21_1-3b_uni3c_baseline_1000_epoch15_2e-5/tensorboardlog/version_0/checkpoints/step400.ckpt")
# state_dict = load_state_dict("/mnt/workspace/checkpoint/wan21_1-3b_uni3c_baseline_dim768_1000_epoch15_1e-5/tensorboardlog/version_0/checkpoints/step200.ckpt")
# state_dict = load_state_dict("/mnt/workspace/checkpoint/wan21_1-3b_uni3c_baseline_1000_epoch15_2e-5_dl3dv/tensorboardlog/version_0/checkpoints/step200.ckpt")
# state_dict = load_state_dict("/mnt/workspace/checkpoint/wan21_1-3b_uni3c_baseline_dim768_1000_epoch10_1e-5_bz32/tensorboardlog/version_0/checkpoints/epoch=0-step=31.ckpt")
# state_dict = load_state_dict("/mnt/data/ssd/user_workspace/yuqifan/unicontrol/models_debug/tensorboardlog/version_0/checkpoints/epoch=0-step=25.ckpt")
state_dict = load_state_dict("/mnt/workspace/checkpoint/wan21_1-3b_plucker_pointcloud_epoch_2/tensorboardlog/version_33/checkpoints/epoch=1-step=2000.ckpt")
pipe.dit.load_state_dict(state_dict, strict=False)
pipe.device = "cuda"
pipe = pipe.to("cuda", dtype=pipe.torch_dtype)
# pipe.enable_vram_management()

# 添加深度估计
# model, transform = depth_pro.create_model_and_transforms(device="cuda")
# model.requires_grad_(False)
# _depth_model = model.eval()
# _depth_transform = transform


dataset = CameraVideoDataset(
    "/mnt/data/camera_datasets_ssd/MuteApo/RealCam-Vid",
    ["/mnt/data/camera_datasets_ssd/MuteApo/RealCam-Vid/RealCam-Vid_train_camera.npz"],
    steps_per_epoch=1,
    max_num_frames=121,
    num_frames=81,
    is_i2v=True,    
    is_camera=True,
)
dataloader = torch.utils.data.DataLoader(
    dataset,
    shuffle=True,
    batch_size=1,
    num_workers=1,
)   

ASPECT_RATIO = get_aspect_ratio(size=632)
aspect_ratio_sample_size = {key : [x for x in ASPECT_RATIO[key]] for key in ASPECT_RATIO.keys()}

for id, data in enumerate(dataloader):
    first_frame = data["first_frame"]
    text = data["text"]
    b, c, h, w  = np.shape(first_frame)
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
    
    first_frame = transform(first_frame)

    render_video = data["render_video"]
    render_mask = data["render_mask"]
    render_video = einops.rearrange(render_video, "B T C H W -> B C T H W")
    render_mask = einops.rearrange(render_mask, "B T C H W -> B C T H W")

    output_png_path = f"./output_{id}_first_frame.png"
    output_mp4_path = f"./output_{id}_render_video.mp4"
    
    # 1. 保存 first_frame (PNG)
    # save_image 会自动处理反归一化（从[-1, 1]到[0, 255]）
    # 我们从批次中取出第一个图像 first_frame_transformed[0]
    print(f"正在保存图像到: {output_png_path}")
    save_image(first_frame[0], output_png_path, normalize=True)

    # 2. 保存 render_video (MP4)
    # 首先，从批次中取出第一个视频
    video_tensor = render_video[0] # Shape: [C, T, H, W]

    # 手动进行反归一化
    video_tensor = (video_tensor * 0.5 + 0.5).clamp(0, 1)

    # 将张量从 PyTorch 格式 (C, T, H, W) 转换为 imageio 期望的格式 (T, H, W, C)
    video_tensor_rearranged = einops.rearrange(video_tensor, 'c t h w -> t h w c')

    # 转换为 NumPy 数组，并将数值范围从 [0, 1] 扩展到 [0, 255] 的 uint8 类型
    video_np = (video_tensor_rearranged.cpu().numpy() * 255).astype(np.uint8)

    # 使用 imageio 保存为 mp4
    print(f"正在保存视频到: {output_mp4_path}")
    imageio.mimsave(output_mp4_path, video_np, fps=30) # fps可以根据你的需要调整
    # render video
    # tiler_kwargs = {"tiled": False, "tile_size": (34,34), "tile_stride": (18,16)}

    first_frame = first_frame.to(pipe.device)

    render_video = render_video.to(pipe.device, dtype=pipe.torch_dtype)

    render_latent = pipe.encode_video(render_video)
    render_mask = render_mask[:, 0:1, :, :, :]
    render_mask[render_mask < 0.5] = 0
    render_mask[render_mask >= 0.5] = 1
    render_latent= render_latent.to(pipe.device)
    render_mask = render_mask.to(pipe.device)

    # video_id = torch.stack(data["video_id"]).detach().cpu().numpy().T
    # print(video_id)
    video_id = data["video_id"]
    # print(data["camera_extrinsics"].shape)
    # print(data["camera_extrinsics"].dtype)
    # camera_extrinsics = torch.from_numpy(data["camera_extrinsics"]).float()
    # camera_intrinsics = torch.from_numpy(data["camera_intrinsics"]).float()

    camera_plucker_embedding = get_plucker_embedding(data["camera_extrinsics"], data["camera_intrinsics"], height=closest_size[0], width=closest_size[1])

    camera_plucker_embedding = camera_plucker_embedding.to(pipe.device)

    # export_to_video(render_video, f"results/test/render_{id}.mp4", fps=16)
    # export_to_video(mask_video, f"results/test/render_mask_{id}.mp4", fps=16)
    video = pipe(
        prompt=text,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=first_frame,
        camera_pose=camera_plucker_embedding,
        render_latent=render_latent,
        render_mask=render_mask,
        num_frames=len(video_id),
        seed=42, tiled=True,
    )
    save_video(video, f"results/test/epoch0_steps_2000_{id}.mp4", fps=16, quality=5)
