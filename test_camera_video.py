
import os
import einops
import torch
from PIL import Image
import torchvision
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.data.camera_utils import get_plucker_embedding_cpu
from diffsynth.data.camera_video import CameraVideoDataset
from diffsynth.data.pointcloud import point_rendering_train_stage
from diffusers.utils import export_to_video
from diffsynth.pipelines.wan_camera_video import WanVideoCameraPipeline, ModelManager
from omegaconf import OmegaConf
from pytorch3d.renderer.points.rasterizer import PointsRasterizationSettings
from torchvision.transforms import v2
# from modelscope import dataset_snapshot_download
import json
import numpy as np

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
], torch_dtype=torch.float32)
pipe = WanVideoCameraPipeline.from_model_manager(model_manager, device="cuda")

dim=pipe.dit.blocks[0].self_attn.q.weight.shape[0]
cfg = OmegaConf.load("/mnt/data/ssd/user_workspace/yuqifan/unicontrol/diffsynth/configs/config.json")
pipe.dit.controlnet_cfg = cfg.controlnet_cfg
pipe.dit.build_controlnet()
pipe = pipe.to(device="cuda", dtype=torch.float32)

# state_dict = load_state_dict("models/train/Wan2.2-TI2V-5B_full_plucker_camera/epoch-1.safetensors")
# state_dict = load_state_dict("models/train/Wan2.2-TI2V-5B_full_plucker_camera_epoch5/epoch-1.safetensors")
# state_dict = load_state_dict("/mnt/workspace/checkpoint/wan21_1-3b_plucker_pointcloud_epoch_2/tensorboardlog/version_0/checkpoints/step600.ckpt")
# state_dict = load_state_dict("/mnt/workspace/checkpoint/wan21_1-3b_plucker_pointcloud_epoch_2_withoutplucker/tensorboardlog/version_0/checkpoints/step200.ckpt")
# state_dict = load_state_dict("/mnt/workspace/checkpoint/wan21_1-3b_uni3c_baseline_1000_epoch15_2e-5/tensorboardlog/version_0/checkpoints/step400.ckpt")
# pipe.dit.load_state_dict(state_dict, strict=False)
pipe.enable_vram_management()

# 添加深度估计
model, transform = depth_pro.create_model_and_transforms(device="cuda")
model.requires_grad_(False)
_depth_model = model.eval()
_depth_transform = transform

height = 480
width = 832
transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((height, width), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.CenterCrop((height, width)),
])

for id in range(1):
    os.makedirs(f'results/test', exist_ok=True)
    camera_params = np.loadtxt(f'data/test/sample_{id}_params.txt')
    extrinsics = []
    intrinsics = []
    Ks = []
    for i in range(camera_params.shape[0]):
        mat_4x4 = np.eye(4)
        mat_4x4[:3, :] = camera_params[i,7:].reshape(3,4)
        extrinsics.append(mat_4x4)
        intrinsics.append(camera_params[i,1:5])
        fx = camera_params[i,1]
        fy = camera_params[i,2]
        cx = camera_params[i,3]
        cy = camera_params[i,4]

        if cx < 1:
            Ks.append(np.array([[fx*width, 0, cx*width],[0, fy*height, cy*height],[0,0,1]], dtype=np.float32))
        else:
            Ks.append(np.array([[fx, 0, cx],[0, fy, cy],[0,0,1]], dtype=np.float32))
    plucker_embedding = get_plucker_embedding_cpu(extrinsics, intrinsics, height=height, width=width)
    plucker_embedding = plucker_embedding.unsqueeze(0)
    input_image = Image.open(f'data/test/sample_{id}_frame_0001.png')

    # batch_size 1: 123G 2m42s/iter depth:1.44s/frame render:7.39s/video
    with torch.no_grad():
        depth_image = transform(v2.ToTensor()(input_image)).float() * 2 - 1
        depth_image = depth_image.to("cuda")
        prediction = _depth_model.infer(depth_image, f_px=None)
        depth = prediction["depth"]
        del depth_image, prediction
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        control_imgs, render_masks = point_rendering_train_stage(K=torch.tensor(np.array(Ks)).float(),
                                                                    w2cs=torch.tensor(np.array(extrinsics)).float(),
                                                                    depth=depth.float(),
                                                                    images=transform(v2.ToTensor()(input_image)).float() * 2 - 1,
                                                                    raster_settings=PointsRasterizationSettings(image_size=(height, width),
                                                                                                                radius=0.008,
                                                                                                                points_per_pixel=8),
                                                                    device="cuda",
                                                                    background_color=[0, 0, 0],
                                                                    sobel_threshold=0.35,
                                                                    sam_mask=None)
        del depth
    
    control_imgs = einops.rearrange(control_imgs, "(b f) c h w -> b c f h w", f=81)
    render_masks = einops.rearrange(render_masks, "(b f) c h w -> b c f h w", f=81)
    render_video = []
    mask_video = []
    pipe.vae = pipe.vae.to(pipe.device)

    # render video
    control_imgs = control_imgs.to(dtype=pipe.torch_dtype, device=pipe.device)
    render_latent = pipe.encode_video(control_imgs)
    
    for i in range(81):
        img = v2.ToPILImage()((control_imgs[0][:, i] + 1) / 2)
        render_video.append(img)
        mask = v2.ToPILImage()(render_masks[0][:, i])
        mask_video.append(mask)

    del control_imgs, render_masks

    render_mask = torch.stack([v2.ToTensor()(frame) for frame in mask_video], dim=0)[:, 0:1][None]  # [f,1,h,w]
    render_mask = einops.rearrange(render_mask, "b f c h w -> b c f h w")
    render_mask[render_mask < 0.5] = 0
    render_mask[render_mask >= 0.5] = 1
    export_to_video(render_video, f"results/test/render_{id}.mp4", fps=16)
    export_to_video(mask_video, f"results/test/render_mask_{id}.mp4", fps=16)
    video = pipe(
        prompt="",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=input_image,
        camera_pose=plucker_embedding,
        render_latent=render_latent,
        render_mask=render_mask,
        num_frames=81,
        seed=1, tiled=True,
    )
    save_video(video, f"results/test/video_Wan21_1-3B_original_{id}.mp4", fps=16, quality=5)
