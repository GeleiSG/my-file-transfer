import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from PIL import Image
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from diffsynth.pipelines.wan_camera_video import WanVideoCameraPipeline
from diffsynth.data.camera_video import CameraVideoDataset
import torch.nn as nn
import imageio
from einops import rearrange
import numpy as np
import subprocess

model_manager = ModelManager(device="cpu")
model_manager.load_models(
    ["/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32, # Image Encoder is loaded with float32
)
model_manager.load_models([
    "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors",
    "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth",
    "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth",
], torch_dtype=torch.bfloat16)

model_manager.load_lora("models_1.3b_lora/lightning_logs/version_1/checkpoints/epoch=9-step=6250.ckpt", lora_alpha=1.0)
pipe = WanVideoCameraPipeline.from_model_manager(model_manager, device="cuda")

# 2. Initialize additional modules introduced in ReCamMaster
dim=pipe.dit.blocks[0].self_attn.q.weight.shape[0]
for block in pipe.dit.blocks:
    block.cam_encoder = nn.Linear(16, dim)
    block.projector = nn.Linear(dim, dim)
    block.cam_encoder.weight.data.zero_()
    block.cam_encoder.bias.data.zero_()
    block.projector.weight = nn.Parameter(torch.eye(dim))
    block.projector.bias = nn.Parameter(torch.zeros(dim))

# 3. Load camera checkpoint
state_dict = torch.load("models_1.3b_lora/lightning_logs/version_1/checkpoints/epoch=9-step=6250.ckpt", map_location="cpu")
# pipe.dit.load_state_dict(state_dict, strict=True)
# pipe.to("cuda")
# pipe.to(dtype=torch.bfloat16)

full_model_keys = []
lora_keys = []
for k in state_dict:
    if 'lora_' in k or 'lora.down' in k or 'lora.up' in k:
        lora_keys.append(k)
    else:
        full_model_keys.append(k)
# 1. 先加载 full model 权重
print(full_model_keys)
print("Loading full model weights...")
try:
    pipe.dit.load_state_dict({k: state_dict[k] for k in full_model_keys}, strict=False)
except RuntimeError as e:
    print("Warning: partial loading error in full model:", str(e))
pipe.to("cuda")
pipe.to(dtype=torch.bfloat16)
pipe.enable_vram_management(num_persistent_param_in_dit=None)

dataset_path = '/mnt/data/hdd/yqf_camera_datasets/MuteApo/RealCam-Vid'
dataset = CameraVideoDataset(
    dataset_path,
    os.path.join(dataset_path, 'RealCam-Vid_test.npz'),
    os.path.join(dataset_path, 'camera_caption_total.json'),
    steps_per_epoch=20,
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
for batch_idx, batch in enumerate(dataloader):
    target_text = batch["text"][0]
    target_camera = batch["camera_embedding"].to(pipe.device)
    # print(target_camera)
    gt_video = batch["video"][0]
    frames = rearrange(gt_video, "C T H W -> T H W C")

    # 如果是 float 类型，转换为 [0, 255] uint8
    if frames.dtype == torch.float32:
        frames = (frames.clamp(0, 1) * 255).to(torch.uint8)
    else:
        frames = frames.clamp(0, 255).to(torch.uint8)

    # 转为 CPU NumPy 数组
    frames = frames.cpu().numpy()

    # 写入视频
    path = batch["path"][0]
    gt_path = f"/root/hdd/yqf/cameractrl/data/examples_results_0529_lora/gt_{batch_idx}.mp4"
    # imageio.mimwrite(gt_path, frames, fps=16, codec='libx264', quality=10, ffmpeg_params=["-pix_fmt", "yuv444p"])
    T, H, W, C = frames.shape
    assert C == 3

    frames = frames.astype(np.uint8)

    process = subprocess.Popen([
        'ffmpeg',
        '-y',  # overwrite
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', f'{W}x{H}',
        '-pix_fmt', 'rgb24',
        '-r', str(16),
        '-i', '-',
        '-an',
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv444p',  # 保持色彩更接近 RGB
        gt_path
    ], stdin=subprocess.PIPE)

    process.stdin.write(frames.tobytes())
    process.stdin.close()
    process.wait()
    print(target_text)
    print(path)
    new_path = f"/root/hdd/yqf/cameractrl/data/examples_results_0529_lora/{batch_idx}_" + path.split('/')[-1]
    os.system(f"cp {path} {new_path}")
    first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
    data[batch_idx] = {
        "camera": target_camera,
        "text": target_text,
        "path": path,
        "first_frame": batch["first_frame"][0].cpu()
    }
    # first_frame = Image.open("data/examples/image.png")
    video = pipe(
        prompt=target_text,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=first_frame,
        camera_pose = target_camera,
        num_inference_steps=50,
        height=480, width=832,
        seed=0, tiled=True
    )
    save_video(video, f"data/examples_results_0529_lora/video_lora_camera_100000_{batch_idx}.mp4", fps=16, quality=5)
torch.save(data, 'data/examples_results_0529_lora/test_20_data.pt')