import json
import torch
import os

from tqdm import tqdm
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download, dataset_snapshot_download
from PIL import Image

from diffsynth.data.video import save_frames


# Download models
# snapshot_download("Wan-AI/Wan2.1-FLF2V-14B-720P", local_dir="/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-FLF2V-14B-720P")

# Load models
model_manager = ModelManager(device="cuda")
model_manager.load_models(
    ["/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-FLF2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32, # Image Encoder is loaded with float32
)
model_manager.load_models(
    [
        [
            "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-FLF2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
        "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-FLF2V-14B-720P/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)


total_data = json.load(open('humman_flf2v_frame_ids.json'))
id = 0
# 2613 + 574 = 3187
for data in tqdm(total_data[id:]):
    input_image_path = data[1]
    last_image_path = data[2]
    prompt = data[3]
    video_name = input_image_path.split('.')[0].split('/')[-1] + '+' + last_image_path.split('.')[0].split('/')[-1]
    if os.path.exists(f"/root/hdd/yqf_camera_datasets/caizhongang/HuMMan/humman_flf2v_video/{video_name}.mp4"):
        continue
    video = pipe(
                prompt=prompt,
                negative_prompt="overexposed, blurry, static, low quality, JPEG artifacts, ugly, distorted, poorly drawn hands, poorly drawn faces, deformed, mutilated, extra fingers, fused fingers, static image, messy background, three-legged, many people in the background, walking backwards",
                num_inference_steps=30,
                input_image=Image.open(input_image_path).resize((832, 480)),
                end_image=Image.open(last_image_path).resize((832, 480)),
                height=480, width=832,
                seed=1, tiled=True
    )
    # save_frames(video, "video_frames")
    save_video(video, f"/root/hdd/yqf_camera_datasets/caizhongang/HuMMan/humman_flf2v_video/{video_name}.mp4", fps=15, quality=5)
    print(f'--------------current id: {id} ---------------------')
    id += 1
