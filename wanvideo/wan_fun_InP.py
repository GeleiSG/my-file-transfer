import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download, dataset_snapshot_download
from PIL import Image


# Download models
# snapshot_download("PAI/Wan2.1-Fun-1.3B-InP", local_dir="/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP")

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [
        "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors",
        "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth",
        "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth",
        "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

# # Download example image
# dataset_snapshot_download(
#     dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#     local_dir="./",
#     allow_file_pattern=f"data/examples/wan/input_image.jpg"
# )
image = Image.open("data/examples/image.png").resize((832,480))

# Image-to-video
video = pipe(
    prompt="The camera pans right, starting from the left side of the scene, capturing a woman dancing gracefully across the photography set, her movements framed against a vibrant, modern backdrop.",
    negative_prompt="overexposed, blurry, static, low quality, JPEG artifacts, ugly, distorted, poorly drawn hands, poorly drawn faces, deformed, mutilated, extra fingers, fused fingers, static image, messy background, three-legged, many people in the background, walking backwards",
    num_inference_steps=50,
    input_image=image,
    # You can input `end_image=xxx` to control the last frame of the video.
    # The model will automatically generate the dynamic content between `input_image` and `end_image`.
    seed=1, tiled=True
)
save_video(video, "video_woman_1_3b.mp4", fps=30, quality=5)
