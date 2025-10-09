import torch
from PIL import Image
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData

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
model_manager.load_lora("models/lightning_logs/version_50/checkpoints/epoch=9-step=5000.ckpt", lora_alpha=1.0)
pipe = WanVideoPipeline.from_model_manager(model_manager, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

image = Image.open("data/examples/image.png").resize((832,480))
video = pipe(
    prompt="The camera pans right, starting from the left side of the scene, capturing a woman dancing gracefully across the photography set, her movements framed against a vibrant, modern backdrop. The camera continues to pan right, maintaining a steady altitude, before gradually zooming out again to capture the broader landscape.",
    negative_prompt="overexposed, blurry, static, low quality, JPEG artifacts, ugly, distorted, poorly drawn hands, poorly drawn faces, deformed, mutilated, extra fingers, fused fingers, static image, messy background, three-legged, many people in the background, walking backwards",
    input_image=image,
    num_inference_steps=50,
    seed=0, tiled=True
)

save_video(video, "video_woman_camera_1.3b_panright.mp4", fps=30, quality=5)