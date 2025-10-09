import torch
from PIL import Image
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData

#   --dit_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors" \
#   --image_encoder_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
#   --text_encoder_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth" \
#   --vae_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth" \
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    ["/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32, # Image Encoder is loaded with float32
)
model_manager.load_models([
        [
            "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
        "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
],
torch_dtype=torch.bfloat16)
model_manager.load_lora("models/lightning_logs/version_49/checkpoints/epoch=0-step=500.ckpt", lora_alpha=1.0)
pipe = WanVideoPipeline.from_model_manager(model_manager, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

image = Image.open("data/examples/image1.png")
video = pipe(
    prompt="A woman is dancing in the laboratory. The camera pans right through a woman who is dancing, starting from the left and ending on the right, always maintaining a consistent horizontal Angle of the eyes.",
    negative_prompt="overexposed, blurry, static, low quality, JPEG artifacts, ugly, distorted, poorly drawn hands, poorly drawn faces, deformed, mutilated, extra fingers, fused fingers, static image, messy background, three-legged, many people in the background, walking backwards",
    input_image=image,
    num_inference_steps=50,
    seed=0, tiled=True
)

save_video(video, "video_camera.mp4", fps=30, quality=5)