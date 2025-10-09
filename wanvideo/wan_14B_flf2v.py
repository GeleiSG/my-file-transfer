import torch
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

# Download example image

# First and last frame to video
# 01345689，是顺时针依次45度
# The camera position is at eye level and the moves along an arc with right rotation angles of 90 degree. (0-8)

# The camera gradually rises, providing a top-down perspective 3 -》2
# The camera is moving in a circular trajectory at eye level steadily to the right, with a rotated angle of 90 degrees.
# The camera moves along an arc with left rotation angles of 90 degree, providing a rotation to left perspective
video = pipe(
    prompt="An aerial view of a luxurious neighborhood with a blend of modern and traditional architecture, surrounded by lush greenery. solar panels on the rooftops indicate an environmentally conscious community. the area is connected by a network of pathways and driveways, with a swimming pool and a white water feature adding to the leisurely feel. the backdrop of a distant city skyline suggests the neighborhood's proximity to urban life. the consistent imagery across the frames highlights the neighborhood's serene and exclusive atmosphere. The camera starts with a wide aerial shot, slowly panning right across a suburban neighborhood. It then tilts slightly downward, providing a closer view of the houses and trees below. The camera continues to pan right, maintaining a steady altitude, before gradually zooming out again to capture the broader landscape.",
    negative_prompt="overexposed, blurry, static, low quality, JPEG artifacts, ugly, distorted, poorly drawn hands, poorly drawn faces, deformed, mutilated, extra fingers, fused fingers, static image, messy background, three-legged, many people in the background, walking backwards",
    num_inference_steps=30,
    input_image=Image.open("data/examples/test_first_image.jpg").resize((832, 480)),
    # end_image=Image.open("data/examples/image_lookup_right.png").resize((832, 480)),
    height=480, width=832,
    seed=1, tiled=True
)
save_frames(video, "video_frames")
save_video(video, "video_camera_1_3b.mp4", fps=30, quality=5)
