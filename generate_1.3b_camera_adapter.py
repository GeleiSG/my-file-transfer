import os

from diffsynth.data.camera_utils import get_camera_sparse_embedding
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
pipe = WanVideoCameraPipeline.from_model_manager(model_manager, device="cuda")
# 2. Initialize additional modules introduced in ReCamMaster
dim=pipe.dit.blocks[0].self_attn.q.weight.shape[0]
for block in pipe.dit.blocks:
    block.cam_encoder = nn.Linear(16, dim)
    block.projector = nn.Linear(dim, dim)
    # block.cam_encoder.weight.data.zero_()
    # block.cam_encoder.bias.data.zero_()
    block.projector.weight.data.zero_()
    block.projector.bias.data.zero_()
    # block.cam_encoder = nn.Linear(16, dim)
    # block.to_gamma_beta = nn.Linear(dim, 2 * dim)
    # nn.init.zeros_(block.to_gamma_beta.weight)
    # nn.init.zeros_(block.to_gamma_beta.bias)
    


# 3. Load ReCamMaster checkpoint
# state_dict = torch.load("models_multi/lightning_logs/version_28/checkpoints/step6250.ckpt", map_location="cpu") # 100k数据全量训练self，caption是long caption
# state_dict = torch.load("models_1.3b_adapter_abs/lightning_logs/version_3/checkpoints/step5000.ckpt", map_location="cpu")  # 236k数据训练self attn，学习率1e-5，绝对尺度训练
# state_dict = torch.load("models_1.3b_adapter_adaln_tensor/lightning_logs/version_0/checkpoints/step2000.ckpt", map_location="cpu")  # 100k数据残差加入相机干预，不训练self attn，学习率8e-5
state_dict = torch.load("models_1.3b_adapter_res_tensor/lightning_logs/version_60/checkpoints/step3000.ckpt", map_location="cpu")
pipe.dit.load_state_dict(state_dict, strict=False)
pipe.to("cuda")
pipe.to(dtype=torch.bfloat16)

pipe.enable_vram_management(num_persistent_param_in_dit=None)


dataset_path = '/mnt/data/hdd/datasets/camera_datasets/MuteApo/RealCam-Vid'
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

# data = {}
# for batch_idx, batch in enumerate(dataloader):
#     target_text = batch["text"][0]
#     target_camera = batch["camera_embedding"].to(pipe.device)
#     # print(target_camera)
#     gt_video = batch["video"][0]
#     frames = rearrange(gt_video, "C T H W -> T H W C")

#     # 如果是 float 类型，转换为 [0, 255] uint8
#     if frames.dtype == torch.float32:
#         frames = (frames.clamp(0, 1) * 255).to(torch.uint8)
#     else:
#         frames = frames.clamp(0, 255).to(torch.uint8)

#     # 转为 CPU NumPy 数组
#     frames = frames.cpu().numpy()

#     # 写入视频
#     path = batch["path"][0]
#     gt_path = f"/root/hdd/yqf/cameractrl/data/examples_results_0529_test/gt_{batch_idx}.mp4"
#     # imageio.mimwrite(gt_path, frames, fps=16, codec='libx264', quality=10, ffmpeg_params=["-pix_fmt", "yuv444p"])
#     T, H, W, C = frames.shape
#     assert C == 3

#     frames = frames.astype(np.uint8)

#     process = subprocess.Popen([
#         'ffmpeg',
#         '-y',  # overwrite
#         '-f', 'rawvideo',
#         '-vcodec','rawvideo',
#         '-s', f'{W}x{H}',
#         '-pix_fmt', 'rgb24',
#         '-r', str(16),
#         '-i', '-',
#         '-an',
#         '-vcodec', 'libx264',
#         '-pix_fmt', 'yuv444p',  # 保持色彩更接近 RGB
#         gt_path
#     ], stdin=subprocess.PIPE)

#     process.stdin.write(frames.tobytes())
#     process.stdin.close()
#     process.wait()
#     print(target_text)
#     print(path)
#     new_path = f"/root/hdd/yqf/cameractrl/data/examples_results_0529_test/{batch_idx}_" + path.split('/')[-1]
#     os.system(f"cp {path} {new_path}")
#     first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
#     data[batch_idx] = {
#         "camera": target_camera,
#         "text": target_text,
#         "path": path,
#         "first_frame": batch["first_frame"][0].cpu()
#     }
#     # first_frame = Image.open("data/examples/image.png")
#     video = pipe(
#         prompt=target_text,
#         negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#         input_image=first_frame,
#         camera_pose = target_camera,
#         num_inference_steps=50,
#         height=480, width=832,
#         seed=0, tiled=True
#     )
#     save_video(video, f"data/examples_results_0529_test/video_test_camera_100000_{batch_idx}.mp4", fps=16, quality=5)
# torch.save(data, 'data/examples_results_0529_test/test_20_data.pt')

total_data = torch.load('data/train_eval/5_data.pt')
for batch_idx, key in enumerate(total_data):
    if batch_idx >= 5:
        break
    batch = total_data[key]
    # target_camera = batch["camera"]
    camera_extrinsic = batch['camera_extrinsic']
    camera_intrinsic = batch['camera_intrinsic']
    video_id = batch['video_id']
    camera_id = video_id[::4]
    # generate camera embedding
    # camera_embedding = get_plucker_embedding(camera_extrinsic[video_id], camera_intrinsic[video_id], height=480, width=832)
    camera_embedding, _ = get_camera_sparse_embedding(camera_extrinsic[camera_id,:,:].cpu().numpy(), camera_intrinsic[camera_id,:].cpu().numpy(), height=480, width=832)

    camera_embedding = camera_embedding[None].to(pipe.device)
    path = batch["path"]

    target_text = batch["text"]
    print(target_text)
    first_frame = Image.fromarray(batch["first_frame"])
    video = pipe(
        prompt=target_text,
        negative_prompt="镜头摇晃，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=first_frame,
        camera_pose = camera_embedding,
        num_inference_steps=50,
        height=480, width=832,
        seed=0, tiled=True
    )
    save_video(video, f"data/train_eval/0630/video_adapter_camera_controlnet_32_8e-5_steps3000_{batch_idx}.mp4", fps=16, quality=5)
exit()
import json
real_img = json.load(open('data/openhumanvid/caption.json'))
for batch_idx, key in enumerate(total_data):
    if batch_idx > 4:
        continue
    batch = total_data[key]
    target_camera = batch["camera"]
    path = batch["path"]
    print(path)
    for image_id in real_img:
        if image_id == "0000":
            continue
        first_frame = Image.open(f"data/openhumanvid/{image_id}.png")
        target_text = real_img[image_id]
        video = pipe(
            prompt=target_text,
            negative_prompt="镜头摇晃，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            input_image=first_frame,
            camera_pose = None,
            num_inference_steps=50,
            height=480, width=832,
            seed=0, tiled=True
        )
        save_video(video, f"data/valid_test/0610/openhuman_video_adapter_non_camera_16_1e-5_steps5000_{image_id}_{batch_idx}.mp4", fps=16, quality=5)