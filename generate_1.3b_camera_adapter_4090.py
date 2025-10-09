import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
for block_i, block in enumerate(pipe.dit.blocks):
    # if block_i < 20:
        block.cam_encoder = nn.Linear(16, dim)
        block.projector = nn.Linear(dim, dim)
        block.cam_encoder.weight.data.zero_()
        block.cam_encoder.bias.data.zero_()
        block.projector.weight = nn.Parameter(torch.eye(dim))
        block.projector.bias = nn.Parameter(torch.zeros(dim))

# 3. Load ReCamMaster checkpoint
steps = 3000_1500
# # state_dict = torch.load(f"models_1.3b_adapter_20/lightning_logs/version_0/checkpoints/step{steps}.ckpt", map_location="cpu")
# state_dict = torch.load(f"models_1.3b_adapter_abs_tensor/lightning_logs/version_7/checkpoints/step{steps}.ckpt", map_location="cpu")
# state_dict = torch.load(f"models_multi/lightning_logs/version_28/checkpoints/step{steps}.ckpt", map_location="cpu")
state_dict = torch.load(f"models_1.3b_abs_pretrain3000_realcam100k/tensorboardlog/version_0/checkpoints/step1500.ckpt", map_location="cpu")
pipe.dit.load_state_dict(state_dict, strict=False)
pipe = pipe.to("cuda")
pipe = pipe.to(dtype=torch.bfloat16)

pipe.enable_vram_management(num_persistent_param_in_dit=None)


dataset_path = '/mnt/data/camera_datasets/MuteApo/RealCam-Vid'
height = 480
width = 832
dataset = CameraVideoDataset(
    dataset_path,
    os.path.join(dataset_path, 'RealCam-Vid_train_camera.npz'),
    os.path.join(dataset_path, 'camera_caption_total.json'),
    steps_per_epoch=5,
    max_num_frames=129,
    frame_interval=1,
    num_frames=81,
    height=height,
    width=width,
    is_i2v=True,     # 根据你的使用情况
    is_camera=True   # 确保启用 camera 相关字段
)
dataloader = torch.utils.data.DataLoader(
    dataset,
    shuffle=True,
    batch_size=1,
    num_workers=4,
)
# root = '/mnt/data/hdd/datasets/camera_datasets/MuteApo/RealCam-Vid'
# metadata = np.load(os.path.join(root, 'RealCam-Vid_train.npz'), allow_pickle=True)["arr_0"]
# # multicam_metadata = np.load('/root/hdd/datasets/camera_datasets/KwaiVGI/MultiCamVideo-Dataset/MultiCamVideo_camera.npz', allow_pickle=True)["arr_0"]
# total_data = dict()
# for entry in metadata:
#     key = os.path.join(root, entry["video_path"])
#     total_data[key] = entry
# data = {}
# for batch_idx, batch in enumerate(dataloader):
#     target_text = batch["text"][0]
#     # target_camera = batch["camera_embedding"].to(pipe.device)
#     # print(target_camera)
#     # gt_video = batch["video"][0]
#     # frames = rearrange(gt_video, "C T H W -> T H W C")

#     # # 如果是 float 类型，转换为 [0, 255] uint8
#     # if frames.dtype == torch.float32:
#     #     frames = (frames.clamp(0, 1) * 255).to(torch.uint8)
#     # else:
#     #     frames = frames.clamp(0, 255).to(torch.uint8)

#     # # 转为 CPU NumPy 数组
#     # frames = frames.cpu().numpy()

#     # # 写入视频
#     path = batch["path"][0]
#     gt_path = f"/root/hdd/user_workspace/yuqifan/cameractrl/data/train_eval/gt_{batch_idx}.mp4"
#     # # imageio.mimwrite(gt_path, frames, fps=16, codec='libx264', quality=10, ffmpeg_params=["-pix_fmt", "yuv444p"])
#     # T, H, W, C = frames.shape
#     # assert C == 3

#     # frames = frames.astype(np.uint8)

#     # process = subprocess.Popen([
#     #     'ffmpeg',
#     #     '-y',  # overwrite
#     #     '-f', 'rawvideo',
#     #     '-vcodec','rawvideo',
#     #     '-s', f'{W}x{H}',
#     #     '-pix_fmt', 'rgb24',
#     #     '-r', str(16),
#     #     '-i', '-',
#     #     '-an',
#     #     '-vcodec', 'libx264',
#     #     '-pix_fmt', 'yuv444p',  # 保持色彩更接近 RGB
#     #     gt_path
#     # ], stdin=subprocess.PIPE)

#     # process.stdin.write(frames.tobytes())
#     # process.stdin.close()
#     # process.wait()
#     # print(target_text)
#     print(path)
#     new_path = f"/root/hdd/user_workspace/yuqifan/cameractrl/data/train_eval/{batch_idx}_" + path.split('/')[-1]
#     os.system(f"cp {path} {new_path}")
#     first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())

#     text, video, path = batch["text"][0], batch["video"], batch["path"][0]
#     video_id = torch.cat(batch["video_id"]).cpu().numpy()
#     camera_extrinsic = batch["camera_extrinsic"][0]
#     camera_intrinsic = batch["camera_intrinsic"][0]

#     # prompt
#     prompt_emb = pipe.encode_prompt(text)
#     # video
#     video = video.to(dtype=pipe.torch_dtype, device=pipe.device)
#     print(video)
#     tiled=True
#     tile_size=(34, 34)
#     tile_stride=(18, 16)
#     tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
#     latents = pipe.encode_video(video, **tiler_kwargs)[0]
#     # image
#     if "first_frame" in batch:
#         first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
#         _, _, num_frames, height, width = video.shape
#         image_emb = pipe.encode_image(first_frame, None, num_frames, height, width)
#     else:
#         image_emb = {}

#     data[batch_idx] = {
#         "gt_camera": total_data[path],
#         "latents": latents,
#         "camera_extrinsic": camera_extrinsic,
#         "camera_intrinsic": camera_intrinsic,
#         "text": target_text,
#         "path": path,
#         "first_frame": batch["first_frame"][0].cpu().numpy(),
#         "video_id": video,
#         "prompt_emb": prompt_emb,
#         "image_emb": image_emb,
#     }
#     continue
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
# torch.save(data, 'data/train_eval/5_data.pt')
# exit()

# steps = 0

# total_data = torch.load('data/test_valid_data.pt')

for batch_idx, batch in enumerate(dataloader):
    # batch = total_data[key]
    target_camera = rearrange(batch["camera_extrinsics"], 'b c d f -> b c (d f)')
    path = batch["path"][0]
    new_path = f"/root/hdd/user_workspace/yuqifan/cameractrl/data/valid_test/train_{batch_idx}_" + path.split('/')[-1]
    os.system(f"cp {path} {new_path}")
    target_text = batch["text"][0]
    first_frame = Image.fromarray(np.array(batch["first_frame"][0]))
    video = pipe(
            prompt=target_text,
            negative_prompt="镜头控制，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            input_image=first_frame,
            camera_pose = target_camera,
            num_inference_steps=50,
            height=height, width=width,
            seed=0, tiled=True
    )
    for i in range(2):
        id = i +1
        first_frame = Image.open(f"data/valid_test/human{id}.png")
        # first_frame = Image.fromarray(np.array(image))
        video = pipe(
            prompt="The girl turns her head 90 degrees.",
            # prompt=target_text,
            negative_prompt="镜头控制，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            input_image=first_frame,
            camera_pose = target_camera,
            num_inference_steps=50,
            height=height, width=width,
            seed=0, tiled=True
        )
        save_video(video, f"data/valid_test/0720/video_adapter_camera_caption_action_16_1e-4_steps{steps}_{batch_idx}_{id}.mp4", fps=16, quality=5)