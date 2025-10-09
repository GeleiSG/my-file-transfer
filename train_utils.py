import gc
import logging
import os
import sys
import random
import numpy as np
import torch
import argparse
from accelerate.logging import get_logger
from diffusers.utils import check_min_version, is_wandb_available
from einops import rearrange
from PIL import Image
from easyanimate.utils.utils import save_videos_grid, read_video_from_path, read_face_video_from_path

from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS
from wan.configs.wan_t2v_ref2v_1_3B import t2v_ref2v_1_3B as wan_config
from wan.modules.model_new import WanModel
# from wan.utils.lib.models.networks.encoder import FanEncoder
import wan
from wan.utils.lib.models.networks.encoder import HFModelFanEncoder as FanEncoder
from wan.modules.audio_proj import FaceProjModel

import os
import torch
import random
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from easyanimate.data.dataset_image_video import (ImageVideoDataset,
                                                  ImageVideoSampler,
                                                  )

from easyanimate.data.bucket_sampler import (ASPECT_RATIO_632,
                                             ASPECT_RATIO_RANDOM_CROP_632,
                                             ASPECT_RATIO_RANDOM_CROP_PROB,
                                             AspectRatioBatchImageSampler,
                                             AspectRatioBatchImageVideoSampler,
                                             RandomSampler, get_closest_ratio)

if is_wandb_available():
    import wandb

def encode_prompt(
    text_encoder,
    input_prompt,
    max_length,
    device: torch.device,
    dtype: torch.dtype,
):  
    prompt_embeds, context_lens = text_encoder(input_prompt, device, max_length)
    return prompt_embeds.to(dtype), context_lens

def encode_img_prompt(
    clip_model,
    img_prompt,
    dtype: torch.dtype,
):
    img_prompt = rearrange(img_prompt, "b h w c -> b c 1 h w")  # (-1., 1.)
    # img_prompt = torch.clip((img_prompt / 255. - 0.5) * 2.0, -1.0, 1.0)
    img_prompt_embeds = clip_model.visual(img_prompt.to(dtype))
    return img_prompt_embeds.to(dtype)

# This way is quicker when batch grows up
def batch_encode_vae(vae, pixel_values, device, dtype, drop_ratio=0., half_ref_pixel_values=None):  # [C, F, H, W]
    pixel_values = pixel_values.to(dtype=vae.dtype, device=device)
    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
    latents = vae.encode(pixel_values)
    if half_ref_pixel_values is not None:
        half_ref_pixel_values = half_ref_pixel_values.to(dtype=vae.dtype, device=device)
        half_ref_pixel_values = rearrange(half_ref_pixel_values, "b f c h w -> b c f h w")
        half_latents = vae.encode(half_ref_pixel_values)
        new_latents, new_half_latents = [], []
        for bs_latents, bs_half_latents in zip(latents, half_latents):
            if random.random() > drop_ratio:
                new_latents.append(bs_latents.unsqueeze(0))
                new_half_latents.append(bs_half_latents.unsqueeze(0))
            else:
                new_latents.append(torch.zeros_like(bs_latents.unsqueeze(0)))
                new_half_latents.append(torch.zeros_like(bs_half_latents.unsqueeze(0)))
        latents = torch.cat(new_latents, dim=0)
        half_latents = torch.cat(new_half_latents, dim=0)
        latents = torch.cat([latents, half_latents], dim=2)
    else:
        new_latents = []
        for bs_latents in latents:
            if random.random() > drop_ratio:
                new_latents.append(bs_latents.unsqueeze(0))
            else:
                new_latents.append(torch.zeros_like(bs_latents.unsqueeze(0)))

        latents = torch.cat(new_latents, dim=0)
    return latents.to(dtype)

class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super(ConfigDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def resize_keep_aspect_ratio(tensor, target_size=224, padding_mode='constant', padding_value=0):
    """
    保持原图比例进行resize

    参数:
        tensor: 输入张量 [batch, channels, height, width]
        target_size: 目标尺寸（正方形）
        padding_mode: 填充模式 ('constant', 'reflect', 'replicate', 'circular')
        padding_value: 填充值（当padding_mode='constant'时使用）

    返回:
        resized_tensor: resize后的张量 [batch, channels, target_size, target_size]
    """
    batch_size, channels, orig_h, orig_w = tensor.shape

    # 计算缩放比例：使较长边等于target_size
    scale = target_size / max(orig_h, orig_w)

    # 计算新的尺寸
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)

    # 使用双线性插值进行resize
    resized = F.interpolate(
        tensor,
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False
    )

    # 计算需要padding的像素数
    pad_h = target_size - new_h
    pad_w = target_size - new_w

    # 计算上下左右的padding
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # 添加padding：[left, right, top, bottom]
    if padding_mode == 'constant':
        padded = F.pad(resized, [pad_left, pad_right, pad_top, pad_bottom],
                       mode=padding_mode, value=padding_value)
    else:
        padded = F.pad(resized, [pad_left, pad_right, pad_top, pad_bottom],
                       mode=padding_mode)

    return padded

def encode_face_fea(face_pixel_values, fan_encoder, batch_size=32, is_train=True, drop_ratio=0.):
    """
    从视频序列中提取面部动作特征，使用预训练的FAN编码器

    参数:
        face_pixel_values (torch.Tensor): 视频帧，形状为(b, f, c, h, w)
                                    其中b是批次大小，f是帧数，c是通道数，h和w是高度和宽度
        fan_encoder: 预训练的FAN编码器模型
        batch_size (int): 处理帧的批次大小

    返回:
        torch.Tensor: 提取的特征
    """
    # 获取输入张量的形状
    b, f, c, h, w = face_pixel_values.shape

    param = next(fan_encoder.parameters())
    model_dtype = param.dtype
    model_device = param.device
    face_pixel_values = face_pixel_values.to(device=model_device, dtype=model_dtype)

    # reshaped_tensor = face_pixel_values.reshape(-1, c, h, w)  # [b, f, c, h, w]转为[b*f, c, h, w]

    # 将face_pixel_values resize到(224,224)，保持比例不变
    target_size = 224
    # 重新整形为 [b*f, c, h, w] 以便批量处理resize
    resized_tensor = face_pixel_values.reshape(-1, c, h, w)
    # 使用双线性插值进行resize
    # reshaped_tensor = F.interpolate(
    #     resized_tensor,
    #     size=(target_size, target_size),
    #     mode='bilinear',
    #     align_corners=False
    # )
    reshaped_tensor = resize_keep_aspect_ratio(
        resized_tensor,
        target_size=target_size,
        padding_mode='constant',  # 黑色填充
        padding_value=0
    )


    if not is_train:
        fan_encoder.eval()

    all_features = []
    if not is_train:
        with torch.no_grad():
            # for i in range(0, reshaped_tensor.shape[0], batch_size):
            #     end_idx = min(i + batch_size, reshaped_tensor.shape[0])
            #     batch_frames = reshaped_tensor[i:end_idx]
            features = fan_encoder.forward(reshaped_tensor)
            all_features.append(features)
    else:
        # for i in range(0, reshaped_tensor.shape[0], batch_size):
        #     end_idx = min(i + batch_size, reshaped_tensor.shape[0])
        #     batch_frames = reshaped_tensor[i:end_idx]
        features = fan_encoder.forward(reshaped_tensor)
        all_features.append(features)

    all_features = torch.cat(all_features, dim=0)

    # [b*f, feature_dim] -> [b, f, feature_dim]
    feature_dim = all_features.shape[1]
    all_features = all_features.reshape(b, f, feature_dim)

    if drop_ratio > 0:
        new_features = []
        for batch_features in all_features:
            if random.random() > drop_ratio:
                new_features.append(batch_features.unsqueeze(0))
            else:
                new_features.append(torch.zeros_like(batch_features.unsqueeze(0)))

        all_features = torch.cat(new_features, dim=0)

    return all_features


def prepare_img_latent(vae, vae_stream_2, video, lat_h, lat_w, device, dtype, random_frame):
    """
    video: (b, f, c, h, w)
    """
    bz = video.size(0)

    n_frames = video.size(1)
    mask = torch.zeros(bz, n_frames, lat_h, lat_w, device=device, dtype=dtype)
    mask_index = random.randint(0, n_frames - 1) if random_frame else 0
    mask[:, mask_index] = 1.

    mask = torch.cat([mask[:, :mask_index],torch.repeat_interleave(mask[:, mask_index:mask_index+1], repeats=4, dim=1),mask[:, mask_index+1:]], dim=1)  # (1, 84, lat_h, lat_w)
    
    mask = mask.view(bz, mask.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)  # (1, 4, 21, lat_h, lat_w)

    input_image = video if video.size(1) == 1 else video[:, mask_index].unsqueeze(1)
    bz, _, _, h, w = input_image.shape
    
    input_image = torch.cat([
        torch.zeros(bz, mask_index, 3, h, w, device=device),
        input_image,
        torch.zeros(bz, n_frames - mask_index - 1, 3, h, w, device=device)
    ], dim=1)

    if vae_stream_2 is not None:
        vae_stream_2.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(vae_stream_2):
            encoded_image = batch_encode_vae(vae, input_image, device, dtype)
    else:
        encoded_image = batch_encode_vae(vae, input_image, device, dtype)

    # mask: 不经过VAE编码的mask, 4个通道
    # encoded_image: 1个frame是图像,其他frame是0, 拼接后经过VAE编码, 16个通道
    result = torch.cat([mask, encoded_image], dim=1)  # (1, 20, 21, lat_h, lat_w)
    return result
      
def get_random_downsample_ratio(sample_size, image_ratio=[], all_choices=False, rng=None):
    def _create_special_list(length):
        if length == 1:
            return [1.0]
        if length >= 2:
            first_element = 0.75
            remaining_sum = 1.0 - first_element
            other_elements_value = remaining_sum / (length - 1)
            special_list = [first_element] + [other_elements_value] * (length - 1)
            return special_list
            
    if sample_size >= 1536:
        number_list = [1, 1.25, 1.5, 2, 2.5, 3] + image_ratio 
    elif sample_size >= 1024:
        number_list = [1, 1.25, 1.5, 2] + image_ratio
    elif sample_size >= 768:
        number_list = [1, 1.25, 1.5] + image_ratio
    elif sample_size >= 512:
        number_list = [1] + image_ratio
    else:
        number_list = [1]

    if all_choices:
        return number_list

    number_list_prob = np.array(_create_special_list(len(number_list)))
    if rng is None:
        return np.random.choice(number_list, p = number_list_prob)
    else:
        return rng.choice(number_list, p = number_list_prob)

def set_vae_device(vae, device, dtype=None):
    vae.model.to(device, dtype=dtype)
    vae.mean = vae.mean.to(device, dtype=dtype)
    vae.std = vae.std.to(device, dtype=dtype)
    vae.scale[0] = vae.scale[0].to(device, dtype=dtype)
    vae.scale[1] = vae.scale[1].to(device, dtype=dtype)


EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}


def log_validation(
    vae, text_encoder, net, clip_model,
    config, args, accelerator, weight_dtype, global_step
):
    # transform3d_val = net.transformer3d

    sample_shift = 5.0

    if args.transformer_path is not None:
        transformer3d_val = WanModel.from_pretrained_t2v(args.transformer_path, audio_model_path=args.audio_model_path).to(weight_dtype)
    else:
        model_type = accelerator.unwrap_model(net.transformer3d).model_type
        transformer3d_val = WanModel.from_pretrained_t2v(args.pretrained_model_name_or_path, model_type=model_type).to(weight_dtype)

    transformer3d_val.load_state_dict(accelerator.unwrap_model(net.transformer3d).state_dict())


    # fan_encoder_path = "/mnt/data/ssd/lisicheng/cogvideox-fun-ref-v1/dream-actor-m1-v2/output_dir/mulloss_back/checkpoint-1000/fan_encoder"
    face_encoder_val = FanEncoder(pose_dim=6, eye_dim=6, motion_only=False)
    face_encoder_val = face_encoder_val.to(accelerator.device, dtype=weight_dtype)
    # face_encoder_val = FanEncoder.from_pretrained(fan_encoder_path).to(weight_dtype)
    face2token_val = FaceProjModel(in_channels=542, intermediate_dim=1024, output_dim=1536, context_tokens=32)
    face2token_val = face2token_val.to(accelerator.device, dtype=weight_dtype)

    face_encoder_val.load_state_dict(accelerator.unwrap_model(net.fan_encoder).state_dict(), strict=False)
    face2token_val.load_state_dict(accelerator.unwrap_model(net.face2token).state_dict(), strict=False)


    logging.info("transformer3d_val has been created.")

    pipeline = wan.WanFace2V(
        config=wan_config,
        checkpoint_dir=args.pretrained_model_name_or_path,
        device=accelerator.device,
        rank=None,
        text_encoder=text_encoder,
        vae=vae,
        model=transformer3d_val,
        validation_mode=True,
        face_encoder=face_encoder_val,
        face2token=face2token_val
    )

    for i in range(len(args.validation_prompts)):
        with torch.no_grad():
            img = Image.open(args.validation_images[i]).convert("RGB")
            if "half" in args.task and args.validation_half_images is not None:
                half_img = Image.open(args.validation_half_images[i]).convert("RGB")
            else:
                half_img = None

            control_video, size = read_video_from_path(args.validation_poses[i], video_sample_n_frames=args.video_sample_n_frames)
            kp_meta_path = args.validation_poses[i][:-4] + '_meta.pkl'

            if "face" in args.task:
                face_video, _ = read_face_video_from_path(args.validation_faces[i],
                                                                    video_sample_n_frames=args.video_sample_n_frames)
            else:
                face_video = None

            # print(len(face_video))
            
            
            face_spheres_video, _ = read_video_from_path(args.validation_face_spheres[i],
                                                                    video_sample_n_frames=args.video_sample_n_frames)
            len_control_video = len(control_video)
            if "face" in args.task:
                len_face_video = len(face_video)
                frame_num = ((min(len_control_video, len_face_video)-1) // 4) * 4 + 1
                face_video = face_video[:frame_num]
            else:
                frame_num = len_control_video

            control_video = control_video[:frame_num]
            face_spheres_video = face_spheres_video[:frame_num]
            if "pose" not in args.task:
                control_video = None
            if "face-sphere" not in args.task:
                face_spheres_video = None
        
            sample = pipeline.generate(
                args.validation_prompts[i],
                img=img,
                half_img=half_img,
                control_video=control_video,
                kp_meta_path=args.validation_face_bboxes[i],
                # face_bbox_path=args.validation_face_bboxes[i],
                face_spheres_video=face_spheres_video,
                face_video=face_video,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=frame_num,
                shift=sample_shift,
                sample_solver='unipc',
                sampling_steps=50,
                guide_scale=7.5,
                guide_scale_img=5.0,
                guide_scale_face=0.0,
                drop_face = args.face_drop_ratio > 0,
                seed=random.randint(0, sys.maxsize),
                offload_model=False)

            sample = sample.unsqueeze(0).cpu()  # b c t h w
            # control_video = np.transpose(control_video, (3, 0, 1, 2))
            # control_video = np.expand_dims(control_video, axis=0)
            # control_video = torch.from_numpy(control_video).to(device=sample.device, dtype=sample.dtype)
            # control_video = control_video / 255.
            # control_video = (control_video - 0.5) * 2.0
            # control_video = torch.nn.functional.interpolate(control_video, size=sample.shape[-3:])
            #
            # faceverse = np.transpose(face_spheres_video, (3, 0, 1, 2))
            # faceverse = np.expand_dims(faceverse, axis=0)
            # faceverse = torch.from_numpy(faceverse).to(device=sample.device, dtype=sample.dtype)
            # faceverse = faceverse / 255.
            # faceverse = (faceverse - 0.5) * 2.0
            # faceverse = torch.nn.functional.interpolate(faceverse, size=sample.shape[-3:])

            face_video = np.transpose(face_video, (3, 0, 1, 2))
            face_video = np.expand_dims(face_video, axis=0)
            face_video = torch.from_numpy(face_video).to(device=sample.device, dtype=sample.dtype)
            face_video = face_video / 255.
            face_video = (face_video - 0.5) * 2.0
            # face_video = torch.nn.functional.interpolate(face_video, size=sample.shape[-3:])
            # Extract dimensions
            target_t, target_h, target_w = sample.shape[-3:]
            _, _, src_t, src_h, src_w = face_video.shape
            # Determine scale to maintain aspect ratio
            scale = min(target_h / src_h, target_w / src_w)
            new_h, new_w = int(src_h * scale), int(src_w * scale)
            # First resize preserving aspect ratio
            face_video = torch.nn.functional.interpolate(
                face_video, size=(target_t, new_h, new_w), mode='trilinear', align_corners=False
            )
            # Calculate padding for centering
            pad_left = (target_w - new_w) // 2
            pad_right = target_w - new_w - pad_left
            pad_top = (target_h - new_h) // 2
            pad_bottom = target_h - new_h - pad_top
            # Apply padding with black (-1.0)
            face_video = torch.nn.functional.pad(
                face_video, (pad_left, pad_right, pad_top, pad_bottom, 0, 0), mode='constant', value=-1.0
            )

            sample = torch.cat([face_video, sample])
            # sample = torch.cat([faceverse, control_video, sample])
        os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
        save_videos_grid(sample, os.path.join(args.output_dir, f"sample/sample-{global_step}-{i}.mp4"), rescale=True, fps=16)
        

    del pipeline
    del transformer3d_val
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()      


def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value

def generate_timestep_with_lognorm(low, high, shape, device="cpu", generator=None):
    u = torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
    t = 1 / (1 + torch.exp(-u)) * (high - low) + low
    return torch.clip(t.to(torch.int32), low, high - 1)


def sanity_check(args, batch, global_step=0, ext="gif", fps=16, use_video_id=True):
    if use_video_id:
        video_id = batch['video_id']
    pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
    if pixel_values.ndim==4:
        pixel_values = pixel_values.unsqueeze(2)
    os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)

    if args.body_mask_weight > 0.:
        body_mask_pixel_values = batch["body_mask_pixel_values"].cpu()
        face_mask_pixel_values = batch["face_mask_pixel_values"].cpu()
        
        body_mask_pixel_values = rearrange(body_mask_pixel_values, "b f c h w -> b c f h w")
        face_mask_pixel_values = rearrange(face_mask_pixel_values, "b f c h w -> b c f h w")

        for idx, (body_mask_pixel_value, face_mask_pixel_value, text) in enumerate(zip(body_mask_pixel_values, face_mask_pixel_values, texts)):
            body_mask_pixel_value = body_mask_pixel_value[None, ...]
            face_mask_pixel_value = face_mask_pixel_value[None, ...]
            if use_video_id:
                gif_name = video_id[idx]
                save_videos_grid(body_mask_pixel_value, f"{args.output_dir}/sanity_check/{gif_name}-body-mask.{ext}", rescale=True, fps=fps)
                save_videos_grid(face_mask_pixel_value, f"{args.output_dir}/sanity_check/{gif_name}-face-mask.{ext}", rescale=True, fps=fps)
            else:
                gif_name = '-'.join(text.replace('/', '').split()[:20]) if not text == '' else f'{global_step}-{idx}'
                save_videos_grid(body_mask_pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:20]}-body-mask.{ext}", rescale=True, fps=fps)
                save_videos_grid(face_mask_pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:20]}-face-mask.{ext}", rescale=True, fps=fps)
    
    if "half" in args.task:
        half_ref_pixel_values = batch['half_ref_pixel_values'].cpu()
        half_ref_pixel_values = (half_ref_pixel_values * 0.5 + 0.5) * 255.
        for idx, (half_ref_pixel_value, text) in enumerate(zip(half_ref_pixel_values, texts)):
            half_ref_pixel_value = half_ref_pixel_value[0].permute(1, 2, 0)
            if use_video_id:
                gif_name = video_id[idx]
                Image.fromarray(np.uint8(half_ref_pixel_value)).save(f"{args.output_dir}/sanity_check/{gif_name}-ref-half.png")
            else:
                gif_name = '-'.join(text.replace('/', '').split()[:20]) if not text == '' else f'{global_step}-{idx}'
                Image.fromarray(np.uint8(half_ref_pixel_value)).save(f"{args.output_dir}/sanity_check/ref_half_{gif_name[:20] if not text == '' else f'{global_step}-{idx}'}.png")

    if "face" in args.task:
        face_pixel_values = batch['face_pixel_values'].cpu()
        # face_pixel_values = (face_pixel_values * 0.5 + 0.5) * 255.
        face_pixel_values = rearrange(face_pixel_values, "b f c h w -> b c f h w")
        for idx, face_pixel_value in enumerate(face_pixel_values):
            face_pixel_value = face_pixel_value[None, ...]
            if use_video_id:
                gif_name = video_id[idx]
                # print(face_pixel_value.shape)
                save_videos_grid(face_pixel_value, f"{args.output_dir}/sanity_check/{gif_name}-face.{ext}", rescale=False, fps=fps)
            else:
                gif_name = '-'.join(text.replace('/', '').split()[:20]) if not text == '' else f'{global_step}-{idx}'
                save_videos_grid(face_pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:20]}-face.{ext}", rescale=True, fps=fps)

    if "sphere" in args.task:
        face_spheres_pixel_values = batch['face_spheres_pixel_values'].cpu()
        face_spheres_pixel_values = rearrange(face_spheres_pixel_values, "b f c h w -> b c f h w")
        for idx, face_pixel_value in enumerate(face_spheres_pixel_values):
            face_pixel_value = face_pixel_value[None, ...]
            if use_video_id:
                gif_name = video_id[idx]
                save_videos_grid(face_pixel_value, f"{args.output_dir}/sanity_check/{gif_name}-face-sphere.{ext}", rescale=True, fps=fps)     
            else:
                gif_name = '-'.join(text.replace('/', '').split()[:20]) if not text == '' else f'{global_step}-{idx}'
                save_videos_grid(face_pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:20]}-face-sphere.{ext}", rescale=True, fps=fps) 

    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
        pixel_value = pixel_value[None, ...]
        if use_video_id:
            gif_name = video_id[idx]
            save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/{gif_name}.{ext}", rescale=True, fps=fps)
        else:
            gif_name = '-'.join(text.replace('/', '').split()[:20]) if not text == '' else f'{global_step}-{idx}'
            save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:20]}.{ext}", rescale=True, fps=fps)
            
    if "pose" in args.task:
        kp_pixel_values = batch['kp_pixel_values'].cpu()
        kp_pixel_values = rearrange(kp_pixel_values, "b f c h w -> b c f h w")
        for idx, kp_pixel_value in enumerate(kp_pixel_values):
            kp_pixel_value = kp_pixel_value[None, ...]
            if use_video_id:
                gif_name = video_id[idx]
                save_videos_grid(kp_pixel_value, f"{args.output_dir}/sanity_check/{gif_name}-pose.{ext}", rescale=True, fps=fps)     
            else:
                gif_name = '-'.join(text.replace('/', '').split()[:20]) if not text == '' else f'{global_step}-{idx}'
                save_videos_grid(kp_pixel_value, f"{args.output_dir}/sanity_check/kp_{gif_name[:20]}.{ext}", rescale=True, fps=fps)      

    if "i2v" in args.task or "ipa" in args.task:
        clip_pixel_values = batch['clip_pixel_values'].cpu()
        for idx, (clip_pixel_value, text) in enumerate(zip(clip_pixel_values, texts)):
            clip_pixel_value = (clip_pixel_value * 0.5 + 0.5) * 255.
            if use_video_id:
                gif_name = video_id[idx]
                Image.fromarray(np.uint8(clip_pixel_value)).save(f"{args.output_dir}/sanity_check/{gif_name}-clip.png")
            else:
                gif_name = '-'.join(text.replace('/', '').split()[:20]) if not text == '' else f'{global_step}-{idx}'
                Image.fromarray(np.uint8(clip_pixel_value)).save(f"{args.output_dir}/sanity_check/clip_{gif_name[:20] if not text == '' else f'{global_step}-{idx}'}.png")
    
    if "ref2v" in args.task:
        ref_pixel_values = batch['ref_pixel_values'].cpu()
        ref_pixel_values = (ref_pixel_values * 0.5 + 0.5) * 255.
        for idx, (ref_pixel_value, text) in enumerate(zip(ref_pixel_values, texts)):
            ref_pixel_value = ref_pixel_value[0].permute(1, 2, 0)
            if use_video_id:
                gif_name = video_id[idx]
                Image.fromarray(np.uint8(ref_pixel_value)).save(f"{args.output_dir}/sanity_check/{gif_name}-ref.png")
            else:
                gif_name = '-'.join(text.replace('/', '').split()[:20]) if not text == '' else f'{global_step}-{idx}'
                Image.fromarray(np.uint8(ref_pixel_value)).save(f"{args.output_dir}/sanity_check/ref_{gif_name[:20] if not text == '' else f'{global_step}-{idx}'}.png")
    

def get_dataloader(args, rng, sample_n_frames_bucket_interval=4):
    # Get the dataset
    train_dataset = ImageVideoDataset(
        args.train_data_meta, 
        args.train_data_dir,
        video_sample_size=args.video_sample_size, 
        video_sample_fps=args.video_sample_fps,
        video_sample_stride=args.video_sample_stride, 
        video_sample_n_frames=args.video_sample_n_frames, 
        video_repeat=args.video_repeat, 
        image_sample_size=args.image_sample_size,
        enable_bucket=args.enable_bucket, 
        enable_inpaint=True if "i2v" in args.task else False,
        enable_ref=True if "ref" in args.task else False,
        enable_kp=True if "pose" in args.task else False,
        enable_face=True if "face" in args.task else False,
        enable_audio=True if "audio" in args.task else False,
        enable_face_spheres=True if "sphere" in args.task else False,
        # enable_fref=True if "fref" in args.task else False,  # reference face
        enable_half_body=True if "half" in args.task else False,
        enable_body_mask=args.body_mask_weight>0.,
        trans_to_vertical=args.trans_to_vertical,
        v_ratio=args.v_ratio,
    )

    if args.enable_bucket:
        aspect_ratio_sample_size = {key : [x / 632 * args.video_sample_size for x in ASPECT_RATIO_632[key]] for key in ASPECT_RATIO_632.keys()}
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = AspectRatioBatchImageVideoSampler(
            sampler=RandomSampler(train_dataset, generator=batch_sampler_generator), dataset=train_dataset.dataset,
            batch_size=args.train_batch_size, train_folder = args.train_data_dir, drop_last=True,
            aspect_ratios=aspect_ratio_sample_size,
        )

        # train_sampler = torch.utils.data.distributed.DistributedSampler(
        #     train_dataset,
        #     num_replicas=accelerator.num_processes,
        #     rank=accelerator.process_index,
        #     shuffle=True,
        #     seed=args.seed
        # )
        #
        # batch_sampler = AspectRatioBatchImageVideoSampler(
        #     sampler=train_sampler,  # 使用分布式采样器替代随机采样器
        #     dataset=train_dataset.dataset,
        #     batch_size=args.train_batch_size,
        #     train_folder=args.train_data_dir,
        #     drop_last=True,
        #     aspect_ratios=aspect_ratio_sample_size,
        # )
        
        # Get the frame length at different resolutions according to token_length
        def get_length_to_frame_num(token_length):
            if args.image_sample_size > args.video_sample_size:
                sample_sizes = list(range(args.video_sample_size, args.image_sample_size + 1, 32))

                if sample_sizes[-1] != args.image_sample_size:
                    sample_sizes.append(args.image_sample_size)
            else:
                sample_sizes = [args.image_sample_size]
            
            length_to_frame_num = {
                sample_size: min(token_length / sample_size / sample_size, args.video_sample_n_frames) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1 for sample_size in sample_sizes
            }


            return length_to_frame_num

        def collate_fn(examples):
            # Get token length
            target_token_length = args.video_sample_n_frames * args.token_sample_size * args.token_sample_size
            length_to_frame_num = get_length_to_frame_num(target_token_length)

            # Create new output
            new_examples = {}
            new_examples["target_token_length"] = target_token_length
            new_examples["pixel_values"] = []
            new_examples["text"] = []
            new_examples["video_id"] = []
            new_examples["data_type"] = []

            # Used in Inpaint mode
            if "i2v" in args.task or "ipa" in args.task:
                new_examples["clip_pixel_values"] = []
            if "ref2v" in args.task:
                new_examples["ref_pixel_values"] = []
            if "half" in args.task:
                new_examples["half_ref_pixel_values"] = []
            if 'pose' in args.task:
                new_examples["kp_pixel_values"] = []
            if 'face' in args.task:
                new_examples["face_pixel_values"] = []
            if 'audio' in args.task:
                new_examples["audio_values"] = []
            if "sphere" in args.task:
                new_examples["face_spheres_pixel_values"] = []
            if args.body_mask_weight > 0.:
                new_examples["body_mask_pixel_values"] = []
                new_examples["face_mask_pixel_values"] = []
                # new_examples["face_features_mask_pixel_values"] = []

            # Get downsample ratio in image and videos
            pixel_value = examples[0]["pixel_values"]
            data_type = examples[0]["data_type"]
            f, h, w, c = np.shape(pixel_value)
            if data_type == 'image':
                random_downsample_ratio = 1 if not args.random_hw_adapt else get_random_downsample_ratio(
                    args.image_sample_size, image_ratio=[args.image_sample_size / args.video_sample_size], rng=rng)

                aspect_ratio_sample_size = {
                    key: [x / 632 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_632[key]] for
                    key in ASPECT_RATIO_632.keys()}
                aspect_ratio_random_crop_sample_size = {
                    key: [x / 632 * args.image_sample_size / random_downsample_ratio for x in
                          ASPECT_RATIO_RANDOM_CROP_632[key]] for key in ASPECT_RATIO_RANDOM_CROP_632.keys()}

                batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
            else:
                if args.random_hw_adapt:
                    if args.training_with_video_token_length:
                        local_min_size = np.min(np.array([np.mean(
                            np.array([np.shape(example["pixel_values"])[1], np.shape(example["pixel_values"])[2]])) for
                            example in examples]))
                        # The video will be resized to a lower resolution than its own.
                        choice_list = [length for length in list(length_to_frame_num.keys()) if
                                       length < local_min_size * 1.25]
                        if len(choice_list) == 0:
                            choice_list = list(length_to_frame_num.keys())
                        if rng is None:
                            local_video_sample_size = np.random.choice(choice_list)
                        else:
                            local_video_sample_size = rng.choice(choice_list)
                        batch_video_length = length_to_frame_num[local_video_sample_size]
                        random_downsample_ratio = args.video_sample_size / local_video_sample_size
                    else:
                        random_downsample_ratio = get_random_downsample_ratio(
                            args.video_sample_size, rng=rng)
                        batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
                else:
                    random_downsample_ratio = 1
                    batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval

                aspect_ratio_sample_size = {
                    key: [x / 632 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_632[key]] for
                    key in ASPECT_RATIO_632.keys()}
                aspect_ratio_random_crop_sample_size = {
                    key: [x / 632 * args.video_sample_size / random_downsample_ratio for x in
                          ASPECT_RATIO_RANDOM_CROP_632[key]] for key in ASPECT_RATIO_RANDOM_CROP_632.keys()}

            closest_size, closest_ratio = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size)
            closest_size = [int(x / 16) * 16 for x in closest_size]
            if args.random_ratio_crop:
                if rng is None:
                    random_sample_size = aspect_ratio_random_crop_sample_size[
                        np.random.choice(list(aspect_ratio_random_crop_sample_size.keys()),
                                         p=ASPECT_RATIO_RANDOM_CROP_PROB)
                    ]
                else:
                    random_sample_size = aspect_ratio_random_crop_sample_size[
                        rng.choice(list(aspect_ratio_random_crop_sample_size.keys()), p=ASPECT_RATIO_RANDOM_CROP_PROB)
                    ]
                random_sample_size = [int(x / 16) * 16 for x in random_sample_size]

            for example in examples:
                if args.random_ratio_crop:
                    # To 0~1
                    pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    if "ref2v" in args.task:
                        ref_pixel_values = torch.from_numpy(example["ref_pixel_values"]).permute(0, 3, 1,
                                                                                                 2).contiguous()
                        ref_pixel_values = ref_pixel_values / 255.
                    if "half" in args.task:
                        half_ref_pixel_values = torch.from_numpy(example["half_ref_pixel_values"]).permute(0, 3, 1,
                                                                                                           2).contiguous()
                        half_ref_pixel_values = half_ref_pixel_values / 255.
                    if "pose" in args.task:
                        kp_pixel_values = torch.from_numpy(example["kp_pixel_values"]).permute(0, 3, 1, 2).contiguous()
                        kp_pixel_values = kp_pixel_values / 255.
                    if 'face' in args.task:
                        assert example.get("face_pixel_values") is not None
                        face_pixel_values = torch.from_numpy(example["face_pixel_values"]).permute(0, 3, 1,
                                                                                                   2).contiguous()
                        face_pixel_values = face_pixel_values / 255.

                    if args.body_mask_weight > 0.:
                        body_mask_pixel_values = torch.from_numpy(example["body_mask_pixel_values"]).permute(0, 3, 1,
                                                                                                             2).contiguous()
                        body_mask_pixel_values = body_mask_pixel_values / 255.
                        
                        face_mask_pixel_values = torch.from_numpy(example["face_mask_pixel_values"]).permute(0, 3, 1,
                                                                                                             2).contiguous()
                        face_mask_pixel_values = face_mask_pixel_values / 255.

                        # face_features_mask_pixel_values = torch.from_numpy(example["face_features_mask_pixel_values"]).permute(0, 3, 1,
                        #                                                         2).contiguous()
                        # face_features_mask_pixel_values = face_features_mask_pixel_values / 255.


                    # Get adapt hw for resize
                    b, c, h, w = pixel_values.size()
                    th, tw = random_sample_size
                    if th / tw > h / w:
                        nh = int(th)
                        nw = int(w / h * nh)
                    else:
                        nw = int(tw)
                        nh = int(h / w * nw)

                    transform = transforms.Compose([
                        transforms.Resize([nh, nw]),
                        transforms.CenterCrop([int(x) for x in random_sample_size]),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])
                else:
                    # To 0~1
                    pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    if "ref2v" in args.task:
                        ref_pixel_values = torch.from_numpy(example["ref_pixel_values"]).permute(0, 3, 1,
                                                                                                 2).contiguous()
                        ref_pixel_values = ref_pixel_values / 255.
                    if "half" in args.task:
                        half_ref_pixel_values = torch.from_numpy(example["half_ref_pixel_values"]).permute(0, 3, 1,
                                                                                                           2).contiguous()
                        half_ref_pixel_values = half_ref_pixel_values / 255.
                    if 'pose' in args.task:
                        kp_pixel_values = torch.from_numpy(example["kp_pixel_values"]).permute(0, 3, 1, 2).contiguous()
                        kp_pixel_values = kp_pixel_values / 255.
                    if 'face' in args.task:
                        face_pixel_values = torch.from_numpy(example["face_pixel_values"]).permute(0, 3, 1,
                                                                                                   2).contiguous()
                        face_pixel_values = face_pixel_values / 255.

                    if "sphere" in args.task:
                        face_spheres_pixel_values = torch.from_numpy(example["face_spheres_pixel_values"]).permute(0, 3, 1,
                                                                                                   2).contiguous()
                        face_spheres_pixel_values = face_spheres_pixel_values / 255.

                    if args.body_mask_weight > 0.:
                        body_mask_pixel_values = torch.from_numpy(example["body_mask_pixel_values"]).permute(0, 3, 1,
                                                                                                             2).contiguous()
                        body_mask_pixel_values = body_mask_pixel_values / 255.
                        face_mask_pixel_values = torch.from_numpy(example["face_mask_pixel_values"]).permute(0, 3, 1,
                                                                                                             2).contiguous()
                        face_mask_pixel_values = face_mask_pixel_values / 255.
                        # face_features_mask_pixel_values = torch.from_numpy(example["face_features_mask_pixel_values"]).permute(0, 3, 1,
                        #                                                                                      2).contiguous()
                        # face_features_mask_pixel_values = face_features_mask_pixel_values / 255.
                        # Get adapt hw for resize
                    closest_size = list(map(lambda x: int(x), closest_size))
                    if closest_size[0] / h > closest_size[1] / w:
                        resize_size = closest_size[0], int(w * closest_size[0] / h)
                    else:
                        resize_size = int(h * closest_size[1] / w), closest_size[1]

                    # transform = transforms.Compose([
                    #     transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
                    #     # Image.BICUBIC
                    #     transforms.CenterCrop(closest_size),
                    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    # ])
                    transform_face = transforms.Compose([
                        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
                        transforms.CenterCrop(closest_size),
                    ])
                    transform_face_mask = transform = transforms.Compose([
                        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
                        # Image.BICUBIC
                        transforms.CenterCrop(closest_size),
                        transforms.Normalize(mean=[0.5], std=[0.5], inplace=True),
                    ])
                state = torch.get_rng_state()
                torch.set_rng_state(state)
                new_examples["pixel_values"].append(transform(pixel_values))
                if "pose" in args.task:
                    torch.set_rng_state(state)
                    new_examples["kp_pixel_values"].append(transform(kp_pixel_values))
                if args.body_mask_weight > 0.:
                    torch.set_rng_state(state)
                    new_examples["body_mask_pixel_values"].append(transform(body_mask_pixel_values))
                    new_examples["face_mask_pixel_values"].append(transform_face_mask(face_mask_pixel_values))
                    # new_examples["face_features_mask_pixel_values"].append(transform_face_mask(face_features_mask_pixel_values))
                if "ref2v" in args.task:
                    torch.set_rng_state(state)
                    new_examples["ref_pixel_values"].append(transform(ref_pixel_values))
                if "half" in args.task:
                    torch.set_rng_state(state)
                    new_examples["half_ref_pixel_values"].append(transform(half_ref_pixel_values))
                if "face" in args.task:
                    torch.set_rng_state(state)
                    new_examples["face_pixel_values"].append(transform_face(face_pixel_values))
                if "sphere" in args.task:
                    torch.set_rng_state(state)
                    new_examples["face_spheres_pixel_values"].append(transform(face_spheres_pixel_values))

                new_examples["text"].append(example["text"])
                new_examples["video_id"].append(example["video_id"])
                new_examples["data_type"].append(example["data_type"])
                if 'audio' in args.task:
                    assert example.get("audio_values") is not None
                    new_examples["audio_values"].append(example["audio_values"])

                # needs the number of frames to be 4n + 1.
                batch_video_length = int(
                    min(
                        batch_video_length,
                        (
                                len(pixel_values) - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1,
                    )
                )

                if batch_video_length == 0:
                    batch_video_length = 1

                # Used in Inpaint mode
                # clip image_prompt 跟 image_latent的image不一定相同
                if "i2v" in args.task or "ipa" in args.task:
                    def get_random_clip_index(low, high):
                        if high - low <= 1.1:
                            return low
                        values = np.arange(low, high)
                        probabilities = np.ones(len(values)) * 0.5 / (len(values) - 1)
                        probabilities[0] = 0.5
                        return np.random.choice(values, p=probabilities)

                    if "ref2v" not in args.task or random.random() < 0.5:
                        clip_index = get_random_clip_index(0, len(new_examples["pixel_values"][-1]))
                        clip_pixel_values = new_examples["pixel_values"][-1][clip_index].permute(1, 2, 0).contiguous()
                    else:
                        # 设置一定概率clip=ref
                        clip_pixel_values = new_examples["ref_pixel_values"][-1][0].permute(1, 2, 0).contiguous()
                    new_examples["clip_pixel_values"].append(clip_pixel_values)

            # Limit the number of frames to the same
            new_examples["pixel_values"] = torch.stack(
                [example[:batch_video_length] for example in new_examples["pixel_values"]])
            if "i2v" in args.task or "ipa" in args.task:
                new_examples["clip_pixel_values"] = torch.stack(
                    [example for example in new_examples["clip_pixel_values"]])
            if "ref2v" in args.task:
                new_examples["ref_pixel_values"] = torch.stack(
                    [example for example in new_examples["ref_pixel_values"]])
            if "pose" in args.task:
                new_examples["kp_pixel_values"] = torch.stack(
                    [example[:batch_video_length] for example in new_examples["kp_pixel_values"]])
            if args.body_mask_weight > 0.:
                new_examples["body_mask_pixel_values"] = torch.stack(
                    [example[:batch_video_length] for example in new_examples["body_mask_pixel_values"]])
                new_examples["face_mask_pixel_values"] = torch.stack(
                    [example[:batch_video_length] for example in new_examples["face_mask_pixel_values"]])
                # new_examples["face_features_mask_pixel_values"] = torch.stack(
                #     [example[:batch_video_length] for example in new_examples["face_features_mask_pixel_values"]])
            if "half" in args.task:
                new_examples["half_ref_pixel_values"] = torch.stack(
                    [example for example in new_examples["half_ref_pixel_values"]])
            if "face" in args.task:
                new_examples["face_pixel_values"] = torch.stack(
                    [example[:batch_video_length] for example in new_examples["face_pixel_values"]])
            if "audio" in args.task:
                new_examples["audio_values"] = torch.stack(
                    [example[:batch_video_length] for example in new_examples["audio_values"]])
            if "sphere" in args.task:
                new_examples["face_spheres_pixel_values"] = torch.stack(
                    [example[:batch_video_length] for example in new_examples["face_spheres_pixel_values"]])

            return new_examples
        
        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
        )
    else:
        # DataLoaders creation:
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = ImageVideoSampler(RandomSampler(train_dataset, generator=batch_sampler_generator), train_dataset, args.train_batch_size)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
        )

        # train_sampler = torch.utils.data.distributed.DistributedSampler(
        #     train_dataset,
        #     num_replicas=accelerator.num_processes,
        #     rank=accelerator.process_index,
        #     shuffle=True,
        #     seed=args.seed
        # )
        #
        # batch_sampler = ImageVideoSampler(
        #     train_sampler,  # 使用分布式采样器替代随机采样器
        #     train_dataset,
        #     args.train_batch_size
        # )
    return train_dataloader, train_dataset, batch_sampler


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. "
        ),
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        default=None,
        help=(
            "A csv containing the training data. "
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )

    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )

    parser.add_argument(
        "--validation_images",
        type=str,
        default=None,
        nargs="+",
        help=("A set of image prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
            )

    parser.add_argument(
        "--validation_poses",
        type=str,
        default=None,
        nargs="+",
        help=("A set of image prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--validation_faces",
        type=str,
        default=None,
        nargs="+",
        help=("A set of image prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--validation_face_bboxes",
        type=str,
        default=None,
        nargs="+",
        help=("A set of image prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--validation_face_spheres",
        type=str,
        default=None,
        nargs="+",
        help=("A set of image prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="datasets/humangen",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--validation_half_images",
        type=str,
        default=None,
        nargs="+",
        help=("A set of image prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--face_model_path",
        type=str,
        default=None,
        help=("face motion feature encoder weight"),
    )
    parser.add_argument(
        "--face2token_path",
        type=str,
        default=None,
        help=("face motion feature encoder weight"),
    )
    parser.add_argument(
        "--audio_model_path",
        type=str,
        default=None,
        help=("audio motion feature encoder weight"),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--use_came",
        action="store_true",
        help="whether to use came",
    )
    parser.add_argument(
        "--multi_stream",
        action="store_true",
        help="whether to use cuda multi-stream",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--vae_mini_batch", type=int, default=32, help="mini batch size for vae."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_model_info", action="store_true", help="Whether or not to report more info about model (such as norm, grad)."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=20000,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=20000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    
    parser.add_argument(
        "--snr_loss", action="store_true", help="Whether or not to use snr_loss."
    )
    parser.add_argument(
        "--uniform_sampling", action="store_true", help="Whether or not to use uniform_sampling."
    )
    parser.add_argument(
        "--loss_type", 
        type=str,
        default="sigma",
        help=(
            'The format of training data. Support `"sigma"`'
            ' (default), `"ddpm"`, `"flow"`.'
        ),
    )
    parser.add_argument(
        "--enable_bucket", action="store_true", help="Whether enable bucket sample in datasets."
    )
    parser.add_argument(
        "--random_ratio_crop", action="store_true", help="Whether enable random ratio crop sample in datasets."
    )
    parser.add_argument(
        "--random_frame_crop", action="store_true", help="Whether enable random frame crop sample in datasets."
    )
    parser.add_argument(
        "--random_hw_adapt", action="store_true", help="Whether enable random adapt height and width in datasets."
    )

    parser.add_argument(
        "--training_with_video_token_length", action="store_true", help="The training stage of the model in training.",
    )
    parser.add_argument(
        "--noise_share_in_frames", action="store_true", help="Whether enable noise share in frames."
    )
    parser.add_argument(
        "--noise_share_in_frames_ratio", type=float, default=0.5, help="Noise share ratio.",
    )
    parser.add_argument(
        "--motion_sub_loss", action="store_true", help="Whether enable motion sub loss."
    )
    parser.add_argument(
        "--motion_sub_loss_ratio", type=float, default=0.25, help="The ratio of motion sub loss."
    )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Run train_sampling_steps.",
    )
    parser.add_argument(
        "--keep_all_node_same_token_length",
        action="store_true", 
        help="Reference of the length token.",
    )
    parser.add_argument(
        "--token_sample_size",
        type=int,
        default=512,
        help="Sample size of the token.",
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=512,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--image_sample_size",
        type=int,
        default=512,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--video_sample_fps",
        type=float,
        default=16,
        help="Sample fps of the video.",
    )
    parser.add_argument(
        "--video_sample_stride",
        type=float,
        default=4,
        help="Sample stride of the video. This will not work if video_sample_fps is set.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=17,
        help="Num frame of video.",
    )
    parser.add_argument(
        "--video_repeat",
        type=int,
        default=0,
        help="Num of repeat video.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=(
            "The config of the model in training."
        ),
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other transformers, input its path."),
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other vaes, input its path."),
    )

    parser.add_argument(
        '--trainable_modules', 
        nargs='+', 
        help='Enter a list of trainable modules'
    )
    parser.add_argument(
        '--untrainable_modules',
        nargs='+',
        help='Enter a list of trainable modules'
    )
    parser.add_argument(
        '--trainable_modules_fan_encoder',
        nargs='+',
        help='Enter a list of trainable modules'
    )
    parser.add_argument(
        '--trainable_modules_low_learning_rate', 
        nargs='+', 
        default=[],
        help='Enter a list of trainable modules with lower learning rate'
    )
    parser.add_argument(
        '--tokenizer_max_length', 
        type=int,
        default=256,
        help='Max length of tokenizer'
    )
    parser.add_argument(
        "--use_deepspeed", action="store_true", help="Whether or not to use deepspeed."
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Whether enable low_vram mode."
    )
    parser.add_argument(
        "--offload_optimizer_device", type=str, help="offload optimizer to: cpu or nvme., Default is None"
    )
    parser.add_argument(
        "--abnormal_norm_clip_start",
        type=int,
        default=1000,
        help=(
            'When do we start doing additional processing on abnormal gradients. '
        ),
    )
    parser.add_argument(
        "--initial_grad_norm_ratio",
        type=int,
        default=5,
        help=(
            'The initial gradient is relative to the multiple of the max_grad_norm. '
        ),
    )
    
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-1.3B",
        help="The task to run.")
    
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--pre_validation",
        action="store_true",
        default=False,
        help="Validation before training.",
    )
    parser.add_argument(
        "--sanity_check",
        action="store_true",
        default=False,
        help="check data before training.",
    )
    parser.add_argument(
        "--random_frame",
        action="store_true",
        default=False,
        help="Use random frame instead of the first frame in image2video training.",
    )

    parser.add_argument(
        "--body_mask_weight",
        default=0.,
        type=float,
        help="Use body mask loss weight. It requires dwpose annotation.",
    )

    parser.add_argument(
        "--v_ratio",
        type=str,
        default=None,
        help="Crop horizontal video to vertical video according to bboxes.",
    )

    parser.add_argument(
        "--ref_drop_ratio",
        type=float,
        default=0.1,
        help="Crop horizontal video to vertical video according to bboxes.",
    )

    parser.add_argument(
        "--face_drop_ratio",
        type=float,
        default=0.,
        help="Crop horizontal video to vertical video according to bboxes.",
    )

    parser.add_argument(
        "--trans_to_vertical",
        choices=["true", True, "false", False, "random"], 
        default=False,
    )


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args
