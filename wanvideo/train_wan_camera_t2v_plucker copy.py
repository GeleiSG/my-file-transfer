import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
from lightning.pytorch.strategies import FSDPStrategy
import pandas as pd
from diffsynth import WanVideoPipeline, ModelManager, load_state_dict
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np
from torch.utils.data._utils.collate import default_collate
import torch.nn as nn
import gc
import random
from tqdm import tqdm
from omegaconf import OmegaConf
from diffsynth.data.bucket_sampler import (
                                             AspectRatioImageVideoSampler,
                                             DistributedRandomBatchSampler,
                                             RandomSampler,
                                             AllInOneAspectRatioSampler,
                                             RobustParallelSampler)
from diffsynth.data.transforms import get_aspect_ratio
from diffsynth.data.utils_data import get_closest_ratio_key
from diffsynth.data.camera_eval import calculate_pose_errors
from diffsynth.data.camera_utils import get_camera_sparse_embedding, get_plucker_embedding, get_plucker_embedding_cpu
from diffsynth.data.camera_video import CameraVideoDataset, get_dataloader
from diffsynth.data.video import save_video
from diffsynth.models.wan_video_camera_adapter import SimpleAdapter
from diffsynth.pipelines.wan_camera_video import WanVideoCameraPipeline
import third_party.depth_pro.depth_pro as depth_pro
from diffsynth.data.pointcloud import point_rendering_train_stage
from pytorch3d.renderer import PointsRasterizationSettings
import einops
from torchvision.transforms import ToTensor, ToPILImage
from torchvision import transforms
from diffusers.utils import export_to_video

from torch.utils.data.distributed import DistributedSampler
from lightning.pytorch.strategies import DDPStrategy

import torch
import traceback
import sys
import os

class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": True, "tile_size": tile_size, "tile_stride": tile_stride}
        
    def test_step(self, batch):
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]

        self.pipe.device = self.device
        if video is not None:
            # prompt
            prompt_emb = self.pipe.encode_prompt(text)
            # video
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)
            # image
            if "first_frame" in batch:
                first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                _, _, num_frames, height, width = video.shape
                image_emb = self.pipe.encode_image(first_frame, None, num_frames, height, width)
            else:
                image_emb = {}
            data = {"latents": latents, "prompt_emb": prompt_emb, "image_emb": image_emb, "camera_emb": batch["camera_embedding"]}
            torch.save(data, path + ".tensors.pth")

class FixedValDataset(torch.utils.data.Dataset):
    def __init__(self, pt_path, max_items=5):
        super().__init__()
        self.data = []
        total_data = torch.load(pt_path)
        for batch_idx, key in enumerate(total_data):
            if batch_idx >= max_items: # 只取前5个
                break
            self.data.append(total_data[key])   

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # dict，比如 {"video": ..., "cond": ...}

class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self, dit_path,
        text_encoder_path, vae_path, image_encoder_path=None, resume_ckpt_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16),
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None
    ):
        super().__init__()
        self.train_architecture = train_architecture
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        # model_manager.load_models(model_path) # load text encoder, vae, image encoder

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        if os.path.isfile(dit_path):
            model_path.append(dit_path)
        else:
            dit_path = dit_path.split(",")
            model_path.append(dit_path)
        model_manager.load_models(model_path)
        if resume_ckpt_path is not None and train_architecture == "lora":
            model_manager.load_lora(resume_ckpt_path, lora_alpha=1.0)
        self.pipe = WanVideoCameraPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # dim=self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        # self.pipe.dit.cam_adapter = SimpleAdapter(24, dim, kernel_size=(2,2), stride=(2,2))
        # for block_i, block in enumerate(self.pipe.dit.blocks):
        #         block.cam_encoder = SimpleAdapter(24, dim, kernel_size=(2,2), stride=(2,2))
        #         block.projector = nn.Linear(dim, dim)
        #         block.projector.weight.data.zero_()
        #         block.projector.bias.data.zero_()
        cfg = OmegaConf.load("/mnt/data/ssd/user_workspace/duanke/unicontrol/diffsynth/configs/config.json")
        self.pipe.dit.controlnet_cfg = cfg.controlnet_cfg
        self.pipe.dit.build_controlnet()
        
        
        if resume_ckpt_path is not None:
            print(f"Loading resume ckpt path: {resume_ckpt_path}")
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")
            self.pipe.dit.load_state_dict(state_dict, strict=False)
            # self.pipe.to("cuda")
            # self.pipe.to(dtype=torch.bfloat16)

        self.freeze_parameters()


        
        # if camera_adapter_path is not None:
        #     camera_state_dict = load_state_dict(camera_adapter_path)
    
        #     # 检查是否包含 camera_adapter 的参数
        #     if 'camera_adapter' in camera_state_dict:
        #         print("Loading camera_adapter parameters.")
        #         self.pipe.dit.camera_adapter.load_state_dict(camera_state_dict['camera_adapter'])
        #     else:
        #         print("No camera_adapter found in the checkpoint.")
        
        # 设置camera adapter为True来训练，考虑对扩散dit进行lora微调或冻结
        # 一般来说，对于相机控制和I2V都是控制vae和encoder是冻结的，dit是要全部微调的，
        # 先尝试把lora和全量的I2V跑通，感觉用camera adapter效果不好，还是要放最前面的token上
        if train_architecture == "adapter":
            for name, module in self.pipe.denoising_model().named_modules():
                # if any(keyword in name for keyword in ["cam_adapter"]):
                if any(keyword in name for keyword in ["controlnet_patch_embedding", "controlnet_mask_embedding", "controlnet", "controlnet_freqs"]):
                    print(f"Trainable: {name}")
                    for param in module.parameters():
                        param.requires_grad = True
            trainable_params = 0
            seen_params = set()
            for name, module in self.pipe.denoising_model().named_modules():
                for param in module.parameters():
                    if param.requires_grad and param not in seen_params:
                        trainable_params += param.numel()
                        seen_params.add(param)
            print(f"Total number of trainable parameters: {trainable_params}")
            # self.pipe.denoising_model().requires_grad_(False)
            # if hasattr(self.pipe, 'dit') and hasattr(self.pipe.dit, 'camera_adapter'):
            #     camera_adapter = self.pipe.dit.camera_adapter  # 获取 dit 中的 camera_adapter
            #     for param in camera_adapter.parameters():
            #         param.requires_grad = True 
            # self.add_lora_to_model(
            #     self.pipe.denoising_model(),
            #     lora_rank=lora_rank,
            #     lora_alpha=lora_alpha,
            #     lora_target_modules=lora_target_modules,
            #     init_lora_weights=init_lora_weights,
            #     pretrained_lora_path=pretrained_lora_path,
            # )
        elif train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
            for name, module in self.pipe.denoising_model().named_modules():
                if any(keyword in name for keyword in ["cam_encoder", "projector", "cam_adapter"]):
                    print(f"Trainable: {name}")
                    for param in module.parameters():
                        param.requires_grad = True
            for name, param in self.pipe.denoising_model().named_parameters():
                if param.requires_grad:
                    if 'lora_' in name:
                        print(f"[LoRA 参数]     {name}")
                    else:
                        print(f"[全量训练参数] {name}")
                else:
                    if 'lora_' in name:
                        print(f"[❌ 冻结 LoRA]  {name}")
                    else:
                        pass  # 常规参数冻结，无需打印
            
        else:
            self.pipe.denoising_model().requires_grad_(True)

        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

        # 添加深度估计
        # model, transform = depth_pro.create_model_and_transforms(device=self.device)
        # model.requires_grad_(False)
        # self._depth_model = model.eval()
        # self._depth_transform = transform
        
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
    
    def training_step(self, batch, batch_idx):
        text, video, path = batch["text"], batch["video"], batch["path"]
        video = rearrange(video, "B T C H W -> B C T H W")
        render_video = batch["render_video"]
        render_mask = batch["render_mask"]
        render_video = rearrange(render_video, "B T C H W -> B C T H W")
        render_mask = rearrange(render_mask, "B T C H W -> B C T H W")

        self.pipe.device = self.device
        if video is not None:
            # prompt
            prompt_emb = self.pipe.encode_prompt(text)
            # video
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)

            if "first_frame" in batch:
                batch_image_emb_clip = []
                batch_image_emb_y = []
                for i in range(batch["first_frame"].shape[0]):
                    first_frame = batch["first_frame"][i]
                    _, _, num_frames, height, width = video.shape
                    image_emb = self.pipe.encode_image(first_frame, None, num_frames, height, width)
                    batch_image_emb_clip.append(image_emb["clip_feature"][0])
                    batch_image_emb_y.append(image_emb["y"][0])
                image_emb = {}                   
                image_emb["clip_feature"] = torch.stack(batch_image_emb_clip)
                image_emb["y"] = torch.stack(batch_image_emb_y)
            else:
                image_emb = {}

            # render video
            render_video = render_video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            render_latent = self.pipe.encode_video(render_video, **self.tiler_kwargs)
            render_latent= render_latent.to(self.device)
            render_mask = None
            latents = latents.to(self.device)
            
            prompt_emb["context"] = prompt_emb["context"].to(self.device)
        
            if "clip_feature" in image_emb:
                image_emb["clip_feature"] = image_emb["clip_feature"].to(self.device)
            if "y" in image_emb:
                image_emb["y"] = image_emb["y"].to(self.device)

            # --- 处理其他视频，非相机参数 ---

            # 1. 创建一个掩码来识别哪些 embedding 是全零的
            #    我们沿着除了batch维度以外的所有维度检查是否全为0
            #    embedding_batch.ndim 是张量的维度数 (例如 4 for B,C,H,W)
            embedding_batch = batch["camera_plucker_embedding"]
            dims_to_check = tuple(range(1, embedding_batch.ndim))
            is_zero_mask = torch.all(embedding_batch == 0, dim=dims_to_check)

            # 2. 找到两种数据的索引
            valid_indices = torch.where(~is_zero_mask)[0]
            zero_indices = torch.where(is_zero_mask)[0]

            outputs = []
            
            self.pipe.device = self.device
            noise = torch.randn_like(latents)
            timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
            timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            extra_input = self.pipe.prepare_extra_input(latents)
            noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
            training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

                # 3. 处理有正常 camera embedding 的数据
            if len(valid_indices) > 0:
                # 提取对应的数据
                latents = latents[valid_indices]
                render_latent = render_latent[valid_indices]
                noise = torch.randn_like(latents)
                extra_input = self.pipe.prepare_extra_input(latents)
                noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)

                camera_embedding = None

                # Compute loss
                noise_pred = self.pipe.denoising_model()(
                    noisy_latents, timestep=timestep, camera_embedding=camera_embedding, render_latent=render_latent, render_mask=render_mask, **prompt_emb, **extra_input, **image_emb,
                    use_gradient_checkpointing=self.use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
                )

                outputs.append(noise_pred)

            # 4. 处理全零 embedding 的数据 (输入None)
            if len(zero_indices) > 0:
                # 提取对应的数据
                latents = latents[zero_indices]
                render_latent = None
                render_mask = None
                camera_embedding = None
                
                # 输入给模型
                noise = torch.randn_like(latents)
                extra_input = self.pipe.prepare_extra_input(latents)
                noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)

                # Compute loss
                noise_pred = self.pipe.denoising_model()(
                    noisy_latents, timestep=timestep, camera_embedding=camera_embedding, render_latent=render_latent, render_mask=render_mask, **prompt_emb, **extra_input, **image_emb,
                    use_gradient_checkpointing=self.use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
                )
                outputs.append(noise_pred)
        
            # --- 核心逻辑结束 ---
            
            # 5. 合并结果进行后续处理（例如计算损失）
            if outputs:
                final_output = torch.cat(outputs, dim=0)
                loss = torch.nn.functional.mse_loss(final_output.float(), training_target.float())
                loss = loss * self.pipe.scheduler.training_weight(timestep)
        
            # 全 GPU 求平均（所有卡算出来的 loss 取平均）
            avg_loss = self.all_gather(loss).mean() if self.trainer.world_size > 1 else loss
            # Record log
            allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            # import swanlab
            # swanlab.log({"train_loss": loss, "avg_loss": avg_loss, "allocated": allocated, "reserved": reserved})
            self.log("train_loss", loss, prog_bar=True)
            self.log("avg_train_loss", avg_loss, prog_bar=True)
            self.log("allocated", allocated, prog_bar=True)
            self.log("reserved", reserved, prog_bar=True)
            return loss


    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        if self.train_architecture == 'adapter':
            checkpoint_dir = self.trainer.checkpoint_callback.dirpath
            print(f"Checkpoint directory: {checkpoint_dir}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            current_step = self.global_step
            print(f"Current step: {current_step}")

            checkpoint.clear()
            trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
            trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
            state_dict = self.pipe.denoising_model().state_dict() # only save trainable params
            update_state_dict = {}
            for name, param in state_dict.items():
                if name in trainable_param_names:
                    update_state_dict[name] = param
            torch.save(update_state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt"))
        else:
            checkpoint_dir = self.trainer.checkpoint_callback.dirpath
            print(f"Checkpoint directory: {checkpoint_dir}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            current_step = self.global_step
            print(f"Current step: {current_step}")
            checkpoint.clear()
            trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
            trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
            state_dict = self.pipe.denoising_model().state_dict()
            torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt")) # save all params
            checkpoint.update(state_dict)
        torch.cuda.empty_cache()
        gc.collect()
    def on_train_batch_end(self, outputs, batch, batch_idx):
        torch.cuda.empty_cache()
        gc.collect()

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--camera_encoder_path",
        type=str,
        default=None,
        help="Path of camera encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=0,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=1,
        help="Batch size of each device",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3", "fsdp"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full", "adapter"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_tensorboard",
        default=False,
        action="store_true",
        help="Whether to use TensorboardX logger.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default="local",
        help="SwanLab mode (cloud or local).",
    )
    parser.add_argument(
        "--image_sample_size",
        type=int,
        default=632
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=632
    )
    parser.add_argument(
        "--token_sample_size",
        type=int,
        default=632
    )
    parser.add_argument(
        "--dataset_list",
        type=parse_comma_separated_list,
        required=True,
        help="dataset file list."
    )
    parser.add_argument(
        "--is_i2v",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--is_camera",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--enable_bucket",
        type=bool,
        default=True
    )
    args = parser.parse_args()
    return args

def val_dataloader(self):
    dataset = FixedValDataset('data/train_eval/5_data.pt', max_items=5)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

def data_process(args):
    dataset = CameraVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, 'RealCam-Vid_MultiCam_camera_train_valid_tensor_idx.npz'),
        os.path.join(args.dataset_path, 'camera_caption_total.json'),
        steps_per_epoch=args.steps_per_epoch,
        max_num_frames=129,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=True,     # 根据你的使用情况
        is_camera=True   # 确保启用 camera 相关字段
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )

    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)
    
def safe_collate(batch):
    batch = [item for item in batch if item is not None]
    return default_collate(batch)

def parse_comma_separated_list(value):
    """解析逗号分隔的字符串为列表"""
    return [item.strip() for item in value.split(',')]

def collate_fn(examples):
        # Get token length
        sample_n_frames_bucket_interval = 4
        video_sample_frames = 81
        video_sample_size = 632
        target_token_length = video_sample_frames * video_sample_size * video_sample_size

        # Create new output
        new_examples = {}
        new_examples["video"] = []
        new_examples["text"] = []
        new_examples["path"] = []
        new_examples["video_id"] = []
        new_examples["type"] = []
        new_examples["render_video"] = []
        new_examples["render_mask"] = []
        new_examples["camera_plucker_embedding"] = []

        new_examples["first_frame"] = []
        # Get downsample ratio in image and videos
        pixel_value = examples[0]["video"]
        data_type = examples[0]["type"]
        batch_video_length = video_sample_frames

        ASPECT_RATIO = get_aspect_ratio(size=video_sample_size)
        aspect_ratio_sample_size = {key : [x for x in ASPECT_RATIO[key]] for key in ASPECT_RATIO.keys()}

        for example in examples:
            # To 0~1
            pixel_values = example["video"]
            f, c, h, w  = np.shape(pixel_value)
            closest_ratio = get_closest_ratio_key(h, w, ratios_dict=aspect_ratio_sample_size)
            closest_size = aspect_ratio_sample_size[closest_ratio]
            closest_size = [int(x / 16) * 16 for x in closest_size]
            
            if closest_size[0] / h > closest_size[1] / w:
                resize_size = closest_size[0], int(w * closest_size[0] / h)
            else:
                resize_size = int(h * closest_size[1] / w), closest_size[1]
            transform = transforms.Compose([
                transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(closest_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])
            if "first_frame" in example:
                first_frame = example["first_frame"]
                first_frame = transform(first_frame)
                new_examples["first_frame"].append(first_frame)
            new_examples["video"].append(transform(pixel_values))
            if "render_video" in example:
                new_examples["render_video"].append(example["render_video"])
                new_examples["render_mask"].append(example["render_mask"])
            else:
                new_examples["render_video"].append(transform(pixel_values))
                new_examples["render_mask"].append(transform(pixel_values))
            new_examples["text"].append(example["text"])
            new_examples["path"].append(example['path'])
            new_examples["video_id"].append(example["video_id"])
            new_examples["type"].append(example["type"])

            video_id = example["video_id"]

            if torch.any(example["camera_extrinsics"]):
                camera_plucker_embedding = get_plucker_embedding_cpu(example["camera_extrinsics"], example["camera_intrinsics"], height=closest_size[0], width=closest_size[1])
                camera_plucker_embedding = camera_plucker_embedding[:,video_id,:,:]
                new_examples["camera_plucker_embedding"].append(camera_plucker_embedding)
            else:
                zero_camera_plucker_embedding = torch.zeros([6, len(video_id), closest_size[0], closest_size[1]])
                new_examples["camera_plucker_embedding"].append(zero_camera_plucker_embedding)

            # needs the number of frames to be 4n + 1.
            batch_video_length = int(
                min(
                    batch_video_length,
                    (len(pixel_values) - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1,
                )
            )
        if batch_video_length == 0:
            batch_video_length = 1
        # Limit the number of frames to the same
        new_examples["video"] = torch.stack(
            [example[:batch_video_length] for example in new_examples["video"]])
        new_examples["render_video"] = torch.stack(
            [example[:batch_video_length] for example in new_examples["render_video"]])
        new_examples["render_mask"] = torch.stack(
            [example[:batch_video_length] for example in new_examples["render_mask"]])
        if "first_frame" in new_examples:
            new_examples["first_frame"] = torch.stack(
                [example for example in new_examples["first_frame"]])
        if 'camera_plucker_embedding' in new_examples:
            new_examples["camera_plucker_embedding"] = torch.stack(
                [example[:,:batch_video_length,:,:] for example in new_examples["camera_plucker_embedding"]]
            )
        return new_examples



class CameraDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        # Using save_hyperparameters allows Lightning to log these settings automatically
        self.save_hyperparameters(args)
        # We can also store args directly for easier access
        self.args = args

    def setup(self, stage: str):
        # This hook is called on each GPU process. It's the perfect place to
        # instantiate the dataset.
        if stage == 'fit' or stage is None:
            self.dataset = CameraVideoDataset(
                self.args.dataset_path,
                self.args.dataset_list,
                steps_per_epoch=self.args.steps_per_epoch,
                max_num_frames=121,
                num_frames=self.args.num_frames,
                is_i2v=self.args.is_i2v,
                is_camera=self.args.is_camera,
            )

    def train_dataloader(self):
        # This hook is called by the Trainer after the distributed environment
        # has been initialized. This is the correct place to create samplers and dataloaders.
        
        if self.args.enable_bucket:
            # --- Bucket-enabled Distributed Logic ---
            ASPECT_RATIO = get_aspect_ratio(size=self.args.video_sample_size)
            aspect_ratio_sample_size = {key: [x for x in ASPECT_RATIO[key]] for key in ASPECT_RATIO.keys()}

            # KEY CHANGE 1: Replace RandomSampler with DistributedSampler.
            # It handles data partitioning across GPUs. shuffle=True ensures randomization each epoch.
            distributed_sampler = DistributedSampler(self.dataset, shuffle=True)

            batch_sampler = AspectRatioImageVideoSampler(
                sampler=distributed_sampler,  # Use the distributed sampler here
                dataset=self.dataset,
                batch_size=self.args.per_device_batch_size,
                drop_last=True,
                aspect_ratios_dict=aspect_ratio_sample_size,
                video_duration_bins=None,
            )

            dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                num_workers=self.args.dataloader_num_workers,
            )
        else:
            # --- Standard Distributed Logic ---
            # KEY CHANGE 2: The simple case also needs a DistributedSampler.
            # Using only `shuffle=True` would lead to all GPUs seeing the same data.
            sampler = DistributedSampler(self.dataset, shuffle=True)
            
            dataloader = torch.utils.data.DataLoader(
                self.dataset,
                sampler=sampler, # Provide the sampler
                shuffle=False,   # IMPORTANT: shuffle must be False when a sampler is provided
                batch_size=self.args.per_device_batch_size,
                num_workers=self.args.dataloader_num_workers,
            )
            
        return dataloader


def train(args):
    torch.set_float32_matmul_precision('high') # or 'medium'
    
    datamodule = CameraDataModule(args)

    model = LightningModelForTrain(
        dit_path=args.dit_path,
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        resume_ckpt_path=args.camera_encoder_path,
        vae_path=args.vae_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
    )
    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan", 
            name="wan",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    elif args.use_tensorboard:
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(args.output_path, name='tensorboardlog', logdir=os.path.join(args.output_path, "tensorboardlog"))
    else:
        logger = None
    if args.training_strategy == "fsdp":
        args.training_strategy = FSDPStrategy(
            sharding_strategy="SHARD_GRAD_OP"
        )
    elif args.training_strategy == "auto":
        args.training_strategy = DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1, every_n_train_steps=200)],
        logger=logger,
        log_every_n_steps=50,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    pl.seed_everything(3407)
    args = parse_args()
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)

