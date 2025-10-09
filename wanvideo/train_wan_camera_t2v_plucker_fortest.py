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
from diffsynth.data.camera_utils import get_camera_sparse_embedding, get_plucker_embedding, get_plucker_embedding_cpu, get_plucker_embedding_cpu_batched
from diffsynth.data.camera_video import CameraVideoDataset, get_dataloader
from diffsynth.data.video import save_video
from diffsynth.models.wan_video_camera_adapter import SimpleAdapter
from diffsynth.pipelines.wan_camera_video import WanVideoCameraPipeline
import third_party.depth_pro.depth_pro as depth_pro
from diffsynth.data.pointcloud import point_rendering_train_stage
# from src.pointcloud import point_rendering_train_stage
from pytorch3d.renderer import PointsRasterizationSettings
import einops
from torchvision.transforms import ToTensor, ToPILImage
from torchvision import transforms
from diffusers.utils import export_to_video

from torch.utils.data.distributed import DistributedSampler
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.profilers import SimpleProfiler
import torch.nn.functional as F
# import time

# from vggt.test import generate_camera_params
# from vggt.vggt.models.vggt import VGGT

import traceback
import sys

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
        # camera_emb = batch["camera_plucker_embedding"][0]

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
            # new_path = os.path.join('data/temp_data', path.split('/')[-1])
            torch.save(data, path + ".tensors.pth")



class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, npz_path, steps_per_epoch):
        metadata = np.load(npz_path, allow_pickle=True)["arr_0"]  # 加载npz，拿arr_0
        self.path = [os.path.join('/mnt/data/camera_datasets/KwaiVGI/MultiCamVideo-Dataset', entry["video_path"]) if 'MultiCamVideo-Dataset' in entry["video_path"] else os.path.join(base_path, entry["video_path"]) for entry in metadata]
        print(len(self.path), "videos in metadata.") # offload check the exist of all tensors
        # self.path = ['data/temp_data/' + i.split('/')[-1] for i in self.path]
        self.path = [i + ".tensors.pth" for i in self.path]
        self.camera_extrinsics = [entry["camera_extrinsics"] for entry in metadata]
        self.camera_intrinsics = [np.repeat(entry["camera_intrinsics"][np.newaxis, :], len(entry["camera_extrinsics"]), axis=0) for entry in metadata]
        print(len(self.path), "tensors cached in metadata.")
        assert len(self.path) > 0
        
        self.empty_text = torch.load('data/empty_text.pth', weights_only=True, map_location="cpu")
        self.steps_per_epoch = steps_per_epoch


    def __getitem__(self, index):
        # data_id = torch.randint(0, len(self.path), (1,))[0]
        # data_id = (data_id + index) % len(self.path) # For fixed seed.
        data_id = index % len(self.path)
        path = self.path[data_id]
        data = torch.load(path, map_location="cpu")
        if random.random() < 0.1:
            # data['prompt_emb']["context"] = torch.zeros_like(data['prompt_emb']["context"])
            data['prompt_emb'] = self.empty_text["prompt_emb"]
        # if random.random() < 0.1:
        #     data['camera_emb'] = torch.zeros_like(data['camera_emb'])
        if 'video_id' in data:
            video_id = data["video_id"]
        else:
            video_id = np.linspace(0, len(self.camera_extrinsics[data_id]) - 1, 81).astype(int)
        height = data["latents"].shape[2] * 8
        width = data["latents"].shape[3] * 8

        data["camera_extrinsics"], data["camera_intrinsics"] = get_camera_sparse_embedding(self.camera_extrinsics[data_id][video_id], self.camera_intrinsics[data_id][video_id], height, width)

        # data["camera_extrinsics"] = self.camera_extrinsics[data_id][video_id]
        # data["camera_intrinsics"] = self.camera_intrinsics[data_id][video_id]

        return data
    

    def __len__(self):
        if self.steps_per_epoch == 0:
            return len(self.path)
        else:
            return self.steps_per_epoch

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
        text_encoder_path, vae_path, image_encoder_path=None,
        id_encoder_path=None, # ===== 新增参数 =====
        resume_ckpt_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16),
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
        if id_encoder_path is not None: # ===== 新增代码 =====
            model_path.append(id_encoder_path)
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

        # ===== 新增代码：为ID图像定义变换 =====
        self.id_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # ==================================

        cfg = OmegaConf.load("/mnt/data/ssd/user_workspace/duanke/unicontrol/diffsynth/configs/config_yqf.json")
        self.pipe.dit.controlnet_cfg = cfg.controlnet_cfg
        self.pipe.dit.build_controlnet()
        
        
        if resume_ckpt_path is not None:
            print(f"Loading resume ckpt path: {resume_ckpt_path}")
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")
            self.pipe.dit.load_state_dict(state_dict, strict=False)
            # self.pipe.to("cuda")
            # self.pipe.to(dtype=torch.bfloat16)

        # self.freeze_parameters()
        

        # validation setting
        ASPECT_RATIO = get_aspect_ratio(size=632)
        self.aspect_ratio_sample_size = {key : [x for x in ASPECT_RATIO[key]] for key in ASPECT_RATIO.keys()}
        
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
            self.setup_for_controlnet_training()
            # if train_architecture == "adapter":
            #     for name, module in self.pipe.denoising_model().named_modules():
            #         # if any(keyword in name for keyword in ["cam_adapter"]):
            #         if any(keyword in name for keyword in ["controlnet_patch_embedding", "controlnet_mask_embedding", "controlnet", "controlnet_freqs"]):
            #             print(f"Trainable: {name}")
            #             for param in module.parameters():
            #                 param.requires_grad = True
            #     trainable_params = 0
            #     seen_params = set()
            #     for name, module in self.pipe.denoising_model().named_modules():
            #         for param in module.parameters():
            #             if param.requires_grad and param not in seen_params:
            #                 trainable_params += param.numel()
            #                 seen_params.add(param)
            #     print(f"Total number of trainable parameters: {trainable_params}")
            dit_norm_training = False
            controlnet_norm_training = True
            for name, module in self.pipe.denoising_model().named_modules():
                # 检查是否是 BatchNorm 或 LayerNorm 的实例
                if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm3d)):
                    print(f"模块: {name}, 类型: {type(module).__name__}, is in training mode: {module.training}")
                    if "controlnet" not in name:
                        if module.training:
                            print("\n结论: denoising_model 中的 Norm 层处于训练模式。")
                            dit_norm_training = True
                    if "controlnet" in name:
                        if not module.training:
                            print("\n结论: denoising_model 中的 Norm 层处于训练模式。")
                            controlnet_norm_training = False

            if dit_norm_training or not controlnet_norm_training:
                print("\n当前的train/eval模式设置存在问题，请检查！")
            else:
                print("\n✅当前的train/eval模式设置正确！")

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
            # 新增：检查train/eval设置是否正确
            dit_norm_training = False
            controlnet_norm_training = True
            for name, module in self.pipe.denoising_model().named_modules():
                # 检查是否是 BatchNorm 或 LayerNorm 的实例
                if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm3d)):
                    print(f"模块: {name}, 类型: {type(module).__name__}, is in training mode: {module.training}")
                    if "controlnet" not in name:
                        if module.training:
                            print("\n结论: denoising_model 中的 Norm 层处于训练模式。")
                            dit_norm_training = True
                    if "controlnet" in name:
                        if not module.training:
                            print("\n结论: denoising_model 中的 Norm 层处于训练模式。")
                            controlnet_norm_training = False

            if dit_norm_training or not controlnet_norm_training:
                print("\n当前的train/eval模式设置存在问题，请检查！")
            else:
                print("\n✅当前的train/eval模式设置正确！")
        elif train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
            # ===== 新增代码：确保ID Encoder可训练 =====
            if self.pipe.id_encoder is not None:
                self.pipe.id_encoder.requires_grad_(True)
                print("ID Encoder parameters are set to trainable.")
            # ======================================
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
            # ===== 新增代码：确保ID Encoder可训练 =====
            if self.pipe.id_encoder is not None:
                self.pipe.id_encoder.requires_grad_(True)
                print("ID Encoder parameters are set to trainable.")
            # ======================================

        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
    

        # 在 trainer.fit() 之前
        print("--- Checking Zero Initialization ---")
        for i, layer in enumerate(self.pipe.dit.controlnet.proj_out):
            weight_sum = layer.weight.abs().sum()
            bias_sum = layer.bias.abs().sum() if layer.bias is not None else torch.tensor(0.0)
            print(f"Layer {i} weight sum: {weight_sum}")
            print(f"Layer {i} bias sum: {bias_sum}")
            if weight_sum > 1e-6 or bias_sum > 1e-6:
                print(f"ERROR: Layer {i} is NOT zero-initialized!")
        print("--------------------------------")

        # 添加深度估计
        # model, transform = depth_pro.create_model_and_transforms(device=self.device)
        # model.requires_grad_(False)
        # self._depth_model = model.eval()
        # self._depth_transform = transform
        # model, transform = depth_pro.create_model_and_transforms(device=self.device)
        # model.requires_grad_(False)
        # self._depth_model = model.eval()
        # self._depth_transform = transform
        
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def setup_for_controlnet_training(self):
        """
        配置 pipeline，冻结主模型并只准备训练 ControlNet 相关模块。
        这个函数应该在模型加载后、训练开始前调用一次。
        """
        print("--- 配置模型用于 ControlNet 训练 ---")
        
        # 步骤 1: 冻结整个 pipeline 的所有参数
        self.pipe.requires_grad_(False)
        
        # 步骤 2: 将整个 pipeline 设置为评估模式。
        # 这会确保所有 BatchNorm 和 Dropout 层都处于非活动、确定性的状态。
        self.pipe.eval()

        if hasattr(self.pipe.denoising_model(), 'gradient_checkpointing_enable'):
            self.pipe.denoising_model().gradient_checkpointing_enable()
            print("✅ 手动为 denoising_model 开启了激活检查点 (Activation Checkpointing)。")
        
        # 步骤 3: 定义需要被训练的 ControlNet 相关模块列表,直接引用模块对象。
        trainable_modules = [
            self.pipe.dit.controlnet,
            self.pipe.dit.controlnet_patch_embedding,
            self.pipe.dit.controlnet_mask_embedding,
            self.pipe.dit.controlnet_rope
        ]
        # ===== 新增代码：将ID Encoder加入可训练模块 =====
        if self.pipe.id_encoder is not None:
            trainable_modules.append(self.pipe.id_encoder)
        # ============================================
        
        # 步骤 4: 遍历列表，解冻这些模块的参数并设置为训练模式
        print("正在解冻并激活以下模块:")
        for module in trainable_modules:
            # 解冻参数
            module.requires_grad_(True)
            # 设置为训练模式 (这对于 ControlNet 内部可能存在的 Dropout/BatchNorm 是必要的)
            module.train()
            print(f"  ✅ {module.__class__.__name__}")

        # 步骤 5: (验证) 确认 denoising_model 本身仍然在评估模式
        if not self.pipe.denoising_model().training:
            print("\n确认: 核心 denoising_model 保持在评估模式 (eval mode)。")
        else:
            print("\n警告: 核心 denoising_model 意外处于训练模式！")
            
        # (可选) 再次打印可训练参数进行最终检查
        print("\n--- 当前可训练参数列表 ---")
        trainable_params = 0
        seen_params = set()
        for name, param in self.pipe.named_parameters():
            if param.requires_grad and param not in seen_params: 
                trainable_params += param.numel()
                seen_params.add(param)
                print(f"  ✅ {name}")
        print(f"Total number of trainable parameters: {trainable_params}")
        
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
    # def on_validation_start(self):
    #     # 此钩子在验证开始时被调用
    #     if self.trainer.is_global_zero:
    #         print("\n--- Checking validation dataloader distribution ---")
        
    #     # 获取当前进程的rank
    #     rank = self.global_rank
    #     # 获取当前进程的dataloader中的batch数量
    #     num_batches = len(self.trainer.val_dataloaders)
        
    #     print(f"[GPU {rank}] Number of validation batches for this process: {num_batches}")
    

    # 在您的 LightningModule 中
    # def on_validation_epoch_end(self):
    #     rank = self.global_rank
    #     print(f"--- [GPU {rank}] --- Reached on_validation_epoch_end hook.")

    #     if self.trainer.is_global_zero:
    #         # # # # # # # # # # # # # # # # # # # #
    #         # 核心修改：注释掉所有实际工作，只保留一个长时间的休眠
    #         # # # # # # # # # # # # # # # # # # # #
            
    #         print(f"--- [GPU {rank}] --- Simulating long task with time.sleep(30)...")
    #         # self.pipe.eval()
    #         # ... (所有获取数据、推理、保存视频的代码全部注释掉) ...
    #         time.sleep(30) # 模拟30秒的耗时
    #         print(f"--- [GPU {rank}] --- Sleep finished.")

    #     print(f"--- [GPU {rank}] --- Reaching barrier...")
    #     if dist.is_initialized():
    #         dist.barrier()
    #     print(f"--- [GPU {rank}] --- Passed barrier. Exiting hook.")

    # def on_validation_epoch_end(self):
    #     """
    #     在每个验证周期结束后，运行此函数来生成一个可视化视频样本。
    #     """
    #     # 仅在主进程中执行，避免在多GPU时重复生成
    #     if not self.trainer.is_global_zero:
    #         return

    #     print(f"\n--- Running validation video generation for epoch {self.current_epoch} ---")

    #     # 1. 从验证数据集中取固定的数据
    #     # trainer.datamodule 属性可以访问到数据模块
    #     try:
    #         val_dataset = self.trainer.datamodule.val_dataset
    #         batch_sample = val_dataset[0]
    #     except Exception as e:
    #         print("Validation dataloader is empty, skipping visualization.")
    #         if torch.distributed.is_initialized():
    #             torch.distributed.barrier()
    #         return

    #     # 1.1 处理数据为batch形式
    #     val_batch = {}
    #     for key, tensor in batch_sample.items():
    #         if isinstance(tensor, torch.Tensor):
    #             val_batch[key] = tensor.unsqueeze(0).to(self.device)
    #         else:
    #             val_batch[key] = [tensor]


    #     # 2. 准备模型和数据
    #     self.pipe.to(self.device, dtype=self.pipe.torch_dtype) # 确保模型在正确的设备上

    #     self.pipe.device = self.device
    #     self.pipe.eval() # 设置为评估模式
    #     torch.set_grad_enabled(False)

    #     # 3. 复制您的数据预处理逻辑
    #     # (这部分与您之前的脚本几乎完全相同，只是`pipe`现在是`self.pipe`)
    #     first_frame = val_batch["first_frame"]
    #     text = val_batch["text"][0]
    #     b, c, h, w = np.shape(first_frame)
    #     closest_ratio = get_closest_ratio_key(h, w, ratios_dict=self.aspect_ratio_sample_size)
    #     closest_size = [int(x / 16) * 16 for x in self.aspect_ratio_sample_size[closest_ratio]]
        
    #     if closest_size[0] / h > closest_size[1] / w:
    #         resize_size = closest_size[0], int(w * closest_size[0] / h)
    #     else:
    #         resize_size = int(h * closest_size[1] / w), closest_size[1]
            
    #     transform = transforms.Compose([
    #         transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
    #         transforms.CenterCrop(closest_size),
    #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    #     ])

    #     first_frame = transform(first_frame).to(self.device)
    #     render_video = einops.rearrange(val_batch["render_video"], "B T C H W -> B C T H W").to(self.device, dtype=self.pipe.torch_dtype)
    #     render_mask = einops.rearrange(val_batch["render_mask"], "B T C H W -> B C T H W").to(self.device)

    #     render_latent = self.pipe.encode_video(render_video)
    #     render_mask = (render_mask[:, 0:1, :, :, :] >= 0.5).float()

    #     video_id = val_batch["video_id"][0]
    #     camera_plucker_embedding = get_plucker_embedding(val_batch["camera_extrinsics"], val_batch["camera_intrinsics"], height=closest_size[0], width=closest_size[1])
    #     camera_plucker_embedding = camera_plucker_embedding[:,:,video_id,:,:].to(self.device)

    #     # 4. 执行推理 (使用 self.pipe)
    #     print("Running inference on the current model weights...")
    #     video = self.pipe(
    #         prompt=text, negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走", # 您的负向提示词
    #         input_image=first_frame,
    #         camera_pose=camera_plucker_embedding,
    #         render_latent=render_latent,
    #         render_mask=render_mask,
    #         num_frames=len(video_id),
    #         seed=42, tiled=True,
    #     )

    #     # 5. 保存结果或记录到TensorBoard
    #     output_dir = os.path.join(self.trainer.log_dir, "validation_videos")
    #     os.makedirs(output_dir, exist_ok=True)
    #     output_gt_path = os.path.join(output_dir, f"epoch_{self.current_epoch:04d}_step_{self.global_step}_gt.mp4")
    #     original_path = val_batch["path"][0]
    #     os.system(f"cp {original_path} {output_gt_path}")
    #     output_path = os.path.join(output_dir, f"epoch_{self.current_epoch:04d}_step_{self.global_step}.mp4")
    #     save_video(video, output_path, fps=16, quality=5)
    #     print(f"Validation video saved to: {output_path}")

    #     # 如果您想记录到TensorBoard
    #     # if hasattr(self.logger.experiment, 'add_video'):
    #     #     # 将视频Tensor的范围从[-1, 1]转换到[0, 255]的uint8
    #     #     video = (video.clamp(-1, 1) + 1) / 2
    #     #     video = (video * 255).to(torch.uint8)
    #     #     self.logger.experiment.add_video(
    #     #         'validation_sample', 
    #     #         video, 
    #     #         global_step=self.global_step,
    #     #         fps=16
    #     #     )

    #     # 恢复模型到训练状态
    #     self.pipe.train()
    #     torch.set_grad_enabled(True)

    #     # 同步
    #     if torch.distributed.is_initialized():
    #         torch.distributed.barrier()

    # def validation_step(self, batch, batch_idx):
    #     print("start validation")
    #     pass
    #     # ... 验证逻辑 ...
    #     text, video, path = batch["text"], batch["video"], batch["path"]
    #     video = rearrange(video, "B T C H W -> B C T H W")
    #     render_video = batch["render_video"]
    #     render_mask = None
    #     camera_embedding = batch["camera_plucker_embedding"][0].to(self.device)

    #     self.pipe.device = self.device
    #     if video is not None:
    #         # prompt
    #         prompt_emb = self.pipe.encode_prompt(text)
    #         # video
    #         video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
    #         latents = self.pipe.encode_video(video, **self.tiler_kwargs)
    #         # render video
    #         render_video = render_video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
    #         render_latent = self.pipe.encode_video(render_video, **self.tiler_kwargs)
    #         # image
    #         if "first_frame" in batch:
    #             batch_image_emb_clip = []
    #             batch_image_emb_y = []
    #             for i in range(batch["first_frame"].shape[0]):
    #                 first_frame = batch["first_frame"][i]
    #                 _, _, num_frames, height, width = video.shape
    #                 image_emb = self.pipe.encode_image(first_frame, None, num_frames, height, width)
    #                 batch_image_emb_clip.append(image_emb["clip_feature"][0])
    #                 batch_image_emb_y.append(image_emb["y"][0])
    #             image_emb = {}                   
    #             image_emb["clip_feature"] = torch.stack(batch_image_emb_clip)
    #             image_emb["y"] = torch.stack(batch_image_emb_y)
    #         else:
    #             image_emb = {}
    #     # generate camera embedding
    #     camera_embedding = batch["camera_plucker_embedding"][0].to(self.device)
    #     print(camera_embedding.shape)

    #     # generate video
    #     self.pipe.device = self.device
    #     video = self.pipe.dit_generation(
    #         prompt_emb_posi=prompt_emb,
    #         render_latent=render_latent,
    #         render_mask=render_mask,
    #         cam_emb=camera_embedding,
    #         image_emb=image_emb,
    #         num_inference_steps=50,
    #         height=480, width=832,
    #         seed=0, tiled=True
    #     )
    #     video_path = f"data/train_eval/temp_videos/video_eval_{batch_idx}_{self.global_step}.mp4"
    #     save_video(video, video_path, fps=16, quality=5)
    #     # estimate extrinsic
    #     vggt_model = VGGT.from_pretrained("/mnt/data/hdd/user_workspace/yuqifan/cameractrl/VGGT-1B").to("cuda")
    #     vggt_model.eval()
    #     with torch.no_grad():
    #         estimated_extrinsic, estimated_intrinsic = generate_camera_params(vggt_model, video_path, 'data/train_eval/temp_images', self.device) # (81, 3, 4) (81, 3, 3)
    #     del vggt_model
    #     torch.cuda.empty_cache()
    #     # calculate metric
    #     gt_extrinsic = camera_extrinsic[video_id].cpu().numpy()[:,:3,:]
    #     trans_err, rot_err, cam_mc = calculate_pose_errors(gt_extrinsic, estimated_extrinsic)
    #     print(trans_err, rot_err, cam_mc)
    #     self.log("trans_err", trans_err, on_epoch=True, prog_bar=True)
    #     self.log("rot_err", rot_err, on_epoch=True, prog_bar=True)
    #     self.log("cam_mc", cam_mc, on_epoch=True, prog_bar=True)
    #     # self.log("val_loss", val_loss, on_epoch=True, prog_bar=True) # 这会为当前的这个验证run记录一个汇总值
    #     return cam_mc
    def training_step(self, batch, batch_idx):
        # for key, value in batch.items():
        #     if isinstance(value, torch.Tensor):
        #         batch[key] = value.to(self.device)
        #  # ========================== 推荐的最终修复方案 ==========================
        # # “预热”日志系统 (Pre-warm the logging system)
        # # 在进入模型计算（和梯度检查点）之前，让 Lightning 提前创建好所有日志指标的状态。
        # # 我们只在训练的第一个 step 执行一次即可。
        # if self.global_step == 0:
        #     # 创建一个在正确设备上的零值张量作为“假”日志值
        #     dummy_value = torch.tensor(0.0, device=self.device)
            
        #     # 将您所有会用到的 self.log 指标都在这里“预热”一遍
        #     self.log("train_loss", dummy_value, prog_bar=True)
        #     self.log("avg_train_loss", dummy_value, prog_bar=True)
        #     self.log("allocated", dummy_value, prog_bar=True)
        #     self.log("reserved", dummy_value, prog_bar=True)
        # ====================================================================
        # Data
        
        text, video, path = batch["text"], batch["video"], batch["path"]
        video = rearrange(video, "B T C H W -> B C T H W")
        render_video = batch["render_video"]
        render_mask = batch["render_mask"]
        render_video = rearrange(render_video, "B T C H W -> B C T H W")
        render_mask = rearrange(render_mask, "B T C H W -> B C T H W")

        self.pipe.device = self.device
        # with sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False):
        if video is not None:
            # prompt
            prompt_emb = self.pipe.encode_prompt(text)
            # video
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)
                # image
                # if "first_frame" in batch:
                #     batch_image_emb_clip = []
                #     batch_image_emb_y = []
                #     for i in range(batch["first_frame"].shape[0]):
                #         first_frame = Image.fromarray(batch["first_frame"][i].cpu().numpy())
                #         _, _, num_frames, height, width = video.shape

                        # --------------------渲染点云视频与mask视频-----------------------------------------------------
                        # batch_size 8: 123G 2m42s/iter depth:1.44s/frame render:7.39s/video
                        # with torch.no_grad():
                        #     depth_image = self._depth_transform(first_frame).to(self.device)
                        #     prediction = self._depth_model.infer(depth_image, f_px=None)
                        #     depth = prediction["depth"]
                        #     del depth_image, prediction
                        # with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                        #     control_imgs, render_masks = point_rendering_train_stage(K=batch["camera_intrinsics"][i].float(),
                        #                                                 w2cs=batch["camera_extrinsics"][i].float(),
                        #                                                 depth=depth.float(),
                        #                                                 images=v2.ToTensor()(first_frame).float() * 2 - 1,
                        #                                                 raster_settings=PointsRasterizationSettings(image_size=(height, width),
                        #                                                                                             radius=0.008,
                        #                                                                                             points_per_pixel=8),
                        #                                                 device=self.device,
                        #                                                 background_color=[0, 0, 0],
                        #                                                 sobel_threshold=0.35,
                        #                                                 sam_mask=None)
                        #     del depth
                        # control_imgs = einops.rearrange(control_imgs, "(b f) c h w -> b c f h w", f=81)
                        # render_masks = einops.rearrange(render_masks, "(b f) c h w -> b c f h w", f=81)
                        # render_video = []
                        # mask_video = []
                        # control_imgs = control_imgs.to(torch.float32).cpu()
                        # render_masks = render_masks.cpu()
                        # for i in range(num_frames):
                        #     img = ToPILImage()((control_imgs[0][:, i] + 1) / 2)
                        #     render_video.append(img)
                        #     mask = ToPILImage()(render_masks[0][:, i])
                        #     mask_video.append(mask)

                        # del control_imgs, render_masks

                        # render_mask = torch.stack([ToTensor()(frame) for frame in mask_video], dim=0)[:, 0:1][None]  # [f,1,h,w]
                        # render_mask = einops.rearrange(render_mask, "b f c h w -> b c f h w")
                        # render_mask[render_mask < 0.5] = 0
                        # render_mask[render_mask >= 0.5] = 1

                        # render_videos.append(render_video)
                        # mask_videos.append(render_mask)

                        # export_to_video(render_video, f"/mnt/data/ssd/user_workspace/duanke/unicontrol/render.mp4", fps=16)
                        # export_to_video(mask_video, f"/mnt/data/ssd/user_workspace/duanke/unicontrol/render_mask.mp4", fps=16)

                        #---------------------渲染完毕，得到render_video和mask_video---------------------------------------------------------------
                #         image_emb = self.pipe.encode_image(first_frame, None, num_frames, height, width)
                #         batch_image_emb_clip.append(image_emb["clip_feature"][0])
                #         batch_image_emb_y.append(image_emb["y"][0])
                #     image_emb = {}
                    
                #     image_emb["clip_feature"] = torch.stack(batch_image_emb_clip)
                #     image_emb["y"] = torch.stack(batch_image_emb_y)
                # else:
                #     image_emb = {}

                # image
            # if "first_frame" in batch:
            #     batch_image_emb_clip = []
            #     batch_image_emb_y = []
            #     for i in range(batch["first_frame"].shape[0]):
            #         first_frame = batch["first_frame"][i]
            #         _, _, num_frames, height, width = video.shape
            #         image_emb = self.pipe.encode_image(first_frame, None, num_frames, height, width)
            #         batch_image_emb_clip.append(image_emb["clip_feature"][0])
            #         batch_image_emb_y.append(image_emb["y"][0])
            #     image_emb = {}                   
            #     image_emb["clip_feature"] = torch.stack(batch_image_emb_clip)
            #     image_emb["y"] = torch.stack(batch_image_emb_y)
            # else:
            #     image_emb = {}

            image_emb = {}
            if "first_frame" in batch:
                first_frames_batch = batch["first_frame"] # 获取整个批次的图像 (B, C, H, W)
                _, _, num_frames, height, width = video.shape

                # 直接将整个批次传入函数，进行一次性处理
                image_emb = self.pipe.encode_image_batch(first_frames_batch, None, num_frames, height, width)
            else:
                image_emb = {}

            # ===== 修改开始: 处理多张ID图 =====
            id_embedding = None
            if "id_image" in batch and self.pipe.id_encoder is not None:
                # batch["id_image"] 的形状现在是 (B, max_k, C, H, W)
                id_images_batch = batch["id_image"].to(self.device, dtype=self.pipe.torch_dtype)
                B, max_k, C, H, W = id_images_batch.shape

                # 1. 预处理图像
                id_images_batch = self.id_transform(id_images_batch.view(B * max_k, C, H, W))
                
                # 2. 通过 ID Encoder 编码
                # 输入形状: (B * max_k, C, 224, 224)
                # 输出形状: (B * max_k, num_patches, embed_dim)
                id_embedding_flat = self.pipe.id_encoder(id_images_batch)
                
                # 3. 恢复批次结构并拼接特征
                _, num_patches, embed_dim = id_embedding_flat.shape
                # 恢复形状: (B, max_k, num_patches, embed_dim)
                id_embedding = id_embedding_flat.view(B, max_k, num_patches, embed_dim)
                # 拼接序列: (B, max_k * num_patches, embed_dim)
                id_embedding = id_embedding.view(B, max_k * num_patches, embed_dim)
            # ===== 修改结束 =====

            # render video
            render_video = render_video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            render_latent = self.pipe.encode_video(render_video, **self.tiler_kwargs)
            render_mask = render_mask[:, 0:1, :, :, :]
            render_mask[render_mask < 0.5] = 0
            render_mask[render_mask >= 0.5] = 1
            # render_latent, render_mask = self.pipe.prepare_camera_controlnet_kwargs(render_video, mask_video, **self.tiler_kwargs)
            render_latent= render_latent.to(self.device)
            render_mask = render_mask.to(self.device)
            render_mask = render_mask.to(self.device)
            # render_latent = None
            # render_mask = None
            # render_mask = None
            latents = latents.to(self.device)
            
            prompt_emb["context"] = prompt_emb["context"].to(self.device)
        
            if "clip_feature" in image_emb:
                image_emb["clip_feature"] = image_emb["clip_feature"].to(self.device)
            if "y" in image_emb:
                image_emb["y"] = image_emb["y"].to(self.device)

            # camera_embedding = batch["camera_emb"].to(self.device)

            # --- 处理其他视频，非相机参数 ---

            # 1. 创建一个掩码来识别哪些 embedding 是全零的
            #    我们沿着除了batch维度以外的所有维度检查是否全为0
            #    embedding_batch.ndim 是张量的维度数 (例如 4 for B,C,H,W)
            embedding_batch = batch["camera_plucker_embedding"]
            dims_to_check = tuple(range(1, embedding_batch.ndim))
            is_zero_mask = torch.all(embedding_batch == 0, dim=dims_to_check)
            # is_zero_mask 的形状会是 [True, False, True, True, False, ...]，长度为8

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
                render_mask = render_mask[valid_indices]
                
                camera_embedding = embedding_batch[valid_indices]
                camera_embedding = embedding_batch[valid_indices]
                # 输入给模型
                # Loss

                noise = torch.randn_like(latents)
                extra_input = self.pipe.prepare_extra_input(latents)
                noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)

                # camera_embedding = None

                # Compute loss
                # noise_pred = self.pipe.denoising_model()(
                #     noisy_latents, timestep=timestep, camera_embedding=camera_embedding, render_latent=render_latent, render_mask=render_mask, **prompt_emb, **extra_input, **image_emb,
                #     use_gradient_checkpointing=self.use_gradient_checkpointing,
                #     use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
                # )
                noise_pred = self.pipe.denoising_model()(
                    noisy_latents[valid_indices], timestep=timestep, camera_embedding=embedding_batch[valid_indices], 
                    render_latent=render_latent[valid_indices], render_mask=render_mask[valid_indices],
                    id_embedding=id_embedding, # <-- 在这里传入
                    **prompt_emb, **extra_input, **image_emb,
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
                # noise_pred = self.pipe.denoising_model()(
                #     noisy_latents, timestep=timestep, camera_embedding=camera_embedding, render_latent=render_latent, render_mask=render_mask, **prompt_emb, **extra_input, **image_emb,
                #     use_gradient_checkpointing=self.use_gradient_checkpointing,
                #     use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
                # )
                noise_pred = self.pipe.denoising_model()(
                    noisy_latents[zero_indices], timestep=timestep, camera_embedding=None, 
                    render_latent=None, render_mask=None,
                    id_embedding=id_embedding[zero_indices] if id_embedding is not None else None, # 传递ID embedding
                    **prompt_emb, **extra_input, **image_emb,
                    use_gradient_checkpointing=self.use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
                )
                outputs.append(noise_pred)
        
            # --- 核心逻辑结束 ---




                # if torch.all(batch["camera_extrinsics"]==0):
                #     camera_embedding = None
                # else:
                #     height = latents.shape[3] * 8
                #     width = latents.shape[4] * 8

                #     camera_embedding = get_plucker_embedding(batch["camera_extrinsics"], batch["camera_intrinsics"], height, width, device=self.device)
                    # control_camera_latents = torch.concat(
                    #     [
                    #         torch.repeat_interleave(camera_embedding[:, :, 0:1], repeats=4, dim=2),
                    #         camera_embedding[:, :, 1:]
                    #     ], dim=2
                    # ).transpose(1, 2)
                    # b, f, c, h, w = control_camera_latents.shape
                    # control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
                    # control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
                    # camera_embedding = control_camera_latents.to(device=self.device, dtype=self.pipe.torch_dtype)
            # only for batch size 1
            # latents = latents.unsqueeze(0)
            # prompt_emb["context"] = prompt_emb["context"].unsqueeze(0)
            # if "clip_feature" in image_emb:
            #     image_emb["clip_feature"] = image_emb["clip_feature"].unsqueeze(0)
            # if "y" in image_emb:
            #     image_emb["y"] = image_emb["y"].unsqueeze(0)

            # Loss
            # self.pipe.device = self.device
            # noise = torch.randn_like(latents)
            # timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
            # timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            # extra_input = self.pipe.prepare_extra_input(latents)
            # noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
            # training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

            # Compute loss
            # noise_pred = self.pipe.denoising_model()(
            #     noisy_latents, timestep=timestep, **prompt_emb, **extra_input, **image_emb, plucker_embedding=camera_plucker_emb,
            #     use_gradient_checkpointing=self.use_gradient_checkpointing,
            #     use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
            # )
            # noise_pred = self.pipe.denoising_model()(
            #     noisy_latents, timestep=timestep, camera_embedding=camera_embedding, render_latent=render_latent, render_mask=render_mask, **prompt_emb, **extra_input, **image_emb,
            #     use_gradient_checkpointing=self.use_gradient_checkpointing,
            #     use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
            # )
            
            
            # 5. 合并结果进行后续处理（例如计算损失）
            if outputs:
                final_output = torch.cat(outputs, dim=0)
                loss = torch.nn.functional.mse_loss(final_output.float(), training_target.float())
                loss = loss * self.pipe.scheduler.training_weight(timestep)
                # ... 在这里计算损失、反向传播等
                # 1. 创建一个值为0的虚拟损失项
                dummy_sum = 0
                
                # 2. 遍历所有正在被优化的参数
                #    我们直接从优化器中获取参数列表，确保和DDP管理的是同一组
                for name, param in self.pipe.named_parameters():
                    if param.requires_grad:
                        # 3. 将每个参数的和乘以0，加到虚拟损失上
                        #    这个操作的梯度是存在的，从而将参数“挂载”到计算图上
                        #    但它对最终loss的数值贡献为0，不影响训练方向
                        dummy_sum = dummy_sum + param.sum() * 0.0

                # 4. 将虚拟损失加到真实的loss上
                loss = loss + dummy_sum
                # print(f"Batch {i}: Final output shape: {final_output.shape}\n")
            # else:
                # print(f"Batch {i}: Empty batch, skipping.\n")
        
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
        # trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        # ===== 修改：将 id_encoder 的参数也加入优化器 =====
        trainable_modules = list(filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters()))
        if self.pipe.id_encoder is not None:
            trainable_modules += list(filter(lambda p: p.requires_grad, self.pipe.id_encoder.parameters()))
        # ==================================================
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        # --- 加入以下验证代码 ---
        print("--- Optimizing Parameters ---")
        total_params = 0
        for name, param in self.pipe.named_parameters():
                if param.requires_grad:
                    # 打印出正在被优化的参数的形状
                    print(name, param.shape)
                    total_params += param.numel()
        print(f"Total number of trainable parameters: {total_params / 1e6:.2f}M")
        print("---------------------------")
        # ---------------------------
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
            checkpoint.update(state_dict)
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
            # lora_state_dict = {}
            # for name, param in state_dict.items():
            #     if name in trainable_param_names:
            #         lora_state_dict[name] = param
            torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt")) # save all params
            checkpoint.update(state_dict)
        torch.cuda.empty_cache()
        gc.collect()
    def on_train_batch_end(self, outputs, batch, batch_idx):
        torch.cuda.empty_cache()
        gc.collect()

    # def on_save_checkpoint(self, checkpoint):
    #     checkpoint_dir = self.trainer.checkpoint_callback.dirpath
    #     print(f"Checkpoint directory: {checkpoint_dir}")
    #     os.makedirs(checkpoint_dir, exist_ok=True)
    #     current_step = self.global_step
    #     print(f"Current step: {current_step}")

    #     checkpoint.clear()
    #     trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
    #     trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
    #     state_dict = self.pipe.denoising_model().state_dict()
    #     torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt"))

# ===== 新增自定义 collate_fn 函数 =====
def custom_collate_fn(batch):
    # 过滤掉 None 样本
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # 1. 找出批次中单个样本包含的最大ID图像数量
    max_k = 0
    if "id_image" in batch[0]:
        max_k = max(len(item["id_image"]) for item in batch)

    # 2. 遍历每个样本进行填充
    for item in batch:
        if "id_image" in item:
            id_images = item["id_image"] # 这是一个Tensor列表
            current_k = len(id_images)
            
            # 如果当前样本的ID图数量小于最大值，则进行填充
            if current_k < max_k:
                # 获取一张图的形状 (C, H, W)
                C, H, W = id_images[0].shape
                # 创建一个全零的Tensor用于填充
                padding_tensor = torch.zeros((C, H, W), dtype=id_images[0].dtype)
                # 将填充Tensor添加到列表末尾
                id_images.extend([padding_tensor] * (max_k - current_k))
            
            # 将列表中的所有Tensor堆叠成一个新的维度
            # 最终形状为 (max_k, C, H, W)
            item["id_image"] = torch.stack(id_images)
        
    # 3. 使用PyTorch默认的collate函数处理已经规整过的batch
    # 它现在可以正确地将所有 (max_k, C, H, W) 的张量堆叠成 (B, max_k, C, H, W)
    return default_collate(batch)
# ===== 自定义 collate_fn 结束 =====


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
    # ===== 新增参数 =====
    parser.add_argument("--id_encoder_path", type=str, default=None, help="Path of ID Encoder.")
    # ==================
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
    # 过滤掉加载失败的样本
    examples = [item for item in examples if item is not None and item]
    if not examples:
        return {}
    # --- 1. 准备工作：确定共享的目标尺寸 ---
    # 由于所有样本都来自同一个分桶，我们可以安全地使用第一个样本来确定共享的 `closest_size`
    first_sample = examples[0]
    # 注意：这些常量和函数(get_aspect_ratio, get_closest_ratio_key)最好在 Dataset 初始化时定义
    # 并通过 functools.partial 传入 collate_fn，这里为了完整性暂时写在内部。
    sample_n_frames_bucket_interval = 4
    video_sample_frames = 81
    video_sample_size = 632 
    ASPECT_RATIO = get_aspect_ratio(size=video_sample_size)
    aspect_ratio_sample_size = {key: [x for x in ASPECT_RATIO[key]] for key in ASPECT_RATIO.keys()}
    
    # 从第一个样本的原始尺寸，计算出整个批次共享的最终裁剪尺寸
    _, _, h, w = first_sample["video"].shape
    closest_ratio = get_closest_ratio_key(h, w, ratios_dict=aspect_ratio_sample_size)
    closest_size = aspect_ratio_sample_size[closest_ratio] # e.g., (512, 768)

    # --- 2. 提前计算出公共的时间长度 ---
    sample_n_frames_bucket_interval = 4
    batch_video_length = min(
        (ex["video"].shape[0] - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1 
        for ex in examples
    )
    if batch_video_length <= 0: batch_video_length = 1


    # --- 3. 循环处理：应用变换（因为原始尺寸不同，循环是必需的）---
    processed_videos = []
    processed_first_frames = []
    processed_render_videos = []
    processed_render_masks = []
    
    for ex in examples:
        # 2a. 为当前样本计算其独特的 resize_size
        _, _, h_i, w_i = ex["video"].shape
        if closest_size[0] / h_i > closest_size[1] / w_i:
            resize_size = closest_size[0], int(w_i * closest_size[0] / h_i)
        else:
            resize_size = int(h_i * closest_size[1] / w_i), closest_size[1]

        # 2b. 创建针对当前样本的变换
        transform = transforms.Compose([
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(closest_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # 2c. 应用变换并收集结果
        processed_videos.append(transform(ex["video"]))
        if "first_frame" in ex:
            processed_first_frames.append(transform(ex["first_frame"]))
        if "render_video" in ex:
            # processed_render_videos.append(transform(ex["render_video"]))
            processed_render_videos.append(ex["render_video"])
        if "render_mask" in ex:
            # processed_render_masks.append(transform(ex["render_mask"]))
            processed_render_masks.append(ex["render_mask"])

    # --- 3. 批量计算：一次性处理所有 Plücker Embeddings ---
    # 此时，所有视频/图像都已被变换到相同的 closest_size
    batch_extrinsics = np.stack([ex["camera_extrinsics"][:batch_video_length] for ex in examples])
    batch_intrinsics = np.stack([ex["camera_intrinsics"][:batch_video_length] for ex in examples])
    
    # 【向量化计算】这是主要的效率提升点
    batch_plucker_embedding = get_plucker_embedding_cpu_batched(
        batch_extrinsics, batch_intrinsics, height=closest_size[0], width=closest_size[1]
    )

    # 截断并堆叠所有张量
    batch = {
        "text": [ex["text"] for ex in examples],
        "path": [ex["path"] for ex in examples],
        "video": torch.stack([v[:batch_video_length] for v in processed_videos]),
        "camera_plucker_embedding": batch_plucker_embedding,
        # "camera_plucker_embedding" : None,
        # 注意：内外参现在是批处理的张量
        "camera_extrinsics": torch.from_numpy(batch_extrinsics),
        "camera_intrinsics": torch.from_numpy(batch_intrinsics),
    }
    if processed_first_frames:
        batch["first_frame"] = torch.stack(processed_first_frames)
    if processed_render_videos:
        batch["render_video"] = torch.stack([rv[:batch_video_length] for rv in processed_render_videos])
    if processed_render_masks:
        batch["render_mask"] = torch.stack([rm[:batch_video_length] for rm in processed_render_masks])

    return batch

# def collate_fn(examples):
#     t_collate = time.time()
    
#     # 过滤掉加载失败的样本
#     examples = [item for item in examples if item is not None and item]
#     if not examples:
#         return {}

#     # 1. 确定 batch 内的公共视频长度 (与之前相同)
#     sample_n_frames_bucket_interval = 4
#     batch_video_length = min(
#         (ex["video"].shape[0] - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1 
#         for ex in examples
#     )
#     if batch_video_length <= 0: # 确保长度至少为1
#         batch_video_length = 1

#     # 2. 收集所有数据并截断到公共长度
#     new_examples = {key: [] for key in examples[0].keys()}
#     for ex in examples:
#         for key, value in ex.items():
#             # 【关键改动】在这里加入对相机参数和 video_id 的截断逻辑
#             if key in ["video", "render_video", "render_mask", "camera_extrinsics", "camera_intrinsics"]:
#                 new_examples[key].append(value[:batch_video_length])
#             elif key == "video_id":
#                 new_examples[key].append(value[:batch_video_length])
#             elif key == "camera_plucker_embedding":
#                 # Plücker embedding 的形状是 [6, F, H, W]，所以要在第二维截断
#                 new_examples[key].append(value[:, :batch_video_length, :, :])
#             else:
#                 # 其他数据如 text, path, first_frame 等不需要截断
#                 new_examples[key].append(value)
    
#     # 3. 使用 default_collate 或手动 stack (与之前相同)
#     batch = {}
#     for key, value_list in new_examples.items():
#         if isinstance(value_list[0], torch.Tensor):
#             try:
#                 batch[key] = torch.stack(value_list)
#             except RuntimeError as e:
#                 print(f"在键 '{key}' 上 stack 失败: {e}")
#                 # 打印每个tensor的形状，方便调试
#                 for i, t in enumerate(value_list):
#                     print(f"  - Tensor {i} shape: {t.shape}")
#                 raise e # 重新抛出异常
#         else:
#             batch[key] = value_list

#     # print(f"time_collate: {time.time()-t_collate:.4f}s") # 可以暂时注释掉，减少打印信息
#     return batch

# def collate_fn(examples):
#         examples = [item for item in examples if item is not None]
#         # Get token length
#         sample_n_frames_bucket_interval = 4
#         video_sample_frames = 81
#         video_sample_size = 632
#         target_token_length = video_sample_frames * video_sample_size * video_sample_size
#         # length_to_frame_num = get_length_to_frame_num(target_token_length)


#         # Create new output
#         new_examples = {}
#         new_examples["video"] = []
#         new_examples["text"] = []
#         new_examples["path"] = []
#         new_examples["video_id"] = []
#         new_examples["type"] = []
#         new_examples["render_video"] = []
#         new_examples["render_mask"] = []
#         # new_examples["camera_extrinsics"] = []
#         # new_examples["camera_intrinsics"] = []
#         new_examples["camera_plucker_embedding"] = []

#         new_examples["first_frame"] = []
#         # Get downsample ratio in image and videos
#         pixel_value = examples[0]["video"]
#         data_type = examples[0]["type"]
#         batch_video_length = video_sample_frames

#         ASPECT_RATIO = get_aspect_ratio(size=video_sample_size)
#         aspect_ratio_sample_size = {key : [x for x in ASPECT_RATIO[key]] for key in ASPECT_RATIO.keys()}

#         for example in examples:
#             # To 0~1
#             pixel_values = example["video"]
#             f, c, h, w  = np.shape(pixel_value)
#             closest_ratio = get_closest_ratio_key(h, w, ratios_dict=aspect_ratio_sample_size)
#             closest_size = aspect_ratio_sample_size[closest_ratio]
#             closest_size = [int(x / 16) * 16 for x in closest_size]
            
#             if closest_size[0] / h > closest_size[1] / w:
#                 resize_size = closest_size[0], int(w * closest_size[0] / h)
#             else:
#                 resize_size = int(h * closest_size[1] / w), closest_size[1]
#             transform = transforms.Compose([
#                 transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
#                 transforms.CenterCrop(closest_size),
#                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
#             ])
#             if "first_frame" in example:
#                 first_frame = example["first_frame"]
#                 first_frame = transform(first_frame)
#                 new_examples["first_frame"].append(first_frame)
#             new_examples["video"].append(transform(pixel_values))
#             if "render_video" in example:
#                 new_examples["render_video"].append(example["render_video"])
#                 new_examples["render_mask"].append(example["render_mask"])
#             else:
#                 new_examples["render_video"].append(transform(pixel_values))
#                 new_examples["render_mask"].append(transform(pixel_values))
#             new_examples["text"].append(example["text"])
#             new_examples["path"].append(example['path'])
#             new_examples["video_id"].append(example["video_id"])
#             new_examples["type"].append(example["type"])

#             video_id = example["video_id"]

#             if torch.any(example["camera_extrinsics"]):
#                 camera_plucker_embedding = get_plucker_embedding_cpu(example["camera_extrinsics"], example["camera_intrinsics"], height=closest_size[0], width=closest_size[1])
#                 camera_plucker_embedding = camera_plucker_embedding[:,video_id,:,:]
#                 new_examples["camera_plucker_embedding"].append(camera_plucker_embedding)
#             else:
#                 zero_camera_plucker_embedding = torch.zeros([6, len(video_id), closest_size[0], closest_size[1]])
#                 new_examples["camera_plucker_embedding"].append(zero_camera_plucker_embedding)

#             # needs the number of frames to be 4n + 1.
#             batch_video_length = int(
#                 min(
#                     batch_video_length,
#                     (len(pixel_values) - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1,
#                 )
#             )
#         if batch_video_length == 0:
#             batch_video_length = 1
#         # Limit the number of frames to the same
#         new_examples["video"] = torch.stack(
#             [example[:batch_video_length] for example in new_examples["video"]])
#         new_examples["render_video"] = torch.stack(
#             [example[:batch_video_length] for example in new_examples["render_video"]])
#         new_examples["render_mask"] = torch.stack(
#             [example[:batch_video_length] for example in new_examples["render_mask"]])
#         if "first_frame" in new_examples:
#             new_examples["first_frame"] = torch.stack(
#                 [example for example in new_examples["first_frame"]])
#         if 'camera_plucker_embedding' in new_examples:
#             new_examples["camera_plucker_embedding"] = torch.stack(
#                 [example[:,:batch_video_length,:,:] for example in new_examples["camera_plucker_embedding"]]
#             )
#         # try:
#         #     new_examples["camera_extrinsics"] = torch.stack(
#         #         [example[:batch_video_length] for example in new_examples["camera_extrinsics"]])
#         #     new_examples["camera_intrinsics"] = torch.stack(
#         #         [example[:batch_video_length] for example in new_examples["camera_intrinsics"]])
#         # except Exception as e:
#         #     print(batch_video_length)
#         #     for example in new_examples["camera_extrinsics"]:
#         #         print(example.shape)
#         #     for example in new_examples["video"]:
#         #         print(example.shape)
#         return new_examples



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
            self.val_dataset = CameraVideoDataset(
                self.args.dataset_path,
                self.args.dataset_list,
                steps_per_epoch=5,
                max_num_frames=121,
                num_frames=self.args.num_frames,
                is_i2v=self.args.is_i2v,
                is_camera=self.args.is_camera,
            )

    def val_dataloader(self):
        sampler = DistributedSampler(self.val_dataset, shuffle=False, drop_last=True)
            
        val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            sampler=sampler, # Provide the sampler
            shuffle=False,   # IMPORTANT: shuffle must be False when a sampler is provided
            batch_size=1,
            num_workers=0,
        )
        return val_dataloader

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
                video_duration_bins=[1.1 ,2.1, 3.1, 4.1, 5.1],
                # video_duration_bins=None
            )

            # batch_sampler = AllInOneAspectRatioSampler(
            #     sampler=distributed_sampler,  # Use the distributed sampler here
            #     dataset=self.dataset,
            #     batch_size=self.args.per_device_batch_size,
            #     drop_last=True,
            #     aspect_ratios_dict=aspect_ratio_sample_size,
            #     video_duration_bins=None,
            # )

            # batch_sampler = RobustParallelSampler(
            #     sampler=distributed_sampler,
            #     dataset=self.dataset,
            #     batch_size=self.args.per_device_batch_size,
            #     aspect_ratios_dict=aspect_ratio_sample_size,
            #     drop_last=True,
            #     # --- 强大的新配置 ---
            #     num_workers=4,                         # 使用16个CPU核心并行计算
            #     cache_path="/mnt/workspace/robust_metadata_cache.json",
            #     save_interval=100,                     # 每处理5000个文件就保存一次进度
            #     n_retries=3
            # )

            dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_sampler=batch_sampler,
                # ===== 修改：使用我们自定义的 collate_fn =====
                collate_fn=custom_collate_fn,
                # ==========================================
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

import time
def train(args):
    # dataset = TensorDataset(
    #     args.dataset_path,
    #     os.path.join(args.dataset_path, 'RealCam-Vid_MultiCam_camera_train_valid_tensor_idx.npz'),
    #     steps_per_epoch=args.steps_per_epoch,
    # )
    # dataset = CameraVideoDataset(
    #     args.dataset_path,
    #     ['/mnt/data/camera_datasets/MuteApo/RealCam-Vid/RealCam-Vid_train_camera.npz', '/mnt/data/hdd/user_workspace/yuqifan/cameractrl/vggt/openhumanvid_81frames_filtered_40000.json'],
    #     # ['/mnt/data/hdd/user_workspace/duanke/video_mixkit_81f_26347/mixkit_vggt_results.json', '/mnt/data/hdd/user_workspace/yuqifan/cameractrl/vggt/openhumanvid_81frames_filtered_40000.json'],
    #     steps_per_epoch=args.steps_per_epoch,
    #     max_num_frames=129,
    #     frame_interval=1,
    #     num_frames=81,
    #     height=480,
    #     width=832,
    #     is_i2v=True,     # 根据你的使用情况
    #     is_camera=True   # 确保启用 camera 相关字段
    # )
    # 加载数据集
    # dataset_path = args.dataset_path
    # dataset = CameraVideoDataset(
    #     dataset_path,
    #     args.dataset_list,
    #     steps_per_epoch=args.steps_per_epoch,
    #     max_num_frames=121,
    #     num_frames=args.num_frames,
    #     is_i2v=args.is_i2v,     # 根据你的使用情况
    #     is_camera=args.is_camera,   # 确保启用 camera 相关字段
    # )
    # batch_size = args.per_device_batch_size
    # num_workers= args.dataloader_num_workers
    # if args.enable_bucket:
    #     ASPECT_RATIO = get_aspect_ratio(size=args.video_sample_size)
    #     aspect_ratio_sample_size = {key : [x for x in ASPECT_RATIO[key]] for key in ASPECT_RATIO.keys()}
    #     batch_sampler_generator = torch.Generator().manual_seed(3407)
    #     batch_sampler = AspectRatioImageVideoSampler(
    #         sampler=RandomSampler(dataset, generator=batch_sampler_generator), dataset=dataset,
    #         batch_size=batch_size, drop_last=True,
    #         aspect_ratios_dict=aspect_ratio_sample_size,
    #         video_duration_bins = None,
    #     )

    #     dataloader = torch.utils.data.DataLoader(
    #         dataset,
    #         batch_sampler=batch_sampler,
    #         collate_fn=collate_fn,
    #         num_workers=num_workers,
    #     )
    # else:
    #     dataloader = torch.utils.data.DataLoader(
    #         dataset,
    #         shuffle=True,
    #         batch_size=args.per_device_batch_size,
    #         num_workers=args.dataloader_num_workers,
    #     )      
    # dataset, dataloader, data_sampler = get_dataloader(args)

    # Set the precision for matrix multiplication to improve performance
    # This often resolves the cuDNN SDPA warning and significantly speeds up attention layers.
    torch.set_float32_matmul_precision('medium') # or 'medium'
 
    tm0 = time.time()

    # =================================================================================
    # 1. Data Loading is now handled by the DataModule.
    #    All the previous dataset/sampler/dataloader code is removed from here.
    # =================================================================================
    datamodule = CameraDataModule(args)

    tm1 = time.time()
    print(f"datamodule:{tm1 -tm0}")
    # =================================================================================
    # 2. Model, Logger, and Strategy setup remains the same.
    # =================================================================================
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        id_encoder_path=args.id_encoder_path, # 传递路径
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
        # logger = TensorBoardLogger(args.output_path, name='tensorboardlog', log_dir=os.path.join(args.output_path, "tensorboardlog"))
        logger = TensorBoardLogger(args.output_path, name='tensorboardlog')
    else:
        logger = None
    if args.training_strategy == "fsdp":
        args.training_strategy = FSDPStrategy(
            sharding_strategy="SHARD_GRAD_OP"
        )
    elif args.training_strategy == "auto":
        args.training_strategy = DDPStrategy(find_unused_parameters=True)
        # args.training_strategy = DDPStrategy()
    # =================================================================================
    # 3. Trainer setup is also the same.
    # =================================================================================
    tm2 = time.time()
    print(f"model:{tm2 -tm1}")
    profiler = SimpleProfiler(dirpath="/mnt/workspace/profiler_logs/", filename="perf_logs")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        # val_check_interval=50,
        # num_sanity_val_steps=0, # 设置为0来禁用
        # check_val_every_n_epoch=1,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
        logger=logger,
        log_every_n_steps=50,
        profiler=profiler
    )
    tm3 = time.time()
    print(f"trainer:{tm3 -tm2}")

    # =================================================================================
    # 4. KEY CHANGE: Pass the datamodule to trainer.fit(), not the old dataloader.
    # =================================================================================
    trainer.fit(model, datamodule=datamodule)
    # try:
    #     # 启动侦测器
    #     add_creation_autofix_checker()
    #     trainer.fit(model, datamodule=datamodule)
    # finally:
    #     # 确保在最后恢复 PyTorch 的原始功能
    #     remove_creation_autofix_checker()


if __name__ == '__main__':
    pl.seed_everything(42)
    args = parse_args()
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)

