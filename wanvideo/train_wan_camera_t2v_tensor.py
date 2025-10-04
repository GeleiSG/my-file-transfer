import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
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

from diffsynth.data.camera_eval import calculate_pose_errors
from diffsynth.data.camera_utils import get_camera_sparse_embedding, get_plucker_embedding
from diffsynth.data.camera_video import CameraVideoDataset
from diffsynth.data.video import save_video
from diffsynth.pipelines.wan_camera_video import WanVideoCameraPipeline
# from vggt.test import generate_camera_params
# from vggt.vggt.models.vggt import VGGT


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
        self.path = [os.path.join('/mnt/data/hdd/datasets/camera_datasets/KwaiVGI/MultiCamVideo-Dataset', entry["video_path"]) if 'MultiCamVideo-Dataset' in entry["video_path"] else os.path.join(base_path, entry["video_path"]) for entry in metadata]

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
            video_id = data["video_id"][::4]
        else:
            video_id = np.linspace(0, len(self.camera_extrinsics[data_id]) - 1, 81).astype(int)
            video_id = video_id[::4]
        height = data['latents'].shape[2] * 8
        width = data['latents'].shape[3] * 8
        data["camera_extrinsics"], data["camera_intrinsics"] = get_camera_sparse_embedding(self.camera_extrinsics[data_id][video_id], self.camera_intrinsics[data_id][video_id], height, width)
        
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
        self, dit_path, vae_path,
        resume_ckpt_path=None,
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None
    ):
        super().__init__()
        self.train_architecture = train_architecture

        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        # model_manager.load_models(model_path) # load text encoder, vae, image encoder

        if os.path.isfile(dit_path):
            model_path = [dit_path]
        else:
            dit_path = dit_path.split(",")
            model_path = [dit_path]
        # model_path.append(vae_path)
        model_manager.load_models(model_path)
        self.pipe = WanVideoCameraPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        dim=self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        for block in self.pipe.dit.blocks:
            # block.cam_encoder = nn.Sequential(
            #     nn.Linear(16, dim),
            #     nn.ReLU(),
            # )
            # block.projector = nn.Sequential(
            #     nn.LayerNorm(dim),
            #     nn.Linear(dim, dim)
            # )
            # # block.cam_encoder.weight.data.zero_()
            # # block.cam_encoder.bias.data.zero_()
            # nn.init.kaiming_normal_(block.cam_encoder[0].weight, mode='fan_out', nonlinearity='relu')
            # nn.init.constant_(block.cam_encoder[0].bias, 0)
            # block.projector[1].weight.data.zero_()
            # block.projector[1].bias.data.zero_()
        
            block.cam_encoder = nn.Linear(16, dim)
            block.projector = nn.Linear(dim, dim)
            # # 零初始化最后一层 projector
            # nn.init.kaiming_normal_(block.cam_encoder.weight, mode='fan_out', nonlinearity='relu')
            # nn.init.constant_(block.cam_encoder.bias, 0)
            block.cam_encoder.weight.data.zero_()
            block.cam_encoder.bias.data.zero_()
            # block.projector.weight.data.zero_()
            # block.projector.bias.data.zero_()
            
            # block.cam_encoder = nn.Linear(16, dim)

        
            # block.to_gamma_beta = nn.Linear(dim, 2 * dim)
            # nn.init.zeros_(block.to_gamma_beta.weight)
            # nn.init.zeros_(block.to_gamma_beta.bias)

            

            # nn.init.normal_(block.projector[-1].weight, mean=0.0, std=1e-3)
            # nn.init.zeros_(block.projector[-1].bias)
            block.projector.weight = nn.Parameter(torch.eye(dim))
            block.projector.bias = nn.Parameter(torch.zeros(dim))
        
        
        if resume_ckpt_path is not None:
            print(f"Loading resume ckpt path: {resume_ckpt_path}")
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")
            self.pipe.dit.load_state_dict(state_dict, strict=False)
            self.pipe.to("cuda")
            self.pipe.to(dtype=torch.bfloat16)

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
                if any(keyword in name for keyword in ["cam_encoder", "projector", "self_attn"]):
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
                if any(keyword in name for keyword in ["cam_encoder", "projector", "self_attn"]):
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
    
    # def val_dataloader(self):
    #     dataset = FixedValDataset('data/train_eval/5_data.pt', max_items=5)
    #     return torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=1,
    #         shuffle=False
    #     )
    # def validation_step(self, batch, batch_idx):
    #     # ... 验证逻辑 ...
    #     prompt_emb = batch['prompt_emb']
    #     prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
    #     image_emb = batch["image_emb"]
    #     if "clip_feature" in image_emb:
    #         image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
    #     if "y" in image_emb:
    #         image_emb["y"] = image_emb["y"][0].to(self.device)
        
    #     camera_extrinsic = batch['camera_extrinsic'][0]
    #     camera_intrinsic = batch['camera_intrinsic'][0]
    #     video_id = batch['video_id'][0]
    #     camera_id = video_id[::4]
    #     # generate camera embedding
    #     # camera_embedding = get_plucker_embedding(camera_extrinsic[video_id], camera_intrinsic[video_id], height=480, width=832)
    #     camera_embedding, _ = get_camera_sparse_embedding(camera_extrinsic[camera_id,:,:].cpu().numpy(), camera_intrinsic[camera_id,:].cpu().numpy(), height=480, width=832)
    #     camera_embedding = camera_embedding[None].to(self.device)

    #     # generate video
    #     self.pipe.device = self.device
    #     video = self.pipe.dit_generation(
    #         prompt_emb_posi=prompt_emb,
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
        # Data
        latents = batch["latents"].to(self.device)
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][:,0,:].to(self.device)
        image_emb = batch["image_emb"]
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][:,0,:].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][:,0,:].to(self.device)
        
        # camera_embedding = torch.cat([batch["camera_intrinsics"], rearrange(batch["camera_extrinsics"], 'b c d f -> b c (d f)')], dim=-1)
        camera_embedding = rearrange(batch["camera_extrinsics"], 'b c d f -> b c (d f)')
        camera_embedding = camera_embedding.to(self.device)
        # only for batch size 1
        # latents = latents.unsqueeze(0)
        # prompt_emb["context"] = prompt_emb["context"].unsqueeze(0)
        # if "clip_feature" in image_emb:
        #     image_emb["clip_feature"] = image_emb["clip_feature"].unsqueeze(0)
        # if "y" in image_emb:
        #     image_emb["y"] = image_emb["y"].unsqueeze(0)

        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        # noise_pred = self.pipe.denoising_model()(
        #     noisy_latents, timestep=timestep, **prompt_emb, **extra_input, **image_emb, plucker_embedding=camera_plucker_emb,
        #     use_gradient_checkpointing=self.use_gradient_checkpointing,
        #     use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        # )
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, camera_embedding=camera_embedding, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)
        # for name, param in self.pipe.denoising_model().named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: grad mean abs = {param.grad.abs().mean().item():.6f}")
        #     else:
        #         if param.requires_grad:
        #             print(f"{name}")
        
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
            lora_state_dict = {}
            for name, param in state_dict.items():
                if name in trainable_param_names:
                    lora_state_dict[name] = param
            torch.save(lora_state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt"))
            checkpoint.update(lora_state_dict)
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
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
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
        os.path.join(args.dataset_path, 'RealCam-Vid_MultiCam_camera_train_tensor_idx.npz'),
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

def train(args):
    dataset = TensorDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, 'RealCam-Vid_MultiCam_camera_train_valid_tensor_idx.npz'),
        steps_per_epoch=args.steps_per_epoch,
    )
    # dataset = CameraVideoDataset(
    #     args.dataset_path,
    #     os.path.join(args.dataset_path, 'RealCam-Vid_train.npz'),
    #     os.path.join(args.dataset_path, 'camera_caption_total.json'),
    #     steps_per_epoch=args.steps_per_epoch,
    #     max_num_frames=129,
    #     frame_interval=1,
    #     num_frames=81,
    #     height=480,
    #     width=832,
    #     is_i2v=True,     # 根据你的使用情况
    #     is_camera=True   # 确保启用 camera 相关字段
    # )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.per_device_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        vae_path=args.vae_path,
        resume_ckpt_path=args.camera_encoder_path,
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
        logger = TensorBoardLogger(args.output_path, name='tensorboardlog')
    else:
        logger = None
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1, every_n_train_steps=1000)],
        logger=logger,
        log_every_n_steps=50,
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    pl.seed_everything(42)
    args = parse_args()
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)

