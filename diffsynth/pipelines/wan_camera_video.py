import types

from diffsynth.models.wan_video_camera_encoder import CameraPoseControlNet


from ..models import ModelManager
from ..models.wan_video_dit import WanModel
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from ..prompters import WanPrompter
import torch, os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional

from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from ..models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_dit import RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_vae import RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_motion_controller import WanMotionControllerModel



class WanVideoCameraPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
         # 初始化 CameraEncoder
        # self.camera_encoder: WanCameraEncoder = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.model_names = ['text_encoder', 'dit', 'vae', 'image_encoder', 'motion_controller', 'vace']
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.use_unified_sequence_parallel = False


    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            enable_vram_management(
                self.vace,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        # if self.camera_encoder is not None:
        #     dtype = next(iter(self.camera_encoder.parameters())).dtype
        #     enable_vram_management(
        #         self.camera_encoder,
        #         module_map = {
        #             torch.nn.Linear: AutoWrappedLinear,
        #             torch.nn.Conv2d: AutoWrappedModule,
        #             torch.nn.LayerNorm: AutoWrappedModule,
        #         },
        #         module_config = dict(
        #             offload_dtype=dtype,
        #             offload_device="cpu",
        #             onload_dtype=dtype,
        #             onload_device="cpu",
        #             computation_dtype=dtype,
        #             computation_device=self.device,
        #         ),
        #     )
        self.enable_cpu_offload()


    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        self.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        self.vace = model_manager.fetch_model("wan_video_vace")
        # self.camera_encoder = model_manager.fetch_model("wan_video_camera_encoder")


    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None, use_usp=False):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = WanVideoCameraPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

            for block in pipe.dit.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            pipe.dit.forward = types.MethodType(usp_dit_forward, pipe.dit)
            pipe.sp_size = get_sequence_parallel_world_size()
            pipe.use_unified_sequence_parallel = True
        return pipe
    
    
    def denoising_model(self):
        return self.dit


    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive, device=self.device)
        return {"context": prompt_emb}
    
    
    def encode_image(self, image, end_image, num_frames, height, width, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = self.preprocess_image(end_image.resize((width, height))).to(self.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            if self.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, self.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        y = self.vae.encode([vae_input.to(dtype=self.torch_dtype, device=self.device)], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=self.torch_dtype, device=self.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}
    
    
    def encode_control_video(self, control_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        control_video = self.preprocess_images(control_video)
        control_video = torch.stack(control_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
        latents = self.encode_video(control_video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=self.torch_dtype, device=self.device)
        return latents
    
    
    def prepare_controlnet_kwargs(self, control_video, num_frames, height, width, clip_feature=None, y=None, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        if control_video is not None:
            control_latents = self.encode_control_video(control_video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            if clip_feature is None or y is None:
                clip_feature = torch.zeros((1, 257, 1280), dtype=self.torch_dtype, device=self.device)
                y = torch.zeros((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=self.torch_dtype, device=self.device)
            else:
                y = y[:, -16:]
            y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}

    def encode_camera(self, camera_pose, camera_intrinsics, height, width):
        # camera_pose = torch.tensor(camera_pose, dtype=torch.float32, device=self.device)
        # camera_intrinsics = torch.tensor(camera_intrinsics, dtype=torch.float32, device=self.device)
        camera_pose = camera_pose.to(dtype=torch.float32, device=self.device)
        camera_intrinsics = camera_intrinsics.to(dtype=torch.float32, device=self.device)
        camera_emb = self.camera_encoder.encode_camera(camera_pose, camera_intrinsics, height, width)
        camera_emb = camera_emb.to(dtype=self.torch_dtype, device=self.device)
        return {"camera_context": camera_emb}

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
    
    
    def prepare_extra_input(self, latents=None):
        return {}
    
    
    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames
    
    
    def prepare_unified_sequence_parallel(self):
        return {"use_unified_sequence_parallel": self.use_unified_sequence_parallel}
    
    
    def prepare_motion_bucket_id(self, motion_bucket_id):
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=self.torch_dtype, device=self.device)
        return {"motion_bucket_id": motion_bucket_id}
    
    
    def prepare_vace_kwargs(
        self,
        latents,
        vace_video=None, vace_mask=None, vace_reference_image=None, vace_scale=1.0,
        height=480, width=832, num_frames=81,
        seed=None, rand_device="cpu",
        tiled=True, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        if vace_video is not None or vace_mask is not None or vace_reference_image is not None:
            self.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=self.torch_dtype, device=self.device)
            else:
                vace_video = self.preprocess_images(vace_video)
                vace_video = torch.stack(vace_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
            
            if vace_mask is None:
                vace_mask = torch.ones_like(vace_video)
            else:
                vace_mask = self.preprocess_images(vace_mask)
                vace_mask = torch.stack(vace_mask, dim=2).to(dtype=self.torch_dtype, device=self.device)
            
            inactive = vace_video * (1 - vace_mask) + 0 * vace_mask
            reactive = vace_video * vace_mask + 0 * (1 - vace_mask)
            inactive = self.encode_video(inactive, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=self.torch_dtype, device=self.device)
            reactive = self.encode_video(reactive, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=self.torch_dtype, device=self.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)
            
            vace_mask_latents = rearrange(vace_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')
            
            if vace_reference_image is None:
                pass
            else:
                vace_reference_image = self.preprocess_images([vace_reference_image])
                vace_reference_image = torch.stack(vace_reference_image, dim=2).to(dtype=self.torch_dtype, device=self.device)
                vace_reference_latents = self.encode_video(vace_reference_image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=self.torch_dtype, device=self.device)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                vace_video_latents = torch.concat((vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :1]), vace_mask_latents), dim=2)
                
                noise = self.generate_noise((1, 16, 1, latents.shape[3], latents.shape[4]), seed=seed, device=rand_device, dtype=torch.float32)
                noise = noise.to(dtype=self.torch_dtype, device=self.device)
                latents = torch.concat((noise, latents), dim=2)
            
            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return latents, {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return latents, {"vace_context": None, "vace_scale": vace_scale}

    @torch.no_grad()
    def dit_generation(
        self,
        prompt_emb_posi,
        prompt_emb_nega=None,
        cam_emb=None,
        image_emb=None,
        control_video=None,
        vace_video=None,
        vace_video_mask=None,
        vace_reference_image=None,
        vace_scale=1.0,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        motion_bucket_id=None,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Initialize noise
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)

        latents = noise
        if prompt_emb_nega is None:
            prompt_emb_nega = torch.load("data/prompt_emb_nega.pth", weights_only=True, map_location="cpu")
            prompt_emb_nega["context"] = prompt_emb_nega["context"].to(self.device)
        # TeaCache
        tea_cache_posi = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        tea_cache_nega = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        
        # Motion Controller
        if self.motion_controller is not None and motion_bucket_id is not None:
            motion_kwargs = self.prepare_motion_bucket_id(motion_bucket_id)
        else:
            motion_kwargs = {}
            
        # Extra input
        extra_input = self.prepare_extra_input(latents)
        # Unified Sequence Parallel
        usp_kwargs = self.prepare_unified_sequence_parallel()

        # Denoise
        self.load_models_to_device(["dit", "motion_controller", "vace"])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # latents_input = torch.cat([latents, source_latents], dim=2)
            # Inference
            noise_pred_posi = model_fn_wan_video(
                self.dit, motion_controller=self.motion_controller, vace=self.vace,
                x=latents, timestep=timestep, cam_emb=cam_emb,
                **prompt_emb_posi, **image_emb, **extra_input,
                **tea_cache_posi, **usp_kwargs, **motion_kwargs,
            )
            if cfg_scale != 1.0:
                noise_pred_nega = model_fn_wan_video(
                    self.dit, motion_controller=self.motion_controller, vace=self.vace,
                    x=latents, timestep=timestep, cam_emb=cam_emb,
                    **prompt_emb_nega, **image_emb, **extra_input,
                    **tea_cache_nega, **usp_kwargs, **motion_kwargs,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
            
        if vace_reference_image is not None:
            latents = latents[:, :, 1:]

        # Decode
        self.load_models_to_device(['vae'])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames


    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        render_latent=None,
        render_mask=None,
        camera_pose=None,
        input_image=None,
        end_image=None,
        input_video=None,
        control_video=None,
        vace_video=None,
        vace_video_mask=None,
        vace_reference_image=None,
        vace_scale=1.0,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        motion_bucket_id=None,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Initialize noise
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        if input_video is not None:
            self.load_models_to_device(['vae'])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
            latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise
        # latents.shape = torch.Size([1, 16, 21, 60, 104]) 832*480（16:9） 1, 16, (num_frames - 1) // 4 + 1, height//8, width//8
        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        prompt_emb_post_camera = self.encode_prompt("The camera zoom in, remains steady throughout.", positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            torch.save(prompt_emb_nega, "data/prompt_emb_nega.pth")
        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(input_image, end_image, num_frames, height, width, **tiler_kwargs)
        else:
            image_emb = {}
        # Encode camera context
        if camera_pose is not None:
            cam_emb = camera_pose.to(dtype=self.torch_dtype, device=self.device)
        else:
            cam_emb = None
        # Encode pointcloud context:
        if render_latent is not None and render_mask is not None:
            render_latent = render_latent.to(dtype=self.torch_dtype, device=self.device)
            render_mask = render_mask.to(dtype=self.torch_dtype, device=self.device)
        else:
            render_latent = None
            render_mask = None   
        # ControlNet
        if control_video is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.prepare_controlnet_kwargs(control_video, num_frames, height, width, **image_emb, **tiler_kwargs)
            
        # Motion Controller
        if self.motion_controller is not None and motion_bucket_id is not None:
            motion_kwargs = self.prepare_motion_bucket_id(motion_bucket_id)
        else:
            motion_kwargs = {}
            
        # Extra input
        extra_input = self.prepare_extra_input(latents) # z_feat = z_feat + cam_feat 直接在这里相加对物体状态信息的变化影响最小
        # 但是要先把i2v的微调跑通
        
        # VACE
        latents, vace_kwargs = self.prepare_vace_kwargs(
            latents, vace_video, vace_video_mask, vace_reference_image, vace_scale,
            height=height, width=width, num_frames=num_frames, seed=seed, rand_device=rand_device, **tiler_kwargs
        )
        
        # TeaCache
        tea_cache_posi = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        tea_cache_nega = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        
        # Unified Sequence Parallel
        usp_kwargs = self.prepare_unified_sequence_parallel()

        # Denoise
        self.load_models_to_device(["dit", "motion_controller", "vace"])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # latents_input = torch.cat([latents, source_latents], dim=2)
            # Inference
            # if progress_id < 40:
            #     print(progress_id)
            #     prompt_emb_posi = prompt_emb_post_camera
                # cam_emb = None
            noise_pred_posi = model_fn_wan_video(
                self.dit, motion_controller=self.motion_controller, vace=self.vace,
                x=latents, timestep=timestep, render_latent=render_latent, render_mask=render_mask, cam_emb=cam_emb, 
                **prompt_emb_posi, **image_emb, **extra_input,
                **tea_cache_posi, **usp_kwargs, **motion_kwargs, **vace_kwargs,
            )
            if cfg_scale != 1.0:
                noise_pred_nega = model_fn_wan_video(
                    self.dit, motion_controller=self.motion_controller, vace=self.vace,
                    x=latents, timestep=timestep,
                    **prompt_emb_nega, **image_emb, **extra_input,
                    **tea_cache_nega, **usp_kwargs, **motion_kwargs, **vace_kwargs,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
            
        if vace_reference_image is not None:
            latents = latents[:, :, 1:]

        # Decode
        self.load_models_to_device(['vae'])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    x: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    render_latent: torch.Tensor = None,
    render_mask: torch.Tensor = None,
    cam_emb: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    vace_context = None,
    vace_scale = 1.0,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    **kwargs,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

        ### process controlnet inputs ###
        if render_latent is None:
            controlnet_inputs = None
            controlnet_rotary_emb = None
        else:
            render_latent = torch.cat([x[:, :20], render_latent], dim=1)
            # render_latent
            controlnet_inputs = dit.controlnet_patch_embedding(render_latent)
            controlnet_rotary_emb = dit.controlnet_rope(render_latent)
            controlnet_inputs = controlnet_inputs.flatten(2).transpose(1, 2)
        
        # additional inputs (mask, camera embedding)
        if cam_emb is None:
            add_inputs = None
        else:
            if render_mask is not None:
                add_inputs = torch.cat([render_mask, cam_emb], dim=1)
            else:
                zero_render_mask = torch.zeros_like(cam_emb[:,0:1,:,:,:]).to(cam_emb.device, cam_emb.dtype)
                add_inputs = torch.cat([zero_render_mask, cam_emb], dim=1)
            add_inputs = dit.controlnet_mask_embedding(add_inputs)

        if controlnet_inputs is not None and add_inputs is not None:
            controlnet_inputs = controlnet_inputs + add_inputs # torch.Size([8, 32760, 5120])
        elif controlnet_inputs is not None:
            controlnet_inputs = controlnet_inputs
        elif add_inputs is not None:
            controlnet_inputs = add_inputs
        else:
            controlnet_inputs = None
        
        ### process controlnet inputs over ###


        if controlnet_inputs is not None:
            controlnet_states = dit.controlnet(hidden_states=controlnet_inputs,
                                        temb=t,
                                        rotary_emb=controlnet_rotary_emb,
                                    )
            # [torch.Size([8, 32760, 1536])] * 15(layers)
        else:
            controlnet_states = []

    x, (f, h, w) = dit.patchify(x)

    
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(x, vace_context, context, t_mod, freqs)


    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        for block_id, block in enumerate(dit.blocks):
            if block_id <100:
                x = block(x, context, t_mod, freqs)
            else:
                x = block(x, context, t_mod, freqs)
            if len(controlnet_states) > 0:
                if block_id < len(controlnet_states):
                    x += controlnet_states[block_id]
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                x = x + vace_hints[vace.vace_layers_mapping[block_id]] * vace_scale
        if tea_cache is not None:
            tea_cache.store(x)
    # print(x.shape)
    # print(t)
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
    x = dit.unpatchify(x, (f, h, w))
    return x
