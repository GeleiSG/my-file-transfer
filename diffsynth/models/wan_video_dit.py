import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from diffsynth.models.wan_video_camera_controlnet import MaskCamEmbed, WanXControlNet
from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed
from einops import rearrange

from diffsynth.models.wan_video_camera_encoder import CameraPoseControlNet
from diffsynth.models.wan_video_camera_self_controlnet import WanControlNet

# from ..models.wan_video_camera_encoder import CameraPoseAdapter
from .utils import hash_state_dict_keys
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    # freqs[k-1] = 0.9 * 2* torch.pi / L_test
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)

class IDCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, id_embedding: torch.Tensor):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(id_embedding))
        v = self.v(id_embedding)
        x = self.attn(q, k, v)
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)

        # ===== 新增代码开始 =====
        # 为ID注入添加一个新的CrossAttention和LayerNorm
        self.id_cross_attn = IDCrossAttention(dim, num_heads, eps)
        self.norm_id = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        # ===== 新增代码结束 =====

        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()
    
    def adaln(self, x, gamma, beta, eps=1e-5):
        # x, gamma, beta: [B, N, D]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + eps) * (1 + gamma) + beta
    def forward(self, x, context, t_mod, freqs, cam_emb=None, id_embedding=None):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)

        # encode camera
        if cam_emb is not None:
            # cam_emb = self.cam_encoder(cam_emb)
            # cam_emb = cam_emb.repeat(1, 2, 1) # for repeat video latents
            # cam_emb = cam_emb.unsqueeze(2).unsqueeze(3).repeat(1, 1, 30, 52, 1)
            # cam_emb = rearrange(cam_emb, 'b f h w d -> b (f h w) d')
            cam_emb = rearrange(cam_emb, 'b c f h w -> b (f h w) c')
            # input_x = input_x + self.projector(cam_emb)
            # x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
            input_x = input_x + cam_emb
            x = self.gate(x, gate_msa, self.projector(self.self_attn(input_x, freqs)))
        else:
            x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        # x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        # if cam_emb is not None:
        #     cam_emb = self.cam_encoder(cam_emb)
        #     cam_emb = cam_emb.unsqueeze(2).unsqueeze(3).repeat(1, 1, 30, 52, 1)
        #     cam_emb = rearrange(cam_emb, 'b f h w d -> b (f h w) d')
        #     gamma_beta = self.to_gamma_beta(cam_emb)
        #     gamma, beta = gamma_beta.chunk(2, dim=-1)
        #     x = self.adaln(x,gamma,beta)    
        #     x = modulate(x, gamma, beta)       
        # print(x.shape)

        x = x + self.cross_attn(self.norm3(x), context)

        # ===== 新增代码开始 =====
        # 注入ID信息
        if id_embedding is not None:
            x = x + self.id_cross_attn(self.norm_id(x), id_embedding)
        # ===== 新增代码结束 =====

        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x

    # camera_adapter_controlnet
    # def forward(self, x, context, t_mod, freqs, gamma_betas):
    #     # msa: multi-head self-attention  mlp: multi-layer perceptron
    #     shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
    #         self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
    #     input_x = modulate(self.norm1(x), shift_msa, scale_msa)


    #     x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
    #     # print(self.camera_adapter.device)
    #     # self.camera_adapter = self.camera_adapter.to(x.device)
    #     # x = self.camera_adapter(x, camera_context)
    #     gamma, beta = gamma_betas.chunk(2, dim=-1)
    #     # _t, _h, _w = grid_sizes[0]
    #     # assert x.shape[1] == _t * _h * _w
    #     # if self.save_mem:
    #     #     x = rearrange(x, 'b (t h w) c -> b t (h w) c', h=_h, w=_w)
    #     #     x = x * (1 + gamma.unsqueeze(dim=2)) + beta.unsqueeze(dim=2)
    #     #     x = rearrange(x, 'b t s c -> b (t s) c')
    #     print("gamma mean:", gamma.mean().item(), "std:", gamma.std().item(), "max:", gamma.max().item())
    #     print("beta mean:", beta.mean().item(), "std:", beta.std().item(), "max:", beta.max().item())

    #     x = x * (1 + gamma) + beta
    #     assert torch.all(torch.isfinite(gamma)), "gamma contains NaN or Inf"
    #     assert torch.all(torch.isfinite(beta)), "beta contains NaN or Inf"
    #     assert torch.all(torch.isfinite(x)), "x contains NaN or Inf before modulation"

    #     x = x + self.cross_attn(self.norm3(x), context)
    #     input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
    #     x = self.gate(x, gate_mlp, self.ffn(input_x))
    #     return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        camera_layer: int = 100,
        controlnet_cfg=None,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.in_dim = in_dim

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        self.has_image_pos_emb = has_image_pos_emb
        self.camera_layer = camera_layer
        self.controlnet_cfg = controlnet_cfg
        self.rope_max_seq_len = 1024
        self.cam_adapter = None
        # 新增：独立控制gradient_checkpoint
        self.gradient_checkpointing = False
    
    def build_controlnet(self):
        # controlnet
        self.controlnet_patch_embedding = nn.Conv3d(
            36, self.controlnet_cfg.conv_out_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.controlnet_mask_embedding = MaskCamEmbed(self.controlnet_cfg)
        self.controlnet = WanXControlNet(self.controlnet_cfg)
        # self.controlnet = WanControlNet(self.controlnet_cfg)
        # for i, controlnet_block in enumerate(self.controlnet.controlnet_blocks):
        #     controlnet_block_params = controlnet_block.state_dict()
        #     pretrained_dict = {k:v for k,v in self.blocks[i].state_dict().items() if k in controlnet_block_params}
        #     controlnet_block_params.update(pretrained_dict)
        #     self.controlnet.controlnet_blocks[i].load_state_dict(controlnet_block_params)
        # self.control_time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.controlnet_cfg.dim, self.controlnet_cfg.dim * 6))
        # self.controlnet_freqs = precompute_freqs_cis_3d(self.controlnet_cfg.dim // self.controlnet_cfg.num_heads)
        self.controlnet_rope = WanRotaryPosEmbed(self.controlnet_cfg.dim // self.controlnet_cfg.num_heads,
                                                 self.patch_size, self.rope_max_seq_len)


    def patchify(self, x: torch.Tensor, camera_embedding: torch.Tensor = None):
        x = self.patch_embedding(x)
        if self.cam_adapter is not None and camera_embedding is not None:
            y_camera = self.cam_adapter(camera_embedding)
            x = [u + v for u, v in zip(x, y_camera)]
            x = torch.stack(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )
    
    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                render_latent: Optional[torch.Tensor] = None,
                render_mask: Optional[torch.Tensor] = None,
                camera_embedding: Optional[torch.Tensor] = None,
                id_embedding: Optional[torch.Tensor] = None, # ===== 新增参数 =====
                enable_render_drop: bool = True,
                enable_camera_drop: bool = True,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        """
        :param render_latent: [b,c,f,h,w]
        :param render_mask: [b,1,f,h,w]
        :param camera_embedding: [b,6,f,h,w]
        """
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        # control_t_mod = self.control_time_projection(t).unflatten(1, (6, self.controlnet_cfg.dim))
        context = self.text_embedding(context)


        # self.has_image_input = True if y is not None else False

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
                 
        ### process controlnet inputs ###
        if render_latent is None or (enable_render_drop and random.random() < 0.05):
            controlnet_inputs = None
            controlnet_rotary_emb = None
        else:
            render_latent = torch.cat([x[:, :20], render_latent], dim=1)
            # render_latent
            controlnet_inputs = self.controlnet_patch_embedding(render_latent)
            # (f, h, w) = controlnet_inputs.shape[2:]

            controlnet_rotary_emb = self.controlnet_rope(render_latent)
            
            # controlnet_rotary_emb = torch.cat([
            #     self.controlnet_freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            #     self.controlnet_freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            #     self.controlnet_freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            # ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

            controlnet_inputs = controlnet_inputs.flatten(2).transpose(1, 2)
        
        
        # additional inputs (mask, camera embedding)
        if camera_embedding is None or (enable_camera_drop and random.random() < 0.05):
            add_inputs = None
        else:
            if render_mask is not None:
                add_inputs = torch.cat([render_mask, camera_embedding], dim=1)
            else:
                zero_render_mask = torch.zeros_like(camera_embedding[:,0:1,:,:,:]).to(camera_embedding.device, camera_embedding.dtype)
                add_inputs = torch.cat([zero_render_mask, camera_embedding], dim=1)
            add_inputs = self.controlnet_mask_embedding(add_inputs)

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
            controlnet_states = self.controlnet(hidden_states=controlnet_inputs,
                                        temb=t,
                                        rotary_emb=controlnet_rotary_emb,
                                        use_gradient_checkpointing=use_gradient_checkpointing)
            # controlnet_states = self.controlnet(hidden_states = controlnet_inputs,
            #                                     t_mod=t_mod,
            #                                     freqs=controlnet_rotary_emb,
            #                                     use_gradient_checkpointing=use_gradient_checkpointing)
            # [torch.Size([8, 32760, 1536])] * 15(layers)
        else:
            controlnet_states = []
        # for controlnet_state in controlnet_states:
        #     print(controlnet_state.mean())
        # for id, controlnet_state in enumerate(controlnet_states):
        #     print(f"layer {id}: ", controlnet_state.mean().item())

        # test_x1, (f, h, w) = self.patchify(x, camera_embedding)
        # print('test_x1_shape:', test_x1.shape)
        # test_x2, (f, h, w) = self.patchify(x, None)
        # print('test_x2_shape:', test_x2.shape)

        x, (f, h, w) = self.patchify(x)
        # camera_embedding = self.cam_encoder(camera_embedding)
    
        # print('pluker:', plucker_embedding.mean().item(), plucker_embedding.max().item(), plucker_embedding.min().item())
        # gamma_betas = self.camera_adapter(plucker_embedding, f, h, w)
        # for gamma_beta in gamma_betas:
        #     print("gamma mean:", gamma_beta.mean().item(), "std:", gamma_beta.std().item(), "max:", gamma_beta.max().item())

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        
        for i_block, block in enumerate(self.blocks):
            # 将 id_embedding 作为参数传递给 block
            block_kwargs = {
                "x": x, "context": context, "t_mod": t_mod, "freqs": freqs, "id_embedding": id_embedding
            }
            # if i_block < self.camera_layer:
            #     if self.training or self.gradient_checkpointing and use_gradient_checkpointing:
            #         if use_gradient_checkpointing_offload:
            #             with torch.autograd.graph.save_on_cpu():
            #                 x = torch.utils.checkpoint.checkpoint(
            #                     create_custom_forward(block),
            #                     x, context, t_mod, freqs,
            #                     use_reentrant=False,
            #                 )
            #         else:
            #             x = torch.utils.checkpoint.checkpoint(
            #                 create_custom_forward(block),
            #                 x, context, t_mod, freqs,
            #                 use_reentrant=False,
            #             )
            #         # adding control features
            #         if i_block < len(controlnet_states):
            #             x += controlnet_states[i_block]
            #     else:
            #         x = block(x, context, t_mod, freqs)
            #         # adding control features
            #         if i_block < len(controlnet_states):
            #             x += controlnet_states[i_block]
            # else:
            #     if self.training or self.gradient_checkpointing and use_gradient_checkpointing:
            #         if use_gradient_checkpointing_offload:
            #             with torch.autograd.graph.save_on_cpu():
            #                 x = torch.utils.checkpoint.checkpoint(
            #                     create_custom_forward(block),
            #                     x, context, t_mod, freqs,
            #                     use_reentrant=False,
            #                 )
            #         else:
            #             x = torch.utils.checkpoint.checkpoint(
            #                 create_custom_forward(block),
            #                 x, context, t_mod, freqs,
            #                 use_reentrant=False,
            #             )
            #     else:
            #         x = block(x, context, t_mod, freqs)
            if i_block >= self.camera_layer:
                block_kwargs["cam_emb"] = None # or some other logic
            
            if self.training or self.gradient_checkpointing and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            **block_kwargs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        **block_kwargs,
                        use_reentrant=False,
                    )
            else:
                x = block(**block_kwargs)
            
            if i_block < len(controlnet_states):
                x += controlnet_states[i_block]


        # for i_block, block in enumerate(self.blocks):
        #     if self.training or self.gradient_checkpointing and use_gradient_checkpointing:
        #         if use_gradient_checkpointing_offload:
        #             with torch.autograd.graph.save_on_cpu():
        #                 x = torch.utils.checkpoint.checkpoint(
        #                     create_custom_forward(block),
        #                     x, context, t_mod, freqs, gamma_betas=gamma_betas[i_block],
        #                     use_reentrant=False,
        #                 )
        #         else:
        #             x = torch.utils.checkpoint.checkpoint(
        #                 create_custom_forward(block),
        #                 x, context, t_mod, freqs, gamma_betas[i_block],
        #                 use_reentrant=False,
        #             )
        #     else:
        #         x = block(x, context, t_mod, freqs, gamma_betas[i_block])

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    
    
class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config
    
    def from_civitai(self, state_dict):
        state_dict = {name: param for name, param in state_dict.items() if not name.startswith("vace")}
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6d6ccde6845b95ad9114ab993d917893":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "349723183fc063b2bfc10bb2835cf677":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "efa44cddf936c70abd0ea28b6cbe946c":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "3ef3b1f8e1dab83d5b71fd7b617f859f":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_image_pos_emb": True
            }
        else:
            config = {}
        return state_dict, config