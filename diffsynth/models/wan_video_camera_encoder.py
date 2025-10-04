import math
import torch
import torch.nn as nn
from einops import rearrange


def get_parameter_dtype(parameter: torch.nn.Module):
    try:
        params = tuple(parameter.parameters())
        if len(params) > 0:
            return params[0].dtype

        buffers = tuple(parameter.buffers())
        if len(buffers) > 0:
            return buffers[0].dtype

    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class PoseAdaptor(nn.Module):
    def __init__(self, unet, pose_encoder):
        super().__init__()
        self.unet = unet
        self.pose_encoder = pose_encoder

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, pose_embedding):
        assert pose_embedding.ndim == 5
        bs = pose_embedding.shape[0]            # b c f h w
        pose_embedding_features = self.pose_encoder(pose_embedding)      # bf c h w
        pose_embedding_features = [rearrange(x, '(b f) c h w -> b c f h w', b=bs)
                                   for x in pose_embedding_features]
        noise_pred = self.unet(noisy_latents,
                               timesteps,
                               encoder_hidden_states,
                               pose_embedding_features=pose_embedding_features).sample
        return noise_pred


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)
    
class Downsample3d(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=3, out_channels=None, padding=1, time_downsample=False, spatial_downsample=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if spatial_downsample:
            self.op = conv_nd(2, self.channels, self.out_channels, 3, stride=2, padding=padding)
        else:
            self.op = nn.Identity()
        self.time_downsample = time_downsample 
        if time_downsample: 
            self.time_op = conv_nd(1, self.channels, self.out_channels, 3, stride=2, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        f = x.shape[2]
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.op(x)
        if self.time_downsample:
            _h, _w = x.shape[2], x.shape[3]
            x = rearrange(x, "(b f) c h w -> (b h w) c f", f=f)
            x = self.time_op(x)
            x = rearrange(x, "(b h w) c f -> b c f h w", h=_h, w=_w)
        else:
            x = rearrange(x, "(b f) c h w -> b c f h w", f=f)
        return x


class ResnetBlock(nn.Module):

    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True, time_downsample=False, spatial_downsample=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        # self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1) 
        self.block1_1 = nn.Conv2d(out_c, out_c, ksize, 1, ps) # spatial
        self.block1_2 = nn.Conv1d(out_c, out_c, ksize, 1, ps) # temporal
        self.act = nn.ReLU()
        self.block2_1 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        self.block2_2 = nn.Conv1d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample3d(in_c, use_conv=use_conv, time_downsample=time_downsample, spatial_downsample=True)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            f = x.shape[2]
            x = rearrange(x, "b c f h w -> (b f) c h w")
            x = self.in_conv(x)
            x = rearrange(x, "(b f) c h w -> b c f h w", f=f)

        f = x.shape[2]
        x = rearrange(x, "b c f h w -> (b f) c h w")
        h = self.block1_1(x)
        _h, _w = h.shape[2], h.shape[3]
        h = rearrange(h, "(b f) c h w -> (b h w) c f", f=f)
        h = self.block1_2(h)

        h = rearrange(h, "(b h w) c f -> (b f) c h w", h=_h, w=_w)
        h = self.act(h)
        h = self.block2_1(h)
        _h, _w = h.shape[2], h.shape[3]
        h = rearrange(h, "(b f) c h w -> (b h w) c f", f=f)
        h = self.block2_2(h)
        h = rearrange(h, "(b h w) c f -> (b f) c h w", h=_h, w=_w)

        if self.skep is not None:
            h = h + self.skep(x)
        else:
            h =  h + x
        h = rearrange(h, "(b f) c h w -> b c f h w", f=f)
        return h


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            d_model,
            dropout=0.,
            max_len=32,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2, ...] = torch.sin(position * div_term)
        pe[0, :, 1::2, ...] = torch.cos(position * div_term)
        pe.unsqueeze_(-1).unsqueeze_(-1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), ...]
        return self.dropout(x)


class CameraPoseEncoder(nn.Module):

    def __init__(self,
                 downscale_factor,
                 channels=[64, 128, 320, 320],
                 nums_rb=3,
                 cin=64,
                 cout=1920,
                 ksize=3,
                 sk=False,
                 use_conv=True,
                 compression_factor=1,
                ):
        super(CameraPoseEncoder, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.channels = channels
        self.nums_rb = nums_rb
        self.encoder_down_conv_blocks = nn.ModuleList()
        
        for i in range(len(channels)):
            conv_layers = nn.ModuleList()
            for j in range(nums_rb):
                if j == 0 and i != 0:
                    in_dim = channels[i - 1]
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(in_dim, out_dim, down=True, ksize=ksize, sk=sk, use_conv=use_conv, time_downsample=(i!=len(channels)-1))
                elif j == 0:
                    in_dim = channels[0]
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv)
                elif j == nums_rb - 1:
                    in_dim = channels[i] / compression_factor
                    out_dim = channels[i]
                    conv_layer = ResnetBlock(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv)
                else:
                    in_dim = int(channels[i] / compression_factor)
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv)
                
                conv_layers.append(conv_layer)
                
            self.encoder_down_conv_blocks.append(conv_layers)

        self.encoder_conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)
        self.out_down = nn.AdaptiveAvgPool2d((4,4))
        self.out_conv = nn.Conv2d(channels[-1], cout, 1, 1)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def forward(self, x):
        # unshuffle
        bs = x.shape[0]
        f = x.shape[2]
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.unshuffle(x)
        x = self.encoder_conv_in(x)
        
        x = rearrange(x, "(b f) c h w -> b c f h w", f=f) 
        
        for res_block in self.encoder_down_conv_blocks:
            for res_layer in res_block:
                x = res_layer(x)
        f = x.shape[2]
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.out_down(x)
        x = self.out_conv(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=f)
        return x

class Interploate(nn.Module):
    def __init__(self, n_frames, height, width):
        super().__init__()
        self.n_frames = n_frames
        self.height = height
        self.width = width
    def forward(self, x):
        # x: b c f h w -> b c self.n_frame self.height self.width
        x = nn.functional.interpolate(x, size=(self.n_frames, self.height, self.width), mode='trilinear')
        return x

class CameraPoseControlNet(nn.Module):
    def __init__(self, 
                 downscale_factor,
                 channels=[64, 128, 320],
                 cin=64,
                 cout=1920,
                 ksize=3,
                 sk=False,
                 use_conv=True,
                 n_blocks=40,
        ):
        super(CameraPoseControlNet, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor) # downsample to 8*8
        self.channels = channels

        self.downsample_blocks = nn.Sequential(
            ResnetBlock(channels[0], channels[1], down=True, ksize=ksize, sk=sk, use_conv=use_conv, time_downsample=True, spatial_downsample=True),
            ResnetBlock(channels[1], channels[2], down=True, ksize=ksize, sk=sk, use_conv=use_conv, time_downsample=True, spatial_downsample=True),
        )
        self.blocks = nn.ModuleList()
        
        for i in range(n_blocks):
            layer = nn.Sequential(
                ResnetBlock(channels[2], channels[2], down=False, ksize=ksize, sk=sk, use_conv=use_conv, time_downsample=False, spatial_downsample=False),
            )
            self.blocks.append(layer)
        
        self.gamma_beta_blocks = nn.ModuleList()
        for i in range(n_blocks):
            

            gemma_beta_layer = nn.ModuleList([
                ResnetBlock(channels[2], channels[2], down=False, ksize=ksize, sk=sk, use_conv=use_conv, time_downsample=False, spatial_downsample=False),
                nn.Conv3d(channels[2], cout*2, 1, 1, 0),
            ])
            self.gamma_beta_blocks.append(gemma_beta_layer)

        self.encoder_conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)
        for i in range(len(channels)):
            self.gamma_beta_blocks[i][1].weight.data.zero_()
            self.gamma_beta_blocks[i][1].bias.data.zero_()

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def forward(self, x, n_frames, height, width):
        # unshuffle
        bs = x.shape[0]
        f = x.shape[2]
        x = rearrange(x, "b c f h w -> (b f) c h w")
        
        x = self.unshuffle(x)
        x = self.encoder_conv_in(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=f)
        
        x = self.downsample_blocks(x)
        gamma_betas = []
        for layer, gamma_beta_layer in zip(self.blocks, self.gamma_beta_blocks):
            x = layer(x)
            gamma_beta = x
            for _gamma_beta in gamma_beta_layer:
                gamma_beta = _gamma_beta(gamma_beta) # b c f h w
            gamma_beta = nn.functional.interpolate(gamma_beta, size=(n_frames, height, width), mode='trilinear')
            gamma_beta = rearrange(gamma_beta, "b c f h w -> b (f h w) c")
            gamma_betas.append(gamma_beta)
        return gamma_betas


if __name__ == "__main__":
    # Test the CameraPoseEncoder
    camera_encoder = CameraPoseEncoder(
        downscale_factor = 8,
        channels = [320, 640, 1280, 1280],
        nums_rb = 2,
        cin = 384,
        cout = 32,
        ksize = 1,
        sk = True,
        use_conv = True,
        compression_factor = 1
    )
    camera_encoder = camera_encoder.to("cuda", dtype=torch.float16)
    x = torch.randn(1, 6, 81, 832, 480).cuda().to(torch.float16)  # Example input tensor
    # out = camera_encoder(x)
    # # out = out + 3
    # print(out.shape)  # Expected output shape: (1, 1920, 21, 4, 4)
    # camera_interpolator = Interploate(81, 832, 480)
    # out = camera_interpolator(out)
    # print(out.shape)  # Expected output shape: (1, 1920, 21, 104, 60)
    
    camera_adapter = CameraPoseControlNet( 
            downscale_factor=8,
            channels=[64, 128, 320],
            cin=6*8*8,
            cout=1536,
            ksize=3,
            sk=True,
            use_conv=True,
            n_blocks=4,
    ).to("cuda", dtype=torch.float16)

    gamma_betas = camera_adapter(x, 21, 52, 30)
    print(gamma_betas[0].shape)  # Expected output shape: (1, 3840, 21, 104, 60)
    gamma_beta_split = torch.split(gamma_betas[0], 1536, dim=2)
    gamma = gamma_beta_split[0]
    beta = gamma_beta_split[1]
    print(gamma.shape)  # Expected output shape: (1, 1920, 21, 104, 60)
    print(beta.shape)  # Expected output shape: (1, 1920, 21, 104, 60)
    content_embedding = torch.randn(1, 1536, 21, 52, 30).cuda().to(torch.float16)
    content_embedding = rearrange(content_embedding, "b c f h w -> b (f h w) c")
    print(content_embedding.shape)  # Expected output shape: (1, 131040, 16)
    adjusted_content_embedding = (gamma + 1) * content_embedding + beta
    print(adjusted_content_embedding.shape)  # Expected output shape: (1, 16, 21, 104, 60)
    ccc
    out.mean().backward()
    print('backward done')
    