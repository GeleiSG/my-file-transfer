from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version as pver
import numpy as np

from diffsynth.models.wan_video_camera_encoder import Camera

def custom_meshgrid(*args):
        # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def get_relative_pose(cam_params):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    cam_to_origin = 0
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses


def ray_condition(K, c2w, H, W, device):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1
    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker


if __name__ == '__main__':
    with open("/home2/wbh/DiffSynth-Studio/camera.txt", 'r') as f:
        poses = f.readlines()
    poses = [pose.strip().split(' ') for pose in poses[1:]]

    cam_params = [[float(x) for x in pose] for pose in poses]
    cam_params = [Camera(cam_param) for cam_param in cam_params]
    image_width = 832
    image_height = 480
    sample_wh_ratio = image_width / image_height
    original_pose_width = cam_params[0].original_pose_width
    original_pose_height = cam_params[0].original_pose_height
    pose_wh_ratio = original_pose_width / original_pose_height
    if pose_wh_ratio > sample_wh_ratio:
        resized_ori_w = image_height * pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fx = resized_ori_w * cam_param.fx / image_width
    else:
        resized_ori_h = image_width / pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fy = resized_ori_h * cam_param.fy / image_height
    intrinsic = np.asarray([[cam_param.fx * image_width,
                                cam_param.fy * image_height,
                                cam_param.cx * image_width,
                                cam_param.cy * image_height]
                            for cam_param in cam_params], dtype=np.float32)

    print(intrinsic.shape)
    K = torch.as_tensor(intrinsic)[None]  # [1, n_frame, 4]
    print(K.shape)
    c2ws = get_relative_pose(cam_params)
    c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]

    plucker_embedding = ray_condition(K, c2ws, image_height, image_width, device='cpu')[0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
    plucker_embedding = plucker_embedding[None]  # B V 6 H W
    print(plucker_embedding.shape)
    plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b c f h w")
    print(plucker_embedding)
    print(plucker_embedding.shape)
