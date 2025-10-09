import contextlib
import os
import sys

import einops
import kornia
import torch
from pytorch3d.renderer import (
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
)
from pytorch3d.structures import Pointclouds
import numpy as np
import logging
import os

import imageio
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return int(os.environ.get("WORLD_SIZE", 1))


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return int(os.environ.get("RANK", 0))


def get_local_rank():
    if torch.cuda.device_count() == 0:
        print("WARNING: No available GPU.")
        return 0
    return get_rank() % torch.cuda.device_count()


def is_distributed():
    return os.environ.get("WORLD_SIZE", None) is not None


def is_main_process():
    return not is_distributed() or dist.get_rank() == 0


def create_logger(logging_dir=None):
    """
    Create a logger that writes to a log file and stdout.
    """
    if is_main_process():  # real logger
        additional_args = dict()
        if logging_dir is not None:
            additional_args["handlers"] = [
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ]
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            **additional_args,
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def load_video(video_path):
    reader = imageio.get_reader(video_path)
    total_frames = reader.count_frames()
    frames = []
    for i in range(total_frames):
        frame = reader.get_data(i)
        frames.append(Image.fromarray(frame))

    reader.close()

    return frames


def points_padding(points):
    padding = torch.ones_like(points)[..., 0:1]
    points = torch.cat([points, padding], dim=-1)
    return points


def np_points_padding(points):
    padding = np.ones_like(points)[..., 0:1]
    points = np.concatenate([points, padding], axis=-1)
    return points


def txt_interpolation(input_list, n, mode='smooth'):
    x = np.linspace(0, 1, len(input_list))
    if mode == 'smooth':
        f = UnivariateSpline(x, input_list, k=3)
    elif mode == 'linear':
        f = interp1d(x, input_list)
    else:
        raise KeyError(f"Invalid txt interpolation mode: {mode}")
    xnew = np.linspace(0, 1, n)
    ynew = f(xnew)
    return ynew


def traj_map(traj_type):
    # pre-defined trajectories
    if traj_type == "free1":  # Zoom out and rotate to the upper left
        cam_traj = "free"
        x_offset = 0.0
        y_offset = 0.0
        z_offset = 0.0
        d_theta = -15.0
        d_phi = 45.0
        d_r = 1.6
    elif traj_type == "free2":  # Rotate to the right horizontally
        cam_traj = "free"
        x_offset = -0.05
        y_offset = 0.0
        z_offset = 0.0
        d_theta = 0.0
        d_phi = -60.0
        d_r = 1.0
    elif traj_type == "free3":  # Move back to the left
        cam_traj = "free"
        x_offset = -0.25
        y_offset = 0.0
        z_offset = 0.0
        d_theta = 0.0
        d_phi = 0.0
        d_r = 1.7
    elif traj_type == "free4":  # Rotate and approach to the upper right
        cam_traj = "free"
        x_offset = 0.0
        y_offset = 0.0
        z_offset = 0.0
        d_theta = -15.0
        d_phi = -60.0
        d_r = 0.75
    elif traj_type == "free5":  # Large-angle camera movement to the upper right
        cam_traj = "free"
        x_offset = 0.0
        y_offset = 0.0
        z_offset = 0.0
        d_theta = -15.0
        d_phi = -120.0
        d_r = 1.6
    elif traj_type == "swing1":  # Swing shot 1
        cam_traj = "swing1"
        x_offset = 0.0
        y_offset = 0.0
        z_offset = 0.0
        d_theta = 0.0
        d_phi = 0.0
        d_r = 1.0
    elif traj_type == "swing2":  # Swing shot 2
        cam_traj = "swing2"
        x_offset = 0.0
        y_offset = 0.0
        z_offset = 0.0
        d_theta = 0.0
        d_phi = 0.0
        d_r = 1.0
    elif traj_type == "orbit":  # 360-degree counterclockwise rotation
        cam_traj = "free"
        x_offset = 0.0
        y_offset = 0.0
        z_offset = 0.0
        d_theta = 0.0
        d_phi = -360.0
        d_r = 1.0
    else:
        raise NotImplementedError
    return cam_traj, x_offset, y_offset, z_offset, d_theta, d_phi, d_r


def set_initial_camera(start_elevation, radius):
    c2w_0 = torch.tensor([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, -radius],
                          [0, 0, 0, 1]], dtype=torch.float32)
    elevation_rad = np.deg2rad(start_elevation)
    R_elevation = torch.tensor([[1, 0, 0, 0],
                                [0, np.cos(-elevation_rad), -np.sin(-elevation_rad), 0],
                                [0, np.sin(-elevation_rad), np.cos(-elevation_rad), 0],
                                [0, 0, 0, 1]], dtype=torch.float32)
    c2w_0 = R_elevation @ c2w_0
    w2c_0 = c2w_0.inverse()

    return w2c_0, c2w_0


def build_cameras(cam_traj, w2c_0, c2w_0, intrinsic, nframe, focal_length,
                  d_theta, d_phi, d_r, radius, x_offset, y_offset, z_offset):
    # build camera viewpoints according to d_theta，d_phi, d_r
    # return: w2cs:[V,4,4], c2ws:[V,4,4], intrinsic:[V,3,3]
    if intrinsic.ndim == 2:
        intrinsic = intrinsic[None].repeat(nframe, 1, 1)

    c2ws = [c2w_0]
    w2cs = [w2c_0]
    d_thetas, d_phis, d_rs = [], [], []
    x_offsets, y_offsets, z_offsets = [], [], []
    if cam_traj == "free":
        for i in range(nframe - 1):
            coef = (i + 1) / (nframe - 1)
            d_thetas.append(d_theta * coef)
            d_phis.append(d_phi * coef)
            d_rs.append(coef * d_r + (1 - coef) * 1.0)
            x_offsets.append(radius * x_offset * ((i + 1) / nframe))
            y_offsets.append(radius * y_offset * ((i + 1) / nframe))
            z_offsets.append(radius * z_offset * ((i + 1) / nframe))
    elif cam_traj == "swing1":
        phis__ = [0, -5, -25, -30, -20, -8, 0]
        thetas__ = [0, -8, -12, -20, -17, -12, -5, -2, 1, 5, 3, 1, 0]
        rs__ = [0, 0.2]
        d_phis = txt_interpolation(phis__, nframe, mode='smooth')
        d_phis[0] = phis__[0]
        d_phis[-1] = phis__[-1]
        d_thetas = txt_interpolation(thetas__, nframe, mode='smooth')
        d_thetas[0] = thetas__[0]
        d_thetas[-1] = thetas__[-1]
        d_rs = txt_interpolation(rs__, nframe, mode='linear')
        d_rs = 1.0 + d_rs
    elif cam_traj == "swing2":
        phis__ = [0, 5, 25, 30, 20, 10, 0]
        thetas__ = [0, -5, -14, -11, 0, 1, 5, 3, 0]
        rs__ = [0, -0.03, -0.1, -0.2, -0.17, -0.1, 0]
        d_phis = txt_interpolation(phis__, nframe, mode='smooth')
        d_phis[0] = phis__[0]
        d_phis[-1] = phis__[-1]
        d_thetas = txt_interpolation(thetas__, nframe, mode='smooth')
        d_thetas[0] = thetas__[0]
        d_thetas[-1] = thetas__[-1]
        d_rs = txt_interpolation(rs__, nframe, mode='smooth')
        d_rs = 1.0 + d_rs
    else:
        raise NotImplementedError("Unknown trajectory type...")

    for i in range(nframe - 1):
        d_theta_rad = np.deg2rad(d_thetas[i])
        R_theta = torch.tensor([[1, 0, 0, 0],
                                [0, np.cos(d_theta_rad), -np.sin(d_theta_rad), 0],
                                [0, np.sin(d_theta_rad), np.cos(d_theta_rad), 0],
                                [0, 0, 0, 1]], dtype=torch.float32)
        d_phi_rad = np.deg2rad(d_phis[i])
        R_phi = torch.tensor([[np.cos(d_phi_rad), 0, np.sin(d_phi_rad), 0],
                              [0, 1, 0, 0],
                              [-np.sin(d_phi_rad), 0, np.cos(d_phi_rad), 0],
                              [0, 0, 0, 1]], dtype=torch.float32)
        c2w_1 = R_phi @ R_theta @ c2w_0
        if i < len(x_offsets) and i < len(y_offsets) and i < len(z_offsets):
            c2w_1[:3, -1] += torch.tensor([x_offsets[i], y_offsets[i], z_offsets[i]])
        c2w_1[:3, -1] *= d_rs[i]
        w2c_1 = c2w_1.inverse()
        c2ws.append(c2w_1)
        w2cs.append(w2c_1)

        intrinsic[i + 1, :2, :2] = intrinsic[i + 1, :2, :2] * focal_length * ((i + 1) / nframe) + \
                                   intrinsic[i + 1, :2, :2] * ((nframe - (i + 1)) / nframe)

    w2cs = torch.stack(w2cs, dim=0)
    c2ws = torch.stack(c2ws, dim=0)

    return w2cs, c2ws, intrinsic


def rotation_matrix_from_vectors(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    cross_product = np.cross(v1, v2)
    dot_product = np.dot(v1, v2)
    angle = np.arccos(dot_product)

    # special issue
    if np.linalg.norm(cross_product) < 1e-10:  # if is parallel
        if dot_product > 0:
            return np.eye(3)
        else:
            arbitrary_axis = np.array([1, 0, 0]) if np.all(v1 != np.array([1, 0, 0])) else np.array([0, 1, 0])
            return Rotation.from_rotvec(np.pi * arbitrary_axis).as_matrix()

    # Rodrigues formula to get rotation
    r = Rotation.from_rotvec(angle * cross_product / np.linalg.norm(cross_product))
    return r.as_matrix()


def get_boundaries_mask(disparity, sobel_threshold=0.3):
    def sobel_filter(disp, mode="sobel", beta=10.0):
        sobel_grad = kornia.filters.spatial_gradient(disp, mode=mode, normalized=False)
        sobel_mag = torch.sqrt(sobel_grad[:, :, 0, Ellipsis] ** 2 + sobel_grad[:, :, 1, Ellipsis] ** 2)
        alpha = torch.exp(-1.0 * beta * sobel_mag).detach()

        return alpha

    sobel_beta = 10.0
    normalized_disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-6)
    return sobel_filter(normalized_disparity, "sobel", beta=sobel_beta) < sobel_threshold


class PointsZbufRenderer(PointsRenderer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, point_clouds, **kwargs):
        fragments = self.rasterizer(point_clouds, **kwargs)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images, fragments.zbuf


@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout_fd = os.dup(sys.stdout.fileno())
        old_stderr_fd = os.dup(sys.stderr.fileno())
        try:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            os.dup2(old_stdout_fd, sys.stdout.fileno())
            os.dup2(old_stderr_fd, sys.stderr.fileno())
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


def point_rendering(K, w2cs, depth, image, raster_settings, device,
                    background_color=[0, 0, 0], sobel_threshold=0.35, contract=8.0,
                    sam_mask=None):
    """
    only support batchsize=1
    :param K: [F,3,3]
    :param w2cs: [F,4,4] opencv
    :param depth: [1,1,H,W]
    :param images: [1,3,H,W]
    :param background_color: [-1,-1,-1]~[1,1,1]
    :param raster_settings:
    :param sam_mask: [1,1,H,W] 0 or 1
    :return: render_rgbs, render_masks
    """
    nframe = w2cs.shape[0]
    _, _, h, w = image.shape

    # depth contract
    depth = depth.to(device)
    K = K.to(device)
    w2cs = w2cs.to(device)
    image = image.to(device)
    c2ws = w2cs.inverse()

    if depth.max() == 0:
        render_rgbs = torch.zeros((nframe, 3, h, w), device=device, dtype=torch.float32)
        render_masks = torch.ones((nframe, 1, h, w), device=device, dtype=torch.float32)
    else:
        mid_depth = torch.median(depth.reshape(-1), dim=0)[0] * contract
        depth[depth > mid_depth] = ((2 * mid_depth) - (mid_depth ** 2 / (depth[depth > mid_depth] + 1e-6)))

        point_depth = einops.rearrange(depth[0], "c h w -> (h w) c")
        disp = 1 / (depth + 1e-7)
        boundary_mask = get_boundaries_mask(disp, sobel_threshold=sobel_threshold)

        x = torch.arange(w).float() + 0.5
        y = torch.arange(h).float() + 0.5
        points = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1).to(device)
        points = einops.rearrange(points, "w h c -> (h w) c")
        # GPU求逆有错
        points_3d = (c2ws[0] @ points_padding((K[0].cpu().inverse().to(device) @ points_padding(points).T).T * point_depth).T).T[:, :3]

        colors = einops.rearrange(image[0], "c h w -> (h w) c")

        boundary_mask = boundary_mask.reshape(-1)
        if sam_mask is not None:
            sam_mask = sam_mask.reshape(-1)
            boundary_mask[sam_mask == True] = True

        points_3d = points_3d[boundary_mask == False]

        if points_3d.shape[0] <= 8:
            render_rgbs = torch.zeros((nframe, 3, h, w), device=device, dtype=torch.float32)
            render_masks = torch.ones((nframe, 1, h, w), device=device, dtype=torch.float32)
            render_rgbs[0:1] = image
            render_masks[0:1] = 0
            return render_rgbs, render_masks

        colors = colors[boundary_mask == False]

        point_cloud = Pointclouds(points=[points_3d.to(device)], features=[colors.to(device)]).extend(nframe)

        # convert opencv to opengl coordinate
        c2ws[:, :, 0] = - c2ws[:, :, 0]
        c2ws[:, :, 1] = - c2ws[:, :, 1]
        w2cs = c2ws.inverse()

        focal_length = torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=1)
        principal_point = torch.stack([K[:, 0, 2], K[:, 1, 2]], dim=1)
        image_shapes = torch.tensor([[h, w]]).repeat(nframe, 1)
        cameras = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point,
                                     R=c2ws[:, :3, :3], T=w2cs[:, :3, -1], in_ndc=False,
                                     image_size=image_shapes, device=device)

        renderer = PointsZbufRenderer(
            rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
            compositor=AlphaCompositor(background_color=background_color)
        )

        try:
            with suppress_stdout_stderr():
                render_rgbs, zbuf = renderer(point_cloud)  # rgb:[f,h,w,3]
        except Exception as e:
            print(f"Error: {e}")
            print("Error rendering, save pointcloud and other data...")
            torch.save(points_3d, "point_3d_debug.pt")
            torch.save(colors, "colors_debug.pt")
            torch.save(boundary_mask, "boundary_mask_debug.pt")

        render_masks = (zbuf[..., 0:1] == -1).float()  # [f,h,w,1]
        render_rgbs = einops.rearrange(render_rgbs, "f h w c -> f c h w")  # [f,3,h,w]
        render_masks = einops.rearrange(render_masks, "f h w c -> f c h w")  # [f,1,h,w]

    # replace the first frame
    render_rgbs[0:1] = image
    render_masks[0:1] = 0

    return render_rgbs, render_masks

def expand_intrinsics(Fx4_intrinsics):
    """
    将形状为 [F, 4] 的相机内参Tensor扩展为 [F, 3, 3] 的3x3内参矩阵Tensor。
    输入格式假设每行为 [fx, fy, cx, cy]。
    
    参数:
        Fx4_intrinsics: Tensor, 形状为 [F, 4]
    
    返回:
        Fx3x3_intrinsics: Tensor, 形状为 [F, 3, 3]
    """
    F = Fx4_intrinsics.shape[0]
    device = Fx4_intrinsics.device  # 保持设备一致（CPU/GPU）
    
    # 初始化一个全零的 [F, 3, 3] Tensor
    Fx3x3 = torch.zeros((F, 3, 3), device=device)
    
    # 填充参数（利用广播机制）
    Fx3x3[:, 0, 0] = Fx4_intrinsics[:, 0]  # fx
    Fx3x3[:, 1, 1] = Fx4_intrinsics[:, 1]  # fy
    Fx3x3[:, 0, 2] = Fx4_intrinsics[:, 2]  # cx
    Fx3x3[:, 1, 2] = Fx4_intrinsics[:, 3]  # cy
    Fx3x3[:, 2, 2] = 1.0                   # 齐次坐标缩放
    
    return Fx3x3

def point_rendering_train_stage(K, w2cs, depth, images, raster_settings, device,
                    background_color=[0, 0, 0], sobel_threshold=0.35, contract=8.0,
                    sam_mask=None):
    """
    only support batchsize=1
    :param K: [F,3,3]
    :param w2cs: [F,4,4] opencv
    :param depth: [F,H,W,1]
    :param images: [F,3,H,W]
    :param background_color: [-1,-1,-1]~[1,1,1]
    :param raster_settings:
    :param sam_mask: [F,1,H,W] 0 or 1
    :return: render_rgbs, render_masks
    """
    nframe = w2cs.shape[0]
    
    if images.ndim == 3:
        images = images.unsqueeze(0)
    
    if depth.ndim == 2:
        depth = depth.unsqueeze(0).unsqueeze(-1)

    if K.shape[1] == 4 and K.ndim == 2:
        K = expand_intrinsics(K)

    _, _, h, w = images.shape

    depth = einops.rearrange(depth, "b h w c -> b c h w")

    # depth contract
    depth = depth.to(device)
    K = K.to(device)
    w2cs = w2cs.to(device)
    images = images.to(device)
    c2ws = w2cs.inverse()

    focal_lengths = []
    principal_points = []
    all_points_3d = []
    all_colors = []

    if depth.max() == 0:
        render_rgbs = torch.zeros((nframe, 3, h, w), device=device, dtype=torch.float32)
        render_masks = torch.ones((nframe, 1, h, w), device=device, dtype=torch.float32)
    else:
        mid_depth = torch.median(depth.reshape(-1), dim=0)[0] * contract
        depth[depth > mid_depth] = ((2 * mid_depth) - (mid_depth ** 2 / (depth[depth > mid_depth] + 1e-6)))

        for i in range(nframe):
            point_depth = einops.rearrange(depth[0], "c h w -> (h w) c") # 用
            disp = 1 / (depth[0].unsqueeze(0) + 1e-7)
            boundary_mask = get_boundaries_mask(disp, sobel_threshold=sobel_threshold)

            x = torch.arange(w).float() + 0.5
            y = torch.arange(h).float() + 0.5
            points = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1).to(device)
            points = einops.rearrange(points, "w h c -> (h w) c")
            # GPU求逆有错，这里要需要3d效果，因此需要用第一帧的相机参数和所有帧的深度图计算点云。
            points_3d = (c2ws[0] @ points_padding((K[0].cpu().inverse().to(device) @ points_padding(points).T).T * point_depth).T).T[:, :3]

            colors = einops.rearrange(images[0], "c h w -> (h w) c")

            boundary_mask = boundary_mask.reshape(-1)
            if sam_mask is not None:
                sam_mask = sam_mask.reshape(-1)
                boundary_mask[sam_mask == True] = True

            points_3d = points_3d[boundary_mask == False]

            if points_3d.shape[0] <= 8:
                render_rgbs = torch.zeros((nframe, 3, h, w), device=device, dtype=torch.float32)
                render_masks = torch.ones((nframe, 1, h, w), device=device, dtype=torch.float32)
                render_rgbs[0:1] = images
                render_masks[0:1] = 0
                return render_rgbs, render_masks

            colors = colors[boundary_mask == False]

            all_points_3d.append(points_3d)
            all_colors.append(colors)

            focal_lengths.append(torch.stack([K[i, 0, 0].unsqueeze(0), K[i, 1, 1].unsqueeze(0)], dim=1))
            principal_points.append(torch.stack([K[i, 0, 2].unsqueeze(0), K[i, 1, 2].unsqueeze(0)], dim=1))

        point_cloud = Pointclouds(points=all_points_3d, features=all_colors)

        # convert opencv to opengl coordinate
        c2ws[:, :, 0] = - c2ws[:, :, 0]
        c2ws[:, :, 1] = - c2ws[:, :, 1]
        w2cs = c2ws.inverse()

        focal_lengths = torch.cat(focal_lengths, dim=0)
        principal_points = torch.cat(principal_points, dim=0)

        image_shapes = torch.tensor([[h, w]]).repeat(nframe, 1)    
        cameras = PerspectiveCameras(focal_length=focal_lengths, principal_point=principal_points,
                                    R=c2ws[:, :3, :3], T=w2cs[:, :3, -1], in_ndc=False,
                                    image_size=image_shapes, device=device)

        renderer = PointsZbufRenderer(
            rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
            compositor=AlphaCompositor(background_color=background_color)
        )

        try:
            with suppress_stdout_stderr():
                render_rgbs, zbuf = renderer(point_cloud)  # rgb:[f,h,w,3]
        except Exception as e:
            print(f"Error: {e}")
            print("Error rendering, save pointcloud and other data...")
            torch.save(points_3d, "point_3d_debug.pt")
            torch.save(colors, "colors_debug.pt")
            torch.save(boundary_mask, "boundary_mask_debug.pt")

        render_masks = (zbuf[..., 0:1] == -1).float()  # [f,h,w,1]
        render_rgbs = einops.rearrange(render_rgbs, "f h w c -> f c h w")  # [f,3,h,w]
        render_masks = einops.rearrange(render_masks, "f h w c -> f c h w")  # [f,1,h,w]


    # replace the first frame
    render_rgbs[0:1] = images[0]
    render_masks[0:1] = 0

    return render_rgbs, render_masks
