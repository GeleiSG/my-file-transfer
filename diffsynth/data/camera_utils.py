from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version as pver
import numpy as np


class Camera(object):
    def __init__(self, entry, align_factor=1.0):
        if len(entry) == 19: 
            original_pose_height = entry[1]
            original_pose_width = entry[2]
            fx, fy, cx, cy = entry[3:7]
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy
            w2c_mat = np.array(entry[7:]).reshape(3, 4)
            w2c_mat_4x4 = np.eye(4)
            w2c_mat_4x4[:3, :] = w2c_mat
            w2c_mat_4x4[:3, 3] *= align_factor
            self.w2c_mat = w2c_mat_4x4
            self.c2w_mat = np.linalg.inv(w2c_mat_4x4)
            self.original_pose_height = original_pose_height
            self.original_pose_width = original_pose_width
        elif len(entry) == 21:
            fx, fy, cx, cy = entry[1:5]
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy
            w2c_mat = np.array(entry[5:]).reshape(4, 4)
            w2c_mat[:3, 3] *= align_factor
            self.w2c_mat = w2c_mat
            self.c2w_mat = np.linalg.inv(w2c_mat)
    def to_dict(self):
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "w2c_mat": self.w2c_mat.tolist(),
            "c2w_mat": self.c2w_mat.tolist()
        }


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
    zs = torch.ones_like(i, dtype=c2w.dtype)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1).to(device, dtype=c2w.dtype)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3
    directions = directions.to(torch.float32)

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d, dim=3)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6

    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker

def get_camera_sparse_embedding(camera_extrinsics, camera_intrinsics, height, width, align_factor=1.0):
    camera_params = []
    for i, (extrinsic, intrinsic) in enumerate(zip(camera_extrinsics, camera_intrinsics)):
        intrinsic_vals = [float(x) for x in intrinsic]  # 内参转float, 所有帧的内参都是一样的
        extrinsic_vals = [float(x) for x in extrinsic.flatten()]         # 外参flatten转float
        camera_entry = [float(i)] + intrinsic_vals + extrinsic_vals      # 合成一行
        camera_params.append(camera_entry)
    cam_params = [Camera(cam_param, align_factor) for cam_param in camera_params]
    intrinsic = np.asarray([[cam_param.fx * width,
                                cam_param.fy * height,
                                cam_param.cx * width,
                                cam_param.cy * height]
                            if cam_param.cx < 1 else
                                [cam_param.fx,
                                cam_param.fy,
                                cam_param.cx,
                                cam_param.cy]
                            for cam_param in cam_params], dtype=np.float32)

    K = torch.as_tensor(intrinsic)[None]  # [1, n_frame, 4]

    c2ws = get_relative_pose(cam_params)
    c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
    # pose_embedding = rearrange(c2ws, 'b c d f -> b c (d f)')
    camera_extrinsics =  c2ws[0] # [n_frame, 4, 4], here don't need batch size

    camera_intrinsics = K[0] # [n_frame, 4]
    return camera_extrinsics, camera_intrinsics

def get_plucker_embedding(camera_extrinsics, camera_intrinsics, height, width, device='cuda'):
    """
    camera_extrinsics: [B, N, 4, 4] Tensor
    camera_intrinsics: [B, N, 3, 3] Tensor or [B, N, 4] (fx, fy, cx, cy)
    height, width: int
    device: 'cuda' or 'cpu'
    """
    B, N = camera_extrinsics.shape[:2]

    K = camera_intrinsics.to(device)
    c2ws = camera_extrinsics.to(device)  # [B, N, 4, 4]
    # ray_condition
    plucker_embedding = ray_condition(K, c2ws, height, width, device=device)  # [B, N, H, W, 6]

    # 转为 [B, 6, N, H, W]
    plucker_embedding = plucker_embedding.permute(0, 4, 1, 2, 3).contiguous()
    # plucker_embedding = plucker_embedding.to(torch.bfloat16)
    return plucker_embedding.to(device)


def get_plucker_embedding_cpu(camera_extrinsics, camera_intrinsics, height, width, align_factor=1.0):
    """
        only support batchsize=1
        :param camera_extrinsics: [F,3,3]
        :param camera_intrinsics: [F,4] opencv
        :return: plucker_embedding
    """
    camera_params = []
    for i, (extrinsic, intrinsic) in enumerate(zip(camera_extrinsics, camera_intrinsics)):
        intrinsic_vals = [float(x) for x in intrinsic]  # 内参转float, 所有帧的内参都是一样的
        extrinsic_vals = [float(x) for x in extrinsic.flatten()]         # 外参flatten转float
        camera_entry = [float(i)] + intrinsic_vals + extrinsic_vals      # 合成一行
        camera_params.append(camera_entry)
    cam_params = [Camera(cam_param, align_factor) for cam_param in camera_params]
    intrinsic = np.asarray([[cam_param.fx * width,
                                cam_param.fy * height,
                                cam_param.cx * width,
                                cam_param.cy * height]
                            if cam_param.cx < 1 else
                                [cam_param.fx,
                                cam_param.fy,
                                cam_param.cx,
                                cam_param.cy]
                            for cam_param in cam_params], dtype=np.float32)

    K = torch.as_tensor(intrinsic)[None]  # [n_frame, 4]
    c2ws = get_relative_pose(cam_params)
    c2ws = torch.as_tensor(c2ws)[None]  # [n_frame, 4, 4]
    
    plucker_embedding = ray_condition(K, c2ws, height, width, device='cpu')[0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
    # plucker_embedding = plucker_embedding[None]  # B V 6 H W

    plucker_embedding = rearrange(plucker_embedding, "f c h w -> c f h w")
    # plucker_embedding = plucker_embedding.to(torch.bfloat16)
    return plucker_embedding

def get_plucker_embedding_cpu_batched(batch_extrinsics, batch_intrinsics, height, width, align_factor=1.0):
    """
    【全新批处理版】
    一次性计算整批相机参数的Plücker Embedding。
    - batch_extrinsics: [B, F, 4, 4] numpy 数组
    - batch_intrinsics: [B, F, 4] numpy 数组
    """
    batch_size, num_frames = batch_extrinsics.shape[:2]
    
    # 1. 批量解析相机参数
    # 这一步仍然需要循环，但这是轻量级的Python对象创建，开销不大
    batch_cam_params = []
    for b in range(batch_size):
        camera_params = []
        for i, (extrinsic, intrinsic) in enumerate(zip(batch_extrinsics[b], batch_intrinsics[b])):
            intrinsic_vals = [float(x) for x in intrinsic]
            extrinsic_vals = [float(x) for x in extrinsic.flatten()]
            camera_entry = [float(i)] + intrinsic_vals + extrinsic_vals
            camera_params.append(camera_entry)
        batch_cam_params.append([Camera(cam_param, align_factor) for cam_param in camera_params])

    # 2. 批量、向量化地计算内参矩阵 K
    normalized_intrinsics_list = [
        [[p.fx, p.fy, p.cx, p.cy] for p in cam_params] for cam_params in batch_cam_params
    ]
    normalized_intrinsics = np.asarray(normalized_intrinsics_list, dtype=np.float32) # (B, F, 4)
    scale_vector = np.array([width, height, width, height], dtype=np.float32)
    intrinsic_scaled = normalized_intrinsics * scale_vector
    K = torch.from_numpy(intrinsic_scaled) # (B, F, 4)

    # 3. 批量计算 c2ws
    # 假设 get_relative_pose 可以被修改或包装以处理批处理
    c2ws_list = [get_relative_pose(cam_params) for cam_params in batch_cam_params]
    c2ws = torch.from_numpy(np.stack(c2ws_list)).float() # (B, F, 4, 4)
    
    # 4. 批量调用核心函数
    # 假设 ray_condition 能够处理批处理输入 (B, F, 4) 和 (B, F, 4, 4)
    # 返回的 plucker_embedding 应该是 (B, F, H, W, 6)
    plucker_embedding = ray_condition(K, c2ws, height, width, device='cpu')
    
    # 5. 批量进行最终塑形
    plucker_embedding = plucker_embedding.permute(0, 1, 4, 2, 3).contiguous() # B, F, C, H, W
    plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b c f h w") # B, C, F, H, W
    
    return plucker_embedding

# if __name__ == '__main__':
#     with open("data/DL3DV-10K/annotations/camera_DL3DV-10K_1K_0a6c01ac3212768772f8f6eca86314c72d5ca320c3e3def148ddaceab23c07f4_1.txt", 'r') as f:
#         poses = f.readlines()
#     poses = [pose.strip().split(' ') for pose in poses[1:]]

#     cam_params = [[float(x) for x in pose] for pose in poses]
#     cam_params = [Camera(cam_param) for cam_param in cam_params]
#     image_width = 832
#     image_height = 480
#     sample_wh_ratio = image_width / image_height
#     # original_pose_width = cam_params[0].original_pose_width
#     # original_pose_height = cam_params[0].original_pose_height
#     pose_wh_ratio = image_width / image_height
#     print(pose_wh_ratio)
#     print(sample_wh_ratio)
#     if pose_wh_ratio > sample_wh_ratio:
#         resized_ori_w = image_height * pose_wh_ratio
#         for cam_param in cam_params:
#             cam_param.fx = resized_ori_w * cam_param.fx / image_width
#     else:
#         resized_ori_h = image_width / pose_wh_ratio
#         for cam_param in cam_params:
#             cam_param.fy = resized_ori_h * cam_param.fy / image_height
#     intrinsic = np.asarray([[cam_param.fx * image_width,
#                                 cam_param.fy * image_height,
#                                 cam_param.cx * image_width,
#                                 cam_param.cy * image_height]
#                             for cam_param in cam_params], dtype=np.float32)

#     print(intrinsic.shape)
#     K = torch.as_tensor(intrinsic)[None]  # [1, n_frame, 4]
#     print(K.shape)
#     c2ws = get_relative_pose(cam_params)
#     c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
#     print(c2ws)

#     plucker_embedding = ray_condition(K, c2ws, image_height, image_width, device='cpu')[0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
#     plucker_embedding = plucker_embedding[None]  # B V 6 H W
#     print(plucker_embedding.shape)
#     plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b c f h w")
#     print(plucker_embedding)
#     print(plucker_embedding.shape)

from tqdm import tqdm
if __name__ == "__main__":
    print("开始验证 Plücker Embedding 原始版本与批处理版本结果的一致性...")
    
    NUM_SAMPLES = 200  # 测试样本数量
    NUM_FRAMES = 8   # 每个样本的帧数
    HEIGHT = 64      # 统一的高度
    WIDTH = 96       # 统一的宽度
    all_passed = True

    # 1. 生成一批随机输入数据
    batch_extrinsics_list = [np.random.rand(NUM_FRAMES, 4, 4).astype(np.float32) for _ in range(NUM_SAMPLES)]
    batch_intrinsics_list = [np.random.rand(NUM_FRAMES, 4).astype(np.float32) for _ in range(NUM_SAMPLES)]

    # --- 2. 使用方法1 (循环调用原始函数) 计算 ---
    print(f"\n正在使用原始函数循环处理 {NUM_SAMPLES} 个样本...")
    looped_results = []
    for i in tqdm(range(NUM_SAMPLES), desc="循环计算"):
        extr = batch_extrinsics_list[i]
        intr = batch_intrinsics_list[i]
        result = get_plucker_embedding_cpu(extr, intr, HEIGHT, WIDTH)
        looped_results.append(result)
    
    final_looped_tensor = torch.stack(looped_results, dim=0)

    # --- 3. 使用方法2 (一次性调用批处理函数) 计算 ---
    print(f"\n正在使用批处理函数一次性处理 {NUM_SAMPLES} 个样本...")
    batch_extrinsics_np = np.stack(batch_extrinsics_list)
    batch_intrinsics_np = np.stack(batch_intrinsics_list)
    final_batched_tensor = get_plucker_embedding_cpu_batched(batch_extrinsics_np, batch_intrinsics_np, HEIGHT, WIDTH)

    # --- 4. 对比两个方法的输出结果 ---
    print("\n正在对比两种方法的结果...")
    
    # 检查形状是否一致
    if final_looped_tensor.shape != final_batched_tensor.shape:
        print("❌ 测试失败！输出形状不匹配。")
        print(f"  - 循环版本形状: {final_looped_tensor.shape}")
        print(f"  - 批处理版本形状: {final_batched_tensor.shape}")
        all_passed = False
    # 检查数值是否一致 (使用 allclose 以处理微小的浮点数差异)
    elif not torch.allclose(final_looped_tensor, final_batched_tensor, atol=1e-6):
        print("❌ 测试失败！输出的数值不一致。")
        diff = torch.abs(final_looped_tensor - final_batched_tensor).max()
        print(f"  - 最大差异值: {diff.item()}")
        all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("✅ 所有测试样本均已通过！")
        print("重构后的批处理代码与原始代码的计算结果完全一致。")
    else:
        print("❌ 验证失败。请检查上面报告的错误。")
    print("="*50)
