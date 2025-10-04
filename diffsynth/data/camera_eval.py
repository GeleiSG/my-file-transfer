import torch
# from vggt.vggt.models.vggt import VGGT
# from vggt.vggt.utils.load_fn import load_and_preprocess_images
# from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
import os
import time
from tqdm import tqdm
import pandas as pd
import numpy as np

def predict_cameras(image_folder, save_path, device, dtype):
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    image_names = [os.path.join(image_folder, name) for name in os.listdir(image_folder) if name.endswith('.png')]  
    image_names.sort()

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = load_and_preprocess_images(image_names).to(device)  
            # add batch dimension
            images = images.unsqueeze(0)
            aggregated_tokens_list, ps_idx = model.aggregator(images)
                    
        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        if save_path and os.path.exists(save_path):
            extrinsics_path = os.path.join(save_path, 'extrinsics.txt')
            intrinsics_path = os.path.join(save_path, 'intrinsics.txt')
            with open(extrinsics_path, 'w') as f:
                for i in range(len(extrinsic[0])):
                    f.write(str(i) + ' ' + ' '.join(map(str, extrinsic[0][i].cpu().numpy().flatten())) + '\n')
            with open(intrinsics_path, 'w') as f:
                for i in range(len(intrinsic[0])):
                    f.write(str(i) + ' ' + ' '.join(map(str, intrinsic[0][i].cpu().numpy().flatten())) + '\n')
        return extrinsic, intrinsic

def convert_w2c_to_relative_c2w(w2c_poses):
    """
    Converts a sequence of absolute world-to-camera (w2c) poses to 
    relative camera-to-world (c2w) poses, with the first camera as the origin.
    
    :param w2c_poses: A torch.Tensor of shape (N, 3, 4) representing absolute w2c matrices.
    :return: A torch.Tensor of shape (N, 3, 4) representing relative c2w poses.
    """
    num_poses = w2c_poses.shape[0]
    device = w2c_poses.device

    bottom_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).unsqueeze(0).repeat(num_poses, 1, 1)
    w2c_homogeneous = torch.cat([w2c_poses, bottom_row], dim=1)
    
    c2w_absolute = torch.inverse(w2c_homogeneous)

    abs_to_rel_transform = w2c_homogeneous[0]

    relative_c2w_homogeneous = torch.matmul(abs_to_rel_transform.unsqueeze(0), c2w_absolute)
    
    return relative_c2w_homogeneous[:, :3, :]

def calculate_pose_errors(gt_extrinsic, test_extrinsic):
    """
    Calculate the pose errors between ground truth and predicted extrinsics.
    :param gt_extrinsic: Ground truth extrinsics in relative scale.
    :param test_extrinsic: Predicted extrinsics in relative scale.
    :return: Pose errors including TransErr, RotErr and CamMC.
    """

    # print(f"GT extrinsics shape: {gt_extrinsic.shape}, Test extrinsics shape: {test_extrinsic.shape}")

    t_gt = gt_extrinsic[:, :3, 3]
    t_test = test_extrinsic[:, :3, 3]
    t_gt = torch.from_numpy(t_gt)
    t_test = torch.from_numpy(t_test)
    trans_err = torch.sum(torch.norm(t_gt - t_test, dim=1))
    
    R_gt = gt_extrinsic[:, :3, :3]
    R_test = test_extrinsic[:, :3, :3]
    R_gt = torch.from_numpy(R_gt)
    R_test = torch.from_numpy(R_test)
    R_rel = torch.matmul(R_test, R_gt.mT)
    trace = R_rel.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
    cos_theta = (trace - 1) / 2
    cos_theta_clamped = torch.clamp(cos_theta, -1.0, 1.0)
    rot_err_rad = torch.acos(cos_theta_clamped)
    # rot_err_deg = rot_err_rad * (180.0 / torch.pi)
    rot_err = torch.sum(rot_err_rad)

    cam_mc = torch.sum(torch.norm(gt_extrinsic - test_extrinsic, p='fro', dim=(1, 2)))

    return trans_err, rot_err, cam_mc

def eval_single():
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    total_trans_err = 0.0
    total_rot_err = 0.0
    total_cam_mc = 0.0
    for i in range(10):
        gt_image_folder = f'/home2/wbh/CamI2V/datasets/gt_video/results/gt_frames/video_{i}'
        gt_save_path = None

        gt_extrinsic, _ = predict_cameras(gt_image_folder, gt_save_path, device, dtype)

        # test_image_folder = f'/home2/wbh/CamI2V/datasets/gt_video/results/cami2v_frames/video_{i}'
        test_image_folder = f'/home2/wbh/CamI2V/datasets/gt_video/results/test_frames/video_{i}'
        test_save_path = None

        test_extrinsic, _ = predict_cameras(test_image_folder, test_save_path, device, dtype)

        gt_extrinsic = gt_extrinsic.squeeze(0)
        test_extrinsic = test_extrinsic.squeeze(0)

        gt_extrinsic = gt_extrinsic[:16]
        test_extrinsic = test_extrinsic[:16]

        # print(f"GT extrinsics shape: {gt_extrinsic.shape}, Test extrinsics shape: {test_extrinsic.shape}")
        # return

        gt_extrinsic_rel_c2w = convert_w2c_to_relative_c2w(gt_extrinsic)
        test_extrinsic_rel_c2w = convert_w2c_to_relative_c2w(test_extrinsic)

        trans_err, rot_err, cam_mc = calculate_pose_errors(gt_extrinsic_rel_c2w, test_extrinsic_rel_c2w)
        print(f"Translation Error: {trans_err.item()}")
        print(f"Rotation Error: {rot_err.item()}")
        print(f"Camera Motion Consistency: {cam_mc.item()}")

        total_trans_err += trans_err.item()
        total_rot_err += rot_err.item()
        total_cam_mc += cam_mc.item()

    print(f"Average Translation Error: {total_trans_err / 10}")
    print(f"Average Rotation Error: {total_rot_err / 10}")
    print(f"Average Camera Motion Consistency: {total_cam_mc / 10}")

if __name__ == "__main__":
    # device = "cuda:2" if torch.cuda.is_available() else "cpu"
    # dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # evaluation_folder = './examples/cami2v_eval'
    # gt_folder = os.path.join(evaluation_folder, 'gt_video')
    # test_folder = os.path.join(evaluation_folder, 'test_video')
    # gt_images_folder = os.path.join(evaluation_folder, 'gt_images')
    # test_images_folder = os.path.join(evaluation_folder, 'test_images')
    # video_names = [name for name in os.listdir(gt_folder) if not os.path.isdir(os.path.join(gt_folder, name))]
    # video_names.sort()
    # print(f"Found {len(video_names)} videos for evaluation.")

    # save_results_path = os.path.join(evaluation_folder, 'results.csv')
    # if os.path.exists(save_results_path):
    #     os.remove(save_results_path)
    # with open(save_results_path, 'w') as f:
    #     f.write('video_name,TransErr,RotErr,CamMC\n')
    #     for video_name in tqdm(video_names):
    #         print(f"Evaluating video: {video_name}")

    #         gt_video_path = os.path.join(gt_folder, video_name)
    #         gt_frames_path = os.path.join(gt_images_folder, video_name)
    #         if not os.path.exists(gt_frames_path):
    #             os.makedirs(gt_frames_path)
    #             os.system(f'ffmpeg -i {gt_video_path} -r 10 -f image2 {gt_frames_path}/frame_%03d.png')

    #         test_video_path = os.path.join(test_folder, video_name)
    #         test_frames_path = os.path.join(test_images_folder, video_name)
    #         if not os.path.exists(test_frames_path):
    #             os.makedirs(test_frames_path)
    #             os.system(f'ffmpeg -i {test_video_path} -r 10 -f image2 {test_frames_path}/frame_%03d.png')

    #         gt_extrinsic, _ = predict_cameras(gt_frames_path, None, device, dtype)
    #         test_extrinsic, _ = predict_cameras(test_frames_path, None, device, dtype)

    #         gt_extrinsic = gt_extrinsic.squeeze(0)
    #         test_extrinsic = test_extrinsic.squeeze(0)

    #         gt_extrinsic_rel_c2w = convert_w2c_to_relative_c2w(gt_extrinsic)
    #         test_extrinsic_rel_c2w = convert_w2c_to_relative_c2w(test_extrinsic)

    #         trans_err, rot_err, cam_mc = calculate_pose_errors(gt_extrinsic_rel_c2w, test_extrinsic_rel_c2w)
    #         # print(f"Translation Error: {trans_err.item()}")
    #         # print(f"Rotation Error: {rot_err.item()}")
    #         # print(f"Camera Motion Consistency: {cam_mc.item()}")

    #         f.write(f"{video_name},{trans_err.item()},{rot_err.item()},{cam_mc.item()}\n")
    eval_single()