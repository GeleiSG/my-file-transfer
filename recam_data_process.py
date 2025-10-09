import os
import json
import numpy as np
from einops import rearrange
import argparse
from tqdm import tqdm

class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)

def parse_matrix(matrix_str):
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert camera parameters for MultiCamVideo dataset into RealCamVid format")
    parser.add_argument('--root_path', type=str, default='/mnt/data/hdd/yqf_camera_datasets/KwaiVGI/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train', help='Root path of the dataset')
    parser.add_argument('--save_path', type=str, default='/mnt/data/hdd/yqf_camera_datasets/KwaiVGI/MultiCamVideo-Dataset/MultiCamVideo.npz', help='Path to save converted metadata')
    args = parser.parse_args()

    all_metadata = []
    root_path = args.root_path
    save_path = args.save_path
    camera_settings_folder = os.listdir(root_path)
    for camera_setting in tqdm(camera_settings_folder):
        if camera_setting.startswith('f'):
            focal_length = float((camera_setting.split('_')[0]).split('f')[1])
            sensor_height = 23.76
            sensor_width = 23.76
            camera_intrinsic = np.array([focal_length / sensor_width, focal_length / sensor_height, 0.5, 0.5])
            
            camera_settings_path = os.path.join(root_path, camera_setting)
            # if not os.path.isdir(camera_settings_path): continue
            scene_folders = os.listdir(camera_settings_path)
            for scene_folder in tqdm(scene_folders):
                if scene_folder.startswith('scene'):
                    scene_path = os.path.join(camera_settings_path, scene_folder)
                    # if not os.path.isdir(scene_path): continue

                    tgt_camera_path = os.path.join(scene_path, 'cameras', 'camera_extrinsics.json')
                    # if not os.path.exists(tgt_camera_path): continue
                    with open(tgt_camera_path, 'r') as file:
                        cam_data = json.load(file)

                    num_frames = 81

                    # cam_idx = list(range(num_frames))[::4]
                    cam_idx = list(range(num_frames))[::1]

                    for cam_type in range(1, 11):
                        metadata = {}
                        metadata['dataset_source'] = "MultiCamVideo"
                        metadata['video_path'] = os.path.join('MultiCamVideo-Dataset/train', camera_setting, scene_folder, 'videos', f'cam{str(cam_type).zfill(2)}.mp4')
                        metadata['camera_intrinsics'] = camera_intrinsic

                        traj = [parse_matrix(cam_data[f"frame{idx}"][f"cam{int(cam_type):02d}"]) for idx in cam_idx]
                        traj = np.stack(traj).transpose(0, 2, 1)
                        c2ws = []
                        for c2w in traj:
                            c2w = c2w[:, [1, 2, 0, 3]]
                            c2w[:3, 1] *= -1.
                            c2w[:3, 3] /= 100
                            c2ws.append(c2w)
                        tgt_cam_params = [Camera(cam_param) for cam_param in c2ws]

                        normed_rel_w2cs = []

                        scales = [np.linalg.norm(rel.w2c_mat[:3, 3]) for rel in tgt_cam_params]

                        baseline = np.mean(scales)

                        for rel, s in zip(tgt_cam_params, scales):
                            rel_norm = rel.w2c_mat.copy()
                            rel_norm[:3, 3] = rel_norm[:3, 3] / baseline
                            normed_rel_w2cs.append(rel_norm)
                        
                        metadata['camera_extrinsics'] = np.array(normed_rel_w2cs)

                        # metadata['align_factor'] = 1 / baseline
                        metadata['align_factor'] = 0.0
                        # metadata['camera_scale'] = np.max(scales)
                        metadata['camera_scale'] = 0.0
                        metadata['vtss_score'] = 0.0

                        unique_key = f"{camera_setting}_{scene_folder}_cam{str(cam_type).zfill(2)}"
                        all_metadata.append(metadata)

    # np.savez_compressed(save_path, **all_metadata)
    np.savez(
        save_path,
        arr_0=np.array(all_metadata, dtype=object)
    )
                        # np.savez(os.path.join('./data/MultiCamVideo-Dataset/train/f35_aperture2.4/scene20/cameras', 'cam' + str(cam_type).zfill(2) + '.npz'), **metadata)