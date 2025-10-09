import numpy as np
import json
import os
import pickle
import math
import traceback
from tqdm import tqdm
import subprocess

# 【新增】多进程支持库
import multiprocessing
from functools import partial
from PIL import Image
from einops import rearrange

def get_video_meta(video_path: str) -> dict | None:
    """
    使用 ffprobe 高效、可靠地获取视频的分辨率和时长。
    (此函数保持不变)
    """
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,duration,nb_frames',
        '-of', 'json',
        video_path
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        meta_data = json.loads(result.stdout)['streams'][0]

        # --- 分辨率 ---
        width = int(meta_data['width'])
        height = int(meta_data['height'])

        # --- 时长计算 (更稳健的方式) ---
        if 'duration' in meta_data and float(meta_data['duration']) > 0:
            duration = float(meta_data['duration'])
            nb_frames = int(meta_data.get('nb_frames', '0'))
        else:
            num, den = map(int, meta_data.get('r_frame_rate', '0/1').split('/'))
            if den == 0: return None
            fps = num / den
            nb_frames = int(meta_data.get('nb_frames', '0'))
            if fps == 0 or nb_frames == 0: return None
            duration = nb_frames / fps

        saved_fps = float(nb_frames / duration) if duration != 0 else None
        
        # 将nb_frames也返回，方便后续使用
        return {'width': width, 'height': height, 'duration': duration, 'nb_frames': int(meta_data.get('nb_frames', '0')), 'fps' : saved_fps}

    except (FileNotFoundError, subprocess.CalledProcessError, KeyError, IndexError, json.JSONDecodeError):
        # 在多进程中，打印过多会混乱，静默失败即可
        return None

def get_image_meta(file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            frame = Image.open(file_path).convert("RGB")
            # frame = self.crop_and_resize(frame)
            width, height = frame.size
            return {'width': width, 'height': height, 'duration': int(0), 'nb_frames': int(0)}
        return {'width': int(0), 'height': int(0), 'duration': float(0.0), 'nb_frames': int(0)}
        

# 【第一步：重构】将单个条目的处理逻辑封装成一个独立的函数
def process_single_entry(entry, base_path):
    """
    处理单个元数据条目。这个函数将在每个子进程中被调用。
    它接收一个原始的 entry 字典，返回一个处理好的 clean_entry 字典或 None。
    """
    try:
        clean_entry = {}
        
        # 2.1 构建完整文件路径
        video_path_suffix = entry["video_path"]
        if "MiraData9K" in video_path_suffix:
            return None
        if 'MultiCamVideo-Dataset' in video_path_suffix:
            full_path = os.path.join('/mnt/data/camera_datasets_ssd/KwaiVGI/MultiCamVideo-Dataset', video_path_suffix)
        elif 'openhumanvid' in video_path_suffix:
            full_path = os.path.join('/mnt/data/omnihuman', video_path_suffix)
        elif 'mixkit' in video_path_suffix:
            full_path = os.path.join('/mnt/data/hdd/user_workspace/duanke/video_mixkit_81f_26347/output', video_path_suffix)
        else:
            full_path = os.path.join(base_path, video_path_suffix)
            
        clean_entry["path"] = full_path
        
        # 2.2 预先生成渲染视频和掩码的路径
        directory_path, file_name = os.path.dirname(full_path), os.path.basename(full_path)
        file_name_without_ext, ext = os.path.splitext(file_name)
        clean_entry["render_video_path"] = os.path.join(directory_path, f"{file_name_without_ext}_render{ext}")
        clean_entry["render_mask_path"] = os.path.join(directory_path, f"{file_name_without_ext}_render_mask{ext}")

        # 2.3 【关键】验证所有必需的文件是否存在
        if not (os.path.exists(clean_entry["path"]) and \
                os.path.exists(clean_entry["render_video_path"]) and \
                os.path.exists(clean_entry["render_mask_path"])):
            return None
            
        # 2.4 【关键】一次性提取并保存视频元数据
        meta_info = get_video_meta(full_path)
        if meta_info is None:
            meta_info = get_image_meta(full_path)
        clean_entry.update(meta_info)

        # 2.5 保存其他需要的信息
        clean_entry["text"] = entry.get("camera_caption", "")
        clean_entry["video_id"] = entry.get("video_id", [])
        clean_entry["camera_extrinsics"] = entry.get("camera_extrinsics")
        
        # 处理内参，确保维度正确
        intrinsics = entry.get("camera_intrinsics")
        if intrinsics.ndim == 1:
            clean_entry["camera_intrinsics"] = np.repeat(intrinsics[np.newaxis, :], len(clean_entry["camera_extrinsics"]), axis=0)
        else:
            clean_entry["camera_intrinsics"] = intrinsics
        
        return clean_entry
    except Exception:
        # 捕获所有未知错误，确保单个条目的失败不会影响其他进程
        return None


def preprocess_metadata(annotation_paths, base_path, output_file, num_frames=81):
    # 1. 加载所有原始标注文件 (这部分保持不变)
    valid_metadata = []
    for meta_file_path in annotation_paths:
        print(f"正在从 {meta_file_path} 加载标注...")
        if meta_file_path.endswith('.npz'):
            metadata = np.load(meta_file_path, allow_pickle=True)["arr_0"]
            valid_metadata.extend(metadata)
        elif meta_file_path.endswith('.json'):
            extend_data1 = json.load(open(meta_file_path))
            for i, data in enumerate(extend_data1):
                # ... (您加载JSON的详细逻辑保持不变) ...
                update_data = {}
                if "extrinsic_array" not in data or not data["extrinsic_array"]:
                    update_data["camera_extrinsics"] = np.zeros((num_frames,4,4))
                    update_data["camera_intrinsics"] = np.zeros((num_frames,4,4))
                else:
                    extrinsics = []
                    intrinsics = []
                    for i in range(np.array(data["extrinsic_array"]).shape[0]):
                        mat_4x4 = np.eye(4)
                        mat_4x4[:3, :] = np.array(data["extrinsic_array"])[i]
                        fx, fy, cx, cy = np.array(data["intrinsic_array"])[i][0][0], np.array(data["intrinsic_array"])[i][1][1], np.array(data["intrinsic_array"])[i][0][2], np.array(data["intrinsic_array"])[i][1][2] 
                        extrinsics.append(mat_4x4)
                        intrinsics.append(np.array([fx,fy,cx,cy]))
                    update_data["camera_extrinsics"] = np.stack(extrinsics)
                    update_data["camera_intrinsics"] = np.stack(intrinsics)
                
                if 'selected_id' in data:
                    update_data["video_id"] = data['selected_id']
                update_data["video_path"] = data['video_path']
                update_data["camera_caption"] = data.get("caption", data.get("text", ""))
                valid_metadata.append(update_data)

    print(f"原始条目总数: {len(valid_metadata)}")
    
    # 【第二步：并行化】使用多进程池替换原来的 for 循环
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"将使用 {num_workers} 个CPU核心进行并行处理...")
    
    # 使用 functools.partial 来固定 process_single_entry 函数的 base_path 参数
    # 这样 Pool 就可以只传递变化的 'entry' 参数
    task_func = partial(process_single_entry, base_path=base_path)

    final_metadata = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        # 使用 pool.imap_unordered 可以获得最佳性能并能与 tqdm 配合显示进度
        results_iterator = pool.imap_unordered(task_func, valid_metadata)
        
        # 遍历结果，过滤掉处理失败的(None)，并添加到最终列表中
        for result in tqdm(results_iterator, total=len(valid_metadata), desc="并行处理元数据"):
            if result is not None:
                final_metadata.append(result)
        
    print(f"有效条目总数: {len(final_metadata)}")
    
    # 3. 将处理好的干净数据保存到文件 (这部分保持不变)
    print(f"正在将 {len(final_metadata)} 个有效条目保存到 {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(final_metadata, f)
    print("预处理完成！")
    return output_file


if __name__ == '__main__':
    # --- !!! 用户需要修改这里的路径 !!! ---
    ANNOTATION_FILES = [
        "/mnt/data/camera_datasets_ssd/MuteApo/RealCam-Vid/RealCam-Vid_train_camera.npz",
        # 添加你所有的标注文件路径
    ]
    BASE_VIDEO_PATH = "/mnt/data/camera_datasets_ssd/MuteApo/RealCam-Vid" # 提供一个基础路径
    OUTPUT_METADATA_FILE = "/mnt/workspace/processed_metadata_parallel.pkl" # 输出文件的路径
    # --- !!! 修改结束 !!! ---

    preprocess_metadata(ANNOTATION_FILES, BASE_VIDEO_PATH, OUTPUT_METADATA_FILE)