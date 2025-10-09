import json
import cv2
import os
import random
from tqdm import tqdm

def extract_and_save_frame(video_path, frame_number, output_folder, video_id, video_sub_id):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    # 设置帧位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    if ret:
        frame_filename = os.path.join(output_folder, f"{video_id}_{frame_number:03d}_00{video_sub_id}.jpg")
        cv2.imwrite(frame_filename, frame)
        cap.release()
        return frame_filename
    else:
        cap.release()
        return None

def process_video_folders(base_folder):
    video_pairs = []
    # 获取首帧文件夹和尾帧文件夹
    first_frame_folder = '/root/hdd/yqf_camera_datasets/caizhongang/HuMMan/humman_first_frames'
    last_frame_folder = '/root/hdd/yqf_camera_datasets/caizhongang/HuMMan/humman_last_frames'

    # 确保文件夹存在
    os.makedirs(first_frame_folder, exist_ok=True)
    os.makedirs(last_frame_folder, exist_ok=True)
    # 遍历目录中的子文件夹
    for root, dirs, files in tqdm(os.walk(base_folder)):
        # 寻找包含 'kinect_color' 的子文件夹
        for dir_name in dirs:
            if 'kinect_color' in dir_name:
                video_folder = os.path.join(root, dir_name)  # 例如 p000438_a000075
                # 01345689，是顺时针依次45度
                valid_pairs = [
                    [0,3,'The camera is moving in a circular trajectory at eye level steadily to the left, with a rotated angle of 90 degrees.'],  # left 90
                    [3,5,'The camera is moving in a circular trajectory at eye level steadily to the left, with a rotated angle of 90 degrees.'], # left 90-180
                    [0,8,'The camera is moving in a circular trajectory at eye level steadily to the right, with a rotated angle of 90 degrees.'], # right 90
                    [8,5,'The camera is moving in a circular trajectory at eye level steadily to the right, with a rotated angle of 90 degrees.'], # right 90-180
                    [3,2,'The camera gradually rises, providing a top-down perspective.'], # left top
                    [8,7,'The camera gradually rises, providing a top-down perspective.'], # right top 
                ]
                for valid_pair in valid_pairs:
                    video_id_0 = valid_pair[0]
                    video_id_1 = valid_pair[1]
                    video_path_0 = os.path.join(video_folder, f'kinect_00{video_id_0}.mp4')
                    video_path_1 = os.path.join(video_folder, f'kinect_00{video_id_1}.mp4')
                    video_id = video_folder.split('/')[-2]  # 获取视频序号（如 p000438_a000075）
                    
                    # 获取视频的总帧数
                    cap = cv2.VideoCapture(video_path_0)
                    total_frames_0 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()

                    cap = cv2.VideoCapture(video_path_1)
                    total_frames_1 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()

                    if total_frames_0 != total_frames_1:
                        continue

                    # 随机选择帧序号（范围在0到total_frames-1之间）
                    frame_number = random.randint(0, total_frames_0 - 1)


                    # 保存选定帧到对应的文件夹
                    frame_1_path = extract_and_save_frame(video_path_0, frame_number, first_frame_folder, video_id, video_id_0)
                    frame_2_path = extract_and_save_frame(video_path_1, frame_number, last_frame_folder, video_id, video_id_1)

                    if frame_1_path and frame_2_path:
                        video_pairs.append((video_id, frame_1_path, frame_2_path, valid_pair[2]))

    return video_pairs


base_folder = '/root/hdd/yqf_camera_datasets/caizhongang/HuMMan/humman_release_v1.0_point'
video_pairs = process_video_folders(base_folder)

# 输出视频对及其帧
with open('humman_flf2v_frame_ids.json', 'w') as f:
    json.dump(video_pairs, f, indent=4)
