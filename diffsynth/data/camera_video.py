import json
import os
import torch
import random
import numpy as np
import torchvision
import torchvision.transforms.v2 as v2
import imageio
import pandas as pd
from PIL import Image
from einops import rearrange

from diffsynth.data.camera_utils import get_camera_sparse_embedding, get_plucker_embedding, get_relative_pose, ray_condition

class CameraVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, npz_path, extend_paths, steps_per_epoch, max_num_frames=100, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False, is_camera=False):
        metadata = np.load(npz_path, allow_pickle=True)["arr_0"]  # 加载npz，拿arr_0
        print("total number of videos: ", len(metadata))
        # metadata = []
        valid_metadata = []
        for entry in metadata:
            # if 'MiraData9K' not in entry['video_path']:
            if 'game' not in entry['long_caption']:
                valid_metadata.append(entry)
        # merge_data
        if extend_paths is not None:
            for extend_path in extend_paths:
                print(extend_path)
                extend_data1 = json.load(open(extend_path))
                for i, data in enumerate(extend_data1):
                    extrinsics = []
                    intrinsics = []
                    update_data = {}
                    if "extrinsic_array" not in data:
                        update_data["camera_extrinsics"] = np.zeros((num_frames,4,4))
                        update_data["camera_intrinsics"] = np.zeros((num_frames,4))
                    else:
                        for i in range(np.array(data["extrinsic_array"]).shape[0]):
                            mat_4x4 = np.eye(4)
                            mat_4x4[:3, :] = np.array(data["extrinsic_array"])[i]
                            fx, fy, cx, cy = np.array(data["intrinsic_array"])[i][0][0], np.array(data["intrinsic_array"])[i][1][1], np.array(data["intrinsic_array"])[i][0][2], np.array(data["intrinsic_array"])[i][1][2] 
                            extrinsics.append(mat_4x4)
                            intrinsics.append(np.array([fx,fy,cx,cy]))
                        camera_extrinsics = np.stack(extrinsics)
                        camera_intrinsics = np.stack(intrinsics)
                        update_data["camera_extrinsics"] = camera_extrinsics
                        update_data["camera_intrinsics"] = camera_intrinsics
                    if 'selected_id' in data:
                        update_data["video_id"] = data['selected_id']
                    update_data["video_path"] = data['video_path']
                    update_data["camera_caption"] = data["caption"] if 'caption' in data else data["text"]
                    valid_metadata.append(update_data)


        print("update total number of videos: ", len(valid_metadata))
        # if camera_caption_path is not None and os.path.exists(camera_caption_path):
        #     camera_captions = json.load(open(camera_caption_path, "r"))
        #     self.camera_captions = {entry["video_path"]: entry["camera_caption"] for entry in camera_captions}

        # self.text = [entry["long_caption"] +  ' ' + self.camera_captions[entry["video_path"]] if entry["video_path"] in self.camera_captions else entry["long_caption"] for entry in valid_metadata]
        # self.text = []
        # for entry in metadata:
        #     if entry["video_path"] in camera_captions:
        #         self.text.append(entry["long_caption"] + ' ' + self.camera_captions[entry["video_path"]] if random.random() < 0.5 else "")
        #     else:
        #         self.text.append(entry["long_caption"] if random.random() < 0.5 else "") 
        # self.text = [entry["camera_caption"] if "camera_caption" in entry else entry["short_caption"] for entry in valid_metadata]
        # self.text = ["" for entry in valid_metadata]
        # self.text = ["" for entry in valid_metadata]
        self.text = [entry["long_caption"] if random.random() < 0.9 else "" for entry in valid_metadata]

        # if "camera_caption" in valid_metadata[0]:
        #     self.text = [entry["camera_caption"] if random.random() < 0.3 else "" for entry in valid_metadata]
        # else:
        #     self.text = [entry["short_caption"] if random.random() < 0.3 else "" for entry in valid_metadata]
        
        self.path = []
        for entry in valid_metadata:
            if 'MultiCamVideo-Dataset' in entry["video_path"]:
                self.path.append(os.path.join('/mnt/data/camera_datasets/KwaiVGI/MultiCamVideo-Dataset', entry["video_path"]))
            elif 'openhumanvid' in entry["video_path"]:
                self.path.append(os.path.join('/mnt/data/omnihuman', entry["video_path"]))
            elif 'mixkit' in entry["video_path"]:
                self.path.append(os.path.join('/mnt/data/hdd/user_workspace/duanke/video_mixkit_81f_26347/output', entry["video_path"]))
            else:
                self.path.append(os.path.join(base_path, entry["video_path"]))
        # os.path.join('/mnt/data/camera_datasets/KwaiVGI/MultiCamVideo-Dataset', entry["video_path"]) if 'MultiCamVideo-Dataset' in entry["video_path"] else os.path.join(base_path, entry["video_path"]) for entry in valid_metadata
        # self.path = [os.path.join(base_path, entry["video_path"]) if '' for entry in valid_metadata]

        # self.align_factor = [float(entry["align_factor"]) for entry in valid_metadata]
        # self.camera_scale = [float(entry["camera_scale"]) for entry in valid_metadata]
        # self.vtss_score = [float(entry["vtss_score"]) for entry in valid_metadata]
        
        self.camera_extrinsics = [entry["camera_extrinsics"] for entry in valid_metadata]
        self.camera_intrinsics = []
        for entry in valid_metadata:
            if entry["camera_intrinsics"].ndim == 1:
                self.camera_intrinsics.append(np.repeat(entry["camera_intrinsics"][np.newaxis, :], len(entry["camera_extrinsics"]), axis=0))
            else:
                self.camera_intrinsics.append(entry["camera_intrinsics"])
        self.video_id = [entry["video_id"] if "video_id" in entry else [] for entry in valid_metadata]
        # self.camera_intrinsics = [np.repeat(entry["camera_intrinsics"][np.newaxis, :], len(entry["camera_extrinsics"]), axis=0) for entry in metadata] # 这里repeat对每一帧保留一个内参，用于统一不同帧不同内参
        # x = get_plucker_embedding(self.camera_extrinsics[0], self.camera_intrinsics[0], height, width)

        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        self.is_camera = is_camera
        self.steps_per_epoch = steps_per_epoch

        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image

    # def load_frames_using_imageio(self, reader, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
    #     if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
    #         reader.close()
    #         return None
        
    #     frames = []
    #     frames_list = []
    #     first_frame = None
    #     for frame_id in range(num_frames):
    #         frame = reader.get_data(start_frame_id + frame_id * interval)
    #         frames_list.append((start_frame_id + frame_id * interval).item())
    #         frame = Image.fromarray(frame)
    #         frame = self.crop_and_resize(frame)
    #         if first_frame is None:
    #             first_frame = frame
    #         frame = frame_process(frame)
    #         frames.append(frame)
    #     reader.close()

    #     frames = torch.stack(frames, dim=0)
    #     frames = rearrange(frames, "T C H W -> C T H W")
        
    #     first_frame = v2.functional.center_crop(first_frame, output_size=(self.height, self.width))
    #     first_frame = np.array(first_frame)
        
    #     if self.is_i2v:
    #         return frames_list, frames, first_frame
    #     else:
    #         return frames_list, frames

    # def load_video(self, file_path):
    #     reader = imageio.get_reader(file_path)
    #     self.max_num_frames = min(reader.count_frames(), self.max_num_frames)
        
    #     if self.max_num_frames < (self.num_frames - 1) * self.frame_interval + 1:
    #         # self.num_frames = 49  # 8n + 1 形式
    #         self.num_frames = (self.max_num_frames - 1) // 8 * 8 + 1  # 8n + 1 形式
    #         # return None
    #     start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
    #     frames = self.load_frames_using_imageio(reader, file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
    #     return frames

    def load_frames_using_imageio(self, reader, file_path, indices, frame_process, max_num_frames):
        
        frames = []
        frames_list = []
        first_frame = None
        for frame_id in indices:
            frame = reader.get_data(frame_id)
            frames_list.append(frame_id.item())
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = frame
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        
        first_frame = v2.functional.center_crop(first_frame, output_size=(self.height, self.width))
        first_frame = np.array(first_frame)
        
        if self.is_i2v:
            return frames_list, frames, first_frame, max_num_frames
        else:
            return frames_list, frames, max_num_frames

    def load_video(self, file_path, video_id):
        try:
            reader = imageio.get_reader(file_path)
        except Exception as e:
            print(e)
            return None
        max_num_frames = reader.count_frames()
        if len(video_id) != 0:
            indices = []
            for frame_id in video_id:
                indices.append(torch.tensor(frame_id))
        else:
            if self.num_frames > reader.count_frames():
                valid_length = min(reader.count_frames(), (reader.count_frames() - 1) // 4 * 4 + 1)
                indices = np.linspace(0, max_num_frames - 1, valid_length).astype(int)
            else:
                start_frame_id = torch.randint(0, max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
                indices = []
                for frame_id in range(self.num_frames):
                    indices.append(start_frame_id + frame_id * self.frame_interval)
        # print(indices)
        # if 'MiraData9K' in file_path or 'MultiCamVideo' in file_path:
        #     if self.num_frames > reader.count_frames():
        #         indices = np.linspace(0, max_num_frames - 1, self.num_frames).astype(int)
        #     else:
        #         start_frame_id = torch.randint(0, max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        #         indices = []
        #         for frame_id in range(self.num_frames):
        #             indices.append(start_frame_id + frame_id * self.frame_interval)
        # elif 'RealEstate10K' in file_path:
        #     indices = np.linspace(10, max_num_frames - 11, self.num_frames).astype(int)
        # else:
        #     indices = np.linspace(0, max_num_frames - 1, self.num_frames).astype(int)
        frames = self.load_frames_using_imageio(reader, file_path, indices, self.frame_process, max_num_frames)
        return frames
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False

    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame

    def __getitem__(self, index):
        # data_id = torch.randint(0, len(self.path), (1,))[0]
        # data_id = (data_id + index) % len(self.path) # For fixed seed.
        # text = self.text[data_id]
        # path = self.path[data_id]
        
        video = None
        while video is None:
            # data_id = torch.randint(0, len(self.path), (1,))[0]
            # data_id = (data_id + index) % len(self.path)
            data_id = index % len(self.path)
            text = self.text[data_id]
            path = self.path[data_id]
            selected_id = self.video_id[data_id]
            if self.is_image(path):
                if self.is_i2v:
                    raise ValueError(f"{path} is not a video. I2V model doesn't support image-to-image training.")
                video = self.load_image(path)
            else:
                video = self.load_video(path, selected_id)
            if video is None:
                index = random.randint(0, len(self.path)-1)

        
        if self.is_i2v:
            video_id, video, first_frame, max_num_frames = video
            data = {"text": text, "video": video, "path": path, "first_frame": first_frame, "video_id": video_id, "max_num_frames": max_num_frames}
        else:
            video_id, video, max_num_frames = video
            data = {"text": text, "video": video, "path": path, "video_id": video_id, "max_num_frames": max_num_frames}

        if self.is_camera:
            cam_idx = video_id
            if not np.any(self.camera_extrinsics[data_id]):
                data["camera_extrinsics"] = torch.as_tensor(np.zeros_like(self.camera_extrinsics[data_id]))
                data["camera_intrinsics"] = torch.as_tensor(np.zeros_like(self.camera_intrinsics[data_id]))
                data['render_video'] = video
                data['render_mask_video'] = video
                data["video_id"] = video_id
                return data
            # data["camera_embedding"], data["camera_intrinsic"] = get_camera_sparse_embedding(self.camera_extrinsics[data_id][cam_idx], self.camera_intrinsics[data_id][cam_idx], self.height, self.width)
            # data["camera_embedding"] = self.camera_plucker_embeddings[data_id][:, :, video_id, :, :]
            # data["camera_embedding"] = get_plucker_embedding(self.camera_extrinsics[data_id][video_id], self.camera_intrinsics[data_id][video_id], self.height, self.width)
            if self.camera_extrinsics[data_id].shape[0] >= max(cam_idx):
                _, render_video, _, _ = self.load_video(path.replace('.mp4', '_render.mp4'), cam_idx) # render_video
                _, render_mask_video, _, _ = self.load_video(path.replace('.mp4', '_render_mask.mp4'), cam_idx) # render_mask
                data['render_video'] = render_video
                data['render_mask_video'] = render_mask_video
                data["camera_extrinsics"], data["camera_intrinsics"] = get_camera_sparse_embedding(self.camera_extrinsics[data_id], self.camera_intrinsics[data_id], self.height, self.width)
                data["camera_extrinsics"] = data["camera_extrinsics"][cam_idx]
                data["camera_intrinsics"] = data["camera_intrinsics"][cam_idx]
                # data["align_factor"] = self.align_factor[data_id]
                # data["camera_scale"] = self.camera_scale[data_id]
                # data["vtss_score"] = self.vtss_score[data_id]
                data["video_id"] = video_id
            
                return data
            else:
                index = random.randint(0, len(self.path)-1)
                data_id = index % len(self.path)
                text = self.text[data_id]
                path = self.path[data_id]
                if self.is_image(path):
                    if self.is_i2v:
                        raise ValueError(f"{path} is not a video. I2V model doesn't support image-to-image training.")
                    video = self.load_image(path)
                else:
                    video = self.load_video(path, [])
                if self.is_i2v:
                    video_id, video, first_frame, max_num_frames = video
                    data = {"text": text, "video": video, "path": path, "first_frame": first_frame, "video_id": video_id, "max_num_frames": max_num_frames}
                else:
                    video_id, video, max_num_frames = video
                    data = {"text": text, "video": video, "path": path, "video_id": video_id, "max_num_frames": max_num_frames}


                cam_idx = video_id
                data["camera_extrinsics"], data["camera_intrinsics"] = get_camera_sparse_embedding(self.camera_extrinsics[data_id], self.camera_intrinsics[data_id], self.height, self.width)
                data["camera_extrinsics"] = data["camera_extrinsics"][cam_idx]
                data["camera_intrinsics"] = data["camera_intrinsics"][cam_idx]
                _, render_video, _, _ = self.load_video(path.replace('.mp4', '_render.mp4'), cam_idx) # render_video
                _, render_mask_video, _, _ = self.load_video(path.replace('.mp4', '_render_mask.mp4'), cam_idx) # render_mask
                data['render_video'] = render_video
                data['render_mask_video'] = render_mask_video
                data["video_id"] = video_id
            
                return data
    
    def __len__(self):
        if self.steps_per_epoch > 0:
            return self.steps_per_epoch
        else:
            return len(self.path)

if __name__ == "__main__":
    # 加载数据集
    dataset_path = 'data'
    dataset = CameraVideoDataset(
        dataset_path,
        os.path.join(dataset_path, "RealCam-Vid_DL3DV_1K.npz"),
        camera_caption_path=os.path.join(dataset_path, "RealCam-Vid_DL3DV_1K.json"),
        max_num_frames=81,
        frame_interval=1,
        num_frames=81,
        height=480,
        width=832,
        is_i2v=True,     # 根据你的使用情况
        is_camera=True,   # 确保启用 camera 相关字段
    )

