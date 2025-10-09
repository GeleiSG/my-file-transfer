import json
import os
import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from accelerate import Accelerator
from tqdm import tqdm



# =============== Dataset 定义 ==============
class CameraVideoInferenceDataset(Dataset):
    def __init__(self, base_path, npz_path, processor, messages_create_func, process_vision_info_func):
        metadata = np.load(npz_path, allow_pickle=True)["arr_0"]
        
        self.video_paths = [os.path.join(base_path, entry["video_path"]) for entry in metadata]
        self.processor = processor
        self.messages_create_func = messages_create_func
        self.process_vision_info_func = process_vision_info_func

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        messages = self.messages_create_func(video_path)

        texts = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs, video_kwargs = self.process_vision_info_func(messages, return_video_kwargs=True)

        data = {
            "video_path": video_path,
            "text": texts,
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "video_kwargs": video_kwargs,
        }
        return data

# =============== Collate function ==============
def inference_collate_fn(batch):
    paths = [item["video_path"] for item in batch]
    texts = [item["text"] for item in batch]
    image_inputs = [item["image_inputs"] for item in batch]
    video_inputs = [item["video_inputs"] for item in batch]
    # video_kwargs = batch[0]["video_kwargs"] if "video_kwargs" in batch[0] else {}
    video_kwargs = {}
    if "video_kwargs" in batch[0]:
        for key in batch[0]["video_kwargs"]:
            video_kwargs[key] = [item["video_kwargs"][key][0] for item in batch]
    return {
        "video_path": paths,
        "texts": texts,
        "image_inputs": image_inputs,
        "video_inputs": video_inputs,
        "video_kwargs": video_kwargs,
    }

# =============== Messages create 示例 ==============
def messages_create(video_path):
    messages = [
        {"role": "system", "content": "When describing the camera trajectory in the video, you must tell its category from the following espects. 1.Scene, including Extreme Long Shot, Long Shot, Full Shot, Medium Long Shot, Medium Shot, Cowboy Shot, Medium Close-up, Close-up, Choker Shot, Extreme Close Up, Wide Angle Shot and Panorama Shot. 2.Angle, including Eye Level, Close-Up Shot, Extreme Close-Up Shot, High Angle Shot, Low Angle Shot, Dutch Angle Shot, Corner Shot, Extreme Angle Shot, Inverted Shot and Perspective Shift Shot. 3.Mode of motion, including Crane/Jib Shot, Whip Pan, Arc Shot, Cinematic Dolly-In, Reverse to Dolly-Out, 360-Degree Spin, Low-Angle Tracking Shot, High-Speed Whip Pan, Drone-like Aerial to Ground Level, Steadicam Sprint, Handheld Follow with Sudden Stop and Slow-Mo Pan. 4.Speed change, including Slow Motion, Fast Motion, Time-Lapse, VFX Shot, Freeze Frame Shot, Multi-Camera Perspective, Long Exposure Shot, Low Frame Rate Shot and Extreme Slow Motion Shot. Before describing the actual camera trajectory, make a statement about classifying the camera of the video in the above 4 espects."},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 832 * 480,
                    "fps": 8.0,
                },
                {"type": "text", "text": "Describe the whole moving trajectory of the camera in less than 30 words. Consider camera's whole translation and rotation and its amplitude from camera's perspective. Describe all camera trajectories in time sequence order, without missing anything. If the camera doesn't zoom in or out, the description shouldn't include zoom in or out. The output should be a few compact descriptions of specific camera motions, without generating redundant or fake desciption. The output should be no more than 30 words."},

            ],
        }
    ]
    return messages

# =============== 主函数 ==============
def setup_distributed():
    accelerator = Accelerator()
    return accelerator

def main():
    # 配置
    base_path = "data"           # 填你的路径
    npz_path = "data/RealCam-Vid_DL3DV_test.npz"     # 填你的npz路径
    batch_size = 1
    model_name = "Qwen2.5-VL-7B-Instruct"
    max_new_tokens = 200

    accelerator = setup_distributed()
    device = accelerator.device

    # 加载processor和dataset
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = 'left'

    dataset = CameraVideoInferenceDataset(
        base_path=base_path,
        npz_path=npz_path,
        processor=processor,
        messages_create_func=messages_create,
        process_vision_info_func=process_vision_info
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=inference_collate_fn
    )

    # 加载模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": accelerator.process_index},
    )
    model = accelerator.prepare(model)
    model.eval()

    # 开始推理
    results = []
    for batch in tqdm(dataloader):
        texts = batch["texts"]
        image_inputs = batch["image_inputs"]
        video_inputs = batch["video_inputs"]
        video_kwargs = batch["video_kwargs"]
        

        # 检查是不是全部 image_inputs是 None
        if all(image is None for image in image_inputs):
            inputs = processor(
                text=texts,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
        else:
            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
        

        with torch.no_grad():
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            generated_ids = model.module.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]

            outputs = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # 逐条处理
            
            for i in range(len(outputs)):
                single_result = {
                    "camera_caption": outputs[i],
                    "video_path": batch["video_path"][i]
                }
                results.append(single_result)

    # 全部推理结束后，统一保存
    if accelerator.is_main_process:
        save_path = "data/RealCam-Vid_DL3DV_test.json"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
