import bisect
import logging
import torch
import numpy as np
import random
from typing import Dict, List

def get_logger(name = None):
    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.INFO)
    return logger

def get_closest_ratio_key(height: float, width: float, ratios_dict: Dict[str, List[float]]) -> str:
    """
    计算给定高宽的最接近的预设宽高比，并返回其在字典中的键。

    Args:
        height (float): 图像或视频帧的高度。
        width (float): 图像或视频帧的宽度。
        ratios_dict (Dict[str, List[float]]): 预设的宽高比字典，键是字符串形式的比例（如 "0.5"），值是[高, 宽]。

    Returns:
        str: 在字典中代表最接近的宽高比的键（字符串形式）。
    """
    aspect_ratio = height / width
    # 将字典的键（字符串）转换为浮点数进行比较
    float_keys = {float(k): k for k in ratios_dict.keys()}
    closest_float_key = min(float_keys.keys(), key=lambda r: abs(r - aspect_ratio))
    return float_keys[closest_float_key]

def get_duration_bin_index(duration: float, video_duration_bins: List[float]) -> int:
    """根据视频时长，计算其所属的桶索引。"""
    #plan b
    # for bin in video_duration_bins:
    #     if duration <= bin:
    #         return bin
    # return max(self.video_duration_bins) 

    # plan a    
    # bisect_right 能够高效地找到插入点，该插入点即为桶的索引
    # 例如 bins=[2, 4, 6], duration=3.5, bisect_right 返回 1, 也就是索引为1的桶 (2s, 4s]
    # duration=8, 返回 3, 也就是索引为3的桶 (6s, +∞)
    return bisect.bisect_right(video_duration_bins, duration)
    
def get_random_mask(shape, image_start_only=True, image_end=False):
    f, c, h, w = shape
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)
    if f != 1:
        if image_start_only and not image_end:
            mask[1:, :, :, :] = 1
        elif image_start_only and image_end:
            mask[1:-1, :, :, :] = 1
        else:
            assert image_start_only == True
    else:
        mask[:, :, :, :] = 1
    return mask

