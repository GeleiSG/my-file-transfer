# data process
# CUDA_VISIBLE_DEVICES="0,1,2,3" python wanvideo/train_wan_camera_t2v.py \
#   --task data_process \
#   --dataset_path /mnt/data/camera_datasets/MuteApo/RealCam-Vid \
#   --output_path ./models \
#   --text_encoder_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth" \
#   --vae_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth" \
#   --image_encoder_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
#   --tiled \
#   --num_frames 81 \
#   --height 480 \
#   --width 832
CUDA_VISIBLE_DEVICES="0,1,2,3" python wanvideo/train_wan_camera_t2v.py \
  --task data_process \
  --dataset_path /mnt/data/camera_datasets/MuteApo/RealCam-Vid \
  --output_path ./models \
  --image_encoder_path "/mnt/data/video_public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --text_encoder_path "/mnt/data/video_public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae_path "/mnt/data/video_public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth" \
  --tiled \
  --num_frames 81 \
  --height 480 \
  --width 832
