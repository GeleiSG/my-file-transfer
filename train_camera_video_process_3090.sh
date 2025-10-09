# data process
CUDA_VISIBLE_DEVICES="0,1,2,3" python wanvideo/train_wan_camera_t2v.py \
  --task data_process \
  --dataset_path /mnt/data/hdd/datasets/camera_datasets/MuteApo/RealCam-Vid \
  --output_path ./models \
  --text_encoder_path "/mnt/data/hdd/user_workspace/yuqifan/wan21_data_process/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae_path "/mnt/data/hdd/user_workspace/yuqifan/wan21_data_process/Wan2.1_VAE.pth" \
  --image_encoder_path "/mnt/data/hdd/user_workspace/yuqifan/wan21_data_process/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --tiled \
  --num_frames 81 \
  --height 480 \
  --width 832
# CUDA_VISIBLE_DEVICES="0,1,2,3" python wanvideo/train_wan_camera_t2v.py \
#   --task data_process \
#   --dataset_path /mnt/data/hdd/yqf_camera_datasets/KwaiVGI/MultiCamVideo-Dataset \
#   --output_path ./models \
#   --text_encoder_path "/root/hdd/user_workspace/yuqifan/wan21_data_process/models_t5_umt5-xxl-enc-bf16.pth" \
#   --vae_path "/root/hdd/user_workspace/yuqifan/wan21_data_process/Wan2.1_VAE.pth" \
#   --image_encoder_path "/root/hdd/user_workspace/yuqifan/wan21_data_process/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
#   --tiled \
#   --num_frames 81 \
#   --height 480 \
#   --width 832 \
# CUDA_VISIBLE_DEVICES="0,1,2,3" python wanvideo/train_wan_camera_t2v.py \
#   --task data_process \
#   --dataset_path /mnt/data/hdd/yqf_camera_datasets/MuteApo/RealCam-Vid \
#   --output_path ./models \
#   --text_encoder_path "/root/hdd/user_workspace/yuqifan/wan21_data_process/models_t5_umt5-xxl-enc-bf16.pth" \
#   --vae_path "/root/hdd/user_workspace/yuqifan/wan21_data_process/Wan2.1_VAE.pth" \
#   --image_encoder_path "/root/hdd/user_workspace/yuqifan/wan21_data_process/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
#   --tiled \
#   --num_frames 81 \
#   --height 480 \
#   --width 832

