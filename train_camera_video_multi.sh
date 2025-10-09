# data process
# CUDA_VISIBLE_DEVICES="0" python wanvideo/train_wan_camera_t2v.py \
#   --task data_process \
#   --dataset_path /root/hdd/yqf_camera_datasets/MuteApo/RealCam-Vid \
#   --output_path ./models \
#   --text_encoder_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth" \
#   --vae_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth" \
#   --image_encoder_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
#   --tiled \
#   --num_frames 81 \
#   --height 480 \
#   --width 832

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python wanvideo/train_wan_camera_t2v.py \
#   --task train \
#   --train_architecture lora \
#   --dataset_path /mnt/data/hdd/yqf_camera_datasets/MuteApo/RealCam-Vid \
#   --output_path ./models \
#   --dit_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors" \
#   --image_encoder_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
#   --text_encoder_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth" \
#   --vae_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth" \
#   --steps_per_epoch 5000 \
#   --max_epochs 10 \
#   --learning_rate 1e-4 \
#   --lora_rank 16 \
#   --lora_alpha 16 \
#   --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
#   --accumulate_grad_batches 1 \
#   --use_gradient_checkpointing \



# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python wanvideo/train_wan_camera_t2v.py \
#   --task train \
#   --train_architecture full \
#   --dataset_path /mnt/data/hdd/yqf_camera_datasets/MuteApo/RealCam-Vid \
#   --output_path ./models \
#   --dit_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors" \
#   --image_encoder_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
#   --text_encoder_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth" \
#   --vae_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth" \
#   --steps_per_epoch 5000 \
#   --max_epochs 10 \
#   --learning_rate 1e-4 \
#   --accumulate_grad_batches 1 \
#   --use_gradient_checkpointing


# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python wanvideo/train_wan_camera_t2v.py \
#   --task train \
#   --train_architecture adapter \
#   --dataset_path /mnt/data/yqf_camera_datasets/MuteApo/RealCam-Vid \
#   --output_path ./models_multi \
#   --dit_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors" \
#   --image_encoder_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
#   --text_encoder_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth" \
#   --vae_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth" \
#   --steps_per_epoch 500 \
#   --max_epochs 10 \
#   --learning_rate 1e-4 \
#   --accumulate_grad_batches 1 \
#   --use_gradient_checkpointing

CUDA_VISIBLE_DEVICES="0,1,2,3" python wanvideo/train_wan_camera_t2v.py \
  --task train \
  --train_architecture lora \
  --dataset_path /mnt/data/hdd/yqf_camera_datasets/MuteApo/RealCam-Vid \
  --output_path /mnt/data/hdd/yqf/cameractrl_ckpt \
  --dit_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors,/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors,/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors,/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors,/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors,/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors,/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors" \
  --image_encoder_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --text_encoder_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae_path "/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth" \
  --steps_per_epoch 10000 \
  --per_device_batch_size 1 \
  --dataloader_num_workers 8 \
  --max_epochs 10 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 16 \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --accumulate_grad_batches 4 \
  --use_gradient_checkpointing \
