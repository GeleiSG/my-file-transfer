import os
from tqdm import tqdm
import subprocess
class CameraState():
    def __init__(self):
        self.d_r = 1.0
        self.d_theta = 0.0
        self.d_phi = 0.0
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.z_offset = 0.0
        self.focal_length = 1.0

baseline = CameraState()
r_1_1 , r_1_2 , r_2_1 , r_2_2 = (CameraState() for _ in range(4))
t_1_1 , t_1_2 , t_2_1 , t_2_2 , t_3_1 , t_3_2 , t_4_1 , t_4_2 , t_5_1 , t_5_2 = (CameraState() for _ in range(10))
c_1_1 , c_1_2 , c_2_1 , c_2_2 , c_3_1 , c_3_2 , c_4_1 , c_4_2 = (CameraState() for _ in range(8))
# s_1_1 = s_1_2 = s_2_1 = s_2_2 = s_3_1 = s_3_2 = s_4_1 = s_4_2 = baseline
r_1_1.d_phi = 30.0
r_1_2.d_phi = -30.0
r_2_1.d_theta = -30.0
r_2_2.d_theta = 30.0
t_1_1.x_offset = -0.1
t_1_2.x_offset = 0.1
t_2_1.y_offset = -0.1
t_2_2.y_offset = 0.1
t_3_1.z_offset = 0.2
t_3_2.z_offset = -0.2
t_4_1.d_r = 0.5
t_4_2.d_r = 1.5
t_5_1.focal_length = 1.5
t_5_2.focal_length = 0.5
c_1_1.d_phi = 30.0
c_1_1.d_theta = -30.0
c_1_2.d_phi = -30.0
c_1_2.d_theta = 30.0
c_2_1.d_phi = 8.0
c_2_1.x_offset = 0.15
c_2_2.d_phi = -8.0
c_2_2.x_offset = -0.15
c_3_1.y_offset = -0.1
c_3_1.d_theta = 30.0
c_3_2.y_offset = 0.1
c_3_2.d_theta = -30.0
c_4_1.z_offset = 0.2
c_4_1.x_offset = -0.05
c_4_2.z_offset = -0.2
c_4_2.x_offset = 0.05

cases = [baseline, r_1_1, r_1_2, r_2_1, r_2_2, t_1_1, t_1_2, t_2_1, t_2_2, t_3_1, t_3_2, t_4_1, t_4_2, t_5_1, t_5_2, c_1_1, c_1_2, c_2_1, c_2_2, c_3_1, c_3_2, c_4_1, c_4_2]

i = 0
for video in tqdm(cases):
    prompt = "The video captures a young woman in a bustling Asian market. She is wearing a white and black baseball cap, a white tank top, and a black purse. Her face is lit up with a smile as she looks directly at the camera. The market is filled with people and colorful signs, creating a lively and vibrant atmosphere. The woman's confident and cheerful demeanor stands out against the busy background. The video is shot in a realistic style, capturing the everyday life of the market and the woman's enjoyment of the experience."
    image_path = f'/home2/wbh/Uni3C/data/openvid/Qv_eZHByTJw_9_0to136.png'
    save_path = f'outputs/cases/case{i}'
    current_env = os.environ.copy()
    current_env["CUDA_VISIBLE_DEVICES"] = "0,1"
    gpu_count_val = "2" 
    cmd = [
        # 'torchrun', f'--nproc_per_node={gpu_count_val}',
        'python',
        'cam_render.py',
        '--reference_image', image_path,
        # '--prompt', prompt,
        '--output_path', save_path,
        '--traj_type', 'custom',
        '--d_r', str(video.d_r),
        '--d_theta', str(video.d_theta),
        '--d_phi', str(video.d_phi),
        '--x_offset', str(video.x_offset),
        '--y_offset', str(video.y_offset),
        '--z_offset', str(video.z_offset),
        '--focal_length', str(video.focal_length),
    ]
    print(cmd)

    try:
        result = subprocess.run(cmd, env=current_env, check=True, capture_output=True, text=True)
        print(f"Successfully processed item {i}")
        print("output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error processing item {i}:")
        print("Command:", e.cmd)
        print("Return code:", e.returncode)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
    except FileNotFoundError:
        print(f"Error: torchrun or generate.py not found. Ensure they are in PATH or provide full paths.")
    i = i+1