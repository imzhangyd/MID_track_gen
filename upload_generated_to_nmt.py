'''
把生扩散模型生成的轨迹 上传到mnt服务器
进而用于生成跟踪的训练轨迹
'''

import numpy as np
import os
import shutil
import glob


# source_folder = '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/experiments/microtubule_low_future1_sample1_dt1_std323_del_neighbor_label_yst/microtubule_low_epoch90_same_density_2025-02-17_12-26-04/stride_1'
source_folder = '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/experiments/microtubule_mid_future1_sample1_dt1_std323_del_neighbor_label_yst/microtubule_mid_epoch90_same_density_2025-02-18_10-19-01/stride_1'
target_folder = '/mnt/data1/ZYDdata/code/MoTT_private_inout/dataset/diffusion_generated/Microtubule_mid_midmodel/'
# target_folder = '/mnt/data1/ZYDdata/code/MoTT_private_inout/dataset/diffusion_generated/Vesicle_low/'
os.makedirs(target_folder, exist_ok=True)
all_generated = glob.glob(os.path.join(source_folder, '**/generate_tracks.csv'))

for generate_tk in all_generated:
    name = generate_tk.split('/')[-2]
    target_path = os.path.join(target_folder, f'{name}.csv')
    shutil.copy(generate_tk, target_path)

