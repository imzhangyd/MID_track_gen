'''
把生扩散模型生成的轨迹 上传到mnt服务器
进而用于生成跟踪的训练轨迹
'''

import numpy as np
import os
import shutil
import glob


source_folder = '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/experiments/helab_vesicle_future1_sample1_dt1_std322_del_neighbor_label_yst/helab_vesicle_epoch90_length8_2025-02-20_11-53-03/stride_5'
target_folder = '/mnt/data1/ZYDdata/code/MoTT_private_inout/dataset/diffusion_generated/helab_vesicle/'
# target_folder = '/mnt/data1/ZYDdata/code/MoTT_private_inout/dataset/diffusion_generated/Vesicle_low/'
os.makedirs(target_folder, exist_ok=True)
all_generated = glob.glob(os.path.join(source_folder, '**/**/generate_tracks.csv'))

for generate_tk in all_generated:
    basemv = generate_tk.split('/')[-3]
    genname = generate_tk.split('/')[-2]
    name = basemv+'_'+genname
    target_path = os.path.join(target_folder, f'{name}.csv')
    shutil.copy(generate_tk, target_path)

