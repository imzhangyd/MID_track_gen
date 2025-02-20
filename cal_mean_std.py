import numpy as np
import pandas as pd
import ipdb

'''
计算一个视频中轨迹的特征的均值和方差
'''

def calculate_mean_std(file_path, dt=1):
    data = np.loadtxt(file_path, delimiter='\t')
    data = pd.read_csv(file_path, sep='\t', header=None, index_col=None)
    data.columns=['frame_index','track_id','pos_x','pos_y']

    pos_x = data['pos_x'].values
    pos_y = data['pos_y'].values
    
    unique_tracks = np.unique(data['track_id'].values)
    
    velocities = []
    accelerations = []
    
    for track in unique_tracks:
        
        this_track = data[data['track_id']== track]
        this_track = this_track.sort_values('frame_index')

        track_pos_x = this_track['pos_x'].values
        track_pos_y = this_track['pos_y'].values

        if len(track_pos_x) < 2:
            continue  # 忽略长度小于2的轨迹
        
        vel_x = np.diff(track_pos_x) / dt
        vel_y = np.diff(track_pos_y) / dt
        
        acc_x = np.diff(vel_x) / dt
        acc_y = np.diff(vel_y) / dt
        
        if vel_x.max() > 30 or vel_y.max() > 30:
            ipdb.set_trace()
            print(this_track)
        velocities.append(np.column_stack((vel_x, vel_y)))
        accelerations.append(np.column_stack((acc_x, acc_y)))
    
    velocities = np.vstack(velocities)
    accelerations = np.vstack(accelerations)
    
    pos_x_mean = np.mean(pos_x)
    pos_x_std = np.std(pos_x)
    pos_y_mean = np.mean(pos_y)
    pos_y_std = np.std(pos_y)
    
    vel_x_mean = np.mean(velocities[:, 0])
    vel_x_std = np.std(velocities[:, 0])
    vel_y_mean = np.mean(velocities[:, 1])
    vel_y_std = np.std(velocities[:, 1])
    
    acc_x_mean = np.mean(accelerations[:, 0])
    acc_x_std = np.std(accelerations[:, 0])
    acc_y_mean = np.mean(accelerations[:, 1])
    acc_y_std = np.std(accelerations[:, 1])
    
    return {
        'pos_x_mean': pos_x_mean,
        'pos_x_std': pos_x_std,
        'pos_y_mean': pos_y_mean,
        'pos_y_std': pos_y_std,
        'vel_x_mean': vel_x_mean,
        'vel_x_std': vel_x_std,
        'vel_y_mean': vel_y_mean,
        'vel_y_std': vel_y_std,
        'acc_x_mean': acc_x_mean,
        'acc_x_std': acc_x_std,
        'acc_y_mean': acc_y_mean,
        'acc_y_std': acc_y_std
    }

file_path = '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/helab_vesicle/train/vesicle_FP_C1_5.txt'
results = calculate_mean_std(file_path)
print(results)