'''
衡量生成轨迹与真实轨迹的相似度

msd曲线
速度分布
转角分布
一致性分布
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.spatial import distance
# from sklearn.metrics import mean_squared_error

# Set the font family to Arial
plt.rcParams['font.family'] = 'Arial'
# Set the font size to 6
plt.rcParams['font.size'] = 6
# Set the axes linewidth to 0.5
plt.rcParams['axes.linewidth'] = 0.5
# Set the major xtick width to 0.5
plt.rcParams['xtick.major.width'] = 0.5
# Set the major ytick width to 0.5
plt.rcParams['ytick.major.width'] = 0.5
# Set the minor xtick width to 0.5
plt.rcParams['xtick.minor.width'] = 0.5
# Set the minor ytick width to 0.5
plt.rcParams['ytick.minor.width'] = 0.5
# Set the xtick direction to 'in'
plt.rcParams['xtick.direction'] = 'in'
# Set the ytick direction to 'in'
plt.rcParams['ytick.direction'] = 'in'
# Disable the top spine
plt.rcParams['axes.spines.top'] = False
# Disable the right spine
plt.rcParams['axes.spines.right'] = False
# Enable the left spine
plt.rcParams['axes.spines.left'] = True
# Enable the bottom spine
plt.rcParams['axes.spines.bottom'] = True
# Set the axes label size to 6
plt.rcParams['axes.labelsize'] = 6
# Set the legend font size to 6
# plt.rcParams['legend.fontsize'] = 6
# Disable the legend frame
# plt.rcParams['legend.frameon'] = False
# Set the legend handle length to 1
# plt.rcParams['legend.handlelength'] = 1
# Set the legend handle text padding to 0.5
# plt.rcParams['legend.handletextpad'] = 0.5
# Set the legend label spacing to 0.5
# plt.rcParams['legend.labelspacing'] = 0.5
# Set the legend location to 'upper right'
# plt.rcParams['legend.loc'] = 'upper left'
# Set the lines linewidth to 0.5
plt.rcParams['lines.linewidth'] = 0.5
# Set the lines markersize to 2
plt.rcParams['lines.markersize'] = 2
# Set the lines marker to 'o'
plt.rcParams['lines.marker'] = 'o'
# Set the lines marker edge width to 0.5
plt.rcParams['lines.markeredgewidth'] = 0.5
# Set the figure DPI to 450
plt.rcParams['figure.dpi'] = 450
# Set the figure size (convert mm to inches)
plt.rcParams['figure.figsize'] = (160/25.4, 50/25.4)
# Set the savefig DPI to 450
plt.rcParams['savefig.dpi'] = 450
# Set the savefig format to 'pdf'
plt.rcParams['savefig.format'] = 'pdf'
# Set the savefig bbox to 'tight'
plt.rcParams['savefig.bbox'] = 'tight'
# Set the savefig pad inches to 0.05
plt.rcParams['savefig.pad_inches'] = 0.05
# Set the PDF font type to 42
plt.rcParams['pdf.fonttype'] = 42
# Set the PDF compression to 9
plt.rcParams['pdf.compression'] = 9
# Use 14 core fonts in PDF
plt.rcParams['pdf.use14corefonts'] = True
# Do not inherit color in PDF
plt.rcParams['pdf.inheritcolor'] = False

def compute_msd_one_track(df):
    '''
    对比 msd_real 和 msd_generated，如果 MSD 曲线形状一致，则运动模式相似。
    '''
    msd_dict = {}
    track_groups = df.groupby("track_id")
    
    for track_id, track in track_groups:
        positions = track[["pos_x", "pos_y"]].values
        frame_count = len(positions)
        msd = []
        
        for tau in range(1, frame_count):
            displacements = positions[tau:] - positions[:-tau]
            squared_displacements = np.sum(displacements**2, axis=1)
            msd.append(np.mean(squared_displacements))
        
        msd_dict[track_id] = msd
    
    return msd_dict


def compute_msd_all_track(df):
    '''
    对比 msd_real 和 msd_generated，如果 MSD 曲线形状一致，则运动模式相似。
    由于这里所有轨迹默认特征是一致的,所以将所有的相同间隔的值都合并取平均和方差
    从而可以绘制一条msd曲线
    '''
    msd_dict = {}
    track_groups = df.groupby("track_id")
    max_frame_count = 1
    for tkid, track in track_groups:
        framenum = len(track)
        if max_frame_count < framenum:
            max_frame_count = framenum
    
    for tau in range(1, max_frame_count):
        msd = []
        for track_id, track in track_groups:
            positions = track[["pos_x", "pos_y"]].values
            displacements = positions[tau:] - positions[:-tau]
            squared_displacements = np.sum(displacements**2, axis=1)
            msd = msd + squared_displacements.tolist()

        msd_dict[tau] = [np.mean(msd), np.std(msd), msd]
    
    return msd_dict


def compute_speed_distribution(df):
    '''
    画出速度分布直方图，若分布形状相似，说明运动速率特性一致。
    '''
    track_groups = df.groupby("track_id")
    all_speeds = []

    for track_id, track in track_groups:
        positions = track[["pos_x", "pos_y"]].values
        displacements = np.diff(positions, axis=0)
        speeds = np.linalg.norm(displacements, axis=1)
        all_speeds.extend(speeds)

    return np.array(all_speeds)


def compute_turning_angles(df):
    '''
    画出转角(相邻速度的夹角)分布直方图，定向运动的角度分布较窄，扩散运动分布较均匀。
    '''
    track_groups = df.groupby("track_id")
    all_angles = []

    for track_id, track in track_groups:
        positions = track[["pos_x", "pos_y"]].values
        displacements = np.diff(positions, axis=0)
        
        for i in range(len(displacements) - 1):
            v1 = displacements[i]
            v2 = displacements[i + 1]
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            degrees = np.degrees(theta)
            all_angles.append(degrees)

    return np.array(all_angles)


def compute_trajectory_persistence(df):
    '''
    计算初始和最终方向的夹角，若分布类似，则说明轨迹方向性一致。
    随着时间间隔增加,轨迹的一致性会变小, 因此需要考虑时间间隔
    '''
    track_groups = df.groupby("track_id")
    persistences = []

    for track_id, track in track_groups:
        positions = track[["pos_x", "pos_y"]].values
        if len(positions) < 2:
            continue

        start_vec = positions[1] - positions[0]
        end_vec = positions[-1] - positions[-2]
        
        cos_theta = np.dot(start_vec, end_vec) / (np.linalg.norm(start_vec) * np.linalg.norm(end_vec))
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        degrees = np.degrees(theta)
        
        persistences.append(degrees)

    return np.array(persistences)


def compare_distribution(real_values, generated_values):

    # 去除 NaN
    real_values = real_values[~np.isnan(real_values)]
    generated_values = generated_values[~np.isnan(generated_values)]

    # 进行 Mann-Whitney U 检验
    u_stat, p_value = mannwhitneyu(real_values, generated_values, alternative='two-sided')
    print(f"Mann-Whitney U-test: U={u_stat:.4f}, p-value={p_value:.4f}")
    if p_value > 0.05:
        print("两组数据分布无显著差异（p > 0.05）✅")
    else:
        print("两组数据分布存在显著差异（p < 0.05）❌")

    ks_stat, ks_pval = ks_2samp(real_values, generated_values)
    print(f"K-S test: statistic={ks_stat:.4f}, p-value={ks_pval:.4f}")
    if ks_pval > 0.05:
        print("两组数据的累积分布函数相似（p > 0.05）✅")
    else:
        print("两组数据的累积分布函数显著不同（p < 0.05）❌")
    
    wasserstein_dist = wasserstein_distance(real_values, generated_values)
    print(f"Wasserstein Distance: {wasserstein_dist}, 越小 形状越接近")  # 越小表示分布越接近

    if len(real_values) > len(generated_values):
        real_values = np.random.choice(real_values, size=len(generated_values), replace=False)
    elif len(generated_values) > len(real_values):
        generated_values = np.random.choice(generated_values, size=len(real_values), replace=False)

    jsd = distance.jensenshannon(real_values, generated_values, base=2)
    print(f"jsd Distance: {jsd}, 越小 形状越接近")  # 越小表示分布越接近



def plot_msd(msd_real,msd_generated,save_path):
    # 设置图形大小
    plt.figure(figsize=(6, 4))

    # 使用Seaborn设置样式
    sns.set(style="ticks")

    # 创建一个颜色调色板
    palette = sns.color_palette("muted", 2)

    # 绘制实际数据的折线图和误差条
    for tau, (mean, std, _) in msd_real.items():
        plt.errorbar(x=tau, y=mean, yerr=std, fmt='-', color=palette[0], capsize=5, capthick=True)

    # 绘制生成数据的折线图和误差条
    for tau, (mean, std, _) in msd_generated.items():
        plt.errorbar(x=tau, y=mean, yerr=std, fmt='-', color=palette[1], capsize=5, capthick=True)

    # 连接均值形成折线图
    plt.plot(list(msd_real.keys()), [v[0] for v in msd_real.values()], linestyle='-', color=palette[0], marker='o', markersize=10, alpha=0.7, label='Real Mean')
    plt.plot(list(msd_generated.keys()), [v[0] for v in msd_generated.values()], linestyle='-', color=palette[1], marker='o', markersize=10, alpha=0.7, label='Generated Mean')

    # 添加图例
    # plt.legend()

    # 添加轮廓背景
    sns.despine()

    # 设置标签
    plt.xlabel("Lag time (tau)")
    plt.ylabel("MSD")
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.png','.pdf'), format='pdf')

def cal_msd_sim(msd_real,msd_generated):
    '''
    计算两个msd的相似性
    (1) 直接计算每个时间间隔上均值的差异
    (2) 每个时间间隔上的Wasserstein距离(概率分布差异)
    '''
    
    # MSE
    real_mean = [v[0] for v in msd_real.values()]
    generated_mean = [v[0] for v in msd_generated.values()]
    mse_msd = np.linalg.norm(np.array(real_mean)-np.array(generated_mean))
    print(f"MSD MSE: {mse_msd:.4f}")

    # 相对误差
    mre_msd = np.mean(np.abs(np.array(real_mean) - np.array(generated_mean)) / (np.array(real_mean) + 1e-8))  # 避免除零
    print(f"MSD Mean Relative Error: {mre_msd:.4f}")

    # 计算 Wasserstein 距离
    wasserstein_dist_list = []
    for tau in msd_real.keys():
        w_distance = wasserstein_distance(msd_real[tau][2], msd_generated[tau][2])
        print(f"when tau = {tau}, Wasserstein Distance: {w_distance:.4f}")
        wasserstein_dist_list.append(w_distance)
    return mse_msd, wasserstein_dist_list


if __name__ == '__main__':

    use_init_as_reference = True
    use_same_length_tracks = True
    vis = True

    # for stride in [1,2,4,5,10,20,50,100]: # steps对应[100,50,25,20,10,5,2,1]
    for stride in [1,2]: # steps对应[100,50,25,20,10,5,2,1]
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>    stride = {stride}   <<<<<<<<<<<<<<<<<<<<<<<<")
        # 读入真实轨迹和生成轨迹
        # real_txtpath = f'/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/experiments/microtubule_low_future1_sample1_dt1_std323_del_neighbor_label_yst/microtubule_low_epoch90_same_density/stride_{stride}/MICROTUBULE_snr_7_density_low.txt'
        real_txtpath = f'/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/experiments/microtubule_low_future1_sample1_dt1_std323_del_neighbor_label_yst/microtubule_low_epoch90/stride_{stride}/MICROTUBULE_snr_7_density_low.txt'
        # real_txtpath = f'/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/experiments/microtubule_mid_future1_sample1_dt1_std323_del_neighbor_label_yst/microtubule_mid_epoch90_same_density_2025-02-18_10-19-01/stride_{stride}/val_MICROTUBULE_snr_7_density_mid/MICROTUBULE_snr_7_density_mid.txt'
        # real_txtpath = f'/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/experiments/vesicle_low_future1_sample1_dt1_std323_del_neighbor_label_yst_repeat/vesicle_low_epoch90/stride_100/VESICLE_snr_7_density_low.txt'
        df_real = pd.read_csv(real_txtpath, sep='\t', header=None, index_col=None)
        df_real.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
        df_real['frame_id'] = df_real['frame_id'] // 10

        if use_init_as_reference:
            # 找到最后一帧还存在的轨迹，并且长度可以长于7帧的
            trackid_list = df_real[df_real['frame_id'] == 99]['track_id'].values.tolist()

            filter_trackid_list = []
            for tkid in trackid_list:
                if len(df_real[df_real['track_id'] == tkid]) > 7:
                    filter_trackid_list.append(tkid)
            
            filter_tracks = df_real[df_real['track_id'].isin(filter_trackid_list)]

            df_real = filter_tracks[filter_tracks['frame_id'] >= 99 - 7]

        # generated_csvpath = f'/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/experiments/microtubule_low_future1_sample1_dt1_std323_del_neighbor_label_yst/microtubule_low_epoch90_same_density/stride_{stride}/generate_tracks.csv'
        generated_csvpath = f'/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/experiments/microtubule_low_future1_sample1_dt1_std323_del_neighbor_label_yst/microtubule_low_epoch90/stride_{stride}/generate_tracks.csv'
        # generated_csvpath = f'/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/experiments/microtubule_mid_future1_sample1_dt1_std323_del_neighbor_label_yst/microtubule_mid_epoch90_same_density_2025-02-18_10-19-01/stride_{stride}/val_MICROTUBULE_snr_7_density_mid/generate_tracks.csv'
        # generated_csvpath = f'/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/experiments/vesicle_low_future1_sample1_dt1_std323_del_neighbor_label_yst_repeat/vesicle_low_epoch90/stride_100/generate_tracks.csv'
        df_generated = pd.read_csv(generated_csvpath,header=0, index_col=None)
        df_generated = df_generated[df_generated['frame_id'] < 50]
        if use_same_length_tracks:
            df_generated = df_generated[df_generated['frame_id']< 8]

        # 计算msd
        msd_real = compute_msd_all_track(df_real)
        msd_generated = compute_msd_all_track(df_generated)
        # 计算msd的相似性
        msd_mse, msd_was_dist_list = cal_msd_sim(msd_real,msd_generated)

        # 计算速度
        speed_real = compute_speed_distribution(df_real)
        speed_generated = compute_speed_distribution(df_generated)
        print('速度 distribution:')
        compare_distribution(speed_real, speed_generated)
        # 计算转角
        angles_real = compute_turning_angles(df_real)
        angles_generated = compute_turning_angles(df_generated)
        print('转角 distribution:')
        compare_distribution(angles_real, angles_generated)
        # 计算一致性
        persistence_real = compute_trajectory_persistence(df_real)
        persistence_generated = compute_trajectory_persistence(df_generated)
        print('一致性 distribution:')
        compare_distribution(persistence_real, persistence_generated)

        if vis:
            savepath = os.path.split(generated_csvpath)[0]
            savepath = os.path.join(savepath, 'figs')
            os.makedirs(savepath, exist_ok=True)
            # 绘制msd曲线
            plot_msd(msd_real,msd_generated,os.path.join(savepath, "sns_mean_msd.png"))

            # 绘制速度分布直方图
            plt.figure()
            bins = np.linspace(0, 10, 30)  # 生成从-3到3的30个等间隔的bins
            plt.hist(speed_real, bins=bins, alpha=0.5, label="Real Speed")
            plt.hist(speed_generated, bins=bins, alpha=0.5, label="Generated Speed")
            plt.xlabel('Speed (px/frame)')
            plt.ylabel('Frequency')
            # plt.legend()
            plt.savefig(os.path.join(savepath, "speed.png"))
            plt.savefig(os.path.join(savepath, "speed.pdf"), format='pdf')
            plt.close()
            # plt.show()

            # 绘制转角分布直方图
            plt.figure()
            bins = np.linspace(0, 180, 30)  # 生成从-3到3的30个等间隔的bins
            plt.hist(angles_real, bins=bins, alpha=0.5, label="Real angles")
            plt.hist(angles_generated, bins=bins, alpha=0.5, label="Generated angles")
            plt.xlabel('Angles (degree)')
            plt.ylabel('Frequency')
            # plt.legend(loc = 'upper right')
            plt.savefig(os.path.join(savepath, "angles.png"))
            plt.savefig(os.path.join(savepath, "angles.pdf"), format='pdf')
            plt.close()

            # 绘制一致性分布直方图
            plt.figure()
            bins = np.linspace(0, 180, 30)  # 生成从-3到3的30个等间隔的bins
            plt.hist(persistence_real, bins=bins, alpha=0.5, label="Real persistence")
            plt.hist(persistence_generated, bins=bins, alpha=0.5, label="Generated persistence")
            # plt.legend(loc = 'upper right')
            plt.xlabel('Angles (degree)')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(savepath, "persistence.png"))
            plt.savefig(os.path.join(savepath, "persistence.pdf"), format='pdf')
            plt.close()
