'''
功能：通过逐步预测生成轨迹，同时生成多条轨迹
先会设定图的size，初始化轨迹的第一帧坐标，
之后，预测每个轨迹的未来坐标，更新轨迹，迭代执行
日期：2025年2月6日
作者：张玉冬

修改:2025年2月19日
基于视频的多个帧初始化,并生成,并合并起来,用于训练跟踪器

'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import time
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm
from easydict import EasyDict
from torch.utils.data import DataLoader
import datetime
from dataset import EnvironmentDataset, collate, get_timesteps_data, restore, get_timesteps_data_for_infer
from models.autoencoder import AutoEncoder
from models.trajectron import Trajectron
from mid_inf import MID_INF
import argparse
import shutil
import os.path as osp
import yaml

import sys
import os
import numpy as np
import pandas as pd
import dill
import pickle

from environment import Environment, Scene, Node, derivative_of

from torch import nn, optim, utils
import ipdb
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind, mannwhitneyu
import time

def seq2env(seq_data,):
    '''
    将历史轨迹的序列，转换成环境类格式（符合原模型的数据格式————pkl存储的内容）
    pkl文件存储的是一个env类的实例，env类包含了多个scene，每个scene包含多个node，每个node包含轨迹的特征序列

    environment————一个场景，包含多个scene，记录节点类型，注意力半径，机器人类型，标准化参数
    scene————一个视频，包含多个node，以及视频帧数，帧间隔，视频名字，增广函数
    node————一条轨迹，包含轨迹的特征序列（坐标，速度，加速度），nodeid就是trackid
    '''

    # desired_max_time = 100
    # pred_indices = [2, 3]
    # state_dim = 6
    # frame_diff = 10
    # desired_frame_diff = 1
    # dt = 0.4 # 两帧之间的时间间隔，s为单位，在求速度的时候是用到了的
    dt = 1.0 # 两帧之间的时间间隔，s为单位，在求速度的时候是用到了的

    data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

    standardization = { # 原模型的标准化参数
        'PEDESTRIAN': {
            'position': {
                'x': {'mean': 0, 'std': 1},
                'y': {'mean': 0, 'std': 1}
            },
            'velocity': {
                'x': {'mean': 0, 'std': 2},
                'y': {'mean': 0, 'std': 2}
            },
            'acceleration': {
                'x': {'mean': 0, 'std': 2},
                'y': {'mean': 0, 'std': 2}
            }
        }
    }
    
    env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
    # 环境类，记录了场景信息，
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
    env.attention_radius = attention_radius

    scenes = [] # 一个视频一个scene
    # 输出pkl的路径
    # data_dict_path = os.path.join(data_folder_name, '_'.join([desired_source, data_class]) + '.pkl')

    data = seq_data
    data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
    data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
    data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')

    data['frame_id'] = data['frame_id'] #// 10 # 转变frame id

    data['frame_id'] -= data['frame_id'].min() # frame_id从0开始

    data['node_type'] = 'PEDESTRIAN'
    data['node_id'] = data['track_id'].astype(str)

    data.sort_values('frame_id', inplace=True)
    # 记录mean，用于恢复轨迹信息
    mean_value = [data['pos_x'].mean(), data['pos_y'].mean()]
    # 以轨迹点的均值为中心，规范化
    data['pos_x'] = data['pos_x'] - data['pos_x'].mean()
    data['pos_y'] = data['pos_y'] - data['pos_y'].mean()

    max_timesteps = data['frame_id'].max() #从0开始的frameid的最大值
    # 每个视频有一个scene
    scene = Scene(timesteps=max_timesteps+1, dt=dt, name='traj_gen', aug_func=None)

    for node_id in pd.unique(data['node_id']): # 遍历每个node，即每个track

        node_df = data[data['node_id'] == node_id]

        node_values = node_df[['pos_x', 'pos_y']].values

        # if node_values.shape[0] < 2: # 如果轨迹历史长度小于2，也就是只有一个历史点 不处理？
        #     continue

        new_first_idx = node_df['frame_id'].iloc[0] # 第一帧出现的位置
        # 求该轨迹的速度和加速度序列
        x = node_values[:, 0]
        y = node_values[:, 1]
        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        data_dict = {('position', 'x'): x,
                        ('position', 'y'): y,
                        ('velocity', 'x'): vx,
                        ('velocity', 'y'): vy,
                        ('acceleration', 'x'): ax,
                        ('acceleration', 'y'): ay}

        node_data = pd.DataFrame(data_dict, columns=data_columns) # 建立一个新的df，不包含t信息。一条轨迹的坐标，速度，加速度信息
        node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data) 
        # Node类，记录一条轨迹的特征序列，nodeid就是trackid
        node.first_timestep = new_first_idx
        # scene类，记录视频信息，视频帧数，帧间隔，视频名字，增广函数，所有的轨迹scene.nodes（Node）
        scene.nodes.append(node)

    # print(scene)
    scenes.append(scene) # 多个视频的scene
    # print(f'Processed {len(scenes):.2f} scene')

    env.scenes = scenes

    
    return env, mean_value # 返回pkl文件的内容



def pred_traj_func(
        traj_hist_env,
        model, # 预测模型
        hyperparams,
        config,
        stride_value):
    model.eval()

    node_type = "PEDESTRIAN"

    scenes = traj_hist_env.scenes

    ph = hyperparams['prediction_horizon']
    max_hl = hyperparams['maximum_history_length']
    min_hl = 0 #hyperparams['minimum_history_length']
    

    for i, scene in enumerate(scenes):
        # 这里只需要取当前帧的往前的历史轨迹 设置一个范围,   
        # timesteps：要取的轨迹的帧的范围
        timesteps = np.arange(max(0, scene.timesteps - max_hl), scene.timesteps)
        # 需要确认这里取数据的方法
        # ipdb.set_trace()
        _mean, _std = traj_hist_env.get_standardize_params(hyperparams['state'][node_type], node_type)
        batch = get_timesteps_data_for_infer(
            env=traj_hist_env, scene=scene, t=timesteps, node_type=node_type, state=hyperparams['state'],
            pred_state=hyperparams['pred_state'], edge_types=traj_hist_env.get_edge_types(),
            min_ht=min_hl, max_ht=max_hl, min_ft=0, # 设置未来长度为0，因为实际没有未来轨迹，就是去预测的
            max_ft=ph, hyperparams=hyperparams)
        # 每个batch 有collate(batch), nodes, out_timesteps
        if batch is None:
            continue
        test_batch = batch[0] # 有9个内容
        nodes = batch[1] # node list
        timesteps_o = batch[2] # 预测帧 list
        # test_batch就是batch数据，包含了历史轨迹的信息，以及需要预测的轨迹的信息
        traj_pred = model.generate(test_batch, node_type, num_points=ph, sample=config.k_eval,bestof=True,sampling = "ddpm", step=stride_value ,v_std=_std[2:4]) # B * 20 * 12 * 2
        # 预测的轨迹traj_pred
        predictions = traj_pred
        predictions_dict = {}
        for i, ts in enumerate(timesteps_o):
            if ts not in predictions_dict.keys():
                predictions_dict[ts] = dict()
            predictions_dict[ts][nodes[i]] = np.transpose(predictions[:, [i]], (1, 0, 2, 3))

    return predictions_dict

def update_traj(
        traj_hist, # 历史轨迹
        predictions_dict,
        mean_value):

    newframeid = traj_hist['frame_id'].max() + 1
    for ts, nodes in predictions_dict.items():
        for node, prediction in nodes.items():
            node_id = node.id
            pos_x, pos_y = prediction[0, 0, 0]
            new_row = {
                'frame_id': newframeid,
                'track_id': int(node_id),
                'pos_x': pos_x,
                'pos_y': pos_y,
                'node_type': 'PEDESTRIAN',
                'node_id': node_id
            }
            traj_hist = traj_hist.append(new_row, ignore_index=True)

    # 反标准化
    traj_hist['pos_x'] = traj_hist['pos_x'] + mean_value[0]
    traj_hist['pos_y'] = traj_hist['pos_y'] + mean_value[1]
    return traj_hist


def init_data_with_dataset(data_txt_path, frame_interval=1,save_root=None):
    '''
    初始化轨迹，用已有的数据集轨迹，而非random随机生成第一帧

    根据需要的历史长度,构建多个初始化状态

    '''
    init_tracks_list = []
    ori_data = pd.read_csv(data_txt_path, sep='\t', header=None, index_col=None)
    ori_data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
    ori_data['frame_id'] = ori_data['frame_id'] // frame_interval
    # 保存基于生成的视频轨迹
    vis_tracks(ori_data, name=os.path.join(save_root,'base_movie_track'))
    
    video_frames = ori_data['frame_id'].max()

    for end_frame in range(8,video_frames,8):
        print(f'准备初始状态,结束帧为{end_frame}')
        # 找到指定帧帧还存在的轨迹，并且长度可以长于7帧的
        trackid_list = ori_data[ori_data['frame_id'] == end_frame]['track_id'].values.tolist()
        filter_tracks = ori_data[ori_data['track_id'].isin(trackid_list)]
        # (截断--后threshold)
        filter_tracks = filter_tracks[filter_tracks['frame_id'] <= end_frame]

        # 过滤历史轨迹的长度要求
        filter_trackid_list = []
        for tkid in trackid_list:
            if len(filter_tracks[filter_tracks['track_id'] == tkid]) > 7:
                filter_trackid_list.append(tkid)

        filter_tracks = filter_tracks[filter_tracks['track_id'].isin(filter_trackid_list)]
        
        # 截断不需要的久远历史(截断--前threshold)
        # filter_tracks = filter_tracks[filter_tracks['frame_id'] >= end_frame - 7]

        init_tracks_list.append(filter_tracks)

    return init_tracks_list
        

def vis_tracks(pred_traj, name=None):
    # 设置图的大小（单位为英寸）
    # 1英寸 = 25.4mm，所以45mm = 45/25.4英寸
    fig, ax = plt.subplots(figsize=(45/25.4, 45/25.4))

    # 获取唯一的轨迹ID
    track_ids = pred_traj['track_id'].unique()

    # 为每个轨迹分配一个颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(track_ids)))  # 使用颜色映射生成颜色
    np.random.shuffle(colors)
    # 绘制每个轨迹
    for idx, track_id in enumerate(track_ids):
        track_data = pred_traj[pred_traj['track_id'] == track_id]
        track_data = track_data.sort_values(by='frame_id')
        # if len(track_data)>8:
        #     ipdb.set_trace()
        ax.plot(track_data['pos_x'], track_data['pos_y'], color=colors[idx], linewidth=0.2)

    # 设置坐标轴范围（根据数据范围调整）
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)

    # 去掉坐标轴刻度
    ax.set_xticks([])
    ax.set_yticks([])

    # 去掉坐标轴边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    if name:
        plt.savefig(f"{name}.pdf", format="pdf", bbox_inches="tight")
    else:
        plt.savefig("trajectory_plot.pdf", format="pdf", bbox_inches="tight")

    # 显示图像
    # plt.show()

def save_generated_traj(traj_df, path, old_framenum):
    traj_df = traj_df[traj_df['frame_id'] >= old_framenum]
    traj_df.loc[:, 'frame_id'] = traj_df['frame_id'] - old_framenum
    traj_df.to_csv(f'{path}', header=True, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    # parser.add_argument('--config', default='/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/configs/vesicle_low_future1_sample1_inf.yaml')
    # parser.add_argument('--config', default='/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/configs/vesicle_low_future1_sample1_dt1_std323_inf.yaml')
    # parser.add_argument('--config', default='/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/configs/vesicle_low_future1_sample1_dt1_std323_del_neighbor_inf.yaml')
    # parser.add_argument('--config', default='/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/configs/vesicle_low_future1_sample1_dt1_std323_del_neighbor_label_yst_inf.yaml')


    # parser.add_argument('--config', default='/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/configs/microtubule_low_future1_sample1_dt1_std323_del_neighbor_label_yst_inf.yaml')
    # parser.add_argument('--config', default='/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/configs/receptor_low_future1_sample1_dt1_std323_del_neighbor_label_yst_inf.yaml')
    # parser.add_argument('--config', default='/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/configs/vesicle_low_future1_sample1_dt1_std323_del_neighbor_label_yst_repeat_inf.yaml')
    # parser.add_argument('--config', default='/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/configs/vesicle_mid_future1_sample1_dt1_std323_del_neighbor_label_yst_inf.yaml')
    # parser.add_argument('--config', default='/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/configs/microtubule_mid_future1_sample1_dt1_std323_del_neighbor_label_yst_inf.yaml')
    parser.add_argument('--config', default='/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/configs/helab_vesicle_future1_sample1_dt1_std322_del_neighbor_label_yst_inf.yaml')
    
    data_txt_path = '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/helab_vesicle/train/vesicle_FP_C1_5.txt'
    # data_txt_path = '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/microtubule_low/val/MICROTUBULE_snr_7_density_low.txt'
    # data_txt_path = '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/receptor_low/val/RECEPTOR_snr_7_density_low.txt'
    
    parser.add_argument('--data_txt_path', default=data_txt_path)

    # parser.add_argument('--dataset', default='vesicle_low')
    # parser.add_argument('--dataset', default='vesicle_mid')
    # parser.add_argument('--dataset', default='microtubule_low')
    parser.add_argument('--dataset', default='helab_vesicle')
    # parser.add_argument('--dataset', default='receptor_low')

    # 初始化轨迹的列表
    parser.add_argument('--data_txt_path_list', nargs="*", type=str, default=None)
    # parser.add_argument('--data_txt_path_list', nargs="*", type=str, default=[
    #     '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/microtubule_low/test/MICROTUBULE_snr_1_density_low.txt',
    #     '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/microtubule_low/test/MICROTUBULE_snr_2_density_low.txt',
    #     '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/microtubule_low/test/MICROTUBULE_snr_4_density_low.txt',
    #     '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/microtubule_low/test/MICROTUBULE_snr_7_density_low.txt',
    # ], )
    # parser.add_argument('--data_txt_path_list', nargs="*", type=str, default=[
    #     '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/microtubule_mid/test/MICROTUBULE_snr_1_density_mid.txt',
    #     '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/microtubule_mid/test/MICROTUBULE_snr_2_density_mid.txt',
    #     '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/microtubule_mid/test/MICROTUBULE_snr_4_density_mid.txt',
    #     '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/microtubule_mid/test/MICROTUBULE_snr_7_density_mid.txt',
    #     '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/microtubule_mid/val/MICROTUBULE_snr_7_density_mid.txt',
    # ], )
    # parser.add_argument('--data_txt_path_list', nargs="*", type=str, default=[
    #     '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/vesicle_mid/test/VESICLE_snr_1_density_mid.txt',
    #     '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/vesicle_mid/test/VESICLE_snr_2_density_mid.txt',
    #     '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/vesicle_mid/test/VESICLE_snr_4_density_mid.txt',
    #     '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/vesicle_mid/test/VESICLE_snr_7_density_mid.txt',
    #     '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/vesicle_mid/val/VESICLE_snr_7_density_mid.txt',
    # ], )
    # parser.add_argument('--data_txt_path_list', nargs="*", type=str, default=[
    #     '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/vesicle_low/test/VESICLE_snr_1_density_low.txt',
    #     '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/vesicle_low/test/VESICLE_snr_2_density_low.txt',
    #     '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/vesicle_low/test/VESICLE_snr_4_density_low.txt',
    #     '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/vesicle_low/test/VESICLE_snr_7_density_low.txt',
    # ], )

    return parser.parse_args()


def main(stride_value,formatted_time):

    np.random.seed(0)
    # 设定视频的空间size 和 时间t的长度
    video_width = 512
    video_length = 512
    video_timepoints = 100

    # 设定轨迹的数量
    num_trajectories = 10

    # -----------------采用原build的方式加载模型----------------

    # parse arguments and load config
    args = parse_args()

    if args.data_txt_path_list is not None:
        # 需要生成多个初始化的轨迹的后续,而忽略data_txt_path参数的设置
        data_txt_path_list = args.data_txt_path_list
    else:
        data_txt_path_list = [args.data_txt_path]
    
    print(f'初始化基于以下{len(data_txt_path_list)}个视频:')
    for ite in data_txt_path_list:
        print(ite)

    with open(args.config) as f:
       config = yaml.safe_load(f)

    for k, v in vars(args).items():
       config[k] = v
    if 'exp_name' in config.keys() and config["exp_name"] is not None:
        pass
    else:
        config["exp_name"] = args.config.split("/")[-1].split(".")[0]
    # modelpath = '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/experiments/vesicle_low_future1_sample1/vesicle_low_epoch28.pt'
    # config["ckpt_path"] = modelpath

    #pdb.set_trace()
    config = EasyDict(config)
    agent = MID_INF(config)

    model_ins = agent.model
    hyperparams = agent.hyperparams

    # 保存路径
    # 创建生成轨迹保存路径
    ckpt_path = config.ckpt_path
    save_root = ckpt_path.replace('.pt', '_length8'+f'_{formatted_time}')
    save_root = os.path.join(save_root, f'stride_{stride_value}')
    os.makedirs(save_root, exist_ok=True)

    this_codepath=os.path.abspath(__file__)
    shutil.copy(this_codepath, os.path.join(save_root,'gencode.py'))

    # -----------------制作当前iter的数据集--------------------
    for data_txt_path in data_txt_path_list:
        save_root = os.path.join(
            save_root, os.path.split(os.path.split(data_txt_path)[0])[-1]+'_'+os.path.split(data_txt_path)[-1].replace('.txt',''))
        os.makedirs(save_root, exist_ok=True)
        # data_txt_path = config['data_txt_path']
        traj_hist_df_list = init_data_with_dataset(data_txt_path,frame_interval=1,save_root=save_root)
        for gen_id, traj_hist_df in enumerate(traj_hist_df_list):
            save_folder = os.path.join(save_root, 'gen_id{}'.format(gen_id))
            os.makedirs(save_folder,exist_ok=True)

            init_traj_df = traj_hist_df.copy()
            vis_tracks(init_traj_df,name=os.path.join(save_folder,'ini_traj'))
            shutil.copy(data_txt_path, os.path.join(save_folder, os.path.split(data_txt_path)[-1]))
            # 1 构建一个初始化的数据分布
            # init_pos = np.random.rand(num_trajectories, 2) * video_width

            # traj_hist_df = pd.DataFrame({
            #     'frame_id': np.zeros(num_trajectories, dtype=int),
            #     'track_id': np.arange(num_trajectories)+1,
            #     'pos_x': init_pos[:, 0],
            #     'pos_y': init_pos[:, 1]
            # })

            # 2 转换成原模型的数据格式 pkl 存储env类的实例
            traj_hist_env, mean_value = seq2env(traj_hist_df)

            # -----------------逐帧推理预测，更新轨迹--------------------
            
            for i in tqdm(range(video_timepoints),desc='生成轨迹'):
                # 预测轨迹
                # print(stride_value , type(stride_value))
                predictions_dict = pred_traj_func(
                    traj_hist_env, 
                    model_ins, 
                    hyperparams,
                    config,
                    stride_value
                    )
                # ipdb.set_trace()
                # 更新轨迹
                traj_hist_df = update_traj(traj_hist_df, predictions_dict,mean_value)
                traj_hist_df = traj_hist_df[['frame_id', 'track_id', 'pos_x', 'pos_y']]
                if i < video_timepoints-1:
                    # 重新制作数据集
                    traj_hist_env, mean_value = seq2env(traj_hist_df)
            
            pred_traj = traj_hist_df.copy()

            # 保存生成的轨迹
            save_generated_traj(pred_traj, path=os.path.join(save_folder,'generate_tracks.csv'), old_framenum=new_gen_frame_start)
            
            # 可视化生成的部分
            vis_tracks(pred_traj[pred_traj['frame_id'] >= new_gen_frame_start],name=os.path.join(save_folder,'generate_traj_all'))
            frame_start = new_gen_frame_start
            frame_end = new_gen_frame_start * 2  # 8---15
            vis_gen_traj = pred_traj[(pred_traj['frame_id'] >= frame_start)& (pred_traj['frame_id'] < frame_end)]
            vis_tracks(vis_gen_traj,name=os.path.join(save_folder,f'generate_traj_frame{frame_start}_{frame_end-1}'))
        

    # return traj_hist_df, init_traj_df, config
    return 0



if __name__ == '__main__':
    # stride = 1
    # for stride in [1,2,4,5,10,20,50,100]: # steps对应[100,50,25,20,10,5,2,1]
    # 获取当前时间
    current_time = datetime.datetime.now()

    # 格式化时间字符串
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    for stride in [100]: # steps对应[100]   用于microtubule数据类型
        pred_traj, init_traj_df, config = None, None, None
        new_gen_frame_start = 8
        print(stride, type(stride))
        main(stride,formatted_time)
        