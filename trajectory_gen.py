'''
功能：通过逐步预测生成轨迹，同时生成多条轨迹
先会设定图的size，初始化轨迹的第一帧坐标，
之后，预测每个轨迹的未来坐标，更新轨迹，迭代执行
日期：2025年2月6日
作者：张玉冬
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
    dt = 0.4 # 两帧之间的时间间隔，s为单位，在求速度的时候是用到了的

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
                'x': {'mean': 0, 'std': 1},
                'y': {'mean': 0, 'std': 1}
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

    print(scene)
    scenes.append(scene) # 多个视频的scene
    print(f'Processed {len(scenes):.2f} scene')

    env.scenes = scenes

    
    return env, mean_value # 返回pkl文件的内容



def pred_traj( # 只是参考，eval流程
        traj_hist_env,
        model, # 预测模型
        hyperparams,
        config,):
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
        
        batch = get_timesteps_data_for_infer(
            env=traj_hist_env, scene=scene, t=timesteps, node_type=node_type, state=hyperparams['state'],
            pred_state=hyperparams['pred_state'], edge_types=traj_hist_env.get_edge_types(),
            min_ht=min_hl, max_ht=max_hl, min_ft=0, # 设置未来长度为0，因为实际没有未来轨迹，就是去预测的
            max_ft=ph, hyperparams=hyperparams)
        # 每个batch 有collate(batch), nodes, out_timesteps
        if batch is None:
            continue
        test_batch = batch[0]
        nodes = batch[1] # n, 8, 6
        timesteps_o = batch[2] # sample是设置采样的数量，也就是生成的结果的个数
        traj_pred = model.generate(test_batch, node_type, num_points=ph, sample=config.k_eval,bestof=True) # B * 20 * 12 * 2
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



def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    parser.add_argument('--config', default='/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/configs/vesicle_low_future1_sample1_inf.yaml')
    parser.add_argument('--dataset', default='vesicle_low')
    return parser.parse_args()

def init_data_with_dataset(data_txt_path):
    '''
    初始化轨迹，用已有的数据集轨迹，而非random随机生成第一帧
    '''
    ori_data = pd.read_csv(data_txt_path, sep='\t', header=None, index_col=None)
    ori_data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
    ori_data['frame_id'] = ori_data['frame_id'] // 10

    # 找到最后一帧还存在的轨迹，并且长度可以长于7帧的
    trackid_list = ori_data[ori_data['frame_id'] == 99]['track_id'].values.tolist()

    filter_trackid_list = []
    for tkid in trackid_list:
        if len(ori_data[ori_data['track_id'] == tkid]) > 7:
            filter_trackid_list.append(tkid)
    
    filter_tracks = ori_data[ori_data['track_id'].isin(filter_trackid_list)]

    filter_tracks = filter_tracks[filter_tracks['frame_id'] >= 99 - 7]

    return filter_tracks
        


def main():

    np.random.seed(0)
    # 设定视频的空间size 和 时间t的长度
    video_width = 512
    video_length = 512
    video_timepoints = 100

    # 设定轨迹的数量
    num_trajectories = 10

    # 设定历史的时间步长
    timesteps = 10

    # 设定模型推理的参数
    sampling = "ddim"
    steps = 5


    # -----------------采用原build的方式加载模型----------------

    # parse arguments and load config
    args = parse_args()
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

    model_dir = osp.join("./experiments", config.exp_name) # 根据config的名字创建文件夹
    shutil.copy(args.config, os.path.join(model_dir, os.path.basename(args.config)))
    shutil.copy('./utils/trajectron_hypers.py', os.path.join(model_dir, 'trajectron_hypers.py'))

    model = agent.model
    hyperparams = agent.hyperparams

    # -----------------制作当前iter的数据集--------------------
    data_txt_path = '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data/vesicle_low/val/VESICLE_snr_7_density_low.txt'
    traj_hist_df = init_data_with_dataset(data_txt_path)
    vis_tracks(traj_hist_df,name='ini_traj')
    # 1 构建一个初始化的数据分布
    # init_pos = np.random.rand(num_trajectories, 2) * video_width

    # traj_hist_df = pd.DataFrame({
    #     'frame_id': np.zeros(num_trajectories, dtype=int),
    #     'track_id': np.arange(num_trajectories)+1,
    #     'pos_x': init_pos[:, 0],
    #     'pos_y': init_pos[:, 1]
    # })


    # 为了方便，将轨迹的第一帧复制一份，作为第二帧
    # fake_df = traj_hist_df.copy()
    # fake_df['frame_id'] = 1
    # traj_hist_df = pd.concat([traj_hist_df, fake_df], ignore_index=True)
    
    # train_data_loader, traj_hist_env = preprocess_input(traj_hist_df, hyperparams, config)

    # 2 转换成原模型的数据格式 pkl 存储env类的实例
    traj_hist_env, mean_value = seq2env(traj_hist_df)

    # -----------------逐帧推理预测，更新轨迹--------------------
    
    for i in tqdm(range(video_timepoints),desc='生成轨迹'):
        # 预测轨迹
        predictions_dict = pred_traj(
            traj_hist_env, 
            model, 
            hyperparams,
            config,
            )
        # ipdb.set_trace()
        # 更新轨迹
        traj_hist_df = update_traj(traj_hist_df, predictions_dict,mean_value)
        traj_hist_df = traj_hist_df[['frame_id', 'track_id', 'pos_x', 'pos_y']]
        if i < video_timepoints-1:
            # 重新制作数据集
            traj_hist_env, mean_value = seq2env(traj_hist_df)

    return traj_hist_df

def vis_tracks(pred_traj, name=None):
    # 设置图的大小（单位为英寸）
    # 1英寸 = 25.4mm，所以45mm = 45/25.4英寸
    fig, ax = plt.subplots(figsize=(45/25.4, 45/25.4))

    # 获取唯一的轨迹ID
    track_ids = pred_traj['track_id'].unique()

    # 为每个轨迹分配一个颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(track_ids)))  # 使用颜色映射生成颜色

    # 绘制每个轨迹
    for idx, track_id in enumerate(track_ids):
        track_data = pred_traj[pred_traj['track_id'] == track_id]
        ax.plot(track_data['pos_x'], track_data['pos_y'], color=colors[idx], linewidth=0.5)

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

if __name__ == '__main__':
    pred_traj = main()
    print(pred_traj)

    # 可视化生成的部分
    vis_gen_traj = pred_traj[pred_traj['frame_id'] > 7]
    vis_tracks(vis_gen_traj,name='generate_traj')
    
    # 计算轨迹的性质


