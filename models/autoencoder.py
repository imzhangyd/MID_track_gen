import torch
from torch.nn import Module
import torch.nn as nn
from .encoders.trajectron import Trajectron
from .encoders import dynamics as dynamic_module
import models.diffusion as diffusion
from models.diffusion import DiffusionTraj,VarianceSchedule
import pdb
import ipdb
import numpy as np


class AutoEncoder(Module):
    '''
    model
    '''
    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.diffnet = getattr(diffusion, config.diffnet)
        '''
        ****设置扩散系数，去噪模型和参数
        '''
        self.diffusion = DiffusionTraj(
            net = self.diffnet(point_dim=2, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False),
            var_sched = VarianceSchedule( # 负责扩散的模块
                num_steps=100,
                beta_T=5e-2,
                mode='linear'

            )
        )

    def encode(self, batch,node_type): # 编码历史信息
        z = self.encoder.get_latent(batch, node_type)
        return z
    
    def generate(self, batch, node_type, num_points, sample, bestof,flexibility=0.0, ret_traj=False, sampling="ddpm", step=100, v_std = None):
        # 推理
        # ipdb.set_trace()
        dynamics = self.encoder.node_models_dict[node_type].dynamic
        encoded_x = self.encoder.get_latent(batch, node_type)
        predicted_y_vel =  self.diffusion.sample(num_points, encoded_x,sample,bestof, flexibility=flexibility, ret_traj=ret_traj, sampling=sampling, step=step)
        if v_std is not None: # 假设输入的v_std是一个list     因为在训练的时候，y用的是非方差标准化数据，所以这里也不用
            v_std = torch.tensor(v_std).float().to(predicted_y_vel.device)
            v_std = v_std.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            predicted_y_vel = predicted_y_vel * v_std
        predicted_y_pos = dynamics.integrate_samples(predicted_y_vel)
        return predicted_y_pos.cpu().detach().numpy()

    def get_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch # dataset 的内容

        feat_x_encoded = self.encode(batch,node_type) # B * 64
        # loss = self.diffusion.get_loss(y_t.cuda(), feat_x_encoded)
        loss = self.diffusion.get_loss(y_st_t.cuda(), feat_x_encoded)
        return loss
