import os
import argparse
import torch
import dill
import pdb
import numpy as np
import os.path as osp
import logging
import time
from torch import nn, optim, utils
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
import pickle

from dataset import EnvironmentDataset, collate, get_timesteps_data, restore
from models.autoencoder import AutoEncoder
from models.trajectron import Trajectron
from utils.model_registrar import ModelRegistrar
from utils.trajectron_hypers import get_traj_hypers
import evaluation
import ipdb


class MID_INF():
    '''
    _INF是用于推理的模型加载作用
    组织训练流程，包括
        build 准备数据、数据loader、模型、优化器
        train 训练多个epoch
        eval eval过程
    更像是一个pipe管理
    '''
    def __init__(self, config):
        self.config = config
        torch.backends.cudnn.benchmark = True
        self._build()

    def _build(self):
        self._build_dir()

        self._build_encoder_config()
        self._build_encoder()
        self._build_model()
        # self._build_train_loader()
        # self._build_eval_loader()

        # self._build_optimizer()

        #self._build_offline_scene_graph()
        #pdb.set_trace()
        print("> Everything built. Have fun :)")

    def _build_dir(self):
        '''
        构建模型的保存路径，日志保存路径（出）, 数据的路径（入）
        '''
        self.model_dir = osp.join("./experiments",self.config.exp_name)
        self.log_writer = SummaryWriter(log_dir=self.model_dir)
        os.makedirs(self.model_dir,exist_ok=True)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        log_name = f"{self.config.dataset}_{log_name}"

        log_dir = osp.join(self.model_dir, log_name)
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        handler = logging.FileHandler(log_dir)
        handler.setLevel(logging.INFO)
        self.log.addHandler(handler)

        self.log.info("Config:")
        self.log.info(self.config)
        self.log.info("\n")
        self.log.info("Eval on:")
        self.log.info(self.config.dataset)
        self.log.info("\n")

        self.train_data_path = osp.join(self.config.data_dir,self.config.dataset + "_train.pkl")
        self.eval_data_path = osp.join(self.config.data_dir,self.config.dataset + "_test.pkl")
        print("> Directory built!")

    

    def _build_encoder_config(self):
        '''
        构建编码器的配置，加载train val 数据
        '''
        self.hyperparams = get_traj_hypers()
        self.hyperparams['enc_rnn_dim_edge'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_edge_influence'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_history'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_future'] = self.config.encoder_dim//2
        # registar
        self.registrar = ModelRegistrar(self.model_dir, "cuda")

        # 添加预训练的checkpoint
        if self.config.ckpt_path:
            checkpoint_path = self.config.ckpt_path
            self.checkpoint = torch.load(checkpoint_path, map_location = "cpu")
            self.registrar.load_models(self.checkpoint['encoder'])

        with open(self.train_data_path, 'rb') as f:
            self.train_env = dill.load(f, encoding='latin1')
        with open(self.eval_data_path, 'rb') as f:
            self.eval_env = dill.load(f, encoding='latin1')

    def _build_encoder(self):
        '''
        构建encoder
        '''
        self.encoder = Trajectron(self.registrar, self.hyperparams, "cuda")

        self.encoder.set_environment(self.train_env)
        self.encoder.set_annealing_params()


    def _build_model(self):
        """ Define Model """
        config = self.config
        model = AutoEncoder(config, encoder = self.encoder)

        self.model = model.cuda()
        # if self.config.eval_mode:
        if self.config.ckpt_path:
            self.model.load_state_dict(self.checkpoint['ddpm'])

        print("> Model built!")

    def _build_train_loader(self):
        config = self.config
        self.train_scenes = []

        with open(self.train_data_path, 'rb') as f:
            train_env = dill.load(f, encoding='latin1')

        # 重写注意力半径，通常为空，不重写
        for attention_radius_override in config.override_attention_radius: # 重写注意力半径，通常为空
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            train_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)


        self.train_scenes = self.train_env.scenes
        # 训练的时候，是否根据频率进行采样
        self.train_scenes_sample_probs = self.train_env.scenes_freq_mult_prop if config.scene_freq_mult_train else None

        self.train_dataset = EnvironmentDataset(train_env,
                                           self.hyperparams['state'], # 这是输入的状态变量定义
                                           self.hyperparams['pred_state'], #输出的状态变量 （位移）
                                           scene_freq_mult=self.hyperparams['scene_freq_mult_train'],
                                           node_freq_mult=self.hyperparams['node_freq_mult_train'],
                                           hyperparams=self.hyperparams,
                                           min_history_timesteps=1, # 历史长度可以是1
                                           min_future_timesteps=self.hyperparams['prediction_horizon'],
                                           return_robot=not self.config.incl_robot_node)
        self.train_data_loader = dict()
        for node_type_data_set in self.train_dataset:
            node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                         collate_fn=collate,
                                                         pin_memory = True,
                                                         batch_size=self.config.batch_size,
                                                         shuffle=True,
                                                         num_workers=self.config.preprocess_workers)
            self.train_data_loader[node_type_data_set.node_type] = node_type_dataloader


    def _build_eval_loader(self):
        config = self.config
        self.eval_scenes = []
        eval_scenes_sample_probs = None

        if config.eval_every is not None:
            with open(self.eval_data_path, 'rb') as f:
                self.eval_env = dill.load(f, encoding='latin1')

            for attention_radius_override in config.override_attention_radius:
                node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
                self.eval_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

            if self.eval_env.robot_type is None and self.hyperparams['incl_robot_node']:
                self.eval_env.robot_type = self.eval_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
                for scene in self.eval_env.scenes:
                    scene.add_robot_from_nodes(self.eval_env.robot_type)

            self.eval_scenes = self.eval_env.scenes
            eval_scenes_sample_probs = self.eval_env.scenes_freq_mult_prop if config.scene_freq_mult_eval else None
            self.eval_dataset = EnvironmentDataset(self.eval_env,
                                              self.hyperparams['state'],
                                              self.hyperparams['pred_state'],
                                              scene_freq_mult=self.hyperparams['scene_freq_mult_eval'],
                                              node_freq_mult=self.hyperparams['node_freq_mult_eval'],
                                              hyperparams=self.hyperparams,
                                              min_history_timesteps=self.hyperparams['minimum_history_length'],
                                              min_future_timesteps=self.hyperparams['prediction_horizon'],
                                              return_robot=not config.incl_robot_node)
            self.eval_data_loader = dict()
            for node_type_data_set in self.eval_dataset:
                node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                             collate_fn=collate,
                                                             pin_memory=True,
                                                             batch_size=config.eval_batch_size,
                                                             shuffle=True,
                                                             num_workers=config.preprocess_workers)
                self.eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

        print("> Dataset built!")

    def _build_offline_scene_graph(self):
        if self.hyperparams['offline_scene_graph'] == 'yes':
            print(f"Offline calculating scene graphs")
            for i, scene in enumerate(self.train_scenes):
                scene.calculate_scene_graph(self.train_env.attention_radius,
                                            self.hyperparams['edge_addition_filter'],
                                            self.hyperparams['edge_removal_filter'])
                print(f"Created Scene Graph for Training Scene {i}")

            for i, scene in enumerate(self.eval_scenes):
                scene.calculate_scene_graph(self.eval_env.attention_radius,
                                            self.hyperparams['edge_addition_filter'],
                                            self.hyperparams['edge_removal_filter'])
                print(f"Created Scene Graph for Evaluation Scene {i}")

    def _build_optimizer(self):
        '''
        构建优化器
        '''
        self.optimizer = optim.Adam([{'params': self.registrar.get_all_but_name_match('map_encoder').parameters()},
                                     {'params': self.model.parameters()}
                                    ],
                                    lr=self.config.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=0.98)
        print("> Optimizer built!")