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


class MID():
    '''
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

    def train(self):
        for epoch in range(1, self.config.epochs + 1):
            self.train_dataset.augment = self.config.augment
            for node_type, data_loader in self.train_data_loader.items():
                pbar = tqdm(data_loader, ncols=80)
                for batch in pbar: # len(batch) = 9
                    # ipdb.set_trace()
                    self.optimizer.zero_grad()
                    train_loss = self.model.get_loss(batch, node_type)
                    if self.log_writer is not None:
                        self.log_writer.add_scalar('Loss/train_step', train_loss.item(), (epoch-1) * len(data_loader) + pbar.n)
                        # 获取当前学习率
                        current_lr = self.optimizer.param_groups[0]['lr']
                        # 记录学习率到 TensorBoard
                        self.log_writer.add_scalar('LearningRate', current_lr, (epoch-1) * len(data_loader) + pbar.n)
                    pbar.set_description(f"Epoch {epoch}, {node_type} MSE: {train_loss.item():.2f}")
                    train_loss.backward()
                    self.optimizer.step()
                    

            self.train_dataset.augment = False
            if epoch % self.config.eval_every == 0:
                self.model.eval()

                node_type = "PEDESTRIAN"
                eval_ade_batch_errors = []
                eval_fde_batch_errors = []

                ph = self.hyperparams['prediction_horizon']
                max_hl = self.hyperparams['maximum_history_length']

                _mean, _std = self.eval_env.get_standardize_params(self.hyperparams['state'][node_type], node_type)

                for i, scene in enumerate(self.eval_scenes): # 遍历每个要eval的视频
                    print(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
                    for t in tqdm(range(0, scene.timesteps, 10)): # 每10帧构成一个batch，取数据
                        timesteps = np.arange(t,t+10)
                        batch = get_timesteps_data(env=self.eval_env, scene=scene, t=timesteps, node_type=node_type, state=self.hyperparams['state'],
                                       pred_state=self.hyperparams['pred_state'], edge_types=self.eval_env.get_edge_types(),
                                       min_ht=max_hl, max_ht=max_hl, min_ft=ph,
                                       max_ft=ph, hyperparams=self.hyperparams)
                        # 每个batch 有collate(batch), nodes, out_timesteps
                        if batch is None:
                            continue
                        test_batch = batch[0]
                        nodes = batch[1]
                        timesteps_o = batch[2] # sample是设置采样的数量，也就是生成的结果的个数
                        traj_pred = self.model.generate(test_batch, node_type, num_points=ph, sample=self.config.k_eval,bestof=True,step=100) #, v_std=_std[2:4]) # B * 20 * 12 * 2
                        # 预测的轨迹traj_pred
                        predictions = traj_pred
                        predictions_dict = {}
                        for i, ts in enumerate(timesteps_o):
                            if ts not in predictions_dict.keys():
                                predictions_dict[ts] = dict()
                            predictions_dict[ts][nodes[i]] = np.transpose(predictions[:, [i]], (1, 0, 2, 3))
                        # 计算误差
                        batch_error_dict = evaluation.compute_batch_statistics(predictions_dict,
                                                                               scene.dt,
                                                                               max_hl=max_hl,
                                                                               ph=ph,
                                                                               node_type_enum=self.eval_env.NodeType,
                                                                               kde=False,
                                                                               map=None,
                                                                               best_of=True,
                                                                               prune_ph_to_future=True)

                        eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[node_type]['ade']))
                        eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[node_type]['fde']))



                ade = np.mean(eval_ade_batch_errors)
                fde = np.mean(eval_fde_batch_errors)

                if self.config.dataset == "eth":
                    ade = ade/0.6
                    fde = fde/0.6
                elif self.config.dataset == "sdd":
                    ade = ade * 50
                    fde = fde * 50
                if self.log_writer is not None:
                    self.log_writer.add_scalar('Eval/ADE', ade, epoch)
                    self.log_writer.add_scalar('Eval/FDE', fde, epoch)

                print(f"Epoch {epoch} | ADE: {ade} FDE: {fde}")
                self.log.info(f"| Epoch {epoch} ADE: {ade} FDE: {fde}")

                # Saving model
                checkpoint = {
                    'encoder': self.registrar.model_dict,
                    'ddpm': self.model.state_dict()
                 }
                torch.save(checkpoint, osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"))

                self.model.train()


    def eval(self, sampling, step):
        epoch = self.config.eval_at

        self.log.info(f"Sampling: {sampling} Stride: {step}")

        node_type = "PEDESTRIAN"
        eval_ade_batch_errors = []
        eval_fde_batch_errors = []
        ph = self.hyperparams['prediction_horizon']
        max_hl = self.hyperparams['maximum_history_length']

        _mean, _std = self.eval_env.get_standardize_params(self.hyperparams['state'][node_type], node_type)

        for i, scene in enumerate(self.eval_scenes):
            print(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t,t+10)
                batch = get_timesteps_data(env=self.eval_env, scene=scene, t=timesteps, node_type=node_type, state=self.hyperparams['state'],
                               pred_state=self.hyperparams['pred_state'], edge_types=self.eval_env.get_edge_types(),
                               min_ht=max_hl, max_ht=max_hl, min_ft=ph, max_ft=ph, # 固定hist长度为最大，future长度
                               hyperparams=self.hyperparams)
                if batch is None:
                    continue
                # ipdb.set_trace()
                test_batch = batch[0] # len=9 
                nodes = batch[1] # 这些node本身包含了hist信息
                timesteps_o = batch[2]
                traj_pred = self.model.generate(test_batch, node_type, num_points=ph, sample=self.config.k_eval, bestof=True, sampling=sampling, step=step) #, v_std=_std[2:4]) # B * 20 * 12 * 2

                predictions = traj_pred
                predictions_dict = {}
                for i, ts in enumerate(timesteps_o):
                    if ts not in predictions_dict.keys():
                        predictions_dict[ts] = dict()
                    predictions_dict[ts][nodes[i]] = np.transpose(predictions[:, [i]], (1, 0, 2, 3))
                # predictions_dict key是timestep value是NODES = 预测的多个值 1,sampletime, futurelen, dim


                batch_error_dict = evaluation.compute_batch_statistics(predictions_dict,
                                                                       scene.dt,
                                                                       max_hl=max_hl,
                                                                       ph=ph,
                                                                       node_type_enum=self.eval_env.NodeType,
                                                                       kde=False,
                                                                       map=None,
                                                                       best_of=True,
                                                                       prune_ph_to_future=True)

                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[node_type]['ade']))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[node_type]['fde']))



        ade = np.mean(eval_ade_batch_errors)
        fde = np.mean(eval_fde_batch_errors)

        if self.config.dataset == "eth":
            ade = ade/0.6
            fde = fde/0.6
        elif self.config.dataset == "sdd":
            ade = ade * 50
            fde = fde * 50
        print(f"Sampling: {sampling} Stride: {step}")
        print(f"Epoch {epoch} | ADE: {ade} FDE: {fde}")
        #self.log.info(f"| Epoch {epoch} ADE: {ade} FDE: {fde}")


    def _build(self):
        self._build_dir()

        self._build_encoder_config()
        self._build_encoder()
        self._build_model()
        self._build_train_loader()
        self._build_eval_loader()

        self._build_optimizer()

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
        self.hyperparams['enc_rnn_dim_edge'] = self.config.encoder_dim#//2
        self.hyperparams['enc_rnn_dim_edge_influence'] = self.config.encoder_dim#//2
        self.hyperparams['enc_rnn_dim_history'] = self.config.encoder_dim#//2
        self.hyperparams['enc_rnn_dim_future'] = self.config.encoder_dim#//2
        # registar
        self.registrar = ModelRegistrar(self.model_dir, "cuda")

        if self.config.eval_mode:
            epoch = self.config.eval_at
            checkpoint_dir = osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt")
            self.checkpoint = torch.load(osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"), map_location = "cpu")

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
        if self.config.eval_mode:
            self.model.load_state_dict(self.checkpoint['ddpm'])

        print("> Model built!")

    def _build_train_loader(self):
        config = self.config
        self.train_scenes = []

        with open(self.train_data_path, 'rb') as f:
            train_env = dill.load(f, encoding='latin1')

        for attention_radius_override in config.override_attention_radius:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            train_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)


        self.train_scenes = self.train_env.scenes
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