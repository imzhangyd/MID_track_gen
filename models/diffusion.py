import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np
import torch.nn as nn
from .common import *
import ipdb
import pdb

class VarianceSchedule(Module):

    def __init__(self, num_steps, mode='linear',beta_1=1e-4, beta_T=5e-2,cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = (
            torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas

class DiffusionTraj(Module):

    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, context, t=None):  # x_0是未归一化的数据，没有做速度的方差归一化
        # x_0 是 bs,futurelen, outdim 是label; context是bs，featdim=256
        batch_size, _, point_dim = x_0.size()
        if t == None: # # 采样出每个样本不同的t（t在step范围内）
            t = self.var_sched.uniform_sample_t(batch_size) 
        
        alpha_bar = self.var_sched.alpha_bars[t] # bs
        beta = self.var_sched.betas[t].cuda() # bs

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).cuda()       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).cuda()   # (B, 1, 1)

        e_rand = torch.randn_like(x_0).cuda()  # (B, N, d)
        # self.net 是 TransformerConcatLinear
        # c0 * x_0 + c1 * e_rand 是加噪了的结果，  self.net是去噪网络，噪声预测网络
        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context) #context是条件
        loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
        return loss

    def sample(self, num_points, context, sample, bestof, point_dim=2, flexibility=0.0, ret_traj=False, sampling="ddpm", step=100):
        traj_list = []
        # 多次采样噪声，在条件控制下，预测噪声，然后恢复实际要预测的未来轨迹,返回预测的轨迹
        for i in range(sample):
            batch_size = context.size(0)
            if bestof:
                x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
            else:
                x_T = torch.zeros([batch_size, num_points, point_dim]).to(context.device)
            traj = {self.var_sched.num_steps: x_T}
            stride = step # 实际上这里的step不是step，而是stride
            # stride = self.var_sched.num_steps // step
            #stride = int(100/stride)
            step_list = list(range(self.var_sched.num_steps, 0, -stride))
            step_nums = len(step_list)
            for step_idx, t in enumerate(step_list):
            # for t in range(self.var_sched.num_steps, 0, -stride):
                if step_idx == step_nums - 1:
                    z = torch.zeros_like(x_T)
                else:
                    z = torch.randn_like(x_T)
                # z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
                alpha = self.var_sched.alphas[t]# 101个  从1到0.95
                alpha_bar = self.var_sched.alpha_bars[t] # 是alpha的累计乘积，“数据保留量”
                alpha_bar_next = self.var_sched.alpha_bars[t-stride] # self.var_sched.alpha_bars 101个
                # ipdb.set_trace()
                sigma = self.var_sched.get_sigmas(t, flexibility)

                c0 = 1.0 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

                x_t = traj[t]
                beta = self.var_sched.betas[[t]*batch_size]
                e_theta = self.net(x_t, beta=beta, context=context) # 在条件下预测噪声参数，self.net 是 TransformerConcatLinear
                if sampling == "ddpm":
                    x_next = c0 * (x_t - c1 * e_theta) + sigma * z
                elif sampling == "ddim":
                    x0_t = (x_t - e_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                    x_next = alpha_bar_next.sqrt() * x0_t + (1 - alpha_bar_next).sqrt() * e_theta
                else:
                    pdb.set_trace()
                traj[t-stride] = x_next.detach()     # Stop gradient and save trajectory.
                traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
                if not ret_traj:
                   del traj[t]

            if ret_traj:
                traj_list.append(traj)
            else:
                traj_list.append(traj[0])
        return torch.stack(traj_list)

'''
这是不同的解码器方案 TrajNet
'''
# class TrajNet(Module):

#     def __init__(self, point_dim, context_dim, residual):
#         super().__init__()
#         self.act = F.leaky_relu
#         self.residual = residual
#         self.layers = ModuleList([
#             ConcatSquashLinear(2, 128, context_dim+3),
#             ConcatSquashLinear(128, 256, context_dim+3),
#             ConcatSquashLinear(256, 512, context_dim+3),
#             ConcatSquashLinear(512, 256, context_dim+3),
#             ConcatSquashLinear(256, 128, context_dim+3),
#             ConcatSquashLinear(128, 2, context_dim+3),

#         ])

#     def forward(self, x, beta, context):
#         """
#         Args:
#             x:  Point clouds at some timestep t, (B, N, d).
#             beta:     Time. (B, ).
#             context:  Shape latents. (B, F).
#         """
#         batch_size = x.size(0)
#         beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
#         context = context.view(batch_size, 1, -1)   # (B, 1, F)

#         time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
#         ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

#         out = x
#         #pdb.set_trace()
#         for i, layer in enumerate(self.layers):
#             out = layer(ctx=ctx_emb, x=out)
#             if i < len(self.layers) - 1:
#                 out = self.act(out)

#         if self.residual:
#             return x + out
#         else:
#             return out


class TransformerConcatLinear(Module):

    def __init__(self, point_dim, context_dim, tf_layer, residual):
        super().__init__()
        self.residual = residual
        self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=24)
        self.concat1 = ConcatSquashLinear(2,2*context_dim,context_dim+3)
        self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=4, dim_feedforward=4*context_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
        self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
        self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
        self.linear = ConcatSquashLinear(context_dim//2, 2, context_dim+3)
        #self.linear = nn.Linear(128,2)


    def forward(self, x, beta, context): # x是添加了噪声的，context是条件
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)
        x = self.concat1(ctx_emb,x) # ctx_emb是条件，x是噪声 bs,futlen,newdim
        final_emb = x.permute(1,0,2) #futlen,bs,newdim
        final_emb = self.pos_emb(final_emb)


        trans = self.transformer_encoder(final_emb).permute(1,0,2)
        trans = self.concat3(ctx_emb, trans) #都是用context生成w,b给future pred
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans) # bs, fut_len, outdim


'''
这是不同的解码器方案 TransformerLinear
'''
# class TransformerLinear(Module):

#     def __init__(self, point_dim, context_dim, residual):
#         super().__init__()
#         self.residual = residual

#         self.pos_emb = PositionalEncoding(d_model=128, dropout=0.1, max_len=24)
#         self.y_up = nn.Linear(2, 128)
#         self.ctx_up = nn.Linear(context_dim+3, 128)
#         self.layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=512)
#         self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=3)
#         self.linear = nn.Linear(128, point_dim)

#     def forward(self, x, beta, context):

#         batch_size = x.size(0)
#         beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
#         context = context.view(batch_size, 1, -1)   # (B, 1, F)

#         time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
#         ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

#         ctx_emb = self.ctx_up(ctx_emb)
#         emb = self.y_up(x)
#         final_emb = torch.cat([ctx_emb, emb], dim=1).permute(1,0,2)
#         #pdb.set_trace()
#         final_emb = self.pos_emb(final_emb)

#         trans = self.transformer_encoder(final_emb)  # 13 * b * 128
#         trans = trans[1:].permute(1,0,2)   # B * 12 * 128, drop the first one which is the z
#         return self.linear(trans)







# class LinearDecoder(Module):
#     def __init__(self):
#             super().__init__()
#             self.act = F.leaky_relu
#             self.layers = ModuleList([
#                 #nn.Linear(2, 64),
#                 nn.Linear(32, 64),
#                 nn.Linear(64, 128),
#                 nn.Linear(128, 256),
#                 nn.Linear(256, 512),
#                 nn.Linear(512, 256),
#                 nn.Linear(256, 128),
#                 nn.Linear(128, 12)
#                 #nn.Linear(2, 64),
#                 #nn.Linear(2, 64),
#             ])
#     def forward(self, code):

#         out = code
#         for i, layer in enumerate(self.layers):
#             out = layer(out)
#             if i < len(self.layers) - 1:
#                 out = self.act(out)
#         return out
