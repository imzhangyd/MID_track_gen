## 环境配置
要求是torch1.7.1，cuda>10.1
结合服务器有的cuda，以及torch版本匹配的cuda，选择在GPU上的cuda10.2
在.zshrc修改cuda路径，source .zshrc

```bash
conda env create -f ./env/mid_env.yaml
```

.whl文件可以手动下载，直接 pip install 就行

## 数据集格式
txt格式存储的dataframe, 四列 
```
frame，trackid，x, y
```

## 先跑一次原数据的pipe

```bash
python process_data.py

python main.py --config configs/reproduce.yaml --dataset eth

```
## dataset
### pkl文件
是一个 Environment类，集合了多个视频
train_env.scenes 是一个 list of Scene
一个Scene对应一个视频
duration是时长 s为单位
dt就是每一帧的时间间隔 s为单位
timesteps 是帧数
如果是训练数据，会有augment的Scene，旋转角度之类的

### EnvironmnetDataset
Dataset 类 场景所有类型的object，每一类object用一个NodeTypeDataset类的示例
    实际上进一步是 NodeTypeDataset类做的初始化和get

### NodeTypeDataset类
在self.index中存储了每个sample

具体是index_env 处理pkl文件中的scenes，变成所有的samples

每个sample是[scene, t, node]
sample的长度就是，所有视频中的检测点的数量


### Node
每一个node都：类型，指定状态的完整序列，开始的frame_id

### get_node_timestep_data
每次训练加载数据都会做滑动窗口的取值操作。
类似滑动窗口取history和future
取neighbor 的state 

first_history_index = 0是不是这个轨迹的开始帧
x_t x_st_t : 8,6
y_t y_st_t : 12,2
neighbors_data_st  adict  m个neighbor的 8,6
neighbors_edge_value  adict  m个值

## train a batch
a batch
len = 9
batch[0].shape=256 : [3,0,0,...2,0,0,...5,4,0,0,0] 第一帧
batch[1].shape=256,8,6 256个sample的轨迹hist
batch[2].shape=256,12,2 对应的未来 
batch[3].shape=256,8,6 
batch[4].shape=256,12,2  
应该是归一化的
batch[5] adict value是 alist Len=256 对应每个sample的邻居 hist
每个sample的邻居也是 alist of m tensor
每个sample的邻居个数是不一样的，
hist也是 shape=8,6 

batch[6] 就是各个邻居的edge value 最大是1

batch[7] 和batch[8]都是空的

### forward-encoder历史和邻居

encoder 用的trajectron get——latent
进而调用Multimodal GenearativeCVAE的get_latent

调用self.obtain_encoded_tensors

编码轨迹自身的history  encode_node_history

run_lstm_on_variable_length_seqs

h_n 和 c_n 是LSTM的隐藏状态和单元状态
单元状态：在整个序列的处理过程中，c_n能够保存从序列开头到结尾的重要信息，使得 LSTM 能够处理长序列数据而不会像传统 RNN 那样容易忘记早期的信息。
隐藏状态：包含了 LSTM 从输入序列中提取的高层语义信息。例如，在自然语言处理中，如果输入是一个句子序列，h_n就包含了句子的语义、语法等综合信息。它可以用于后续的任务，如分类任务（将句子分类为不同的情感类别）、生成任务（基于前面的语义生成后续的句子内容）等。


编码得到
output.hsape=[bs, hist_len, feat_dim=128],h_n, c_n shape=[1, bs, feat_dim]
但没有用h_n 和 c_n

输出：每个序列的最后一个位置的编码向量

然后编码邻居，邻居的历史直接sum，然后与轨迹历史cat，过LSTM，取轨迹最新的feat，得到bs,128的维度。
通过attention，将邻居embedding映射到track空间。

最后把两者cat一起，得到x
y就是轨迹的future标准化，
还有x最后位置的标准化


self.node_modules.keys()
dict_keys(['PEDESTRIAN/node_history_encoder', 'PEDESTRIAN/node_future_encoder', 'PEDESTRIAN/node_future_encoder/initial_h', 'PEDESTRIAN/node_future_encoder/initial_c', 'PEDESTRIAN/edge_influence_encoder', 'PEDESTRIAN/p_z_x', 'PEDESTRIAN/hx_to_z', 'PEDESTRIAN/hxy_to_z', 'PEDESTRIAN/decoder/state_action', 'PEDESTRIAN/decoder/rnn_cell', 'PEDESTRIAN/decoder/initial_h', 'PEDESTRIAN/decoder/proj_to_GMM_log_pis', 'PEDESTRIAN/decoder/proj_to_GMM_mus', 'PEDESTRIAN/decoder/proj_to_GMM_log_sigmas', 'PEDESTRIAN/decoder/proj_to_GMM_corrs', 'PEDESTRIAN->PEDESTRIAN/edge_encoder'])


### forward-diffusion 获得loss

DuffusionTraj--get_loss
先给x_0添加噪声
然后经过self.net--TransformerConcatLinear

大概就是用contex生成w和b给future_pred,中间future_pred还在bs维度做了transformer

然后最终生成 bs, futurelen, 2

这是预测的噪声，与原本添加的噪声做MSE loss

## eval/val
To evaluate a trained-model, please set ```eval_mode``` in config file to True and set the epoch you'd like to evaluate at from ```eval_at``` and run
```bash
python main.py --config configs/reproduce.yaml --dataset eth
```

### 取数据
遍历eval视频，每隔10帧取其中的node聚合一个batch

也是先给轨迹和邻居 encoder，采样噪声，denose网络预测噪声
恢复预测轨迹结果

计算指标
ADE是各个位置上的均方误差
fde是最后一个位置上的均方误差


##
这个任务就是，input 一个场景下，坐标点和历史轨迹，然后预测未来的位置。

如果想用这个方法生成数据
初始化第一帧：生成一张点的分布
然后送入模型预测各个点的未来轨迹，预测一个就可以
可以设置只预测下两帧这样，接受两帧的结果
然后更新轨迹和场景，然后再继续预测

需要做的就是，每一帧点的出现和消失，需要我们来设置

需要写的就是一个，外部的控制过程，内部的模型训练还是用这个就可以。

或者有没有可能：让这个模型可以学习到停止的情况。

总的来说，就是先看一下用这个模型训练粒子轨迹的预测效果如何，
不同的粒子运动状态，以及一些细胞的轨迹都可以。


## 制作粒子/细胞数据集


python main.py --config configs/microtubule_low.yaml --dataset microtubule_low

python main.py --config configs/receptor_low.yaml --dataset receptor_low



