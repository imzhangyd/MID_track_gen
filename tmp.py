
# %%
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
    
# 输入特征 (序列长度=4, 特征维度=8)
d_model = 128
x = torch.rand(7, 4, d_model)  
max_len = 10
pos_encoder = PositionalEncoding(d_model, max_len=max_len)
# 添加位置编码
x_encoded = pos_encoder(x)
print(x_encoded.shape)  # 输出: torch.Size([4, 8])
import matplotlib.pyplot as plt

# 可视化位置编码
pe = pos_encoder.pe.squeeze(0).numpy()
print(pe.shape) # 10,1,d_model

plt.figure(figsize=(10, 6))
# plt.imshow(pe, cmap="viridis", aspect="auto")
for i in range(d_model):
    plt.plot(pe[:,0, i], label=f"dim {i}",color=[i/d_model,i/d_model,i/d_model])
# plt.legend()
plt.xlabel("Position")
plt.ylabel("encoder value")
# plt.colorbar()
# plt.title("Positional Encoding")
plt.show()
# plt.savefig('Positional Encoding.png')

# %%