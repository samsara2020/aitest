import numpy as np
import torch
import torch.nn as nn

a=nn.Embedding(10, 3) # vocab_size=10, embedding_dim=3
print(a.weight) # 10x3的矩阵，随机初始化的词向量
print(a.weight.shape) # torch.Size([10, 3])
print(a) 