import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层1: 输入1通道(灰度), 输出32通道, 核大小3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 池化层: 最大池化，将尺寸减半
        self.pool = nn.MaxPool2d(2, 2)
        # 卷积层2: 输入32通道, 输出64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 全连接层: 经过两次池化后，28x28 -> 14x14 -> 7x7
        # 输入维度: 64通道 * 7 * 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) # 输出10个数字类别

    def forward(self, x):
        # 卷积 -> ReLU激活 -> 池化
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # 展平操作，准备进入全连接层
        x = x.view(-1, 64 * 7 * 7)
        
        # 全连接层 -> ReLU -> 输出
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型
model = SimpleCNN()
print(model)