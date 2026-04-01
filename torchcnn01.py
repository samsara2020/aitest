import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 定义 CNN 模型结构
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # --- 特征提取部分 (卷积 + 池化) ---
        
        # 第一层卷积：输入通道1 (灰度图), 输出通道32, 卷积核3x3
        # 输入尺寸: 28x28 -> 输出尺寸: 26x26 (无padding)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu1 = nn.ReLU()
        # 最大池化：2x2 窗口
        # 尺寸: 26x26 -> 13x13
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # 第二层卷积：输入32, 输出64, 卷积核3x3
        # 尺寸: 13x13 -> 11x11
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.relu2 = nn.ReLU()
        # 最大池化：2x2 窗口
        # 尺寸: 11x11 -> 5x5
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # --- 分类部分 (全连接层) ---
        # 展平后的维度计算: 通道数(64) * 高(5) * 宽(5) = 1600
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10) # 输出10个类别 (0-9)

    def forward(self, x):
        # 前向传播流程
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 展平数据以输入全连接层
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# 2. 准备数据 (MNIST 手写数字)
transform = transforms.Compose([
    transforms.ToTensor(), # 将图片转换为 Tensor 并归一化到 [0, 1]
    transforms.Normalize((0.1307,), (0.3081,)) # 使用 MNIST 的均值和标准差进行标准化
])

# 下载并加载训练集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# 下载并加载测试集
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# 3. 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss() # 分类问题常用交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练循环
print(f"开始在 {device} 上训练...")
epochs = 3

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播 & 优化
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 5. 测试模型
print("\n开始测试模型...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"测试集准确率: {100 * correct / total:.2f}%")