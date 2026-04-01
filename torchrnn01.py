import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 1. 定义简单的数据集 (模拟递增/递减序列)
class SequenceDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=10):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.data = []
        self.labels = []
        
        for _ in range(num_samples):
            # 随机生成起始点和步长
            start = np.random.randint(0, 50)
            step = np.random.randint(1, 5)
            
            # 50% 概率生成递增序列，50% 概率生成递减序列
            if np.random.rand() > 0.5:
                sequence = [start + i * step for i in range(seq_length)]
                label = 1 # 1 代表递增
            else:
                sequence = [start - i * step for i in range(seq_length)]
                label = 0 # 0 代表递减
            
            # 归一化数据到 0-1 之间 (有助于训练稳定)
            sequence = [(x - min(sequence)) / (max(sequence) - min(sequence) + 1e-6) for x in sequence]
            
            self.data.append(sequence)
            self.labels.append(label)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # RNN 输入形状要求: (seq_length, input_size)
        # 这里 input_size = 1 (每个时间步只有一个数字)
        x = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

# 2. 定义 RNN 模型
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        
        # 核心 RNN 层
        # input_size: 每个时间步的输入特征维度 (这里是1)
        # hidden_size: 隐藏层状态的大小 (记忆单元的维度)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        
        # 全连接层，将隐藏层状态映射到输出类别
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Sigmoid 用于二分类输出 (0-1之间)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x 形状: (batch_size, seq_length, input_size)
        
        # rnn_output: (batch_size, seq_length, hidden_size)
        # h_n: (num_layers, batch_size, hidden_size) -> 最后一个时间步的隐藏状态
        out, h_n = self.rnn(x)
        
        # 我们只取最后一个时间步的隐藏状态 (h_n) 作为整个序列的特征表示
        # h_n 形状: (1, batch_size, hidden_size)，去掉第一维
        last_hidden_state = h_n[-1, :, :]
        
        # 通过全连接层得到最终预测
        output = self.fc(last_hidden_state)
        return self.sigmoid(output)

# 3. 准备数据
batch_size = 32
seq_length = 10
input_size = 1
hidden_size = 32
output_size = 1

train_dataset = SequenceDataset(num_samples=2000, seq_length=seq_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = SequenceDataset(num_samples=200, seq_length=seq_length)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 4. 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleRNN(input_size, hidden_size, output_size).to(device)
criterion = nn.BCELoss() # 二元交叉熵损失，用于二分类
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. 训练循环
print(f"开始在 {device} 上训练 RNN...")
epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.unsqueeze(1) # 调整标签形状为 (batch, 1)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        
        # 防止梯度爆炸 (RNN 常见问题)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# 6. 测试模型
print("\n开始测试模型...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predicted = (outputs >= 0.5).float()
        total += labels.size(0)
        correct += (predicted.squeeze() == labels).sum().item()

print(f"测试集准确率: {100 * correct / total:.2f}%")

# 7. 手动验证一个例子
print("\n--- 手动验证示例 ---")
model.eval()
# 创建一个明显的递增序列: 1, 2, 3, 4, 5
test_seq = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float32).unsqueeze(0).to(device)
# 归一化 (模拟数据集的处理逻辑，这里简化处理)
test_seq = (test_seq - 1) / 4 

with torch.no_grad():
    pred = model(test_seq)
    result = "递增" if pred.item() >= 0.5 else "递减"
    print(f"输入序列: [1, 2, 3, 4, 5] -> 预测结果: {result} (置信度: {pred.item():.4f})")