import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        1. 初始化网络参数
        权重用小的随机数初始化，偏置初始化为0。
        """
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        """2. 定义激活函数"""
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        """
        3. 前向传播
        计算从输入到输出的预测值，并缓存中间结果供反向传播使用。
        """
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_pred = self.sigmoid(self.z2)
        return self.y_pred
    
    def backward(self, X, y, lr=1.0):
        """
        4. 反向传播
        计算损失函数对各个参数的梯度，并更新参数。
        """
        m = X.shape[0]  # 样本数量
        
        # 1) 计算输出层的误差   
        dz2 = (self.y_pred - y) / m
        
        # 2) 计算输出层参数的梯度
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # 3) 将误差反向传播到隐藏层
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.a1 * (1 - self.a1)  # Sigmoid的导数
        
        # 4) 计算隐藏层参数的梯度
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # 5) 梯度下降，更新参数
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
    
    def train(self, X, y, epochs, lr=1.0):
        """5. 定义训练循环"""
        for i in range(epochs):
            # 前向传播
            y_pred = self.forward(X)
            
            # 计算均方误差损失
            loss = np.mean((y - y_pred) ** 2)
            
            # 反向传播
            self.backward(X, y, lr)
            
            # 打印训练进度
            if i % 1000 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")

# ==========================================
# 6. 准备数据并开始训练
# ==========================================

# XOR 数据集
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# 创建网络实例 (2输入 -> 4隐藏层 -> 1输出)
nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)

# 开始训练
print("开始训练...")
nn.train(X, y, epochs=5000, lr=1.0)

# 训练完成后进行预测
print("\n训练完成，进行预测:")
for i in range(len(X)):
    pred = nn.forward(X[i:i+1])  # 逐个预测
    print(f"输入: {X[i]} -> 预测: {pred[0][0]:.4f} (真实值: {y[i][0]})")