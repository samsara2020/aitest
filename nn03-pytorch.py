import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class LinearRegressionTorch(nn.Module):
    """1. 单输出层神经元，无隐藏层，单特征，线性回归"""
    def __init__(self):
        super().__init__()
        # 输入特征数=1，输出特征数=1
        # 权重形状: (1, 1), 偏置形状: (1)
        self.linear = nn.Linear(1, 1)
        # 参数数量: 1*1 + 1 = 2个标量
        
    def forward(self, x):
        """
        x: 特征数据 - 形状 (batch_size, 特征数=1)
        """
        return self.linear(x)

class LogisticRegressionTorch(nn.Module):
    """2. 单输出层神经元，无隐藏层，单特征，逻辑回归"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.sigmoid(self.linear(x))

class MultiFeatureLinearTorch(nn.Module):
    """3. 单输出层神经元，无隐藏层，多特征"""
    def __init__(self, n_features):
        super().__init__()
        # n_features: 输入特征数，输出特征数=1
        # 权重形状: (1, n_features), 偏置形状: (1)
        self.linear = nn.Linear(n_features, 1)
        # 参数数量: n_features + 1
        
    def forward(self, x):
        return self.linear(x)

class MultiOutputLinearTorch(nn.Module):
    """4. 多输出层神经元，无隐藏层，多特征"""
    def __init__(self, n_features, n_outputs):
        super().__init__()
        # n_features: 输入特征数，n_outputs: 输出特征数
        # 权重形状: (n_outputs, n_features), 偏置形状: (n_outputs)
        self.linear = nn.Linear(n_features, n_outputs)
        # 参数数量: n_outputs * n_features + n_outputs
        
    def forward(self, x):
        return self.linear(x)

class SingleHiddenSingleOutputTorch(nn.Module):
    """5. 单输出层神经元，1隐藏层，多特征"""
    def __init__(self, n_features, hidden_size=4):
        super().__init__()
        # 输入层: n_features -> 隐藏层: hidden_size
        # 权重形状: (hidden_size, n_features), 偏置形状: (hidden_size)
        self.fc1 = nn.Linear(n_features, hidden_size)
        # 隐藏层: hidden_size -> 输出层: 1
        # 权重形状: (1, hidden_size), 偏置形状: (1)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        # 参数数量: hidden_size*n_features + hidden_size + 1*hidden_size + 1
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SingleHiddenMultiOutputTorch(nn.Module):
    """6. 多输出层神经元，1隐藏层，多特征"""
    def __init__(self, n_features, n_outputs, hidden_size=4):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_outputs)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FourHiddenSingleOutputTorch(nn.Module):
    """7. 单输出层神经元，4隐藏层，多特征"""
    def __init__(self, n_features, hidden_sizes=[64, 32, 16, 8]):
        super().__init__()
        layers = []
        prev_size = n_features
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))  # 输出层
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class FourHiddenMultiOutputTorch(nn.Module):
    """8. 多输出层神经元，4隐藏层，多特征"""
    def __init__(self, n_features, n_outputs, hidden_sizes=[64, 32, 16, 8]):
        super().__init__()
        layers = []
        prev_size = n_features
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, n_outputs))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_torch_model(model, X, Y, epochs=500, lr=0.01):
    """PyTorch训练函数"""
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    return losses

def run_torch_menu():
    """PyTorch版本交互菜单"""
    models = {
        1: ("线性回归 - 单特征", LinearRegressionTorch),
        2: ("逻辑回归 - 单特征", LogisticRegressionTorch),
        3: ("线性回归 - 多特征", lambda: MultiFeatureLinearTorch(n_features=5)),
        4: ("多输出线性回归 - 多特征", lambda: MultiOutputLinearTorch(n_features=5, n_outputs=3)),
        5: ("1隐藏层单输出 - 多特征", lambda: SingleHiddenSingleOutputTorch(n_features=5)),
        6: ("1隐藏层多输出 - 多特征", lambda: SingleHiddenMultiOutputTorch(n_features=5, n_outputs=3)),
        7: ("4隐藏层单输出 - 多特征", lambda: FourHiddenSingleOutputTorch(n_features=5)),
        8: ("4隐藏层多输出 - 多特征", lambda: FourHiddenMultiOutputTorch(n_features=5, n_outputs=3))
    }
    
    while True:
        print("\n" + "="*60)
        print("PyTorch神经网络演示程序")
        print("="*60)
        for k, (name, _) in models.items():
            print(f"{k}. {name}")
        print("0. 退出")
        
        try:
            choice = int(input("\n请选择模型 (0-8): "))
            if choice == 0:
                print("退出程序")
                break
            if choice not in models:
                print("无效选择")
                continue
            
            name, model_class = models[choice]
            print(f"\n运行: {name}")
            
            # 生成数据
            if choice in [1, 2]:
                X_np = np.random.randn(200, 1).astype(np.float32)
                if choice == 1:
                    Y_np = (2.5 * X_np + 1 + np.random.randn(200, 1) * 0.5).astype(np.float32)
                else:
                    prob = 1 / (1 + np.exp(-(2 * X_np + 1)))
                    Y_np = (prob > 0.5).astype(np.float32)
            else:
                n_samples = 200
                n_features = 5
                X_np = np.random.randn(n_samples, n_features).astype(np.float32)
                if choice in [3, 5, 7]:  # 单输出
                    true_weights = np.random.randn(1, n_features).astype(np.float32)
                    Y_np = (np.dot(X_np, true_weights.T) + 0.5 + np.random.randn(n_samples, 1) * 0.1).astype(np.float32)
                else:  # 多输出
                    n_outputs = 3
                    true_weights = np.random.randn(n_outputs, n_features).astype(np.float32)
                    Y_np = (np.dot(X_np, true_weights.T) + 0.5 + np.random.randn(n_samples, n_outputs) * 0.1).astype(np.float32)
            
            X = torch.from_numpy(X_np)
            Y = torch.from_numpy(Y_np)
            
            # 创建模型
            if choice in [1, 2]:
                model = model_class()
                lr = 0.1
            elif choice == 3:
                model = MultiFeatureLinearTorch(n_features=5)
                lr = 0.1
            elif choice == 4:
                model = MultiOutputLinearTorch(n_features=5, n_outputs=3)
                lr = 0.1
            elif choice == 5:
                model = SingleHiddenSingleOutputTorch(n_features=5)
                lr = 0.1
            elif choice == 6:
                model = SingleHiddenMultiOutputTorch(n_features=5, n_outputs=3)
                lr = 0.1
            elif choice == 7:
                model = FourHiddenSingleOutputTorch(n_features=5)
                lr = 0.05
            elif choice == 8:
                model = FourHiddenMultiOutputTorch(n_features=5, n_outputs=3)
                lr = 0.05
            
            # 打印参数信息
            print(f"\n特征数据形状: {X.shape} (batch_size, 特征数)")
            print(f"输出数据形状: {Y.shape} (batch_size, 输出神经元数)")
            print("\n参数形状:")
            total_params = 0
            for name, param in model.named_parameters():
                print(f"  {name}: {param.shape} -> {param.numel()}个参数")
                total_params += param.numel()
            print(f"总参数标量数量: {total_params}")
            
            # 训练
            print("\n开始训练...")
            losses = train_torch_model(model, X, Y, epochs=500, lr=lr)
            
            # 评估
            with torch.no_grad():
                Y_pred = model(X)
                final_loss = nn.MSELoss()(Y_pred, Y).item()
            print(f"\n最终损失: {final_loss:.6f}")
            
            # 绘图
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(losses)
            plt.title(f'{name} - 损失曲线')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            
            plt.subplot(1, 2, 2)
            Y_np_pred = Y_pred.numpy()
            if Y.shape[1] == 1:
                plt.scatter(X_np[:100, 0], Y_np[:100, 0], alpha=0.5, label='真实值')
                plt.scatter(X_np[:100, 0], Y_np_pred[:100, 0], alpha=0.5, label='预测值')
                plt.legend()
                plt.title('预测 vs 真实')
            else:
                plt.bar(range(Y.shape[1]), np.mean((Y_np - Y_np_pred)**2, axis=0))
                plt.title('各输出维度MSE')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_torch_menu()