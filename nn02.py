import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 全局设置
# ==========================================
torch.manual_seed(42)  # 固定随机种子

# 定义设备 (如果有GPU则使用GPU，否则CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前设备: {device}")

# ==========================================
# 1. 单神经元，单特征，线性回归
# ==========================================
def model_1():
    print("\n" + "="*60)
    print("1. PyTorch: 单神经元，单特征，线性回归")
    print("="*60)
    
    # [数据定义]
    # PyTorch 数据必须是 float 类型
    X = torch.linspace(0, 2, 100).reshape(-1, 1) 
    y = 4 * X + 3 + torch.randn(100, 1) * 0.5 # y = 4x + 3 + noise
    
    print(f"[数据] X形状: {X.shape}, y形状: {y.shape}")

    # [模型定义]
    # nn.Linear(输入特征数, 输出特征数)
    model = nn.Linear(1, 1)
    
    # [损失函数]
    criterion = nn.MSELoss()
    
    # [优化器]
    # SGD (随机梯度下降), lr是学习率, model.parameters()是要更新的参数
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    # [训练循环]
    for epoch in range(1000):
        # --- 前向传播 ---
        y_pred = model(X)
        
        # --- 计算损失 ---
        loss = criterion(y_pred, y)
        
        # --- 反向传播 ---
        # 1. 清空梯度 (PyTorch默认会累加梯度，所以每步必须清零)
        optimizer.zero_grad()
        
        # 2. 计算梯度 (自动计算 dL/dw 和 dL/db)
        loss.backward()
        
        # 3. 更新参数
        optimizer.step()
        
        if epoch % 200 == 0:
            # 获取权重和偏置
            w = model.weight.data[0, 0]
            b = model.bias.data[0]
            print(f"Epoch {epoch} | Loss: {loss.item():.4f} | w: {w:.2f}, b: {b:.2f}")

# ==========================================
# 2. 单神经元，单特征，逻辑回归
# ==========================================
def model_2():
    print("\n" + "="*60)
    print("2. PyTorch: 单神经元，单特征，逻辑回归")
    print("="*60)
    
    # [数据定义]
    X = torch.linspace(0, 2, 100).reshape(-1, 1)
    # 标签生成
    y = (X > 1).float() 
    # 添加噪声
    noise_idx = torch.randint(0, 100, (10,))
    y[noise_idx] = 1 - y[noise_idx]

    print(f"[数据] X形状: {X.shape}, y形状: {y.shape}")

    # [模型定义]
    model = nn.Linear(1, 1)
    
    # [损失函数]
    # BCEWithLogitsLoss 结合了 Sigmoid 和 BCELoss，数值更稳定
    # 如果模型最后没有加 Sigmoid，就用这个。
    criterion = nn.BCEWithLogitsLoss()
    
    # [优化器]
    optimizer = optim.SGD(model.parameters(), lr=1.0)

    # [训练循环]
    for epoch in range(1000):
        # --- 前向传播 ---
        # 注意：BCEWithLogitsLoss 期望输入是原始 logits (未过 Sigmoid)
        y_pred = model(X)
        
        # --- 计算损失 ---
        loss = criterion(y_pred, y)
        
        # --- 反向传播 ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# ==========================================
# 3. 单神经元，多特征
# ==========================================
def model_3():
    print("\n" + "="*60)
    print("3. PyTorch: 单神经元，多特征 (多元线性回归)")
    print("="*60)
    
    # [数据定义]
    n_features = 5
    X = torch.rand(100, n_features)
    # 真实权重
    true_w = torch.tensor([[1.5], [-2.0], [0.5], [3.0], [-1.0]])
    y = torch.matmul(X, true_w) + torch.randn(100, 1) * 0.1

    print(f"[数据] X形状: {X.shape}, y形状: {y.shape}")

    # [模型定义]
    model = nn.Linear(n_features, 1)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # [训练循环]
    for epoch in range(1000):
        y_pred = model(X)
        loss = criterion(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# ==========================================
# 4. 多神经元，多特征，无隐藏层，不同激活函数
# ==========================================
def model_4():
    print("\n" + "="*60)
    print("4. PyTorch: 多神经元，多特征，无隐藏层")
    print("="*60)
    
    n_features = 4
    n_outputs = 3
    X = torch.rand(100, n_features)
    y = torch.rand(100, n_outputs)

    # 定义不同的激活函数类
    activations = [
        ('Sigmoid', nn.Sigmoid()),
        ('ReLU', nn.ReLU()),
        ('Tanh', nn.Tanh())
    ]

    for name, act_func in activations:
        print(f"\n--- 测试激活函数: {name} ---")
        
        # 模型: 线性层 -> 激活函数
        # nn.Sequential 允许我们将层串联起来
        model = nn.Sequential(
            nn.Linear(n_features, n_outputs),
            act_func
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        for epoch in range(1000):
            y_pred = model(X)
            loss = criterion(y_pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 200 == 0:
                print(f"  Epoch {epoch} | Loss: {loss.item():.4f}")

# ==========================================
# 5. 1个隐藏层
# ==========================================
def model_5():
    print("\n" + "="*60)
    print("5. PyTorch: 1个隐藏层")
    print("="*60)
    
    n_features, n_hidden, n_outputs = 4, 5, 1
    X = torch.rand(100, n_features)
    y = torch.rand(100, n_outputs)

    # [模型定义]
    # 使用 Sequential 构建: 输入 -> 隐藏层(Linear+ReLU) -> 输出层(Linear)
    model = nn.Sequential(
        nn.Linear(n_features, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_outputs)
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(1000):
        y_pred = model(X)
        loss = criterion(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# ==========================================
# 6. 4个隐藏层
# ==========================================
def model_6():
    print("\n" + "="*60)
    print("6. PyTorch: 4个隐藏层 (深层网络)")
    print("="*60)
    
    n_features = 4
    n_outputs = 1
    
    X = torch.rand(100, n_features)
    y = torch.rand(100, n_outputs)

    # [模型定义]
    # 结构: 4 -> 10 -> 8 -> 6 -> 4 -> 1
    model = nn.Sequential(
        nn.Linear(4, 10), nn.ReLU(),
        nn.Linear(10, 8), nn.ReLU(),
        nn.Linear(8, 6),  nn.ReLU(),
        nn.Linear(6, 4),  nn.ReLU(),
        nn.Linear(4, 1)
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01) # 深层网络常用 Adam 优化器

    for epoch in range(2000):
        y_pred = model(X)
        loss = criterion(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 400 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# ==========================================
# 主菜单
# ==========================================
def main_menu():
    while True:
        print("\n" + "#"*30)
        print("PyTorch 神经网络演示菜单")
        print("#"*30)
        print("1. 单神经元，单特征，线性回归")
        print("2. 单神经元，单特征，逻辑回归")
        print("3. 单神经元，多特征")
        print("4. 多神经元，多特征，无隐藏层")
        print("5. 1个隐藏层")
        print("6. 4个隐藏层")
        print("0. 退出")
        
        choice = input("\n请输入选项 (0-6): ")
        
        if choice == '1':
            model_1()
        elif choice == '2':
            model_2()
        elif choice == '3':
            model_3()
        elif choice == '4':
            model_4()
        elif choice == '5':
            model_5()
        elif choice == '6':
            model_6()
        elif choice == '0':
            print("退出程序。")
            break
        else:
            print("无效选项，请重新输入。")

if __name__ == "__main__":
    main_menu()