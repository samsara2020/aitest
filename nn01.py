import numpy as np

# ==========================================
# 全局设置与辅助函数
# ==========================================
np.random.seed(42)

# --- 激活函数及其导数 ---
def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_derivative(z): s = sigmoid(z); return s * (1 - s)
def relu(z): return np.maximum(0, z)
def relu_derivative(z): return np.where(z > 0, 1, 0)
def tanh(z): return np.tanh(z)
def tanh_derivative(z): return 1 - np.tanh(z) ** 2
def linear(z): return z
def linear_derivative(z): return np.ones_like(z)

# --- 损失函数 ---
def mse_loss(y_true, y_pred): return np.mean((y_true - y_pred) ** 2)
def bce_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# ==========================================
# 1. 单神经元，单特征，线性回归
# ==========================================
def model_1():
    print("\n" + "="*60)
    print("1. 单神经元，单特征，线性回归")
    print("="*60)
    
    # [数据定义]
    X = 2 * np.random.rand(100, 1)
    y = 4 * X + 3 + np.random.randn(100, 1)
    
    # [类型检查]
    print(f"[数据] X 类型: {type(X)}, 形状: {X.shape}, 数据类型: {X.dtype}")
    print(f"[数据] y 类型: {type(y)}, 形状: {y.shape}, 数据类型: {y.dtype}")

    w = np.random.randn(1, 1)
    b = np.random.randn(1)
    # [类型检查]
    print(f"[变量] w 类型: {type(w)}, 形状: {w.shape}, 数据类型: {w.dtype}")
    print(f"[变量] b 类型: {type(b)}, 形状: {b.shape}, 数据类型: {b.dtype}")

    lr = 0.1

    for epoch in range(1000):
        z = np.dot(X, w) + b
        y_pred = linear(z)
        loss = mse_loss(y, y_pred)
        
        # 反向传播
        dz = (2 / len(X)) * (y_pred - y)
        dw = np.dot(X.T, dz)
        db = np.sum(dz)
        
        w -= lr * dw
        b -= lr * db
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f} | w: {w[0,0]:.2f}, b: {b[0]:.2f}")

# ==========================================
# 2. 单神经元，单特征，逻辑回归
# ==========================================
def model_2():
    print("\n" + "="*60)
    print("2. 单神经元，单特征，逻辑回归")
    print("="*60)
    
    X = 2 * np.random.rand(100, 1)
    y = (X > 1).astype(int) 
    noise_idx = np.random.choice(100, 10)
    y[noise_idx] = 1 - y[noise_idx]

    # [类型检查]
    print(f"[数据] X 类型: {type(X)}, 形状: {X.shape}, 数据类型: {X.dtype}")
    print(f"[数据] y 类型: {type(y)}, 形状: {y.shape}, 数据类型: {y.dtype}")

    w = np.random.randn(1, 1)
    b = np.random.randn(1)
    # [类型检查]
    print(f"[变量] w 类型: {type(w)}, 形状: {w.shape}, 数据类型: {w.dtype}")
    print(f"[变量] b 类型: {type(b)}, 形状: {b.shape}, 数据类型: {b.dtype}")

    lr = 1.0

    for epoch in range(1000):
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)
        loss = bce_loss(y, y_pred)
        
        dz = y_pred - y
        dw = np.dot(X.T, dz)
        db = np.sum(dz)
        
        w -= lr * dw
        b -= lr * db
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f} | w: {w[0,0]:.2f}, b: {b[0]:.2f}")

# ==========================================
# 3. 单神经元，多特征
# ==========================================
def model_3():
    print("\n" + "="*60)
    print("3. 单神经元，多特征 (多元线性回归)")
    print("="*60)
    
    n_features = 5
    X = np.random.rand(100, n_features)
    true_w = np.array([[1.5], [-2.0], [0.5], [3.0], [-1.0]])
    y = np.dot(X, true_w) + 0.1 * np.random.randn(100, 1)

    # [类型检查]
    print(f"[数据] X 类型: {type(X)}, 形状: {X.shape}, 数据类型: {X.dtype}")
    print(f"[数据] y 类型: {type(y)}, 形状: {y.shape}, 数据类型: {y.dtype}")

    w = np.random.randn(n_features, 1)
    b = np.random.randn(1)
    # [类型检查]
    print(f"[变量] w 类型: {type(w)}, 形状: {w.shape}, 数据类型: {w.dtype}")
    print(f"[变量] b 类型: {type(b)}, 形状: {b.shape}, 数据类型: {b.dtype}")

    lr = 0.1

    for epoch in range(1000):
        z = np.dot(X, w) + b
        y_pred = linear(z)
        loss = mse_loss(y, y_pred)
        
        dz = (2 / len(X)) * (y_pred - y)
        dw = np.dot(X.T, dz)
        db = np.sum(dz)
        
        w -= lr * dw
        b -= lr * db
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")

# ==========================================
# 4. 多神经元，多特征，无隐藏层，不同激活函数
# ==========================================
def model_4():
    print("\n" + "="*60)
    print("4. 多神经元，多特征，无隐藏层 (不同激活函数)")
    print("="*60)
    
    n_features = 4
    n_outputs = 3
    X = np.random.rand(100, n_features)
    y = np.random.rand(100, n_outputs)

    # [类型检查]
    print(f"[数据] X 类型: {type(X)}, 形状: {X.shape}, 数据类型: {X.dtype}")
    print(f"[数据] y 类型: {type(y)}, 形状: {y.shape}, 数据类型: {y.dtype}")

    activations = [
        ('Sigmoid', sigmoid, sigmoid_derivative),
        ('ReLU', relu, relu_derivative),
        ('Tanh', tanh, tanh_derivative)
    ]

    for name, act_func, act_deriv in activations:
        print(f"\n--- 测试激活函数: {name} ---")
        W = np.random.randn(n_features, n_outputs)
        b = np.random.randn(1, n_outputs)
        # [类型检查]
        print(f"[变量] W 类型: {type(W)}, 形状: {W.shape}, 数据类型: {W.dtype}")
        print(f"[变量] b 类型: {type(b)}, 形状: {b.shape}, 数据类型: {b.dtype}")
        
        lr = 0.1
        
        for epoch in range(1000):
            z = np.dot(X, W) + b
            a = act_func(z)
            loss = mse_loss(y, a)
            
            da = (2 / len(X)) * (a - y)
            dz = da * act_deriv(z)
            dW = np.dot(X.T, dz)
            db = np.sum(dz, axis=0, keepdims=True)
            
            W -= lr * dW
            b -= lr * db
            
            if epoch % 200 == 0:
                print(f"  Epoch {epoch} | Loss: {loss:.4f}")

# ==========================================
# 5. 1个隐藏层
# ==========================================
def model_5():
    print("\n" + "="*60)
    print("5. 1个隐藏层 (隐藏层:ReLU, 输出层:Linear)")
    print("="*60)
    
    n_features, n_hidden, n_outputs = 4, 5, 1
    X = np.random.rand(100, n_features)
    y = np.random.rand(100, n_outputs)

    # [类型检查]
    print(f"[数据] X 类型: {type(X)}, 形状: {X.shape}, 数据类型: {X.dtype}")
    print(f"[数据] y 类型: {type(y)}, 形状: {y.shape}, 数据类型: {y.dtype}")

    W1 = np.random.randn(n_features, n_hidden)
    b1 = np.random.randn(1, n_hidden)
    W2 = np.random.randn(n_hidden, n_outputs)
    b2 = np.random.randn(1, n_outputs)
    
    # [类型检查]
    print(f"[变量] W1 类型: {type(W1)}, 形状: {W1.shape}, 数据类型: {W1.dtype}")
    print(f"[变量] b1 类型: {type(b1)}, 形状: {b1.shape}, 数据类型: {b1.dtype}")
    print(f"[变量] W2 类型: {type(W2)}, 形状: {W2.shape}, 数据类型: {W2.dtype}")
    print(f"[变量] b2 类型: {type(b2)}, 形状: {b2.shape}, 数据类型: {b2.dtype}")
    
    lr = 0.1

    for epoch in range(1000):
        Z1 = np.dot(X, W1) + b1
        A1 = relu(Z1)
        
        Z2 = np.dot(A1, W2) + b2
        A2 = linear(Z2)
        
        loss = mse_loss(y, A2)
        
        dA2 = (2 / len(X)) * (A2 - y)
        dZ2 = dA2 * linear_derivative(Z2)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")

# ==========================================
# 6. 4个隐藏层
# ==========================================
def model_6():
    print("\n" + "="*60)
    print("6. 4个隐藏层 (深层网络)")
    print("="*60)
    
    n_features = 4
    n_outputs = 1
    layers = [n_features, 10, 8, 6, 4, n_outputs]
    
    X = np.random.rand(100, n_features)
    y = np.random.rand(100, n_outputs)

    # [类型检查]
    print(f"[数据] X 类型: {type(X)}, 形状: {X.shape}, 数据类型: {X.dtype}")
    print(f"[数据] y 类型: {type(y)}, 形状: {y.shape}, 数据类型: {y.dtype}")

    params = {}
    for i in range(1, len(layers)):
        params[f'W{i}'] = np.random.randn(layers[i-1], layers[i])
        params[f'b{i}'] = np.random.randn(1, layers[i])
        
    # [类型检查]
    for i in range(1, len(layers)):
        print(f"[变量] W{i} 类型: {type(params[f'W{i}'])}, 形状: {params[f'W{i}'].shape}, 数据类型: {params[f'W{i}'].dtype}")
        print(f"[变量] b{i} 类型: {type(params[f'b{i}'])}, 形状: {params[f'b{i}'].shape}, 数据类型: {params[f'b{i}'].dtype}")
        
    lr = 0.01

    for epoch in range(2000):
        cache = {'A0': X}
        for i in range(1, len(layers)):
            Z = np.dot(cache[f'A{i-1}'], params[f'W{i}']) + params[f'b{i}']
            act = relu if i < len(layers) - 1 else linear
            cache[f'A{i}'] = act(Z)
            cache[f'Z{i}'] = Z
            
        A_out = cache[f'A{len(layers)-1}']
        loss = mse_loss(y, A_out)
        
        grads = {}
        dA = (2 / len(X)) * (A_out - y)
        
        for i in range(len(layers) - 1, 0, -1):
            act = relu if i < len(layers) - 1 else linear
            act_deriv = relu_derivative if i < len(layers) - 1 else linear_derivative
            
            dZ = dA * act_deriv(cache[f'Z{i}'])
            grads[f'dW{i}'] = np.dot(cache[f'A{i-1}'].T, dZ)
            grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True)
            
            if i > 1:
                dA = np.dot(dZ, params[f'W{i}'].T)
        
        for i in range(1, len(layers)):
            params[f'W{i}'] -= lr * grads[f'dW{i}']
            params[f'b{i}'] -= lr * grads[f'db{i}']
            
        if epoch % 400 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")

# ==========================================
# 主菜单
# ==========================================
def main_menu():
    while True:
        print("\n" + "#"*30)
        print("神经网络算法演示菜单")
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