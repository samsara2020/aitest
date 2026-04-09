import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler

class NeuralNetworkNumPy:
    """NumPy实现的神经网络基类"""
    def __init__(self, layer_dims, learning_rate=0.01):
        """
        参数:
        layer_dims: list, 每层神经元数量 [输入特征数, 隐藏层1..., 输出层特征数]
        learning_rate: float, 学习率
        """
        self.layer_dims = layer_dims
        self.lr = learning_rate
        self.parameters = {}
        self.cache = {}
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """初始化权重和偏置"""
        np.random.seed(42)
        for l in range(1, len(self.layer_dims)):
            # 权重形状: (当前层神经元数, 上一层神经元数)
            self.parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
            # 偏置形状: (当前层神经元数, 1)
            self.parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))
    
    def linear_forward(self, A_prev, W, b):
        """线性前向传播 Z = W·A_prev + b"""
        Z = np.dot(W, A_prev) + b
        return Z
    
    def sigmoid(self, Z):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-Z))
    
    def relu(self, Z):
        """ReLU激活函数"""
        return np.maximum(0, Z)
    
    def linear_backward(self, dZ, A_prev, W):
        """线性反向传播"""
        m = A_prev.shape[1]
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        return dW, db, dA_prev
    
    def update_parameters(self, grads):
        """梯度下降更新参数"""
        for l in range(1, len(self.layer_dims)):
            self.parameters[f'W{l}'] -= self.lr * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= self.lr * grads[f'db{l}']

class LinearRegressionNumPy(NeuralNetworkNumPy):
    """1. 单输出层神经元，无隐藏层，单特征，线性回归"""
    def __init__(self, learning_rate=0.01):
        # 输入特征数=1，输出神经元数=1
        super().__init__(layer_dims=[1, 1], learning_rate=learning_rate)
        # 参数形状: W1: (1,1), b1: (1,1)
        # 权重参数标量数量: 1*1 + 1 = 2个 (W和b)
    
    def forward(self, X):
        """
        X: 特征数据 - 形状 (特征数=1, 样本数)
        """
        self.cache['A0'] = X
        Z1 = self.linear_forward(X, self.parameters['W1'], self.parameters['b1'])
        self.cache['Z1'] = Z1
        return Z1  # 线性回归无激活函数
    
    def compute_loss(self, Y_pred, Y_true):
        """MSE损失"""
        m = Y_true.shape[1]
        loss = np.mean((Y_pred - Y_true) ** 2)
        return loss
    
    def backward(self, Y_pred, Y_true, X):
        """反向传播"""
        m = X.shape[1]
        dZ = (Y_pred - Y_true)  # 线性回归梯度: 2*(y_pred-y_true)/m，简化后
        dW, db, dA_prev = self.linear_backward(dZ, X, self.parameters['W1'])
        self.grads = {'dW1': dW, 'db1': db}
        return self.grads

class LogisticRegressionNumPy(NeuralNetworkNumPy):
    """2. 单输出层神经元，无隐藏层，单特征，逻辑回归"""
    def __init__(self, learning_rate=0.01):
        super().__init__(layer_dims=[1, 1], learning_rate=learning_rate)
        # 参数形状: W1: (1,1), b1: (1,1)
    
    def forward(self, X):
        self.cache['A0'] = X
        Z1 = self.linear_forward(X, self.parameters['W1'], self.parameters['b1'])
        A1 = self.sigmoid(Z1)  # Sigmoid输出概率
        self.cache['Z1'] = Z1
        self.cache['A1'] = A1
        return A1
    
    def compute_loss(self, Y_pred, Y_true):
        """二元交叉熵损失"""
        m = Y_true.shape[1]
        loss = -np.mean(Y_true * np.log(Y_pred + 1e-8) + (1 - Y_true) * np.log(1 - Y_pred + 1e-8))
        return loss
    
    def backward(self, Y_pred, Y_true, X):
        dZ = Y_pred - Y_true  # 逻辑回归梯度简化形式
        dW, db, dA_prev = self.linear_backward(dZ, X, self.parameters['W1'])
        self.grads = {'dW1': dW, 'db1': db}
        return self.grads

class MultiFeatureLinear(NeuralNetworkNumPy):
    """3. 单输出层神经元，无隐藏层，多特征"""
    def __init__(self, n_features, learning_rate=0.01):
        # n_features: 输入特征数，输出神经元数=1
        super().__init__(layer_dims=[n_features, 1], learning_rate=learning_rate)
        # 参数形状: W1: (1, n_features), b1: (1,1)
        # 权重参数标量数量: n_features + 1
    
    def forward(self, X):
        self.cache['A0'] = X
        Z1 = self.linear_forward(X, self.parameters['W1'], self.parameters['b1'])
        return Z1
    
    def compute_loss(self, Y_pred, Y_true):
        return np.mean((Y_pred - Y_true) ** 2)
    
    def backward(self, Y_pred, Y_true, X):
        dZ = Y_pred - Y_true
        dW, db, dA_prev = self.linear_backward(dZ, X, self.parameters['W1'])
        self.grads = {'dW1': dW, 'db1': db}
        return self.grads

class MultiOutputLinear(NeuralNetworkNumPy):
    """4. 多输出层神经元，无隐藏层，多特征"""
    def __init__(self, n_features, n_outputs, learning_rate=0.01):
        # n_features: 输入特征数，n_outputs: 输出神经元数
        super().__init__(layer_dims=[n_features, n_outputs], learning_rate=learning_rate)
        # 参数形状: W1: (n_outputs, n_features), b1: (n_outputs, 1)
        # 权重参数标量数量: n_outputs * n_features + n_outputs
    
    def forward(self, X):
        self.cache['A0'] = X
        Z1 = self.linear_forward(X, self.parameters['W1'], self.parameters['b1'])
        return Z1
    
    def compute_loss(self, Y_pred, Y_true):
        return np.mean((Y_pred - Y_true) ** 2)
    
    def backward(self, Y_pred, Y_true, X):
        dZ = Y_pred - Y_true
        dW, db, dA_prev = self.linear_backward(dZ, X, self.parameters['W1'])
        self.grads = {'dW1': dW, 'db1': db}
        return self.grads

class SingleHiddenSingleOutput(NeuralNetworkNumPy):
    """5. 单输出层神经元，1隐藏层，多特征"""
    def __init__(self, n_features, hidden_size=4, learning_rate=0.01):
        # 输入层: n_features, 隐藏层: hidden_size, 输出层: 1
        super().__init__(layer_dims=[n_features, hidden_size, 1], learning_rate=learning_rate)
        # 参数形状:
        # W1: (hidden_size, n_features), b1: (hidden_size, 1)
        # W2: (1, hidden_size), b2: (1, 1)
        # 权重参数标量数量: hidden_size*n_features + hidden_size + 1*hidden_size + 1
    
    def forward(self, X):
        self.cache['A0'] = X
        # 隐藏层
        Z1 = self.linear_forward(X, self.parameters['W1'], self.parameters['b1'])
        A1 = self.relu(Z1)
        # 输出层
        Z2 = self.linear_forward(A1, self.parameters['W2'], self.parameters['b2'])
        self.cache['Z1'], self.cache['A1'], self.cache['Z2'] = Z1, A1, Z2
        return Z2
    
    def compute_loss(self, Y_pred, Y_true):
        return np.mean((Y_pred - Y_true) ** 2)
    
    def backward(self, Y_pred, Y_true, X):
        m = X.shape[1]
        # 输出层梯度
        dZ2 = Y_pred - Y_true
        dW2, db2, dA1 = self.linear_backward(dZ2, self.cache['A1'], self.parameters['W2'])
        # 隐藏层梯度
        dZ1 = dA1 * (self.cache['Z1'] > 0)  # ReLU导数
        dW1, db1, dA0 = self.linear_backward(dZ1, X, self.parameters['W1'])
        self.grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
        return self.grads

class SingleHiddenMultiOutput(NeuralNetworkNumPy):
    """6. 多输出层神经元，1隐藏层，多特征"""
    def __init__(self, n_features, n_outputs, hidden_size=4, learning_rate=0.01):
        super().__init__(layer_dims=[n_features, hidden_size, n_outputs], learning_rate=learning_rate)
        # 参数形状:
        # W1: (hidden_size, n_features), b1: (hidden_size, 1)
        # W2: (n_outputs, hidden_size), b2: (n_outputs, 1)
    
    def forward(self, X):
        self.cache['A0'] = X
        Z1 = self.linear_forward(X, self.parameters['W1'], self.parameters['b1'])
        A1 = self.relu(Z1)
        Z2 = self.linear_forward(A1, self.parameters['W2'], self.parameters['b2'])
        self.cache['Z1'], self.cache['A1'], self.cache['Z2'] = Z1, A1, Z2
        return Z2
    
    def compute_loss(self, Y_pred, Y_true):
        return np.mean((Y_pred - Y_true) ** 2)
    
    def backward(self, Y_pred, Y_true, X):
        dZ2 = Y_pred - Y_true
        dW2, db2, dA1 = self.linear_backward(dZ2, self.cache['A1'], self.parameters['W2'])
        dZ1 = dA1 * (self.cache['Z1'] > 0)
        dW1, db1, dA0 = self.linear_backward(dZ1, X, self.parameters['W1'])
        self.grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
        return self.grads

class FourHiddenSingleOutput(NeuralNetworkNumPy):
    """7. 单输出层神经元，4隐藏层，多特征"""
    def __init__(self, n_features, hidden_sizes=[64, 32, 16, 8], learning_rate=0.01):
        layer_dims = [n_features] + hidden_sizes + [1]
        super().__init__(layer_dims=layer_dims, learning_rate=learning_rate)
        # 各层参数形状: 每层 W: (当前层神经元数, 前一层神经元数), b: (当前层神经元数, 1)
    
    def forward(self, X):
        self.cache['A0'] = X
        A = X
        for l in range(1, len(self.layer_dims)):
            Z = self.linear_forward(A, self.parameters[f'W{l}'], self.parameters[f'b{l}'])
            self.cache[f'Z{l}'] = Z
            if l < len(self.layer_dims) - 1:
                A = self.relu(Z)
            else:
                A = Z  # 输出层无线激活
            self.cache[f'A{l}'] = A
        return A
    
    def compute_loss(self, Y_pred, Y_true):
        return np.mean((Y_pred - Y_true) ** 2)
    
    def backward(self, Y_pred, Y_true, X):
        m = X.shape[1]
        grads = {}
        dA = Y_pred - Y_true  # 输出层梯度
        
        for l in reversed(range(1, len(self.layer_dims))):
            if l == len(self.layer_dims) - 1:
                dZ = dA  # 输出层无线激活
            else:
                dZ = dA * (self.cache[f'Z{l}'] > 0)  # ReLU导数
            
            A_prev = self.cache[f'A{l-1}']
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dA = np.dot(self.parameters[f'W{l}'].T, dZ)
            grads[f'dW{l}'], grads[f'db{l}'] = dW, db
        
        self.grads = grads
        return self.grads

class FourHiddenMultiOutput(NeuralNetworkNumPy):
    """8. 多输出层神经元，4隐藏层，多特征"""
    def __init__(self, n_features, n_outputs, hidden_sizes=[64, 32, 16, 8], learning_rate=0.01):
        layer_dims = [n_features] + hidden_sizes + [n_outputs]
        super().__init__(layer_dims=layer_dims, learning_rate=learning_rate)
    
    def forward(self, X):
        self.cache['A0'] = X
        A = X
        for l in range(1, len(self.layer_dims)):
            Z = self.linear_forward(A, self.parameters[f'W{l}'], self.parameters[f'b{l}'])
            self.cache[f'Z{l}'] = Z
            if l < len(self.layer_dims) - 1:
                A = self.relu(Z)
            else:
                A = Z
            self.cache[f'A{l}'] = A
        return A
    
    def compute_loss(self, Y_pred, Y_true):
        return np.mean((Y_pred - Y_true) ** 2)
    
    def backward(self, Y_pred, Y_true, X):
        m = X.shape[1]
        grads = {}
        dA = Y_pred - Y_true
        
        for l in reversed(range(1, len(self.layer_dims))):
            if l == len(self.layer_dims) - 1:
                dZ = dA
            else:
                dZ = dA * (self.cache[f'Z{l}'] > 0)
            
            A_prev = self.cache[f'A{l-1}']
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dA = np.dot(self.parameters[f'W{l}'].T, dZ)
            grads[f'dW{l}'], grads[f'db{l}'] = dW, db
        
        self.grads = grads
        return self.grads

# 训练辅助函数
def train_model(model, X, Y, epochs=1000, verbose=False):
    """通用训练函数"""
    losses = []
    for i in range(epochs):
        Y_pred = model.forward(X)
        loss = model.compute_loss(Y_pred, Y)
        model.backward(Y_pred, Y, X)
        model.update_parameters(model.grads)
        losses.append(loss)
        if verbose and i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss:.6f}")
    return losses

def generate_data(model_type, n_samples=200):
    """生成演示数据"""
    if model_type in [1, 2]:  # 单特征
        X = np.random.randn(1, n_samples) * 2
        if model_type == 1:  # 线性回归
            Y = 2.5 * X + 1 + np.random.randn(1, n_samples) * 0.5
        else:  # 逻辑回归
            Y_prob = 1 / (1 + np.exp(-(2 * X + 1)))
            Y = (Y_prob > 0.5).astype(float)
    else:  # 多特征
        n_features = 5
        X = np.random.randn(n_features, n_samples)
        if model_type in [3, 5, 7]:  # 单输出
            true_weights = np.random.randn(1, n_features)
            Y = np.dot(true_weights, X) + 0.5 + np.random.randn(1, n_samples) * 0.1
        else:  # 多输出
            n_outputs = 3
            true_weights = np.random.randn(n_outputs, n_features)
            Y = np.dot(true_weights, X) + 0.5 + np.random.randn(n_outputs, n_samples) * 0.1
    return X, Y

def run_menu():
    """交互菜单"""
    models = {
        1: ("线性回归 - 单特征", LinearRegressionNumPy),
        2: ("逻辑回归 - 单特征", LogisticRegressionNumPy),
        3: ("线性回归 - 多特征", lambda: MultiFeatureLinear(n_features=5)),
        4: ("多输出线性回归 - 多特征", lambda: MultiOutputLinear(n_features=5, n_outputs=3)),
        5: ("1隐藏层单输出 - 多特征", lambda: SingleHiddenSingleOutput(n_features=5)),
        6: ("1隐藏层多输出 - 多特征", lambda: SingleHiddenMultiOutput(n_features=5, n_outputs=3)),
        7: ("4隐藏层单输出 - 多特征", lambda: FourHiddenSingleOutput(n_features=5)),
        8: ("4隐藏层多输出 - 多特征", lambda: FourHiddenMultiOutput(n_features=5, n_outputs=3))
    }
    
    while True:
        print("\n" + "="*60)
        print("NumPy神经网络演示程序")
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
                X, Y = generate_data(choice)
            else:
                if choice == 3:
                    X, Y = generate_data(3)
                elif choice == 4:
                    X, Y = generate_data(4)
                elif choice == 5:
                    X, Y = generate_data(5)
                elif choice == 6:
                    X, Y = generate_data(6)
                elif choice == 7:
                    X, Y = generate_data(7)
                else:
                    X, Y = generate_data(8)
            
            # 创建模型
            if choice in [1, 2]:
                model = model_class(learning_rate=0.1)
            elif choice == 3:
                model = MultiFeatureLinear(n_features=5, learning_rate=0.1)
            elif choice == 4:
                model = MultiOutputLinear(n_features=5, n_outputs=3, learning_rate=0.1)
            elif choice == 5:
                model = SingleHiddenSingleOutput(n_features=5, learning_rate=0.1)
            elif choice == 6:
                model = SingleHiddenMultiOutput(n_features=5, n_outputs=3, learning_rate=0.1)
            elif choice == 7:
                model = FourHiddenSingleOutput(n_features=5, learning_rate=0.05)
            elif choice == 8:
                model = FourHiddenMultiOutput(n_features=5, n_outputs=3, learning_rate=0.05)
            
            # 打印参数信息
            print(f"\n特征数据形状: {X.shape} (特征数, 样本数)")
            print(f"输出数据形状: {Y.shape} (输出神经元数, 样本数)")
            print("\n参数形状:")
            total_params = 0
            for l in range(1, len(model.layer_dims)):
                w_shape = model.parameters[f'W{l}'].shape
                b_shape = model.parameters[f'b{l}'].shape
                params = w_shape[0] * w_shape[1] + b_shape[0]
                total_params += params
                print(f"  W{l}: {w_shape}, b{l}: {b_shape} -> {params}个参数")
            print(f"总参数标量数量: {total_params}")
            
            # 训练
            print("\n开始训练...")
            losses = train_model(model, X, Y, epochs=500, verbose=True)
            
            # 评估
            Y_pred = model.forward(X)
            final_loss = model.compute_loss(Y_pred, Y)
            print(f"\n最终损失: {final_loss:.6f}")
            
            # 绘图
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(losses)
            plt.title(f'{name} - 损失曲线')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            
            plt.subplot(1, 2, 2)
            if Y.shape[0] == 1:
                plt.scatter(X[0, :100], Y[0, :100], alpha=0.5, label='真实值')
                plt.scatter(X[0, :100], Y_pred[0, :100], alpha=0.5, label='预测值')
                plt.legend()
                plt.title('预测 vs 真实')
            else:
                plt.bar(range(Y.shape[0]), np.mean((Y - Y_pred)**2, axis=1))
                plt.title('各输出维度MSE')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_menu()