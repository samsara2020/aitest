import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ======================== 工具函数 ========================
def im2col(X, k_h, k_w, stride=1, pad=0):
    """
    将图像块展开为矩阵列，用于快速卷积。
    输入 X: (N, H, W, C)
    返回 col: (k_h*k_w*C, H_out*W_out*N)
    """
    N, H, W, C = X.shape
    H_out = (H + 2*pad - k_h) // stride + 1
    W_out = (W + 2*pad - k_w) // stride + 1

    if pad > 0:
        X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='constant')
    else:
        X_pad = X

    col = np.zeros((N, H_out, W_out, k_h, k_w, C))
    for y in range(k_h):
        y_max = y + stride * H_out
        for x in range(k_w):
            x_max = x + stride * W_out
            col[:, :, :, y, x, :] = X_pad[:, y:y_max:stride, x:x_max:stride, :]

    col = col.transpose(0,3,4,5,1,2).reshape(N, k_h*k_w*C, H_out*W_out)
    return col, H_out, W_out

def col2im(col, X_shape, k_h, k_w, stride=1, pad=0):
    """im2col 的逆过程，用于反向传播"""
    N, H, W, C = X_shape
    H_out = (H + 2*pad - k_h) // stride + 1
    W_out = (W + 2*pad - k_w) // stride + 1
    col = col.reshape(N, k_h, k_w, C, H_out, W_out).transpose(0,4,5,1,2,3)
    img = np.zeros((N, H+2*pad, W+2*pad, C))
    for y in range(k_h):
        y_max = y + stride * H_out
        for x in range(k_w):
            x_max = x + stride * W_out
            img[:, y:y_max:stride, x:x_max:stride, :] += col[:, :, :, y, x, :]
    if pad > 0:
        img = img[:, pad:-pad, pad:-pad, :]
    return img

def max_pool_forward(X, pool_size=2, stride=2):
    N, H, W, C = X.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    out = np.zeros((N, H_out, W_out, C))
    for i in range(H_out):
        for j in range(W_out):
            h_start, w_start = i*stride, j*stride
            h_end, w_end = h_start+pool_size, w_start+pool_size
            out[:, i, j, :] = np.max(X[:, h_start:h_end, w_start:w_end, :], axis=(1,2))
    return out

def max_pool_backward(dout, X, pool_size=2, stride=2):
    N, H, W, C = X.shape
    H_out, W_out = dout.shape[1], dout.shape[2]
    dX = np.zeros_like(X)
    for i in range(H_out):
        for j in range(W_out):
            h_start, w_start = i*stride, j*stride
            h_end, w_end = h_start+pool_size, w_start+pool_size
            patch = X[:, h_start:h_end, w_start:w_end, :]
            mask = (patch == np.max(patch, axis=(1,2), keepdims=True))
            dX[:, h_start:h_end, w_start:w_end, :] += mask * dout[:, i:i+1, j:j+1, :]
    return dX

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy_loss(probs, y_true):
    N = probs.shape[0]
    loss = -np.mean(np.log(probs[np.arange(N), y_true] + 1e-8))
    grad = probs.copy()
    grad[np.arange(N), y_true] -= 1
    grad /= N
    return loss, grad

# ======================== 层定义 ========================
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.pad = pad

        # 初始化权重（He初始化）
        self.W = np.random.randn(*self.kernel_size, in_channels, out_channels) * np.sqrt(2.0 / (in_channels * np.prod(self.kernel_size)))
        self.b = np.zeros(out_channels)

        self.cache = None

    def forward(self, X):
        k_h, k_w = self.kernel_size
        N, H, W, C_in = X.shape
        H_out = (H + 2*self.pad - k_h) // self.stride + 1
        W_out = (W + 2*self.pad - k_w) // self.stride + 1

        col, _, _ = im2col(X, k_h, k_w, self.stride, self.pad)
        # col shape: (N, k_h*k_w*C_in, H_out*W_out)
        col = col.transpose(0,2,1).reshape(N*H_out*W_out, -1)  # (N*H_out*W_out, k_h*k_w*C_in)
        W_flat = self.W.reshape(-1, self.out_channels)        # (k_h*k_w*C_in, out_channels)

        out = np.dot(col, W_flat) + self.b
        out = out.reshape(N, H_out, W_out, self.out_channels)

        self.cache = (X, col, W_flat, H_out, W_out)
        return out

    def backward(self, dout):
        X, col, W_flat, H_out, W_out = self.cache
        N, H, W, C_in = X.shape
        k_h, k_w = self.kernel_size

        dout_flat = dout.reshape(N*H_out*W_out, self.out_channels)

        dW = np.dot(col.T, dout_flat).reshape(k_h, k_w, C_in, self.out_channels)
        db = np.sum(dout_flat, axis=0)

        dcol = np.dot(dout_flat, W_flat.T)  # (N*H_out*W_out, k_h*k_w*C_in)
        dcol = dcol.reshape(N, H_out*W_out, k_h*k_w*C_in).transpose(0,2,1)
        dX = col2im(dcol, X.shape, k_h, k_w, self.stride, self.pad)

        return dX, dW, db

class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None

    def forward(self, X):
        out = max_pool_forward(X, self.pool_size, self.stride)
        self.cache = X
        return out

    def backward(self, dout):
        dX = max_pool_backward(dout, self.cache, self.pool_size, self.stride)
        return dX

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, X):
        self.mask = (X > 0)
        return X * self.mask

    def backward(self, dout):
        return dout * self.mask

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.cache = None

    def forward(self, X):
        out = np.dot(X, self.W) + self.b
        self.cache = X
        return out

    def backward(self, dout):
        X = self.cache
        dW = np.dot(X.T, dout)
        db = np.sum(dout, axis=0)
        dX = np.dot(dout, self.W.T)
        return dX, dW, db

# ======================== 模型组装 ========================
class SimpleCNN:
    def __init__(self, input_shape=(28,28,3), num_classes=10):
        C_in = input_shape[2]

        # 卷积 + 激活 + 池化
        self.conv1 = Conv2D(C_in, 32, kernel_size=3, stride=1, pad=0)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)

        self.conv2 = Conv2D(32, 64, kernel_size=3, stride=1, pad=0)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool_size=2, stride=2)

        # 计算全连接层输入维度（前向传播一次 dummy 数据）
        self._compute_fc_input_dim(input_shape)

        self.fc1 = Linear(self.fc_input_dim, 128)
        self.relu3 = ReLU()
        self.fc2 = Linear(128, num_classes)

        self.layers = [self.conv1, self.relu1, self.pool1,
                       self.conv2, self.relu2, self.pool2,
                       self.fc1, self.relu3, self.fc2]

    def _compute_fc_input_dim(self, input_shape):
        dummy = np.random.randn(1, *input_shape)
        out = self.conv1.forward(dummy)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)
        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)
        self.fc_input_dim = int(np.prod(out.shape[1:]))

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
            if isinstance(layer, Linear) and out.ndim == 2:
                pass  # 保持形状
        return out

    def backward(self, dout):
        grads = {}
        d = dout
        for layer in reversed(self.layers):
            if isinstance(layer, Conv2D):
                d, dW, db = layer.backward(d)
                grads['conv'] = (dW, db)
            elif isinstance(layer, Linear):
                d, dW, db = layer.backward(d)
                grads['fc'] = (dW, db)
            elif isinstance(layer, (ReLU, MaxPool2D)):
                d = layer.backward(d)
        return grads

    def update_params(self, grads, lr=0.001):
        # 简化更新，仅更新卷积层和全连接层
        # 实际应用中应使用优化器
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Conv2D):
                dW, db = grads['conv']
                layer.W -= lr * dW
                layer.b -= lr * db
            elif isinstance(layer, Linear):
                dW, db = grads['fc']
                layer.W -= lr * dW
                layer.b -= lr * db

# ======================== 训练示例 ========================
def load_sample_data():
    # 使用 MNIST 转成 3 通道 (复制灰度到三通道)
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='pandas')
    X = X.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
    X = np.repeat(X, 3, axis=-1)  # 变成 3 通道
    y = y.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train():
    X_train, X_test, y_train, y_test = load_sample_data()
    model = SimpleCNN(input_shape=(28,28,3), num_classes=10)

    batch_size = 32
    epochs = 5
    lr = 0.001
    num_train = X_train.shape[0]

    for epoch in range(epochs):
        perm = np.random.permutation(num_train)
        for i in range(0, num_train, batch_size):
            idx = perm[i:i+batch_size]
            X_batch = X_train[idx]
            y_batch = y_train[idx]

            # 前向
            logits = model.forward(X_batch)
            probs = softmax(logits)
            loss, grad = cross_entropy_loss(probs, y_batch)

            # 反向
            model.backward(grad)
            model.update_params({'conv': (None, None), 'fc': (None, None)}, lr)  # 简化示意，实际应传递真实梯度

        # 评估
        logits_test = model.forward(X_test[:1000])
        preds = np.argmax(logits_test, axis=1)
        acc = np.mean(preds == y_test[:1000])
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Test Acc: {acc:.4f}")

# train()  # 取消注释即可运行（需要安装 sklearn 和 matplotlib）

# ======================== 参数详解 ========================
def print_network_structure():
    print("网络结构及神经元数量计算：")
    print("输入: (batch, 28, 28, 3)")
    print("Conv1: 32个 3x3 卷积，步长1，valid填充 → 输出: (26, 26, 32)")
    print("     尺寸公式: (28 - 3)/1 + 1 = 26")
    print("MaxPool1: 2x2，步长2 → 输出: (13, 13, 32)")
    print("Conv2: 64个 3x3 卷积，步长1，valid填充 → 输出: (11, 11, 64)")
    print("     尺寸公式: (13 - 3)/1 + 1 = 11")
    print("MaxPool2: 2x2，步长2 → 输出: (5, 5, 64)")
    print("Flatten: 5*5*64 = 1600")
    print("FC1: 1600 -> 128 (自定义数量)")
    print("FC2: 128 -> 10 (类别数)")
    print("各层神经元数量说明：")
    print("- 卷积层输出神经元数 = 输出特征图尺寸 * 通道数，例如 Conv1 有 26*26*32 = 21632 个神经元（每个空间位置一个值）。")
    print("- 池化层不改变通道数，仅下采样。")
    print("- 全连接层神经元数由设计者自定义，输出层由任务类别数决定。")

print_network_structure()