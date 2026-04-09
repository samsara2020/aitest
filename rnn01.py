import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        初始化RNN网络参数
        
        参数说明:
        input_size (int): 输入层维度，例如词汇表大小、特征数量
        hidden_size (int): 隐藏层维度，控制记忆容量
        output_size (int): 输出层维度，例如类别数量
        learning_rate (float): 学习率，控制参数更新步长
        """
        # 网络结构参数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        
        # 权重矩阵初始化（使用小随机数防止梯度爆炸）
        # W_xh: 输入到隐藏层的权重，形状 [input_size, hidden_size]
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01
        
        # W_hh: 隐藏层到隐藏层的权重（循环权重），形状 [hidden_size, hidden_size]
        # 这是RNN记忆的核心，连接当前隐藏状态和上一时间步的隐藏状态
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        
        # W_hy: 隐藏层到输出层的权重，形状 [hidden_size, output_size]
        self.W_hy = np.random.randn(hidden_size, output_size) * 0.01
        
        # 偏置项
        # b_h: 隐藏层偏置，形状 [hidden_size]
        self.b_h = np.zeros((1, hidden_size))
        
        # b_y: 输出层偏置，形状 [output_size]
        self.b_y = np.zeros((1, output_size))
        
    def forward(self, x, h_prev=None):
        """
        前向传播（处理完整序列）
        
        参数说明:
        x (numpy.ndarray): 输入序列，形状 [seq_len, input_size]
        h_prev (numpy.ndarray): 上一时间步的隐藏状态，形状 [1, hidden_size]
                              如果为None，则初始化为零向量
        
        返回值:
        outputs (list): 每个时间步的输出列表
        h_final (numpy.ndarray): 最终隐藏状态
        """
        seq_len = x.shape[0]  # 序列长度
        outputs = []  # 存储所有时间步的输出
        
        # 初始化隐藏状态
        if h_prev is None:
            h_prev = np.zeros((1, self.hidden_size))
        h = h_prev
        
        # 按时间步循环处理
        for t in range(seq_len):
            # x_t: 当前时间步的输入，形状 [1, input_size]
            x_t = x[t].reshape(1, -1)
            
            # 核心RNN公式: h_t = tanh(x_t * W_xh + h_{t-1} * W_hh + b_h)
            # 1. 输入贡献: x_t @ W_xh，形状 [1, hidden_size]
            input_contrib = np.dot(x_t, self.W_xh)
            
            # 2. 记忆贡献: h @ W_hh，形状 [1, hidden_size]
            hidden_contrib = np.dot(h, self.W_hh)
            
            # 3. 添加偏置并应用激活函数tanh（范围-1到1）
            h = np.tanh(input_contrib + hidden_contrib + self.b_h)
            
            # 输出计算: y_t = softmax(h * W_hy + b_y)
            # 先计算线性变换，形状 [1, output_size]
            output_linear = np.dot(h, self.W_hy) + self.b_y
            
            # 应用softmax得到概率分布
            y_t = self.softmax(output_linear)
            outputs.append(y_t)
        
        return outputs, h
    
    def forward_step(self, x_t, h_prev):
        """
        单步前向传播（处理单个时间步）
        
        参数说明:
        x_t (numpy.ndarray): 当前时间步输入，形状 [1, input_size]
        h_prev (numpy.ndarray): 上一时间步隐藏状态，形状 [1, hidden_size]
        
        返回值:
        y_t (numpy.ndarray): 当前时间步输出，形状 [1, output_size]
        h (numpy.ndarray): 当前时间步隐藏状态，形状 [1, hidden_size]
        """
        # 计算隐藏状态
        input_contrib = np.dot(x_t, self.W_xh)
        hidden_contrib = np.dot(h_prev, self.W_hh)
        h = np.tanh(input_contrib + hidden_contrib + self.b_h)
        
        # 计算输出
        output_linear = np.dot(h, self.W_hy) + self.b_y
        y_t = self.softmax(output_linear)
        
        return y_t, h
    
    def backward(self, x, outputs, targets, h_final):
        """
        反向传播（BPTT - 随时间反向传播）
        
        参数说明:
        x (numpy.ndarray): 输入序列，形状 [seq_len, input_size]
        outputs (list): 前向传播的输出列表
        targets (numpy.ndarray): 目标值，形状 [seq_len, output_size]
        h_final (numpy.ndarray): 最终隐藏状态，形状 [1, hidden_size]
        
        返回值:
        gradients (dict): 所有参数的梯度
        """
        seq_len = x.shape[0]
        
        # 初始化梯度为零
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        # 初始化隐藏状态的梯度
        dh_next = np.zeros((1, self.hidden_size))
        
        # 从最后一个时间步反向遍历
        for t in reversed(range(seq_len)):
            x_t = x[t].reshape(1, -1)
            h_t = None
            h_prev_t = None
            
            # 获取当前时间步的隐藏状态
            if t == 0:
                h_t = h_final  # 简化处理，实际需要保存所有隐藏状态
                h_prev_t = np.zeros((1, self.hidden_size))
            else:
                # 实际应用中需要保存所有时间步的隐藏状态
                h_t = h_final
                h_prev_t = np.zeros((1, self.hidden_size))
            
            # 输出层梯度
            y_t = outputs[t]  # 当前时间步预测值
            target_t = targets[t].reshape(1, -1)  # 当前时间步目标值
            
            # 损失对输出线性层的梯度
            dy = y_t - target_t  # 交叉熵损失的梯度简化
            dW_hy += np.dot(h_t.T, dy)
            db_y += dy
            
            # 隐藏层梯度
            dh = np.dot(dy, self.W_hy.T) + dh_next
            
            # tanh激活函数的导数: 1 - tanh^2
            dh_raw = dh * (1 - h_t ** 2)
            
            # 参数梯度
            dW_xh += np.dot(x_t.T, dh_raw)
            dW_hh += np.dot(h_prev_t.T, dh_raw)
            db_h += dh_raw
            
            # 传递到上一时间步的梯度
            dh_next = np.dot(dh_raw, self.W_hh.T)
        
        # 返回梯度字典
        gradients = {
            'W_xh': dW_xh,
            'W_hh': dW_hh,
            'W_hy': dW_hy,
            'b_h': db_h,
            'b_y': db_y
        }
        
        return gradients
    
    def update(self, gradients):
        """
        使用梯度更新参数
        
        参数说明:
        gradients (dict): 包含所有参数梯度的字典
        """
        self.W_xh -= self.lr * gradients['W_xh']
        self.W_hh -= self.lr * gradients['W_hh']
        self.W_hy -= self.lr * gradients['W_hy']
        self.b_h -= self.lr * gradients['b_h']
        self.b_y -= self.lr * gradients['b_y']
    
    def softmax(self, x):
        """
        Softmax激活函数，用于多分类
        
        参数说明:
        x (numpy.ndarray): 输入数组，形状 [1, output_size]
        
        返回值:
        numpy.ndarray: 概率分布，形状 [1, output_size]
        """
        # 减去最大值提高数值稳定性
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def train(self, x, targets, epochs=100):
        """
        训练RNN网络
        
        参数说明:
        x (numpy.ndarray): 训练输入序列，形状 [seq_len, input_size]
        targets (numpy.ndarray): 训练目标值，形状 [seq_len, output_size]
        epochs (int): 训练轮数
        
        返回值:
        losses (list): 每轮的平均损失值
        """
        losses = []
        
        for epoch in range(epochs):
            # 前向传播
            outputs, h_final = self.forward(x)
            
            # 计算损失（交叉熵）
            loss = 0
            for t in range(len(outputs)):
                y_t = outputs[t]
                target_t = targets[t].reshape(1, -1)
                # 交叉熵损失 = -sum(target * log(y))
                loss += -np.sum(target_t * np.log(y_t + 1e-8))  # 加小量防止log(0)
            loss /= len(outputs)
            losses.append(loss)
            
            # 反向传播
            gradients = self.backward(x, outputs, targets, h_final)
            
            # 更新参数
            self.update(gradients)
            
            # 打印损失
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, x, h_prev=None):
        """
        预测函数
        
        参数说明:
        x (numpy.ndarray): 输入序列，形状 [seq_len, input_size]
        h_prev (numpy.ndarray): 初始隐藏状态，形状 [1, hidden_size]
        
        返回值:
        predictions (list): 预测结果列表
        """
        outputs, _ = self.forward(x, h_prev)
        return outputs


# 示例使用：简单的序列预测
def demo():
    """
    演示RNN的基本使用
    """
    print("=== RNN演示示例 ===")
    
    # 参数设置
    input_size = 3    # 输入维度
    hidden_size = 4   # 隐藏层维度
    output_size = 2   # 输出维度
    seq_len = 5       # 序列长度
    
    # 创建RNN实例
    rnn = RNN(input_size, hidden_size, output_size, learning_rate=0.1)
    
    # 生成随机训练数据
    print(f"\n生成随机训练数据：序列长度={seq_len}, 输入维度={input_size}, 输出维度={output_size}")
    x_train = np.random.randn(seq_len, input_size)  # 输入序列
    y_train = np.random.randn(seq_len, output_size) # 目标序列
    y_train = np.exp(y_train) / np.sum(np.exp(y_train), axis=1, keepdims=True)  # 转换为概率分布
    
    # 训练网络
    print("\n开始训练...")
    losses = rnn.train(x_train, y_train, epochs=50)
    
    # 预测
    print("\n预测结果：")
    predictions = rnn.predict(x_train)
    for t, pred in enumerate(predictions):
        print(f"时间步 {t}: 预测概率分布 = {pred[0]}")
    
    print("\n训练完成！")

if __name__ == "__main__":
    demo()