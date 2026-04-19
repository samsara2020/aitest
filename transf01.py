"""
纯NumPy实现Transformer大模型训练
不使用PyTorch/TensorFlow等深度学习框架
包含手动反向传播、梯度计算、参数更新
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import random

# ==================== 1. 基础工具函数 ====================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax函数 - 数值稳定版本"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
               eps: float = 1e-6) -> Tuple[np.ndarray, Dict]:
    """
    层归一化前向传播
    
    输入形状: [batch_size, seq_len, d_model]
    gamma/beta: [d_model] - 可训练参数
    """
    mean = np.mean(x, axis=-1, keepdims=True)  # [batch, seq, 1]
    var = np.var(x, axis=-1, keepdims=True)    # [batch, seq, 1]
    
    x_norm = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_norm + beta
    
    # 保存反向传播所需的值
    cache = {
        'x': x,
        'mean': mean,
        'var': var,
        'x_norm': x_norm,
        'gamma': gamma,
        'eps': eps
    }
    
    return out, cache


def layer_norm_backward(dout: np.ndarray, cache: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    层归一化反向传播
    
    返回: dx, dgamma, dbeta
    """
    x = cache['x']
    mean = cache['mean']
    var = cache['var']
    x_norm = cache['x_norm']
    gamma = cache['gamma']
    eps = cache['eps']
    
    N, D = x.shape[-1], x.size // x.shape[-1]
    
    # dbeta: 直接求和
    dbeta = np.sum(dout, axis=(0, 1))
    
    # dgamma
    dgamma = np.sum(dout * x_norm, axis=(0, 1))
    
    # dx_norm
    dx_norm = dout * gamma
    
    # dx
    dx = (1. / np.sqrt(var + eps)) * (dx_norm - np.mean(dx_norm, axis=-1, keepdims=True) 
                                      - x_norm * np.mean(dx_norm * x_norm, axis=-1, keepdims=True))
    
    return dx, dgamma, dbeta


def dropout_forward(x: np.ndarray, dropout_prob: float, training: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Dropout前向传播
    
    输入形状: 任意
    返回: 输出和mask
    """
    if not training or dropout_prob == 0:
        return x, None
    
    mask = np.random.binomial(1, 1 - dropout_prob, size=x.shape) / (1 - dropout_prob)
    return x * mask, mask


def dropout_backward(dout: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Dropout反向传播"""
    if mask is None:
        return dout
    return dout * mask


# ==================== 2. 模型层实现 ====================

class Linear:
    """
    线性层 (全连接层)
    
    概念定义：
    - 权重W: [in_features, out_features] - 需要训练得出
    - 偏置b: [out_features] - 需要训练得出
    
    数据形状：
    - 输入: [batch_size, seq_len, in_features]
    - 输出: [batch_size, seq_len, out_features]
    """
    
    def __init__(self, in_features: int, out_features: int):
        """
        Args:
            in_features: 输入维度 - 提前设定不变
            out_features: 输出维度 - 提前设定不变
        """
        self.in_features = in_features
        self.out_features = out_features
        
        # 参数初始化 - Xavier初始化
        limit = np.sqrt(6 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        self.b = np.zeros(out_features)
        
        # 梯度存储
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        # 缓存前向传播的输入，用于反向传播
        self.cache = {}
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        输入: [batch_size, seq_len, in_features]
        输出: [batch_size, seq_len, out_features]
        """
        self.cache['x'] = x
        
        # y = x @ W + b
        # x: [batch, seq, in] -> reshape: [batch*seq, in]
        batch_size, seq_len = x.shape[0], x.shape[1]
        x_flat = x.reshape(-1, self.in_features)
        
        out_flat = x_flat @ self.W + self.b
        out = out_flat.reshape(batch_size, seq_len, self.out_features)
        
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播
        
        输入梯度: [batch_size, seq_len, out_features]
        返回: 输入的梯度 [batch_size, seq_len, in_features]
        """
        x = self.cache['x']
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # 展平
        x_flat = x.reshape(-1, self.in_features)
        dout_flat = dout.reshape(-1, self.out_features)
        
        # 计算梯度
        self.dW = x_flat.T @ dout_flat  # [in_features, out_features]
        self.db = np.sum(dout_flat, axis=0)  # [out_features]
        
        # 计算输入的梯度
        dx_flat = dout_flat @ self.W.T  # [batch*seq, in_features]
        dx = dx_flat.reshape(batch_size, seq_len, self.in_features)
        
        return dx
    
    def update(self, lr: float):
        """更新参数"""
        self.W -= lr * self.dW
        self.b -= lr * self.db
        
    def get_params(self) -> List[np.ndarray]:
        """获取所有可训练参数"""
        return [self.W, self.b]
    
    def get_grads(self) -> List[np.ndarray]:
        """获取所有梯度"""
        return [self.dW, self.db]


class Embedding:
    """
    词嵌入层
    
    概念定义：
    - 嵌入矩阵: [vocab_size, d_model] - 需要训练得出
    - 每个token ID映射到一个d_model维向量
    
    数据形状：
    - 输入: [batch_size, seq_len] (token IDs)
    - 输出: [batch_size, seq_len, d_model]
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
            vocab_size: 词表大小 - 提前设定不变
            d_model: 嵌入维度 - 提前设定不变
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 参数初始化
        limit = np.sqrt(6 / vocab_size)
        self.weight = np.random.uniform(-limit, limit, (vocab_size, d_model))
        self.dweight = np.zeros_like(self.weight)
        
        self.cache = {}
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        输入: [batch_size, seq_len] - token IDs
        输出: [batch_size, seq_len, d_model]
        """
        self.cache['x'] = x
        return self.weight[x]
    
    def backward(self, dout: np.ndarray) -> None:
        """
        反向传播
        由于嵌入层没有可训练的输入，只累积梯度
        """
        x = self.cache['x']
        self.dweight.fill(0)
        
        # 使用np.add.at累积相同索引的梯度
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                token_id = x[i, j]
                self.dweight[token_id] += dout[i, j]
    
    def update(self, lr: float):
        """更新参数"""
        self.weight -= lr * self.dweight
    
    def get_params(self) -> List[np.ndarray]:
        """获取所有可训练参数"""
        return [self.weight]
    
    def get_grads(self) -> List[np.ndarray]:
        """获取所有梯度"""
        return [self.dweight]


class PositionalEncoding:
    """
    位置编码 - 固定不变，无训练参数
    
    数据形状：
    - 输入: [batch_size, seq_len, d_model]
    - 输出: [batch_size, seq_len, d_model]
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            d_model: 模型维度 - 提前设定不变
            max_len: 最大序列长度 - 提前设定不变
        """
        self.d_model = d_model
        self.max_len = max_len
        
        # 创建固定的位置编码矩阵
        self.pe = np.zeros((max_len, d_model))
        position = np.arange(max_len).reshape(-1, 1)
        
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        self.pe[:, 0::2] = np.sin(position * div_term)
        self.pe[:, 1::2] = np.cos(position * div_term)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播：直接加上位置编码
        """
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播：直接传递梯度
        """
        return dout


class MultiHeadAttention:
    """
    多头注意力机制
    
    包含4个线性层: W_q, W_k, W_v, W_o
    所有参数都需要训练得出
    
    数据形状：
    - Q/K/V输入: [batch_size, seq_len, d_model]
    - 输出: [batch_size, seq_len, d_model]
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度 - 提前设定不变
            n_heads: 注意力头数 - 提前设定不变
            dropout: dropout比率 - 提前设定不变
        """
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        
        # 线性变换层
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)
        
        self.cache = {}
        
    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        分割多头
        输入: [batch, seq, d_model]
        输出: [batch, n_heads, seq, d_k]
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = x.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """
        合并多头
        输入: [batch, n_heads, seq, d_k]
        输出: [batch, seq, d_model]
        """
        batch_size = x.shape[0]
        seq_len = x.shape[2]
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None, training: bool = True) -> np.ndarray:
        """
        前向传播
        """
        batch_size = query.shape[0]
        
        # 1. 线性变换
        Q = self.W_q.forward(query)  # [batch, q_len, d_model]
        K = self.W_k.forward(key)    # [batch, k_len, d_model]
        V = self.W_v.forward(value)  # [batch, v_len, d_model]
        
        # 2. 分割多头
        Q = self._split_heads(Q)  # [batch, n_heads, q_len, d_k]
        K = self._split_heads(K)  # [batch, n_heads, k_len, d_k]
        V = self._split_heads(V)  # [batch, n_heads, v_len, d_k]
        
        # 3. 计算注意力分数
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        
        # 4. 应用mask
        if mask is not None:
            scores = np.where(mask, scores, -1e9)
        
        # 5. Softmax
        attn_weights = softmax(scores, axis=-1)
        
        # 6. Dropout
        attn_weights, attn_mask = dropout_forward(attn_weights, self.dropout, training)
        
        # 7. 应用注意力权重
        context = attn_weights @ V  # [batch, n_heads, q_len, d_k]
        
        # 8. 合并多头
        context = self._combine_heads(context)  # [batch, q_len, d_model]
        
        # 9. 输出线性变换
        output = self.W_o.forward(context)
        
        # 保存缓存
        self.cache = {
            'Q': Q, 'K': K, 'V': V,
            'scores': scores,
            'attn_weights': attn_weights,
            'attn_mask': attn_mask,
            'mask': mask
        }
        
        return output
    
    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        反向传播
        返回: dQ, dK, dV
        """
        cache = self.cache
        Q, K, V = cache['Q'], cache['K'], cache['V']
        attn_weights = cache['attn_weights']
        attn_mask = cache['attn_mask']
        
        # 1. 输出线性层的反向传播
        d_context = self.W_o.backward(dout)
        
        # 2. 分割多头的反向传播
        d_context = self._split_heads(d_context)  # [batch, n_heads, q_len, d_k]
        
        # 3. 注意力加权的反向传播
        dV = attn_weights.transpose(0, 1, 3, 2) @ d_context
        d_attn_weights = d_context @ V.transpose(0, 1, 3, 2)
        
        # 4. Dropout反向传播
        d_attn_weights = dropout_backward(d_attn_weights, attn_mask)
        
        # 5. Softmax反向传播
        d_scores = d_attn_weights * attn_weights
        d_scores -= attn_weights * np.sum(d_scores, axis=-1, keepdims=True)
        d_scores /= np.sqrt(self.d_k)
        
        # 6. Mask反向传播（忽略masked位置）
        if cache['mask'] is not None:
            d_scores = np.where(cache['mask'], d_scores, 0)
        
        # 7. Q, K, V的反向传播
        dQ = d_scores @ K
        dK = d_scores.transpose(0, 1, 3, 2) @ Q
        dV = dV
        
        # 8. 合并多头
        dQ = self._combine_heads(dQ)
        dK = self._combine_heads(dK)
        dV = self._combine_heads(dV)
        
        # 9. 输入线性层的反向传播
        d_query = self.W_q.backward(dQ)
        d_key = self.W_k.backward(dK)
        d_value = self.W_v.backward(dV)
        
        return d_query, d_key, d_value
    
    def update(self, lr: float):
        """更新所有参数"""
        self.W_q.update(lr)
        self.W_k.update(lr)
        self.W_v.update(lr)
        self.W_o.update(lr)


class FeedForward:
    """
    前馈神经网络
    
    两个线性层，中间ReLU激活
    
    数据形状：
    - 输入: [batch_size, seq_len, d_model]
    - 输出: [batch_size, seq_len, d_model]
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度 - 提前设定不变
            d_ff: 前馈网络中间层维度 - 提前设定不变
            dropout: dropout比率 - 提前设定不变
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        
        self.cache = {}
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        前向传播
        """
        # 第一层线性 + ReLU
        out1 = self.linear1.forward(x)
        out1_relu = np.maximum(0, out1)  # ReLU
        
        # Dropout
        out1_dropout, mask = dropout_forward(out1_relu, self.dropout, training)
        
        # 第二层线性
        out2 = self.linear2.forward(out1_dropout)
        
        self.cache = {
            'out1': out1,
            'out1_relu': out1_relu,
            'dropout_mask': mask
        }
        
        return out2
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播
        """
        cache = self.cache
        
        # 第二层线性的反向传播
        d_out1_dropout = self.linear2.backward(dout)
        
        # Dropout反向传播
        d_out1_relu = dropout_backward(d_out1_dropout, cache['dropout_mask'])
        
        # ReLU反向传播
        d_out1 = d_out1_relu * (cache['out1'] > 0)
        
        # 第一层线性的反向传播
        dx = self.linear1.backward(d_out1)
        
        return dx
    
    def update(self, lr: float):
        """更新参数"""
        self.linear1.update(lr)
        self.linear2.update(lr)


class TransformerEncoderLayer:
    """
    Transformer编码器层
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # LayerNorm参数
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)
        
        # 梯度
        self.dgamma1 = np.zeros_like(self.gamma1)
        self.dbeta1 = np.zeros_like(self.beta1)
        self.dgamma2 = np.zeros_like(self.gamma2)
        self.dbeta2 = np.zeros_like(self.beta2)
        
        self.dropout = dropout
        self.cache = {}
        
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None,
                training: bool = True) -> np.ndarray:
        """
        前向传播
        """
        # 自注意力子层
        attn_out = self.self_attn.forward(x, x, x, mask, training)
        attn_out, _ = dropout_forward(attn_out, self.dropout, training)
        
        # 残差连接 + 层归一化
        x1 = x + attn_out
        x1, cache1 = layer_norm(x1, self.gamma1, self.beta1)
        
        # 前馈网络子层
        ff_out = self.feed_forward.forward(x1, training)
        ff_out, _ = dropout_forward(ff_out, self.dropout, training)
        
        # 残差连接 + 层归一化
        x2 = x1 + ff_out
        x2, cache2 = layer_norm(x2, self.gamma2, self.beta2)
        
        self.cache = {
            'x': x,
            'attn_out': attn_out,
            'x1': x1,
            'cache1': cache1,
            'ff_out': ff_out,
            'cache2': cache2
        }
        
        return x2
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播
        """
        cache = self.cache
        
        # 第二个残差连接 + 层归一化
        dx2, self.dgamma2, self.dbeta2 = layer_norm_backward(dout, cache['cache2'])
        d_ff_out = dx2
        d_x1 = dx2
        
        # 前馈网络反向传播
        d_x1_ff = self.feed_forward.backward(d_ff_out)
        d_x1 += d_x1_ff
        
        # 第一个残差连接 + 层归一化
        dx1, self.dgamma1, self.dbeta1 = layer_norm_backward(d_x1, cache['cache1'])
        d_attn_out = dx1
        d_x = dx1
        
        # 自注意力反向传播
        d_attn_q, _, _ = self.self_attn.backward(d_attn_out)
        d_x += d_attn_q
        
        return d_x
    
    def update(self, lr: float):
        """更新参数"""
        self.self_attn.update(lr)
        self.feed_forward.update(lr)
        
        self.gamma1 -= lr * self.dgamma1
        self.beta1 -= lr * self.dbeta1
        self.gamma2 -= lr * self.dgamma2
        self.beta2 -= lr * self.dbeta2


class TransformerDecoderLayer:
    """
    Transformer解码器层
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # LayerNorm参数
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)
        self.gamma3 = np.ones(d_model)
        self.beta3 = np.zeros(d_model)
        
        # 梯度
        self.dgamma1 = np.zeros_like(self.gamma1)
        self.dbeta1 = np.zeros_like(self.beta1)
        self.dgamma2 = np.zeros_like(self.gamma2)
        self.dbeta2 = np.zeros_like(self.beta2)
        self.dgamma3 = np.zeros_like(self.gamma3)
        self.dbeta3 = np.zeros_like(self.beta3)
        
        self.dropout = dropout
        self.cache = {}
        
    def forward(self, x: np.ndarray, enc_output: np.ndarray,
                src_mask: Optional[np.ndarray] = None,
                tgt_mask: Optional[np.ndarray] = None,
                training: bool = True) -> np.ndarray:
        """
        前向传播
        """
        # 1. 掩码自注意力
        attn_out1 = self.self_attn.forward(x, x, x, tgt_mask, training)
        attn_out1, _ = dropout_forward(attn_out1, self.dropout, training)
        
        x1 = x + attn_out1
        x1, cache1 = layer_norm(x1, self.gamma1, self.beta1)
        
        # 2. 交叉注意力
        attn_out2 = self.cross_attn.forward(x1, enc_output, enc_output, src_mask, training)
        attn_out2, _ = dropout_forward(attn_out2, self.dropout, training)
        
        x2 = x1 + attn_out2
        x2, cache2 = layer_norm(x2, self.gamma2, self.beta2)
        
        # 3. 前馈网络
        ff_out = self.feed_forward.forward(x2, training)
        ff_out, _ = dropout_forward(ff_out, self.dropout, training)
        
        x3 = x2 + ff_out
        x3, cache3 = layer_norm(x3, self.gamma3, self.beta3)
        
        self.cache = {
            'x': x, 'enc_output': enc_output,
            'attn_out1': attn_out1, 'x1': x1, 'cache1': cache1,
            'attn_out2': attn_out2, 'x2': x2, 'cache2': cache2,
            'ff_out': ff_out, 'cache3': cache3
        }
        
        return x3
    
    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        反向传播
        返回: d_x, d_enc_output
        """
        cache = self.cache
        
        # 第三个残差连接 + 层归一化
        dx3, self.dgamma3, self.dbeta3 = layer_norm_backward(dout, cache['cache3'])
        d_ff_out = dx3
        d_x2 = dx3
        
        # 前馈网络反向传播
        d_x2_ff = self.feed_forward.backward(d_ff_out)
        d_x2 += d_x2_ff
        
        # 第二个残差连接 + 层归一化
        dx2, self.dgamma2, self.dbeta2 = layer_norm_backward(d_x2, cache['cache2'])
        d_attn_out2 = dx2
        d_x1 = dx2
        
        # 交叉注意力反向传播
        d_cross_q, d_cross_k, d_cross_v = self.cross_attn.backward(d_attn_out2)
        d_x1 += d_cross_q
        d_enc_output = d_cross_k  # 传递给编码器的梯度
        
        # 第一个残差连接 + 层归一化
        dx1, self.dgamma1, self.dbeta1 = layer_norm_backward(d_x1, cache['cache1'])
        d_attn_out1 = dx1
        d_x = dx1
        
        # 自注意力反向传播
        d_self_q, _, _ = self.self_attn.backward(d_attn_out1)
        d_x += d_self_q
        
        return d_x, d_enc_output
    
    def update(self, lr: float):
        """更新参数"""
        self.self_attn.update(lr)
        self.cross_attn.update(lr)
        self.feed_forward.update(lr)
        
        self.gamma1 -= lr * self.dgamma1
        self.beta1 -= lr * self.dbeta1
        self.gamma2 -= lr * self.dgamma2
        self.beta2 -= lr * self.dbeta2
        self.gamma3 -= lr * self.dgamma3
        self.beta3 -= lr * self.dbeta3


class Transformer:
    """
    完整的Transformer模型
    """
    
    def __init__(self, 
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 d_ff: int = 512,
                 max_len: int = 100,
                 dropout: float = 0.1):
        
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        # 嵌入层
        self.src_embedding = Embedding(src_vocab_size, d_model)
        self.tgt_embedding = Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # 编码器层
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ]
        
        # 解码器层
        self.decoder_layers = [
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ]
        
        # 输出层
        self.output_layer = Linear(d_model, tgt_vocab_size)
        
    def generate_causal_mask(self, sz: int) -> np.ndarray:
        """生成因果掩码"""
        mask = np.triu(np.ones((sz, sz)), k=1) == 0
        return mask
    
    def forward(self, src: np.ndarray, tgt: np.ndarray,
                src_mask: Optional[np.ndarray] = None,
                tgt_mask: Optional[np.ndarray] = None,
                training: bool = True) -> np.ndarray:
        """
        前向传播
        
        输入:
        - src: [batch_size, src_len]
        - tgt: [batch_size, tgt_len]
        """
        # 嵌入 + 位置编码
        src_emb = self.src_embedding.forward(src) * np.sqrt(self.d_model)
        src_emb = self.positional_encoding.forward(src_emb)
        
        tgt_emb = self.tgt_embedding.forward(tgt) * np.sqrt(self.d_model)
        tgt_emb = self.positional_encoding.forward(tgt_emb)
        
        # 准备注意力掩码
        if src_mask is not None:
            src_attn_mask = src_mask[:, np.newaxis, np.newaxis, :]
        else:
            src_attn_mask = None
            
        if tgt_mask is not None:
            tgt_len = tgt.shape[1]
            causal_mask = self.generate_causal_mask(tgt_len)
            tgt_attn_mask = tgt_mask[:, np.newaxis, np.newaxis, :] & causal_mask
        else:
            tgt_attn_mask = None
        
        # 编码器
        enc_output = src_emb
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer.forward(enc_output, src_attn_mask, training)
        
        # 解码器
        dec_output = tgt_emb
        for decoder_layer in self.decoder_layers:
            dec_output = decoder_layer.forward(
                dec_output, enc_output, src_attn_mask, tgt_attn_mask, training
            )
        
        # 输出层
        output = self.output_layer.forward(dec_output)
        
        # 保存反向传播所需的值
        self.cache = {
            'enc_output': enc_output
        }
        
        return output
    
    def backward(self, dout: np.ndarray) -> None:
        """
        反向传播
        """
        # 输出层反向传播
        d_dec_output = self.output_layer.backward(dout)
        
        # 解码器反向传播
        d_enc_output_total = np.zeros_like(self.cache['enc_output'])
        
        for decoder_layer in reversed(self.decoder_layers):
            d_dec_output, d_enc_output = decoder_layer.backward(d_dec_output)
            d_enc_output_total += d_enc_output
        
        # 编码器反向传播
        d_enc_output = d_enc_output_total
        for encoder_layer in reversed(self.encoder_layers):
            d_enc_output = encoder_layer.backward(d_enc_output)
        
        # 嵌入层反向传播（位置编码无参数，直接传递梯度）
        self.tgt_embedding.backward(d_dec_output)
        self.src_embedding.backward(d_enc_output)
    
    def update(self, lr: float):
        """更新所有参数"""
        self.src_embedding.update(lr)
        self.tgt_embedding.update(lr)
        
        for layer in self.encoder_layers:
            layer.update(lr)
        
        for layer in self.decoder_layers:
            layer.update(lr)
        
        self.output_layer.update(lr)


# ==================== 3. 优化器 ====================

class Adam:
    """
    Adam优化器
    """
    
    def __init__(self, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        
        self.m = {}  # 一阶矩
        self.v = {}  # 二阶矩
        
    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> None:
        """
        更新参数
        """
        self.t += 1
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            if i not in self.m:
                self.m[i] = np.zeros_like(param)
                self.v[i] = np.zeros_like(param)
            
            # 更新矩估计
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # 偏差修正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # 更新参数
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ==================== 4. 损失函数 ====================

def cross_entropy_loss(predictions: np.ndarray, targets: np.ndarray, 
                       ignore_index: int = 0) -> Tuple[float, np.ndarray]:
    """
    交叉熵损失函数
    
    输入:
    - predictions: [batch_size * seq_len, vocab_size] - logits
    - targets: [batch_size * seq_len] - 目标token IDs
    - ignore_index: 忽略的索引（如padding）
    
    返回: loss, gradient
    """
    # Softmax
    probs = softmax(predictions, axis=-1)
    
    # 防止log(0)
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    
    # 计算损失
    batch_size = predictions.shape[0]
    loss = 0
    count = 0
    
    # 梯度
    grad = probs.copy()
    
    for i in range(batch_size):
        target = targets[i]
        if target != ignore_index:
            loss -= np.log(probs[i, target])
            grad[i, target] -= 1
            count += 1
    
    if count > 0:
        loss /= count
        grad /= count
    
    return loss, grad


# ==================== 5. 数据准备 ====================

class DataLoader:
    """简单的数据加载器"""
    
    def __init__(self, src_data: List[List[int]], tgt_data: List[List[int]], 
                 batch_size: int, shuffle: bool = True):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(src_data)))
        
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        
        for start_idx in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[start_idx:start_idx + self.batch_size]
            
            # 获取batch数据
            batch_src = [self.src_data[i] for i in batch_indices]
            batch_tgt = [self.tgt_data[i] for i in batch_indices]
            
            # Padding
            max_src_len = max(len(seq) for seq in batch_src)
            max_tgt_len = max(len(seq) for seq in batch_tgt)
            
            src_padded = np.zeros((len(batch_indices), max_src_len), dtype=np.int32)
            tgt_padded = np.zeros((len(batch_indices), max_tgt_len), dtype=np.int32)
            src_mask = np.zeros((len(batch_indices), max_src_len), dtype=np.bool_)
            tgt_mask = np.zeros((len(batch_indices), max_tgt_len), dtype=np.bool_)
            
            for i, (src, tgt) in enumerate(zip(batch_src, batch_tgt)):
                src_padded[i, :len(src)] = src
                tgt_padded[i, :len(tgt)] = tgt
                src_mask[i, :len(src)] = True
                tgt_mask[i, :len(tgt)] = True
            
            yield {
                'src': src_padded,
                'tgt': tgt_padded,
                'src_mask': src_mask,
                'tgt_mask': tgt_mask
            }


# ==================== 6. 训练循环 ====================

def train(model: Transformer, dataloader: DataLoader, 
          optimizer: Adam, epochs: int, vocab_size: int):
    """
    训练函数
    """
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            src = batch['src']
            tgt = batch['tgt']
            src_mask = batch['src_mask']
            tgt_mask = batch['tgt_mask']
            
            # 准备输入输出
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask_input = tgt_mask[:, :-1]
            
            # 前向传播
            predictions = model.forward(src, tgt_input, src_mask, tgt_mask_input, training=True)
            
            # 计算损失
            batch_size, seq_len = tgt_output.shape
            predictions_flat = predictions.reshape(-1, vocab_size)
            targets_flat = tgt_output.reshape(-1)
            
            loss, grad = cross_entropy_loss(predictions_flat, targets_flat, ignore_index=0)
            
            # 反向传播
            grad = grad.reshape(predictions.shape)
            model.backward(grad)
            
            # 获取所有参数并更新
            params = collect_params(model)
            grads = collect_grads(model)
            optimizer.update(params, grads)
            
            total_loss += loss
            num_batches += 1
            
            if num_batches % 10 == 0:
                print(f"  Batch {num_batches}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / num_batches
        perplexity = np.exp(avg_loss)
        
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print()


def collect_params(model: Transformer) -> List[np.ndarray]:
    """收集所有可训练参数"""
    params = []
    
    params.extend(model.src_embedding.get_params())
    params.extend(model.tgt_embedding.get_params())
    
    for layer in model.encoder_layers:
        params.extend(layer.self_attn.W_q.get_params())
        params.extend(layer.self_attn.W_k.get_params())
        params.extend(layer.self_attn.W_v.get_params())
        params.extend(layer.self_attn.W_o.get_params())
        params.extend(layer.feed_forward.linear1.get_params())
        params.extend(layer.feed_forward.linear2.get_params())
        params.append(layer.gamma1)
        params.append(layer.beta1)
        params.append(layer.gamma2)
        params.append(layer.beta2)
    
    for layer in model.decoder_layers:
        params.extend(layer.self_attn.W_q.get_params())
        params.extend(layer.self_attn.W_k.get_params())
        params.extend(layer.self_attn.W_v.get_params())
        params.extend(layer.self_attn.W_o.get_params())
        params.extend(layer.cross_attn.W_q.get_params())
        params.extend(layer.cross_attn.W_k.get_params())
        params.extend(layer.cross_attn.W_v.get_params())
        params.extend(layer.cross_attn.W_o.get_params())
        params.extend(layer.feed_forward.linear1.get_params())
        params.extend(layer.feed_forward.linear2.get_params())
        params.append(layer.gamma1)
        params.append(layer.beta1)
        params.append(layer.gamma2)
        params.append(layer.beta2)
        params.append(layer.gamma3)
        params.append(layer.beta3)
    
    params.extend(model.output_layer.get_params())
    
    return params


def collect_grads(model: Transformer) -> List[np.ndarray]:
    """收集所有梯度"""
    grads = []
    
    grads.extend(model.src_embedding.get_grads())
    grads.extend(model.tgt_embedding.get_grads())
    
    for layer in model.encoder_layers:
        grads.extend(layer.self_attn.W_q.get_grads())
        grads.extend(layer.self_attn.W_k.get_grads())
        grads.extend(layer.self_attn.W_v.get_grads())
        grads.extend(layer.self_attn.W_o.get_grads())
        grads.extend(layer.feed_forward.linear1.get_grads())
        grads.extend(layer.feed_forward.linear2.get_grads())
        grads.append(layer.dgamma1)
        grads.append(layer.dbeta1)
        grads.append(layer.dgamma2)
        grads.append(layer.dbeta2)
    
    for layer in model.decoder_layers:
        grads.extend(layer.self_attn.W_q.get_grads())
        grads.extend(layer.self_attn.W_k.get_grads())
        grads.extend(layer.self_attn.W_v.get_grads())
        grads.extend(layer.self_attn.W_o.get_grads())
        grads.extend(layer.cross_attn.W_q.get_grads())
        grads.extend(layer.cross_attn.W_k.get_grads())
        grads.extend(layer.cross_attn.W_v.get_grads())
        grads.extend(layer.cross_attn.W_o.get_grads())
        grads.extend(layer.feed_forward.linear1.get_grads())
        grads.extend(layer.feed_forward.linear2.get_grads())
        grads.append(layer.dgamma1)
        grads.append(layer.dbeta1)
        grads.append(layer.dgamma2)
        grads.append(layer.dbeta2)
        grads.append(layer.dgamma3)
        grads.append(layer.dbeta3)
    
    grads.extend(model.output_layer.get_grads())
    
    return grads


# ==================== 7. 主程序 ====================

def main():
    """主函数"""
    
    print("=" * 60)
    print("纯NumPy实现Transformer训练")
    print("=" * 60)
    
    # 超参数设置（提前设定，训练中不变）
    SRC_VOCAB_SIZE = 100
    TGT_VOCAB_SIZE = 100
    D_MODEL = 128
    N_HEADS = 4
    N_LAYERS = 2
    D_FF = 256
    MAX_LEN = 50
    DROPOUT = 0.1
    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 0.001
    
    print(f"\n模型配置:")
    print(f"  d_model: {D_MODEL}")
    print(f"  n_heads: {N_HEADS}")
    print(f"  n_layers: {N_LAYERS}")
    print(f"  d_ff: {D_FF}")
    print(f"  batch_size: {BATCH_SIZE}")
    print(f"  epochs: {EPOCHS}")
    
    # 创建模拟数据
    print("\n准备模拟数据...")
    num_samples = 200
    
    # 生成随机序列（模拟token IDs）
    src_data = []
    tgt_data = []
    
    for _ in range(num_samples):
        src_len = random.randint(5, 20)
        tgt_len = random.randint(5, 20)
        
        # 添加特殊token: 1=<sos>, 2=<eos>
        src_seq = [1] + [random.randint(3, SRC_VOCAB_SIZE-1) for _ in range(src_len-2)] + [2]
        tgt_seq = [1] + [random.randint(3, TGT_VOCAB_SIZE-1) for _ in range(tgt_len-2)] + [2]
        
        src_data.append(src_seq)
        tgt_data.append(tgt_seq)
    
    dataloader = DataLoader(src_data, tgt_data, BATCH_SIZE, shuffle=True)
    
    # 创建模型
    print("\n创建模型...")
    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT
    )
    
    # 统计参数量
    params = collect_params(model)
    total_params = sum(p.size for p in params)
    print(f"总参数量: {total_params:,}")
    
    # 创建优化器
    optimizer = Adam(lr=LEARNING_RATE)
    
    # 训练
    print("\n开始训练...")
    print("-" * 60)
    train(model, dataloader, optimizer, EPOCHS, TGT_VOCAB_SIZE)
    
    print("\n训练完成！")
    
    # 推理示例
    print("\n推理示例:")
    test_src = np.array([[1, 5, 8, 12, 2]])  # <sos> token1 token2 token3 <eos>
    test_tgt = np.array([[1]])  # 只给<sos>开始
    
    # 自回归生成
    for _ in range(10):
        output = model.forward(test_src, test_tgt, training=False)
        next_token_logits = output[0, -1, :]
        next_token = np.argmax(next_token_logits)
        
        test_tgt = np.concatenate([test_tgt, [[next_token]]], axis=1)
        
        if next_token == 2:  # <eos>
            break
    
    print(f"输入序列: {test_src[0].tolist()}")
    print(f"生成序列: {test_tgt[0].tolist()}")


if __name__ == "__main__":
    main()