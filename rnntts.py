import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. 定义核心组件：注意力机制
# ==============================
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super(Attention, self).__init__()
        # 计算注意力分数的线性层
        self.score_layer = nn.Linear(encoder_dim + decoder_dim, decoder_dim)
        self.v = nn.Linear(decoder_dim, 1, bias=False)
        
    def forward(self, encoder_outputs, decoder_hidden):
        """
        encoder_outputs: (Batch, Seq_Len_Enc, Encoder_Dim)
        decoder_hidden: (Batch, Decoder_Dim)
        """
        batch_size, seq_len_enc, _ = encoder_outputs.size()
        
        # 将 decoder_hidden 重复以匹配 encoder 的时间步
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, seq_len_enc, -1)
        
        # 拼接并计算能量分数
        combined = torch.cat((encoder_outputs, decoder_hidden_expanded), dim=2)
        energy = torch.tanh(self.score_layer(combined))
        scores = self.v(energy).squeeze(2) # (Batch, Seq_Len_Enc)
        
        # Softmax 得到权重分布
        weights = torch.softmax(scores, dim=1)
        
        # 加权求和得到上下文向量 (Context Vector)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, weights

# ==============================
# 2. 定义 TTS 模型 (Encoder-Decoder with Attention)
# ==============================
class SimpleTacotron(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encoder_dim, decoder_dim, output_dim):
        super(SimpleTacotron, self).__init__()
        
        # --- 编码器 (Encoder) ---
        # 将字符索引转换为向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 双向 LSTM 提取文本特征
        self.encoder_lstm = nn.LSTM(embedding_dim, encoder_dim // 2, 
                                    bidirectional=True, batch_first=True)
        
        # --- 注意力 ---
        self.attention = Attention(encoder_dim, decoder_dim)
        
        # --- 解码器 (Decoder) ---
        # 输入：上一帧的频谱图 + 上下文向量
        decoder_input_dim = output_dim + encoder_dim 
        self.decoder_lstm = nn.LSTMCell(decoder_input_dim, decoder_dim)
        
        # 输出层：预测当前帧的频谱图
        self.fc_out = nn.Linear(decoder_dim, output_dim)
        
        # 停止标记预测 (判断是否生成结束)
        self.stop_layer = nn.Linear(decoder_dim, 1)
        
        self.decoder_dim = decoder_dim
        self.output_dim = output_dim

    def forward(self, text_inputs, mel_targets=None, teacher_forcing_ratio=0.5):
        """
        text_inputs: (Batch, Text_Len)
        mel_targets: (Batch, Mel_Len, Output_Dim) - 训练时使用
        """
        batch_size = text_inputs.size(0)
        
        # 1. 编码阶段
        embedded = self.embedding(text_inputs) # (B, T_text, Emb)
        encoder_outputs, _ = self.encoder_lstm(embedded) # (B, T_text, Enc_Dim)
        
        # 2. 解码阶段初始化
        # 解码器隐藏状态
        h_dec = torch.zeros(batch_size, self.decoder_dim).to(text_inputs.device)
        c_dec = torch.zeros(batch_size, self.decoder_dim).to(text_inputs.device)
        
        # 初始输入 (全零帧)
        go_frame = torch.zeros(batch_size, self.output_dim).to(text_inputs.device)
        decoder_input = go_frame
        
        mel_outputs = []
        stop_outputs = []
        alignments = []
        
        # 决定输出长度 (训练时用目标长度，推理时需设定最大长度)
        max_len = mel_targets.size(1) if mel_targets is not None else 50 
        
        for t in range(max_len):
            # A. 计算注意力
            context, weights = self.attention(encoder_outputs, h_dec)
            alignments.append(weights)
            
            # B. 准备 LSTM 输入: [上一帧频谱, 上下文向量]
            lstm_input = torch.cat((decoder_input, context), dim=1)
            
            # C. RNN 步进
            h_dec, c_dec = self.decoder_lstm(lstm_input, (h_dec, c_dec))
            
            # D. 预测当前帧
            mel_pred = self.fc_out(h_dec)
            stop_pred = torch.sigmoid(self.stop_layer(h_dec))
            
            mel_outputs.append(mel_pred)
            stop_outputs.append(stop_pred)
            
            # E. 准备下一帧输入 (Teacher Forcing)
            if mel_targets is not None and np.random.rand() < teacher_forcing_ratio:
                decoder_input = mel_targets[:, t, :] # 使用真实值
            else:
                decoder_input = mel_pred # 使用预测值
        
        # 整理输出
        mel_outputs = torch.stack(mel_outputs, dim=1) # (B, Max_Len, Dim)
        stop_outputs = torch.stack(stop_outputs, dim=1) # (B, Max_Len, 1)
        alignments = torch.stack(alignments, dim=1) # (B, Max_Len, Text_Len)
        
        return mel_outputs, stop_outputs, alignments

# ==============================
# 3. 模拟数据与训练循环
# ==============================

# 参数设置
vocab_size = 50       # 模拟字符集大小
embedding_dim = 256
encoder_dim = 512
decoder_dim = 1024
output_dim = 80       # 梅尔频谱图的频段数 (通常是 80)
batch_size = 4
text_len = 20
mel_len = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleTacotron(vocab_size, embedding_dim, encoder_dim, decoder_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion_mel = nn.MSELoss()
criterion_stop = nn.BCELoss()

print(f"模型已构建在 {device} 上")
print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

# 生成假数据
# 文本输入: 随机整数序列
fake_text = torch.randint(0, vocab_size, (batch_size, text_len)).to(device)
# 目标频谱: 随机浮点数序列 (模拟真实的 Mel 频谱)
fake_mel = torch.randn(batch_size, mel_len, output_dim).to(device)
# 停止标记: 最后一帧为 1，其余为 0
fake_stop = torch.zeros(batch_size, mel_len, 1).to(device)
fake_stop[:, -1, :] = 1.0

print("\n开始训练演示 (仅 10 个 step)...")

model.train()
for step in range(10):
    optimizer.zero_grad()
    
    # 前向传播
    # 注意：实际训练中 teacher_forcing_ratio 会随 epoch 逐渐降低
    mel_out, stop_out, align = model(fake_text, fake_mel, teacher_forcing_ratio=0.8)
    
    # 截断输出以匹配目标长度
    mel_out = mel_out[:, :mel_len, :]
    stop_out = stop_out[:, :mel_len, :]
    
    # 计算损失
    loss_mel = criterion_mel(mel_out, fake_mel)
    loss_stop = criterion_stop(stop_out, fake_stop)
    loss = loss_mel + loss_stop
    
    # 反向传播
    loss.backward()
    
    # 梯度裁剪 (防止 RNN 梯度爆炸)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    print(f"Step {step+1}, Loss: {loss.item():.4f} (Mel: {loss_mel.item():.4f}, Stop: {loss_stop.item():.4f})")

# ==============================
# 4. 可视化注意力 (模拟推理)
# ==============================
print("\n生成注意力热力图...")
model.eval()
with torch.no_grad():
    # 推理时不使用 teacher forcing，完全靠模型自己生成
    mel_out, stop_out, align = model(fake_text[:1], None, teacher_forcing_ratio=0.0)

# 绘制注意力权重 (横轴: 文本长度, 纵轴: 生成的频谱帧)
plt.figure(figsize=(10, 6))
# align 形状: (1, Max_Len, Text_Len) -> 取第一个样本
attention_weights = align[0].cpu().numpy() 
plt.imshow(attention_weights.T, aspect='auto', cmap='hot')
plt.xlabel("Generated Frames")
plt.ylabel("Input Text Tokens")
plt.title("Attention Alignment (Simulated)")
plt.colorbar(label='Weight')
plt.show()

print("完成！这是一个 TTS 模型的骨架。要生成真实声音，你需要：")
print("1. 加载真实的文本 - 音频数据集 (如 LJSpeech)。")
print("2. 提取真实的梅尔频谱图作为训练目标。")
print("3. 训练更多 Epoch。")
print("4. 使用声码器 (如 Griffin-Lim 或 HiFi-GAN) 将输出的梅尔频谱图转换为波形音频。")