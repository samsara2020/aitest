import numpy as np

# ==========================================
# 1. 数据准备与预处理
# ==========================================
# 简单的语料库
corpus_text = [
    "I love natural language processing",
    "I love machine learning",
    "Machine learning is great",
    "Natural language processing is fun"
]

# 构建词汇表
words = []
for sentence in corpus_text:
    words.extend(sentence.lower().split())
words = list(set(words)) # 去重
vocab_size = len(words)
word2idx = {word: i for i, word in enumerate(words)}
idx2word = {i: word for i, word in enumerate(words)}

print(f"词汇表: {words}")
print(f"word2idx: {word2idx}")
print(f"idx2word: {idx2word}")
print(f"词汇表大小: {vocab_size}\n")

# 生成 Skip-gram 训练样本 (中心词, 上下文词)
def create_training_data(sentences, window_size=2):
    pairs = []
    for sentence in sentences:
        sentence_words = sentence.lower().split()
        for i, center_word in enumerate(sentence_words):
            # 确定上下文范围
            
            start = max(0, i - window_size)
            end = min(len(sentence_words), i + window_size + 1)
            temp = end -	start
            print(f"i : {i}")
            print(f"start - end : {temp}")
            for j in range(start, end):
                if i != j: # 排除中心词自己
                    context_word = sentence_words[j]
                    pairs.append((word2idx[center_word], word2idx[context_word]))
    return pairs

training_pairs = create_training_data(corpus_text, window_size=2)

print(f"training_pairs len: {training_pairs.__len__()}")
print(f"type(training_pairs[0]): {type(training_pairs[0])}")
print(f"training_pairs: {training_pairs}")
# ==========================================
# 2. 模型参数初始化
# ==========================================
embedding_dim = 5  # 词向量维度 (设小一点方便观察)
learning_rate = 0.01
epochs = 5000

# 随机初始化权重矩阵
# W1: 输入矩阵 (Vocab x Dim)
# W2: 输出矩阵 (Dim x Vocab)
np.random.seed(42)
W1 = np.random.rand(vocab_size, embedding_dim)
W2 = np.random.rand(embedding_dim, vocab_size)

# ==========================================
# 3. 辅助函数 (Softmax)
# ==========================================
def softmax(x):
    # 减去最大值以防止数值溢出
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ==========================================
# 4. 训练循环 (核心算法)
# ==========================================
print("开始训练...")
for epoch in range(epochs):
    total_loss = 0
    
    for center_idx, context_idx in training_pairs:
        # --- 1. 前向传播 ---
        
        # 输入层 -> 隐藏层
        # x 是 One-hot 向量，x @ W1 相当于取出 W1 中对应的那一行
        x = np.zeros(vocab_size)
        x[center_idx] = 1
        
        h = np.dot(x, W1)  # 隐藏层输出 (1 x Dim)
        
        # 隐藏层 -> 输出层
        u = np.dot(h, W2)  # 输出层得分 (1 x Vocab)
        
        # Softmax 归一化
        y_pred = softmax(u) # 预测概率分布
        
        # --- 2. 计算损失 (交叉熵) ---
        # 真实标签也是一个 One-hot 向量 (target)
        target = np.zeros(vocab_size)
        target[context_idx] = 1
        
        # 交叉熵损失: -sum(target * log(y_pred))
        loss = -np.sum(target * np.log(y_pred + 1e-8)) 
        total_loss += loss
        
        # --- 3. 反向传播 (计算梯度) ---
        
        # 输出层误差 (y_pred - target)
        # 形状: (Vocab,)
        dl_du = y_pred - target 
        
        # 计算 W2 的梯度: h.T @ dl_du
        # h: (Dim,), dl_du: (Vocab,) -> (Dim x Vocab)
        dl_dW2 = np.outer(h, dl_du)
        
        # 计算隐藏层误差: dl_du @ W2.T
        # 形状: (Dim,)
        dl_dh = np.dot(dl_du, W2.T)
        
        # 计算 W1 的梯度: x.T @ dl_dh
        # x: (Vocab,), dl_dh: (Dim,) -> (Vocab x Dim)
        dl_dW1 = np.outer(x, dl_dh)
        
        # --- 4. 更新权重 ---
        W1 -= learning_rate * dl_dW1
        W2 -= learning_rate * dl_dW2

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# ==========================================
# 5. 结果展示
# ==========================================
print("\n训练完成！\n")

# 通常我们取 W1 作为最终的词向量 (或者 W1+W2.T 的平均)
final_embeddings = W1

def get_similar_words(word, top_n=3):
    if word not in word2idx:
        return "词不在词汇表中"
    
    word_vec = final_embeddings[word2idx[word]]
    # 计算余弦相似度
    similarities = np.dot(final_embeddings, word_vec) / (np.linalg.norm(final_embeddings, axis=1) * np.linalg.norm(word_vec))
    
    # 获取相似度最高的词索引 (排除自身)
    similar_indices = np.argsort(similarities)[::-1]
    
    result = []
    for idx in similar_indices:
        if idx != word2idx[word]:
            result.append((idx2word[idx], similarities[idx]))
            if len(result) == top_n:
                break
    return result

# 测试
test_word = "love"
print(f"与 '{test_word}' 最相似的词:")
for word, score in get_similar_words(test_word):
    print(f"- {word}: {score:.4f}")

print(f"\n词向量示例 ('machine'): \n{final_embeddings[word2idx['machine']]}")