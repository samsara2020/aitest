import numpy as np

def softmax(z):
    """
    Softmax 函数实现
    z: 输入向量或矩阵
    """
    # 减去最大值防止数值溢出 (数值稳定性技巧)
    print(f"输入: {z}")
    print(f"np.max(z): {np.max(z)}")
    print(f"z-np.max(z): {z - np.max(z)}")
    exp_z = np.exp(z - np.max(z))
    print(f"exp_z: {exp_z}")
    print(f"np.sum(exp_z): {np.sum(exp_z)}")
    return exp_z / np.sum(exp_z)

# 示例
z = np.array((2.0, 1.0))
probs = softmax(z)

print(f"Softmax 输出: {probs}")
print(f"概率之和: {np.sum(probs):.6f}")  # 应为 1.0