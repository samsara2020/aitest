def conv2d_pure_python(X, K, stride=1, padding=0):
    """
    纯 Python 实现二维卷积
    
    参数:
        X: 输入特征图，二维列表，形状 (H, W)，例如: [[1,2],[3,4]]
        K: 卷积核，二维列表，形状 (k_h, k_w)，例如: [[1,0],[0,1]]
        stride: 步长，整数，卷积核滑动的步长，默认值为1
        padding: 填充大小，整数，在特征图四周填充0的行列数，默认值为0
    
    返回:
        Y: 输出特征图，二维列表，形状 (out_h, out_w)
    """
    
    # ========== 1. 获取输入尺寸 ==========
    # h: 输入特征图的高度（行数）
    h = len(X)
    # w: 输入特征图的宽度（列数），如果h>0则取第一行的长度，否则为0
    w = len(X[0]) if h > 0 else 0
    
    # k_h: 卷积核的高度（行数）
    k_h = len(K)
    # k_w: 卷积核的宽度（列数），如果k_h>0则取第一行的长度，否则为0
    k_w = len(K[0]) if k_h > 0 else 0
    
    # ========== 2. 添加填充 ==========
    # padded_h: 填充后的特征图高度 = 原高度 + 2 * 填充大小
    padded_h = h + 2 * padding
    # padded_w: 填充后的特征图宽度 = 原宽度 + 2 * 填充大小
    padded_w = w + 2 * padding
    
    # X_padded: 填充后的特征图，初始化为全0的二维列表
    X_padded = [[0] * padded_w for _ in range(padded_h)]
    
    # 将原特征图的值复制到填充后的中心位置
    for i in range(h):          # i: 原特征图的行索引，范围 0 到 h-1
        for j in range(w):      # j: 原特征图的列索引，范围 0 到 w-1
            # 目标行 = i + padding，目标列 = j + padding
            X_padded[i + padding][j + padding] = X[i][j]
    
    # ========== 3. 计算输出尺寸 ==========
    # out_h: 输出特征图的高度
    # 公式: (填充后高度 - 卷积核高度) // 步长 + 1
    out_h = (padded_h - k_h) // stride + 1
    # out_w: 输出特征图的宽度
    # 公式: (填充后宽度 - 卷积核宽度) // 步长 + 1
    out_w = (padded_w - k_w) // stride + 1
    
    # Y: 输出特征图，初始化为全0的二维列表，形状为 (out_h, out_w)
    Y = [[0] * out_w for _ in range(out_h)]
    
    # ========== 4. 滑动窗口卷积计算 ==========
    # i: 输出特征图的行索引，范围 0 到 out_h-1
    for i in range(out_h):
        # j: 输出特征图的列索引，范围 0 到 out_w-1
        for j in range(out_w):
            # start_i: 卷积核在填充后特征图上的起始行位置
            # 公式: i * stride
            start_i = i * stride
            # start_j: 卷积核在填充后特征图上的起始列位置
            # 公式: j * stride
            start_j = j * stride
            
            # total: 当前输出位置的值，累加卷积核与对应区域的乘积
            total = 0
            
            # ki: 卷积核的行索引，范围 0 到 k_h-1
            for ki in range(k_h):
                # kj: 卷积核的列索引，范围 0 到 k_w-1
                for kj in range(k_w):
                    # 累加: 填充后特征图上的像素值 × 卷积核对应位置的权重
                    # X_padded[start_i + ki][start_j + kj]: 卷积核覆盖的像素值
                    # K[ki][kj]: 卷积核的权重
                    total += X_padded[start_i + ki][start_j + kj] * K[ki][kj]
            
            # 将计算结果存储到输出特征图的对应位置
            Y[i][j] = total
    
    # 返回输出特征图
    return Y


def conv2d_multichannel(X, K, stride=1, padding=0):
    """
    支持多输入通道和多输出通道的纯 Python 卷积
    
    参数:
        X: 输入特征图，三维列表，形状 (C_in, H, W)
           C_in: 输入通道数，例如 3（RGB图像）
           H: 特征图高度
           W: 特征图宽度
        K: 卷积核，四维列表，形状 (C_out, C_in, k_h, k_w)
           C_out: 输出通道数（卷积核个数）
           C_in: 输入通道数（必须与X的通道数匹配）
           k_h: 卷积核高度
           k_w: 卷积核宽度
        stride: 步长，整数
        padding: 填充大小，整数
    
    返回:
        Y: 输出特征图，三维列表，形状 (C_out, out_h, out_w)
    """
    
    # ========== 1. 输入验证 ==========
    # 检查输入特征图是否为空
    if not X or not K:
        return []
    
    # C_in: 输入通道数
    C_in = len(X)
    if C_in == 0:
        return []
    
    # H: 输入特征图高度（第一个通道的行数）
    H = len(X[0])
    # W: 输入特征图宽度（第一个通道第一行的列数）
    W = len(X[0][0]) if H > 0 else 0
    
    # C_out: 输出通道数（卷积核的数量）
    C_out = len(K)
    
    # k_h: 卷积核高度（第一个输出通道的第一个输入通道的卷积核行数）
    k_h = len(K[0][0]) if C_out > 0 and C_in > 0 else 0
    # k_w: 卷积核宽度
    k_w = len(K[0][0][0]) if k_h > 0 else 0
    
    # ========== 2. 对每个输入通道添加填充 ==========
    # X_padded: 存储填充后的特征图，形状 (C_in, padded_h, padded_w)
    X_padded = []
    
    # ic: 输入通道索引，范围 0 到 C_in-1
    for ic in range(C_in):
        # 计算填充后的高度和宽度
        padded_h = H + 2 * padding
        padded_w = W + 2 * padding
        
        # 创建全0的填充特征图
        padded = [[0] * padded_w for _ in range(padded_h)]
        
        # 将原通道数据复制到填充后的中心位置
        for i in range(H):      # i: 原特征图的行索引
            for j in range(W):  # j: 原特征图的列索引
                # 目标位置: 行偏移padding，列偏移padding
                padded[i + padding][j + padding] = X[ic][i][j]
        
        # 将填充后的通道添加到列表
        X_padded.append(padded)
    
    # ========== 3. 计算输出尺寸 ==========
    # padded_h: 填充后的特征图高度（所有通道相同）
    padded_h = len(X_padded[0])
    # padded_w: 填充后的特征图宽度
    padded_w = len(X_padded[0][0])
    
    # out_h: 输出特征图高度
    out_h = (padded_h - k_h) // stride + 1
    # out_w: 输出特征图宽度
    out_w = (padded_w - k_w) // stride + 1
    
    # Y: 输出特征图，形状 (C_out, out_h, out_w)
    # 初始化为全0的三维列表
    Y = [[[0] * out_w for _ in range(out_h)] for _ in range(C_out)]
    
    # ========== 4. 多通道卷积计算 ==========
    # oc: 输出通道索引，范围 0 到 C_out-1
    for oc in range(C_out):
        # ic: 输入通道索引，范围 0 到 C_in-1
        for ic in range(C_in):
            # i: 输出特征图的行索引，范围 0 到 out_h-1
            for i in range(out_h):
                # j: 输出特征图的列索引，范围 0 到 out_w-1
                for j in range(out_w):
                    # start_i: 卷积核在填充后特征图上的起始行
                    start_i = i * stride
                    # start_j: 卷积核在填充后特征图上的起始列
                    start_j = j * stride
                    
                    # total: 当前通道组合的卷积结果累加值
                    total = 0
                    
                    # ki: 卷积核的行索引，范围 0 到 k_h-1
                    for ki in range(k_h):
                        # kj: 卷积核的列索引，范围 0 到 k_w-1
                        for kj in range(k_w):
                            # 累加: 填充后的输入像素 × 卷积核权重
                            # X_padded[ic][start_i+ki][start_j+kj]: 第ic个输入通道的像素值
                            # K[oc][ic][ki][kj]: 第oc个输出通道、第ic个输入通道的卷积核权重
                            total += X_padded[ic][start_i + ki][start_j + kj] * K[oc][ic][ki][kj]
                    
                    # 将当前输入通道的卷积结果累加到输出特征图
                    # Y[oc][i][j]: 第oc个输出通道、位置(i,j)的输出值
                    Y[oc][i][j] += total
    
    # 返回输出特征图
    return Y


# ========== 示例代码 ==========
if __name__ == "__main__":
    print("=" * 50)
    print("单通道卷积示例")
    print("=" * 50)
    
    # X: 输入特征图，形状 (4, 4)
    X = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]]
    
    # K: 卷积核，形状 (2, 2)
    K = [[1, 0],
         [0, 1]]
    
    print("输入特征图 X (4x4):")
    for row in X:
        print(row)
    
    print("\n卷积核 K (2x2):")
    for row in K:
        print(row)
    
    # stride=1, padding=0
    result = conv2d_pure_python(X, K, stride=1, padding=0)
    print("\n卷积结果 Y (3x3):")
    for row in result:
        print(row)
    
    print("\n" + "=" * 50)
    print("多通道卷积示例")
    print("=" * 50)
    
    # X_multi: 多通道输入，形状 (2, 2, 2)
    # 2个输入通道，每个通道尺寸 2x2
    X_multi = [
        [[1, 2], [3, 4]],   # 输入通道 0
        [[5, 6], [7, 8]]    # 输入通道 1
    ]
    
    # K_multi: 多通道卷积核，形状 (1, 2, 2, 2)
    # 1个输出通道，2个输入通道，卷积核尺寸 2x2
    K_multi = [
        [                   # 输出通道 0
            [[1, 0], [0, 1]],  # 对应输入通道 0 的卷积核
            [[0, 1], [1, 0]]   # 对应输入通道 1 的卷积核
        ]
    ]
    
    print("多通道输入 X (2个通道，每个2x2):")
    for c, channel in enumerate(X_multi):
        print(f"通道 {c}:")
        for row in channel:
            print(row)
    
    print("\n多通道卷积核 K (1个输出通道，2个输入通道，每个2x2):")
    for oc, out_ch in enumerate(K_multi):
        print(f"输出通道 {oc}:")
        for ic, kernel in enumerate(out_ch):
            print(f"  输入通道 {ic} 的卷积核:")
            for row in kernel:
                print(f"    {row}")
    
    result_multi = conv2d_multichannel(X_multi, K_multi, stride=1, padding=0)
    print("\n多通道卷积结果 (1个输出通道，1x1):")
    for oc, channel in enumerate(result_multi):
        print(f"输出通道 {oc}:")
        for row in channel:
            print(row)