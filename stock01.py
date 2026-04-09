import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 获取当前日期及30天前的日期
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

# 收集所有数据
all_data = []
print("正在获取数据...")

for i in range(30):
    current_date = start_date + timedelta(days=i)
    date_param = current_date.strftime("%Y%m%d")
    try:
        day_data = ak.get_cffex_daily(date=date_param)
        if day_data is not None and not day_data.empty:
            all_data.append(day_data)
            print(f"成功获取 {date_param} 的数据，数据列: {day_data.columns.tolist()}")
    except Exception as e:
        print(f"获取 {date_param} 的数据失败: {e}")

if not all_data:
    print("未获取到任何数据")
    exit()

# 合并所有数据
df = pd.concat(all_data, ignore_index=True)

# 查看数据结构和列名
print(f"\n数据总行数: {len(df)}")
print(f"所有列名: {df.columns.tolist()}")
print(f"\n前几行数据:")
print(df.head())

# 筛选出沪深300股指期货（IF合约）的数据
# 根据实际列名调整，可能是"合约"、"symbol"、"code"等
if "合约" in df.columns:
    if_data = df[df["合约"].str.contains("IF", na=False)].copy()
elif "symbol" in df.columns:
    if_data = df[df["symbol"].str.contains("IF", na=False)].copy()
elif "code" in df.columns:
    if_data = df[df["code"].str.contains("IF", na=False)].copy()
else:
    print("未找到合约字段，请检查数据列名")
    # 列出所有可能的合约相关字段
    for col in df.columns:
        if "合约" in col or "symbol" in col.lower() or "code" in col.lower():
            print(f"可能的合约字段: {col}")
    exit()

if if_data.empty:
    print("未找到IF合约数据")
    # 查看所有可用的合约代码
    contract_col = [col for col in df.columns if "合约" in col or "symbol" in col.lower() or "code" in col.lower()][0]
    print(f"所有可用的合约代码: {df[contract_col].unique()}")
    exit()

# 确定日期列名
date_col = None
for col in ["日期", "date", "trade_date", "trading_date"]:
    if col in if_data.columns:
        date_col = col
        break

if not date_col:
    print("未找到日期字段")
    exit()

# 确定持仓量列名
oi_col = None
for col in ["持仓量", "open_interest", "oi", "position"]:
    if col in if_data.columns:
        oi_col = col
        break

if not oi_col:
    print("未找到持仓量字段")
    print(f"可用的列名: {if_data.columns.tolist()}")
    exit()

# 转换日期格式
if_data[date_col] = pd.to_datetime(if_data[date_col])

# 按日期排序
if_data = if_data.sort_values(by=date_col)

# 绘制未平仓合约走势图
plt.figure(figsize=(12, 6))
plt.plot(if_data[date_col], if_data[oi_col], marker='o', linestyle='-', linewidth=2, markersize=4)
plt.title(f"沪深300股指期货（IF合约）近30天未平仓合约走势", fontsize=16)
plt.xlabel("日期", fontsize=12)
plt.ylabel(f"{oi_col}（手）", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 打印数据概览
print(f"\n沪深300股指期货（IF合约）数据概览:")
print(if_data[[date_col, contract_col, oi_col]].to_string(index=False))

# 如果有多个合约，可以分别查看
if len(if_data[contract_col].unique()) > 1:
    print(f"\n发现多个IF合约: {if_data[contract_col].unique()}")
    for contract in if_data[contract_col].unique():
        contract_data = if_data[if_data[contract_col] == contract]
        print(f"\n{contract} 合约:")
        print(f"  日期范围: {contract_data[date_col].min()} 到 {contract_data[date_col].max()}")
        print(f"  最新持仓量: {contract_data[oi_col].iloc[-1]}")