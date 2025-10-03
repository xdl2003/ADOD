from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# ========== 1. 生成正常点 (来自三个高斯分布) ==========
# 设定三个高斯分布的中心（均值）
# 这里将它们放在一个等边三角形的三个顶点上，边长为 5
centers = [
    [0.0, 0.0],      # 第一个簇中心
    [5.0, 0.0],      # 第二个簇中心
    [2.5, 4.33]      # 第三个簇中心 (5 * sin(60°) ≈ 4.33)
]
# 每个簇的大小（425个点尽量平均分配）
cluster_sizes = [140, 140, 145]  # 总和为 425

# 生成三个高斯分布的点
inliers, _ = make_blobs(
    n_samples=cluster_sizes,
    centers=centers,
    cluster_std=[0.6, 1.2, 0.3],  # 对应论文中的 σ=[0.6, 1.2, 0.3]
    random_state=42  # 为了结果可复现
)

# ========== 2. 生成异常点 (均匀分布) ==========
# 确定均匀分布的范围
# 为了确保异常点是离群的，范围应该比正常点的范围大
x_min, x_max = inliers[:, 0].min() - 5, inliers[:, 0].max() + 5
y_min, y_max = inliers[:, 1].min() - 5, inliers[:, 1].max() + 5

# 生成75个在 (x_min, x_max) x (y_min, y_max) 范围内均匀分布的点
outliers = np.random.uniform(
    low=[x_min, y_min],
    high=[x_max, y_max],
    size=(75, 2)  # 75个点，每个点2维
)

# ========== 3. 合并数据集并标记标签 ==========
# 将正常点和异常点合并
X = np.vstack([inliers, outliers])

# 创建标签：0 表示正常点 (inlier)，1 表示异常点 (outlier)
labels = np.hstack([np.zeros(len(inliers)), np.ones(len(outliers))])

# ========== 4. 可视化 (可选) ==========
plt.figure(figsize=(8, 6))
# 绘制正常点（白色）
plt.scatter(inliers[:, 0], inliers[:, 1], c='white', edgecolors='k', s=30, label='Inliers (Normal)')
# 绘制异常点（黑色）
plt.scatter(outliers[:, 0], outliers[:, 1], c='black', s=30, label='Outliers')
plt.title('ThreeBlob Outlier Dataset (Synthetic)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.axis('equal')  # 保持坐标轴比例一致
plt.grid(True, alpha=0.3)


# ========== 5. 保存数据 (可选) ==========
# 可以将数据保存为 .npy 文件
# 定义路径
filepath = Path('../data/threeblob_X.npy')
filepath.parent.mkdir(exist_ok=True, parents=True)  # 自动创建所有上级目录
np.save(filepath, X)
filepath = Path('../data/threeblob_label.npy')
np.save(filepath, labels)
filepath = Path('../data/threeblob.png')
plt.savefig(filepath)

plt.show()

print(f"数据集总样本数: {len(X)}")
print(f"正常点数量: {np.sum(labels == 0)}")
print(f"异常点数量: {np.sum(labels == 1)}")