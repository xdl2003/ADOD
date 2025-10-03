import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from sklearn.metrics import roc_auc_score

threeblob_X_file = '../data/threeblob_X.npy'
threeblob_label_file = '../data/threeblob_label.npy'


def knn_outlier_detection(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    使用 kNN 算法进行异常检测。baseline。

    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, n_features)
        输入的数据矩阵。
    k : int
        最近邻的数量（k值）。

    Returns:
    --------
    scores : np.ndarray, shape (n_samples,)
        每个样本的异常分数（到第k个最近邻的距离），分数越高越可能是异常。
    ranks : np.ndarray, shape (n_samples,)
        按异常分数从高到低排序的样本索引。

    Example:
    --------
    scores, ranks = knn_outlier_detection(X, k=20)
    """
    n_samples = X.shape[0]

    # 使用 sklearn 的 NearestNeighbors 找到每个点的 k 个最近邻
    # 注意：它会包含自己（距离为0），所以我们需要找 k+1 个邻居，然后取第2到第k+1个
    n_neighbors = k + 1
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)

    # distances.shape = (n_samples, k+1), 第一列是到自身的距离（0）
    distances, _ = nn.kneighbors(X)

    # 取第 k 个最近邻的距离（即索引为 k 的列，因为索引从0开始）
    # 注意：distances[:, 0] 是到自身的距离（0），distances[:, 1] 是到第1个邻居，... distances[:, k] 是到第k个邻居
    kth_distances = distances[:, k]  # 异常分数：越大越异常

    # 按分数从高到低排序（异常优先）
    ranks = np.argsort(-kth_distances)  # 负号实现降序排序

    return kth_distances, ranks

def load_data():
    """
    加载特征数据和标签数据
    """
    try:
        threeblob_X = np.load(threeblob_X_file)
        threeblob_y = np.load(threeblob_label_file)
        print("✅ 数据加载成功！")
        print(f"特征数据 X 的形状: {threeblob_X.shape}")
        print(f"标签数据 y 的形状: {threeblob_y.shape}")
        print(f"标签分布: 正常点 (0) = {np.sum(threeblob_y == 0)}, 异常点 (1) = {np.sum(threeblob_y == 1)}")
        return threeblob_X, threeblob_y
    except FileNotFoundError as e:
        print(f"❌ 错误：找不到文件 {e}")
        print("请确保 threeblob_X.npy 和 threeblob_labels.npy 在当前目录下。")
        return None, None
    except Exception as e:
        print(f"❌ 加载数据时发生未知错误: {e}")
        return None, None

def get_result(X, y, scores, ranks, filepath=None, title=None):
    """
    计算 ROC AUC 和 P@N，并可选地绘制高质量决策边界图（论文风格）
    使用插值生成平滑的异常分数热力图 + 等高线
    """
    from sklearn.metrics import roc_auc_score

    # ---------- 1. 计算评估指标 ----------
    roc_auc = roc_auc_score(y, scores)

    N = np.sum(y == 1)
    if N == 0 or N == len(y):
        raise ValueError("数据集需要包含正常和异常样本，才能计算 P@N。")

    top_N_indices = ranks[:N]
    p_at_n = np.mean(y[top_N_indices] == 1)
    errno = int(N * p_at_n)
    title = title + f" ({errno} errors)"

    # ---------- 2. 高质量可视化 ----------
    if filepath is not None and title is not None:
        plt.figure(figsize=(8, 6), dpi=150)
        ax = plt.gca()

        # ---- 创建网格用于插值 ----
        margin = 0.3
        x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
        y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

        nx, ny = 200, 200
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                             np.linspace(y_min, y_max, ny))

        # ---- 插值异常分数到网格 ----
        grid_scores = griddata(
            points=X,
            values=scores,
            xi=(xx, yy),
            method='cubic',  # 可选 'linear', 'nearest'
            fill_value=np.nan
        )

        # ---- 绘制热力图（异常分数） ----
        # 使用柔和的 colormap，突出异常区域
        cmap = plt.cm.YlOrRd  # Yellow -> Orange -> Red
        contourf = ax.contourf(xx, yy, grid_scores, levels=20, cmap=cmap, alpha=0.6, zorder=1)

        # ---- 绘制等高线 ----
        contour = ax.contour(xx, yy, grid_scores, levels=10, colors='orange', alpha=0.8, linewidths=1.2, zorder=2)

        # ---- 绘制数据点 ----
        # 正常点：白色带黑边
        inliers = X[y == 0]
        ax.scatter(inliers[:, 0], inliers[:, 1],
                   c='white', edgecolors='black', s=50, alpha=0.9, zorder=3, label='Inliers')

        # 异常点：黑色实心
        outliers = X[y == 1]
        ax.scatter(outliers[:, 0], outliers[:, 1],
                   c='black', s=50, alpha=0.9, zorder=3, label='Outliers')

        # ---- 美化样式 ----
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('Feature 1', fontsize=12)
        ax.set_ylabel('Feature 2', fontsize=12)

        # 去掉上/右边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

        # 坐标轴刻度
        ax.tick_params(axis='both', which='major', labelsize=10, width=0.5, length=3)

        # 图例
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right', fontsize=10, frameon=True, fancybox=False, edgecolor='black', framealpha=0.9)

        # 调整布局
        plt.tight_layout()

        # 保存为高分辨率 PNG 或 PDF（推荐论文用 PDF）
        plt.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        print(f"✅ 论文级图像已保存至: {filepath}")

    return roc_auc, p_at_n


if __name__ == "__main__":
    tb_X, tb_y = load_data()
    scores, ranks = knn_outlier_detection(tb_X, int(math.sqrt(tb_X.shape[0])))
    roc_auc, p_at_n = get_result(tb_X, tb_y, scores, ranks, "./output/tb_knn", "knn outcome on threeblob data")
    print(roc_auc)
    print(p_at_n)

