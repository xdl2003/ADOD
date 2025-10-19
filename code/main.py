import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd
from adod import adod_outlier_detection

threeblob_X_file = '../data/threeblob_X.npy'
threeblob_label_file = '../data/threeblob_label.npy'
pendigits_X_file = '../data/pendigits_X.npy'
pendigits_y_file = '../data/pendigits_y.npy'
speech_X_file = '../data/speech_X.npy'
speech_y_file = '../data/speech_y.npy'
file0 = '../data/5_campaign.npz'
file1 = '../data/22_magic.gamma.npz'
file2 = '../data/3_backdoor.npz'


def knn_outlier_detection(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Use KNN algorithm to detect outliers。baseline。

    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, n_features)
        Input data matrix
    k : int
        number of neighbors (k value)

    Returns:
    --------
    scores : np.ndarray, shape (n_samples,)
        anomalous scores(distance to k-th neighbor), the higher socres, the higher probablity of anomalous points
    ranks : np.ndarray, shape (n_samples,)
        sample index by anomalous scores from high to low

    Example:
    --------
    scores, ranks = knn_outlier_detection(X, k=20)
    """
    n_samples = X.shape[0]

    # Use NearesrNeighbors of sklearn to find k neighbors of each point 使用 sklearn 的 NearestNeighbors 找到每个点的 k 个最近邻
    # Note：It woudl contain itself(distance = 0), so we need to find k+1 neighbors then take 2-nd to n+1-th
    n_neighbors = k + 1
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)

    # distances.shape = (n_samples, k+1), distance to itself is in the first col
    distances, _ = nn.kneighbors(X)

    # Take k-th nearest distance (index at k due to index start at 0)
    # Note: distance[:, 0] is the distance to itself(0), distance[:, 1] is to 1-st neighbor,..., distance[:, k] is to k-th neighbor
    kth_distances = distances[:, k]  # 异常分数：越大越异常

    # Descending sort by scores  (anomalous points first)
    ranks = np.argsort(-kth_distances)  # Negative sign implements descending sort

    return kth_distances, ranks

def load_data():
    """
    Loading feature data and label data
    """
    try:
        threeblob_X = np.load(threeblob_X_file)
        threeblob_y = np.load(threeblob_label_file)
        pendigits_X = np.load(pendigits_X_file)
        pendigits_y = np.load(pendigits_y_file)
        speech_X = np.load(speech_X_file)
        speech_y = np.load(speech_y_file)
        campaign_file = np.load(file0)
        campaign_X = campaign_file['X']
        campaign_y = campaign_file['y']
        magic_file = np.load(file1, allow_pickle=True)
        # print(magic_file.files)
        magic_X = magic_file['X']
        magic_y = magic_file['y']
        bd_file = np.load(file2, allow_pickle=True)
        # print(magic_file.files)
        bd_X = bd_file['X']
        bd_y = bd_file['y']
        print("✅ Load succesfully!")
        print(f"shape of threeblob feature data X: {threeblob_X.shape}")
        print(f"shape of threeblob label data y: {threeblob_y.shape}")
        print(f"threeblob label distribution: normal points (0) = {np.sum(threeblob_y == 0)}, anomalous points (1) = {np.sum(threeblob_y == 1)}")
        print(f"shape of pendigits feature data X: {pendigits_X.shape}")
        print(f"shape of pendigits label data y: {pendigits_y.shape}")
        print(f"pendigits label distribution: normal points (0) = {np.sum(pendigits_y == 0)}, anomalous points (1) = {np.sum(pendigits_y == 1)}")
        print(f"shape of speech feature data X: {speech_X.shape}")
        print(f"shape of speech label data y: {speech_y.shape}")
        print(f"speech label distribution: noraml points (0) = {np.sum(speech_y == 0)}, anomalous points (1) = {np.sum(speech_y == 1)}")
        print(f"shape of campaign feature data X: {campaign_X.shape}")
        print(f"shape of campaign label data y: {campaign_y.shape}")
        print(f"campaign label distribution: noraml points (0) = {np.sum(campaign_y == 0)}, anomalous points (1) = {np.sum(campaign_y == 1)}")
        print(f"shape of magic_gamma feature data X: {magic_X.shape}")
        print(f"shape of magic_gamma label data y: {magic_y.shape}")
        print(
            f"magic_gamma label distribution: noraml points (0) = {np.sum(magic_y == 0)}, anomalous points (1) = {np.sum(magic_y == 1)}")
        print(f"shape of backdoor feature data X: {bd_X.shape}")
        print(f"shape of backdoor label data y: {bd_y.shape}")
        print(
            f"backdoor label distribution: noraml points (0) = {np.sum(bd_y == 0)}, anomalous points (1) = {np.sum(bd_y == 1)}")

        return threeblob_X, threeblob_y, pendigits_X, pendigits_y, speech_X, speech_y, campaign_X, campaign_y, magic_X, magic_y, bd_X, bd_y
    except FileNotFoundError as e:
        print(f"❌ Error: Cannont find the files {e}")
        print("Please ensure threeblob_X.npy and threeblob_labels.npy under current directory.")
        return None, None, None, None, None, None, None, None, None, None
    except Exception as e:
        print(f"❌ Unknown errors happened while loading: {e}")
        return None, None, None, None, None, None, None, None, None, None

def get_result(X, y, scores, ranks, filepath=None, title=None):
    """
    Calculate ROC AUC and P@N, and optionally plot high-quality decision boundary diagram (Essay style)
    Generate smooth anomalous scores heatmap + contour lines by using interpolation
    """
    from sklearn.metrics import roc_auc_score

    # ---------- 1. Calculate estimation scores ----------
    roc_auc = roc_auc_score(y, scores)

    N = np.sum(y == 1)
    if N == 0 or N == len(y):
        raise ValueError("Dataset needs to contain both normal and anomalous samples to calculate P@N")

    top_N_indices = ranks[:N]
    p_at_n = np.mean(y[top_N_indices] == 1)
    errno = int(N * p_at_n)

    # ---------- 2. High-quality visualization ----------
    if filepath is not None and title is not None:
        title = title + f" ({errno} errors)"
        plt.figure(figsize=(8, 6), dpi=150)
        ax = plt.gca()

        # ---- Create grid for interpolation ----
        margin = 0.3
        x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
        y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

        nx, ny = 200, 200
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                             np.linspace(y_min, y_max, ny))

        # ---- Interpolate anomalous scores to grid ----
        grid_scores = griddata(
            points=X,
            values=scores,
            xi=(xx, yy),
            method='cubic',  # optional 'linear', 'nearest'
            fill_value=np.nan
        )

        # ---- Plot heatmap（anomalous scores） ----
        # Use soft colormap to emphasize anomalous region
        cmap = plt.cm.YlOrRd  # Yellow -> Orange -> Red
        contourf = ax.contourf(xx, yy, grid_scores, levels=20, cmap=cmap, alpha=0.6, zorder=1)

        # ---- Plot contour lines----
        contour = ax.contour(xx, yy, grid_scores, levels=10, colors='orange', alpha=0.8, linewidths=1.2, zorder=2)

        # ---- Plot data points ----
        # Normal points：white with black edges
        inliers = X[y == 0]
        ax.scatter(inliers[:, 0], inliers[:, 1],
                   c='white', edgecolors='black', s=50, alpha=0.9, zorder=3, label='Inliers')

        # Anomalous points：black
        outliers = X[y == 1]
        ax.scatter(outliers[:, 0], outliers[:, 1],
                   c='black', s=50, alpha=0.9, zorder=3, label='Outliers')

        # ---- Beatify style ----
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('Feature 1', fontsize=12)
        ax.set_ylabel('Feature 2', fontsize=12)

        # Remove top/right boundary
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

        # Axis scale
        ax.tick_params(axis='both', which='major', labelsize=10, width=0.5, length=3)

        # Legenf
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right', fontsize=10, frameon=True, fancybox=False, edgecolor='black', framealpha=0.9)

        # Adjust the layout
        plt.tight_layout()

        # Save as high-resolution PNG or PDF(Essay is recommanded as PDF)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        print(f"✅ Paper-level images saved to:: {filepath}")

    return roc_auc, p_at_n


if __name__ == "__main__":
    # tb is threeblob data，pd is pendigits data，sp is speech data
    load_data()

    tb_X, tb_y, pd_X, pd_y, sp_X, sp_y, ca_X, ca_y, ma_X, ma_y, bd_X, bd_y = load_data()
    
    print("\n" + "=" * 60)
    print("Running kNN outlier detection algorithm")
    print("=" * 60)
    tb_scores, tb_ranks = knn_outlier_detection(tb_X, int(math.sqrt(tb_X.shape[0])))
    tb_roc_auc, tb_p_at_n = get_result(tb_X, tb_y, tb_scores, tb_ranks, "./output/tb_knn", "knn outcome on threeblob data")
    pd_scores, pd_ranks = knn_outlier_detection(pd_X, int(math.sqrt(pd_X.shape[0])))
    pd_roc_auc, pd_p_at_n = get_result(pd_X, pd_y, pd_scores, pd_ranks)
    sp_scores, sp_ranks = knn_outlier_detection(sp_X, int(math.sqrt(sp_X.shape[0])))
    sp_roc_auc, sp_p_at_n = get_result(sp_X, sp_y, sp_scores, sp_ranks)
    ca_scores, ca_ranks = knn_outlier_detection(ca_X, int(math.sqrt(ca_X.shape[0])))
    ca_roc_auc, ca_p_at_n = get_result(ca_X, ca_y, ca_scores, ca_ranks)
    ma_scores, ma_ranks = knn_outlier_detection(ma_X, int(math.sqrt(ma_X.shape[0])))
    ma_roc_auc, ma_p_at_n = get_result(ma_X, ma_y, ma_scores, ma_ranks)
    bd_scores, bd_ranks = knn_outlier_detection(bd_X, int(math.sqrt(bd_X.shape[0])))
    bd_roc_auc, bd_p_at_n = get_result(bd_X, bd_y, bd_scores, bd_ranks)

    # print("\n" + "=" * 60)
    # print("Running ADOD outlier detection algorithm")
    # print("=" * 60)
    # tb_adod_scores, tb_adod_ranks, _, _ = adod_outlier_detection(tb_X, None, 0.999, False, 3)
    # tb_adod_roc_auc, tb_adod_p_at_n = get_result(tb_X, tb_y, tb_adod_scores, tb_adod_ranks, "./code/output/tb_adod", "ADOD outcome on threeblob data")
    # pd_adod_scores, pd_adod_ranks, _, _ = adod_outlier_detection(pd_X, None, 0.999, False, 3)
    # pd_adod_roc_auc, pd_adod_p_at_n = get_result(pd_X, pd_y, pd_adod_scores, pd_adod_ranks)
    # sp_adod_scores, sp_adod_ranks, _, _ = adod_outlier_detection(sp_X, None, 0.999, False, 3)
    # sp_adod_roc_auc, sp_adod_p_at_n = get_result(sp_X, sp_y, sp_adod_scores, sp_adod_ranks)

    # Results statistic form is shown as below
    models = {}
    knn_results = {}
    knn_results["threeblob"] = {"roc_auc": tb_roc_auc, "p_at_n": tb_p_at_n}
    knn_results["speech"] = {"roc_auc": sp_roc_auc, "p_at_n": sp_p_at_n}
    knn_results["pendigits"] = {"roc_auc": pd_roc_auc, "p_at_n": pd_p_at_n}
    knn_results['campaign'] = {"roc_auc": ca_roc_auc, "p_at_n": ca_p_at_n}
    knn_results['magic-gamma'] = {"roc_auc": ma_roc_auc, "p_at_n": ma_p_at_n}
    knn_results['backdoor'] = {"roc_auc": bd_roc_auc, "p_at_n": bd_p_at_n}
    models['knn'] = knn_results
    
    # adod_results = {}
    # adod_results["threeblob"] = {"roc_auc": tb_adod_roc_auc, "p_at_n": tb_adod_p_at_n}
    # adod_results["speech"] = {"roc_auc": sp_adod_roc_auc, "p_at_n": sp_adod_p_at_n}
    # adod_results["pendigits"] = {"roc_auc": pd_adod_roc_auc, "p_at_n": pd_adod_p_at_n}
    # models['adod'] = adod_results

    # Denote dataset name and acorresponding data
    datasets = {
        'threeblob': (tb_X, tb_y),
        'pendigits': (pd_X, pd_y),
        'speech': (sp_X, sp_y),
        'campaign': (ca_X, ca_y),
        'magic-gamma': (ma_X, ma_y),
        'backdoor': (bd_X, bd_y)
    }
    # === Build ROC-AUC form ===
    roc_df = pd.DataFrame({
        model_name: [results[ds]['roc_auc'] for ds in datasets.keys()]
        for model_name, results in models.items()
    }, index=list(datasets.keys())).T  # Transposition：row as model，column as dataset
    roc_df.index.name = "Model"
    roc_df.columns.name = "Dataset"

    # === Build P@N form ===
    p_at_n_df = pd.DataFrame({
        model_name: [results[ds]['p_at_n'] for ds in datasets.keys()]
        for model_name, results in models.items()
    }, index=list(datasets.keys())).T
    p_at_n_df.index.name = "Model"
    p_at_n_df.columns.name = "Dataset"

    # === Print form ===
    print("\n" + "=" * 50)
    print("📊 ROC-AUC Form")
    print("=" * 50)
    print(roc_df.round(3))

    print("\n" + "=" * 50)
    print("📊 Precision @ N Form")
    print("=" * 50)
    print(p_at_n_df.round(3))

    roc_df.round(3).to_csv("./output/roc_auc.csv")
    p_at_n_df.round(3).to_csv("./output/p_at_n.csv")
