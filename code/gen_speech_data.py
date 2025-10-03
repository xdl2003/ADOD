import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


def process_speech_arff(arff_file='', output_dir='.'):
    """
    处理 speech.arff 文件，提取特征和标签，标准化，保存为 .npy 文件。

    Parameters:
    -----------
    arff_file : str
        ARFF 文件路径
    output_dir : str
        输出目录

    Returns:
    --------
    X_scaled : np.ndarray, shape (n_samples, 16)
    y : np.ndarray, shape (n_samples,)
    """
    # 检查文件是否存在
    if not os.path.exists(arff_file):
        raise FileNotFoundError(f"未找到文件: {arff_file}")

    print("📥 正在加载 ARFF 数据...")
    data, meta = arff.loadarff(arff_file)

    # 转为 Pandas DataFrame
    df = pd.DataFrame(data)

    # 查看基本信息
    print(f"原始数据形状: {df.shape}")

    # 删除缺失值（虽然 Quality 显示为 0，仍检查）
    if df.isnull().any().any():
        print("⚠️ 发现缺失值，正在删除...")
        df = df.dropna()
    else:
        print("✅ 无缺失值")

    # 提取特征（前16个 input 字段）
    feature_columns = [f'input{i}' for i in range(1, 17)]
    X = df[feature_columns].values.astype(np.float32)

    # 提取标签：binaryClass 是 bytes 类型，需解码
    y_bytes = df['binaryClass'].values
    y = np.array([1 if label == b'P' else 0 for label in y_bytes])  # N=异常(1), P=正常(0)

    # 标准化
    print("🔄 正在标准化特征...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, '../data/pendigits_X.npy'), X_scaled)
    np.save(os.path.join(output_dir, '../data/pendigits_y.npy'), y)

    # ✅ 输出统计信息
    n, d = X_scaled.shape
    n_normal = np.sum(y == 0)
    n_anomaly = np.sum(y == 1)
    anomaly_ratio = n_anomaly / n

    print("\n" + "=" * 50)
    print("📊 Pendigits 数据集（ARFF 二分类版本）")
    print("=" * 50)
    print(f"🔹 特征数 (d):          {d}")
    print(f"🔹 总数据量 (n):           {n}")
    print(f"🔹 正常样本大小 (P):      {n_normal}")
    print(f"🔹 异常样本大小 (N):      {n_anomaly}")
    print(f"🔹 异常比例:              {anomaly_ratio:.3f} ({anomaly_ratio * 100:.1f}%)")
    print(f"✅ 已保存: {os.path.join(output_dir, 'pendigits_X.npy')}")
    print(f"✅ 已保存: {os.path.join(output_dir, 'pendigits_y.npy')}")
    print("=" * 50)

    return X_scaled, y



if __name__ == "__main__":
    X, y = process_pendigits_arff('../data/pendigits.arff', output_dir='.')