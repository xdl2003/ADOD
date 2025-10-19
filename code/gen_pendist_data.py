import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


def process_pendigits_arff(arff_file='', output_dir='.'):
    """
    Process pendigits.arff file, take features and labels, normalize, save as .npy file.

    Parameters:
    -----------
    arff_file : str
        ARFF file path
    output_dir : str
        output directory

    Returns:
    --------
    X_scaled : np.ndarray, shape (n_samples, 16)
    y : np.ndarray, shape (n_samples,)
    """
    # Check whether files exist
    if not os.path.exists(arff_file):
        raise FileNotFoundError(f"File not found: {arff_file}")

    print("📥 Loading ARFF data...")
    data, meta = arff.loadarff(arff_file)

    # Convert to Pandas DataFrame
    df = pd.DataFrame(data)

    # Check basic info
    print(f"Primitive data shape: {df.shape}")

    # Delete missing values(Check though quality = 0)
    if df.isnull().any().any():
        print("⚠️ Find missing values, deleting...")
        df = df.dropna()
    else:
        print("✅ No missing value.")

    # Take features（first 16 input chars）
    feature_columns = [f'input{i}' for i in range(1, 17)]
    X = df[feature_columns].values.astype(np.float32)

    # Take labels：binaryClass is bytes type，need to be decoded
    y_bytes = df['binaryClass'].values
    y = np.array([1 if label == b'P' else 0 for label in y_bytes])  # N=anomalous(1), P=normal(0)

    # Noramlize
    print("🔄 Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, '../data/pendigits_X.npy'), X_scaled)
    np.save(os.path.join(output_dir, '../data/pendigits_y.npy'), y)

    # ✅ Output statistic info
    n, d = X_scaled.shape
    n_normal = np.sum(y == 0)
    n_anomaly = np.sum(y == 1)
    anomaly_ratio = n_anomaly / n

    print("\n" + "=" * 50)
    print("📊 Pendigits dataset(ARFF binary classification)")
    print("=" * 50)
    print(f"🔹 Number of features (d):          {d}")
    print(f"🔹 Number of data (n):           {n}")
    print(f"🔹 Size of normal samples (P):      {n_normal}")
    print(f"🔹 Size of anomalous samples (N):      {n_anomaly}")
    print(f"🔹 Anomalous propotion:              {anomaly_ratio:.3f} ({anomaly_ratio * 100:.1f}%)")
    print(f"✅ Saved: {os.path.join(output_dir, 'pendigits_X.npy')}")
    print(f"✅ Saved: {os.path.join(output_dir, 'pendigits_y.npy')}")
    print("=" * 50)

    return X_scaled, y



if __name__ == "__main__":
    X, y = process_pendigits_arff('../data/pendigits.arff', output_dir='.')