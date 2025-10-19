from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# ========== 1. Generate normal points (from three Gaussian distributions) ==========
# Set the centers (means) of the three Gaussian distributions
# Here, we place them at the vertices of an equilateral triangle with a side length of 5
centers = [
    [0.0, 0.0],      # Center of the first cluster
    [5.0, 0.0],      # Center of the second cluster
    [2.5, 4.33]      # Center of the third cluster (5 * sin(60°) ≈ 4.33)
]
# Size of each cluser (425 points distributed uniformly)
cluster_sizes = [140, 140, 145]  # sum as 425

# Generate points of three Gaussian distributions
inliers, _ = make_blobs(
    n_samples=cluster_sizes,
    centers=centers,
    cluster_std=[0.6, 1.2, 0.3],  # σ=[0.6, 1.2, 0.3] in essay
    random_state=42  # To make the results reproducible
)

# ========== 2. Generate Outliers (Uniform Distribution) ==========
# Determine the Uniform Distribution Range
# To ensure that outliers are outliers, the range should be larger than that of normal points.
x_min, x_max = inliers[:, 0].min() - 5, inliers[:, 0].max() + 5
y_min, y_max = inliers[:, 1].min() - 5, inliers[:, 1].max() + 5

# Generate 75 points evenly distributed in the range (x_min, x_max) x (y_min, y_max)
outliers = np.random.uniform(
    low=[x_min, y_min],
    high=[x_max, y_max],
    size=(75, 2)  # 75 points，2-d for each
)

# ========== 3. Merge datasets and label them ==========
# Merge normal and abnormal points
X = np.vstack([inliers, outliers])

# Create labels: 0 represents a normal point (inlier), 1 represents an abnormal point (outlier)
labels = np.hstack([np.zeros(len(inliers)), np.ones(len(outliers))])

# ========== 4. Visualization (Optional) ==========
plt.figure(figsize=(8, 6))
# Plot normal points (white)
plt.scatter(inliers[:, 0], inliers[:, 1], c='white', edgecolors='k', s=30, label='Inliers (Normal)')
# Plot anomalous points (black)
plt.scatter(outliers[:, 0], outliers[:, 1], c='black', s=30, label='Outliers')
plt.title('ThreeBlob Outlier Dataset (Synthetic)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.axis('equal')  # Keep the axis proportions consistent
plt.grid(True, alpha=0.3)


# ========== 5. Save (Optional) ==========
# Data can be saved as a .npy file
# Define the path
filepath = Path('../data/threeblob_X.npy')
filepath.parent.mkdir(exist_ok=True, parents=True)  # Automatically create all parent directories
np.save(filepath, X)
filepath = Path('../data/threeblob_label.npy')
np.save(filepath, labels)
filepath = Path('../data/threeblob.png')
plt.savefig(filepath)

plt.show()

print(f"Number of samples: {len(X)}")
print(f"Number of normal points: {np.sum(labels == 0)}")
print(f"Number of anomalous points: {np.sum(labels == 1)}")