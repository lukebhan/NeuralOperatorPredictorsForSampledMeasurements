"""Compare fixed-vs-multistep predictor datasets (visual diagnostics)."""

import numpy as np
import matplotlib.pyplot as plt

# Load datasets
data_fixed = np.load("dataset/multistep_predictor_dataset_small_fixed.npz")
data_multistep = np.load("dataset/multistep_predictor_dataset_small.npz")

# Inspect keys
print("fixed keys:", data_fixed.files)
print("Multistep keys:", data_multistep.files)

# Extract arrays (adjust names if needed)
X1, Y1 = data_fixed["X"], data_fixed["Y"]
X2, Y2 = data_multistep["X"], data_multistep["Y"]

print("Shapes:", X1.shape, Y1.shape, X2.shape, Y2.shape)

# Compute differences
min_len = min(len(X1), len(X2))
dx = np.linalg.norm(X1[:min_len] - X2[:min_len], axis=1)
dy = np.linalg.norm(Y1[:min_len] - Y2[:min_len], axis=1)

# Plot differences
plt.figure()
plt.plot(dx, label="X difference norm")
plt.plot(dy, label="Y difference norm")
plt.legend()
plt.title("Dataset difference")
plt.xlabel("sample")
plt.ylabel("L2 difference")
plt.show()

# Scatter comparison of first dimension
plt.figure()
plt.scatter(X1[:min_len,0], X2[:min_len,0], s=2)
plt.xlabel("X fixed")
plt.ylabel("X multistep")
plt.title("Input comparison")
plt.show()