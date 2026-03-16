import numpy as np
from scipy.optimize import minimize

# ============================
# Parameters
# ============================
EJ = 80.0

# ============================
# Potential energy
# V = -2EJ [ cos(t1-p1)+cos(t1-p2)+cos(t2-p1)-cos(t2-p2) ]
# ============================
def V(x):
    t1, t2, p1, p2 = x
    return -2.0 * EJ * (
        np.cos(t1 - p1)
        + np.cos(t1 - p2)
        + np.cos(t2 - p1)
        - np.cos(t2 - p2)
    )

# ============================
# Bounds (one 2π cell is enough)
# ============================
bounds = [(-np.pi, np.pi)] * 4

# ============================
# Multi-start global search
# ============================
n_restarts = 50
best_val = np.inf
best_x = None

for i in range(n_restarts):
    x0 = np.random.uniform(-np.pi, np.pi, 4)

    res = minimize(
        V,
        x0,
        method="L-BFGS-B",
        bounds=bounds
    )

    if res.fun < best_val:
        best_val = res.fun
        best_x = res.x

print("Minimum value:", best_val)
print("Argmin (t1, t2, p1, p2):", best_x)