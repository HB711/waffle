import math
import torch
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# Device / dtype
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
print("device:", device)

TWO_PI = 2.0 * math.pi
PI = math.pi

def wrap_pi(x: torch.Tensor) -> torch.Tensor:
    """Map to (-pi, pi]. Works elementwise for any shape."""
    return torch.remainder(x + PI, TWO_PI) - PI

def circ_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Circular distance on angles represented in (-pi,pi]."""
    d = torch.abs(a - b)
    return torch.minimum(d, TWO_PI - d)

# ============================================================
# Physics: 6D potential
# ============================================================
EJ = 80.0

def potential6D(t1, t2, t3, p1, p2, p3, EJ=EJ):
    alpha = 2 * torch.pi / 3
    term_phi1 = -2 * torch.cos(t1 - p1) - 2 * torch.cos(t2 - p1) - 2 * torch.cos(t3 - p1)
    term_phi2 = -2 * torch.cos(t1 - p2) - 2 * torch.cos(t2 - p2 + alpha) - 2 * torch.cos(t3 - p2 - alpha)
    term_phi3 = -2 * torch.cos(t1 - p3) - 2 * torch.cos(t2 - p3 - alpha) - 2 * torch.cos(t3 - p3 + alpha)
    return EJ * (term_phi1 + term_phi2 + term_phi3) / math.sqrt(3.0)

# ============================================================
# Your orthogonal transform U
#   Phi_tilde = (x0, x2, x3, x4, x5, x6)
#   Phi = Phi_tilde @ U  -> (t1,t2,t3,p1,p2,p3)
# ============================================================
sqrt6 = math.sqrt(6.0)
sqrt2 = math.sqrt(2.0)
U = torch.tensor([
    [ 1/sqrt6,  1/sqrt6,  1/sqrt6,  1/sqrt6,  1/sqrt6,  1/sqrt6],
    [ 1/sqrt2, -1/sqrt2,  0.0,      0.0,      0.0,      0.0     ],
    [ 1/sqrt6,  1/sqrt6, -2/sqrt6,  0.0,      0.0,      0.0     ],
    [ 0.0,      0.0,      0.0,      1/sqrt2, -1/sqrt2,  0.0     ],
    [ 0.0,      0.0,      0.0,      1/sqrt6,  1/sqrt6, -2/sqrt6 ],
    [-1/sqrt6, -1/sqrt6, -1/sqrt6,  1/sqrt6,  1/sqrt6,  1/sqrt6 ],
], device=device, dtype=dtype)

def x5_to_phi6(x5: torch.Tensor, x0: float = 0.0) -> torch.Tensor:
    """
    x5: (5,) or (N,5) representing (x2..x6)
    returns Phi: (6,) or (N,6) representing (t1,t2,t3,p1,p2,p3), wrapped to (-pi,pi]
    """
    if x5.ndim == 1:
        x5 = x5[None, :]
    N = x5.shape[0]
    x0t = torch.full((N, 1), float(x0), device=x5.device, dtype=x5.dtype)
    Phi_tilde = torch.cat([x0t, x5], dim=1)  # (N,6)
    Phi = Phi_tilde @ U                      # (N,6)
    Phi = wrap_pi(Phi)                       # represent on (-pi,pi]
    return Phi.squeeze(0) if N == 1 else Phi

def potential5D(x5: torch.Tensor, x0: float = 0.0) -> torch.Tensor:
    """
    x5: (N,5) returns V: (N,)
    """
    Phi = x5_to_phi6(x5, x0=x0)  # (N,6)
    t1, t2, t3, p1, p2, p3 = Phi.T
    return potential6D(t1, t2, t3, p1, p2, p3, EJ=EJ)

# ============================================================
# Clustering unique minima in Phi space (6 angles)
# ============================================================
def greedy_cluster_phi(Phi: torch.Tensor, tol_angle: float = 2e-3):
    """
    Phi: (K,6) in (-pi,pi]
    returns representative indices for unique clusters
    """
    K = Phi.shape[0]
    if K == 0:
        return []
    used = torch.zeros(K, device=Phi.device, dtype=torch.bool)
    reps = []
    thr = (tol_angle * tol_angle) * 6.0
    for i in range(K):
        if used[i]:
            continue
        reps.append(i)
        d = circ_dist(Phi, Phi[i:i+1])  # (K,6)
        d2 = (d * d).sum(dim=1)
        used |= (d2 <= thr)
    return reps

@torch.no_grad()
def collect_all_global_minima(x_all, V_all, x0=0.0, tol_energy=1e-8, tol_angle=2e-3):
    """
    x_all: (n,5) final optimized points
    V_all: (n,) energies
    returns:
      x_u: (m,5)
      Phi_u: (m,6)
      V_u: (m,)
      vmin: float
    """
    vmin = V_all.min()
    mask = (V_all <= vmin + tol_energy)
    idx = torch.nonzero(mask, as_tuple=False).flatten()

    x_sel = x_all[idx]                    # (K,5)
    Phi_sel = x5_to_phi6(x_sel, x0=x0)    # (K,6)
    V_sel = V_all[idx]                    # (K,)

    reps = greedy_cluster_phi(Phi_sel, tol_angle=tol_angle)
    x_u = x_sel[reps].detach().cpu()
    Phi_u = Phi_sel[reps].detach().cpu()
    V_u = V_sel[reps].detach().cpu()

    order = torch.argsort(V_u)
    return x_u[order], Phi_u[order], V_u[order], float(vmin.detach().cpu())

def print_one(name, x5, V, x0=0.0):
    Phi = x5_to_phi6(x5, x0=x0)
    x2,x3,x4,x5v,x6 = [float(v) for v in x5.detach().cpu()]
    t1,t2,t3,p1,p2,p3 = [float(v) for v in Phi.detach().cpu()]
    print(f"\n==== {name} ====")
    print(f"V = {float(V.detach().cpu()):.12f}")
    print(f"(x2,x3,x4,x5,x6) = ({x2:.6f}, {x3:.6f}, {x4:.6f}, {x5v:.6f}, {x6:.6f})")
    print(f"(t1,t2,t3,p1,p2,p3) = ({t1:.6f}, {t2:.6f}, {t3:.6f}, {p1:.6f}, {p2:.6f}, {p3:.6f})")

# ============================================================
# Multi-start minimization in 5D
# ============================================================
def find_minima_5D(
    x0=0.0,
    n_starts=50000,
    steps=2500,
    lr=0.05,
    print_every=200,
    seed=0,
    tol_energy=1e-8,
    tol_angle=2e-3,
):
    torch.manual_seed(seed)

    # IMPORTANT: x2..x6 are NOT independent angles; do NOT wrap them.
    # Initialize broadly.
    x = (torch.randn(n_starts, 5, device=device, dtype=dtype) * 4*PI).requires_grad_(True)

    opt = torch.optim.Adam([x], lr=lr)
    best_curve = []

    for it in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        V = potential5D(x, x0=x0)   # (n_starts,)
        loss = V.mean()
        loss.backward()
        opt.step()

        with torch.no_grad():
            best_curve.append(float(V.min().detach().cpu()))

        if (it == 1) or (it % print_every == 0) or (it == steps):
            with torch.no_grad():
                print(f"[it={it:4d}] V_min={float(V.min()):.12f}   V_mean={float(V.mean()):.12f}")

    with torch.no_grad():
        V = potential5D(x, x0=x0)
        idx_best = torch.argmin(V)
        x_best = x[idx_best].detach()
        V_best = V[idx_best].detach()

        x_u, Phi_u, V_u, vmin = collect_all_global_minima(
            x_all=x.detach(),
            V_all=V.detach(),
            x0=x0,
            tol_energy=tol_energy,
            tol_angle=tol_angle,
        )

    return x_best, V_best, best_curve, x_u, Phi_u, V_u, vmin

# ============================================================
# Main
# ============================================================
def main():
    # Gauge-fix center-of-mass coordinate x0 (can be any number; energy should not depend on it)
    x0 = 0

    x_best, V_best, curve, x_u, Phi_u, V_u, vmin = find_minima_5D(
        x0=x0,
        n_starts=50000,
        steps=2500,
        lr=0.05,
        print_every=250,
        seed=0,
        tol_energy=1e-8,
        tol_angle=2e-3,
    )

    print_one("Global min (one representative)", x_best, V_best, x0=x0)

    print(f"\nFound {len(V_u)} unique global minima (clustered in Phi-space).")
    print(f"V_global_min = {vmin:.12f}")

    for k in range(len(V_u)):
        x2,x3,x4,x5v,x6 = [float(v) for v in x_u[k]]
        t1,t2,t3,p1,p2,p3 = [float(v) for v in Phi_u[k]]
        print(f"\n[{k:02d}] V={float(V_u[k]):.12f}")
        print(f"  x5=(x2..x6)=({x2:.6f}, {x3:.6f}, {x4:.6f}, {x5v:.6f}, {x6:.6f})")
        print(f"  Phi=(t1..p3)=({t1:.6f}, {t2:.6f}, {t3:.6f}, {p1:.6f}, {p2:.6f}, {p3:.6f})")

    # ============================================================
    # Build table of all unique minima
    # ============================================================

    rows = []
    for k in range(len(V_u)):
        row = {
            "index": k,
            "V": float(V_u[k]),
            "x2": float(x_u[k, 0]),
            "x3": float(x_u[k, 1]),
            "x4": float(x_u[k, 2]),
            "x5": float(x_u[k, 3]),
            "x6": float(x_u[k, 4]),
            "t1": float(Phi_u[k, 0]),
            "t2": float(Phi_u[k, 1]),
            "t3": float(Phi_u[k, 2]),
            "p1": float(Phi_u[k, 3]),
            "p2": float(Phi_u[k, 4]),
            "p3": float(Phi_u[k, 5]),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    print("\n================ ALL UNIQUE GLOBAL MINIMA =================")
    print(df.to_string(index=False))

    # ============================================================
    # Convert angles to k*pi/12 form
    # ============================================================

    def to_pi_over_12_string(x):
        """
        Convert float angle to string k*pi/12
        """
        k = int(round(float(x) / (math.pi / 12)))
        if k == 0:
            return "0"
        if k == 1:
            return r"\frac{\pi}{12}"
        if k == -1:
            return r"-\frac{\pi}{12}"
        return rf"{k}\frac{{\pi}}{{12}}"

    rows = []

    for i in range(len(Phi_u)):
        row = {"index": i}
        for j, name in enumerate(["t1", "t2", "t3", "p1", "p2", "p3"]):
            row[name] = to_pi_over_12_string(Phi_u[i, j])
        rows.append(row)

    df_pi = pd.DataFrame(rows)

    print("\n===== ALL 36 MINIMA (π form) =====")
    print(df_pi)

    excel_filename = "all_36_minima_pi.xlsx"
    df_pi.to_excel(excel_filename, index=False)
    print(f"\nExcel file saved as: {excel_filename}")
    latex_code = df_pi.to_latex(index=False, escape=False)

    with open("all_36_minima_pi.tex", "w") as f:
        f.write(latex_code)

    print("\nLaTeX table saved as: all_36_minima_pi.tex")
    print("\n===== LaTeX code preview =====")
    print(latex_code)
    # Plot best-curve
    # plt.figure()
    # plt.plot(curve)
    # plt.xlabel("iteration")
    # plt.ylabel("best V_min among starts")
    # plt.grid(True)
    # plt.title("Convergence of best-found minimum (5D gauge-fixed)")
    # plt.show()

if __name__ == "__main__":
    main()