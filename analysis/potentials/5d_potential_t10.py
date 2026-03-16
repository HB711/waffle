import math
import torch
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
    # always interpret angles on (-pi,pi]
    t1, t2, t3, p1, p2, p3 = map(wrap_pi, (t1, t2, t3, p1, p2, p3))

    alpha = 2 * torch.pi / 3
    term_phi1 = -2 * torch.cos(t1 - p1) - 2 * torch.cos(t2 - p1) - 2 * torch.cos(t3 - p1)
    term_phi2 = -2 * torch.cos(t1 - p2) - 2 * torch.cos(t2 - p2 + alpha) - 2 * torch.cos(t3 - p2 - alpha)
    term_phi3 = -2 * torch.cos(t1 - p3) - 2 * torch.cos(t2 - p3 - alpha) - 2 * torch.cos(t3 - p3 + alpha)
    return EJ * (term_phi1 + term_phi2 + term_phi3) / math.sqrt(3.0)

def potential6D_from_var(phi_var: torch.Tensor, theta1_fixed: float = 0.0):
    """
    phi_var: (N,5) where columns are (t2,t3,p1,p2,p3) in unconstrained real values.
    Enforces:
      t1 = wrap_pi(theta1_fixed) but also theta1 mod 2pi = 0 => in [-pi,pi] it's exactly 0.
    Returns V: (N,)
    """
    if phi_var.ndim != 2 or phi_var.shape[1] != 5:
        raise ValueError("phi_var must have shape (N,5) = (t2,t3,p1,p2,p3)")

    # represent all angles within (-pi,pi]
    phi_var = wrap_pi(phi_var)

    t1 = torch.zeros((phi_var.shape[0],), device=phi_var.device, dtype=phi_var.dtype) + float(theta1_fixed)
    t1 = wrap_pi(t1)  # this will be 0 if theta1_fixed is 0

    t2 = phi_var[:, 0]
    t3 = phi_var[:, 1]
    p1 = phi_var[:, 2]
    p2 = phi_var[:, 3]
    p3 = phi_var[:, 4]

    return potential6D(t1, t2, t3, p1, p2, p3, EJ=EJ)

def build_Phi6(phi_var_5: torch.Tensor, theta1_fixed: float = 0.0) -> torch.Tensor:
    """
    phi_var_5: (N,5) or (5,)
    returns Phi6: (N,6) in (-pi,pi]
    """
    if phi_var_5.ndim == 1:
        phi_var_5 = phi_var_5[None, :]
    phi_var_5 = wrap_pi(phi_var_5)

    N = phi_var_5.shape[0]
    t1 = torch.zeros((N, 1), device=phi_var_5.device, dtype=phi_var_5.dtype) + float(theta1_fixed)
    t1 = wrap_pi(t1)

    Phi6 = torch.cat([t1, phi_var_5[:, 0:1], phi_var_5[:, 1:2], phi_var_5[:, 2:3], phi_var_5[:, 3:4], phi_var_5[:, 4:5]], dim=1)
    return wrap_pi(Phi6)

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
def collect_all_global_minima_phi(Phi_var_all, V_all, theta1_fixed=0.0, tol_energy=1e-8, tol_angle=2e-3):
    """
    Phi_var_all: (n,5) final optimized vars (t2,t3,p1,p2,p3)
    V_all: (n,) energies
    returns:
      Phi6_u: (m,6)
      V_u: (m,)
      vmin: float
    """
    vmin = V_all.min()
    mask = (V_all <= vmin + tol_energy)
    idx = torch.nonzero(mask, as_tuple=False).flatten()

    Phi_var_sel = Phi_var_all[idx]                 # (K,5)
    Phi6_sel = build_Phi6(Phi_var_sel, theta1_fixed=theta1_fixed)  # (K,6)
    V_sel = V_all[idx]                              # (K,)

    reps = greedy_cluster_phi(Phi6_sel, tol_angle=tol_angle)
    Phi6_u = Phi6_sel[reps].detach().cpu()
    V_u = V_sel[reps].detach().cpu()

    order = torch.argsort(V_u)
    return Phi6_u[order], V_u[order], float(vmin.detach().cpu())

def print_one_phi(name, Phi6, V):
    t1,t2,t3,p1,p2,p3 = [float(v) for v in Phi6.detach().cpu()]
    print(f"\n==== {name} ====")
    print(f"V = {float(V.detach().cpu()):.12f}")
    print(f"(t1,t2,t3,p1,p2,p3) = ({t1:.6f}, {t2:.6f}, {t3:.6f}, {p1:.6f}, {p2:.6f}, {p3:.6f})")

# ============================================================
# Multi-start minimization directly in 6D with t1 fixed = 0
# ============================================================
def find_minima_6D_theta1_fixed(
    theta1_fixed=0.0,
    n_starts=50000,
    steps=2500,
    lr=0.05,
    print_every=250,
    seed=0,
    tol_energy=1e-8,
    tol_angle=2e-3,
):
    torch.manual_seed(seed)

    # optimize only (t2,t3,p1,p2,p3) as unconstrained reals, but we always wrap to (-pi,pi]
    phi_var = (torch.randn(n_starts, 5, device=device, dtype=dtype) * 4*PI).requires_grad_(True)
    opt = torch.optim.Adam([phi_var], lr=lr)

    best_curve = []

    for it in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)

        V = potential6D_from_var(phi_var, theta1_fixed=theta1_fixed)  # (n_starts,)
        loss = V.mean()
        loss.backward()
        opt.step()

        # hard project angles back to (-pi,pi] after each step so "range [-pi,pi]" is respected
        with torch.no_grad():
            phi_var[:] = wrap_pi(phi_var)
            best_curve.append(float(V.min().detach().cpu()))

        if (it == 1) or (it % print_every == 0) or (it == steps):
            with torch.no_grad():
                print(f"[it={it:4d}] V_min={float(V.min()):.12f}   V_mean={float(V.mean()):.12f}")

    with torch.no_grad():
        V = potential6D_from_var(phi_var, theta1_fixed=theta1_fixed)
        idx_best = torch.argmin(V)
        phi_best_var = phi_var[idx_best].detach()
        Phi6_best = build_Phi6(phi_best_var, theta1_fixed=theta1_fixed).squeeze(0)
        V_best = V[idx_best].detach()

        Phi6_u, V_u, vmin = collect_all_global_minima_phi(
            Phi_var_all=phi_var.detach(),
            V_all=V.detach(),
            theta1_fixed=theta1_fixed,
            tol_energy=tol_energy,
            tol_angle=tol_angle,
        )

    return Phi6_best, V_best, best_curve, Phi6_u, V_u, vmin

# ============================================================
# Main
# ============================================================
def main():
    # Your constraint: theta_1 mod 2pi = 0, and angles in [-pi,pi]
    # => theta_1 must be exactly 0 in this representation.
    theta1_fixed = 0.0

    Phi6_best, V_best, curve, Phi6_u, V_u, vmin = find_minima_6D_theta1_fixed(
        theta1_fixed=theta1_fixed,
        n_starts=50000,
        steps=2500,
        lr=0.05,
        print_every=250,
        seed=0,
        tol_energy=1e-8,
        tol_angle=2e-3,
    )

    print_one_phi("Global min (one representative)", Phi6_best, V_best)

    print(f"\nFound {len(V_u)} unique global minima (clustered in Phi-space).")
    print(f"V_global_min = {vmin:.12f}")

    for k in range(len(V_u)):
        t1,t2,t3,p1,p2,p3 = [float(v) for v in Phi6_u[k]]
        print(f"\n[{k:02d}] V={float(V_u[k]):.12f}")
        print(f"  Phi=(t1..p3)=({t1:.6f}, {t2:.6f}, {t3:.6f}, {p1:.6f}, {p2:.6f}, {p3:.6f})")

    # ============================================================
    # Build table of all unique minima
    # ============================================================
    rows = []
    for k in range(len(V_u)):
        row = {
            "index": k,
            "V": float(V_u[k]),
            "t1": float(Phi6_u[k, 0]),
            "t2": float(Phi6_u[k, 1]),
            "t3": float(Phi6_u[k, 2]),
            "p1": float(Phi6_u[k, 3]),
            "p2": float(Phi6_u[k, 4]),
            "p3": float(Phi6_u[k, 5]),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n================ ALL UNIQUE GLOBAL MINIMA (6D) =================")
    print(df.to_string(index=False))

    # ============================================================
    # Convert angles to k*pi/12 form
    # ============================================================
    def to_pi_over_12_string(x):
        k = int(round(float(x) / (math.pi / 12)))
        if k == 0:
            return "0"
        if k == 1:
            return r"\frac{\pi}{12}"
        if k == -1:
            return r"-\frac{\pi}{12}"
        return rf"{k}\frac{{\pi}}{{12}}"

    rows_pi = []
    for i in range(len(Phi6_u)):
        row = {"index": i}
        for j, name in enumerate(["t1", "t2", "t3", "p1", "p2", "p3"]):
            row[name] = to_pi_over_12_string(Phi6_u[i, j])
        rows_pi.append(row)

    df_pi = pd.DataFrame(rows_pi)
    print("\n===== ALL MINIMA (π/12 form) =====")
    print(df_pi)

    excel_filename = "all_minima_theta1_fixed_pi.xlsx"
    df_pi.to_excel(excel_filename, index=False)
    print(f"\nExcel file saved as: {excel_filename}")

    latex_code = df_pi.to_latex(index=False, escape=False)
    with open("all_minima_theta1_fixed_pi.tex", "w") as f:
        f.write(latex_code)

    print("\nLaTeX table saved as: all_minima_theta1_fixed_pi.tex")
    print("\n===== LaTeX code preview =====")
    print(latex_code)

if __name__ == "__main__":
    main()