import math
import torch
import matplotlib.pyplot as plt

# ============================================================
# Device / dtype
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
print("device:", device)

TWO_PI = 2.0 * math.pi
PI = math.pi

def wrap_centered(x: torch.Tensor) -> torch.Tensor:
    # maps to (-pi, pi]
    return torch.remainder(x + PI, TWO_PI) - PI

def clamp_penalty_interval(x: torch.Tensor, lo=-PI, hi=PI) -> torch.Tensor:
    # quadratic penalty for violating [lo,hi]
    return torch.relu(lo - x) ** 2 + torch.relu(x - hi) ** 2

def circ_dist_centered(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = torch.abs(a - b)
    return torch.minimum(d, TWO_PI - d)

# ----------------------------
# Your 6D classical potential
# ----------------------------
def potential6D(t1, t2, t3, p1, p2, p3, EJ=80.0):
    alpha = 2 * torch.pi / 3
    term_phi1 = -2 * torch.cos(t1 - p1) - 2 * torch.cos(t2 - p1) - 2 * torch.cos(t3 - p1)
    term_phi2 = -2 * torch.cos(t1 - p2) - 2 * torch.cos(t2 - p2 + alpha) - 2 * torch.cos(t3 - p2 - alpha)
    term_phi3 = -2 * torch.cos(t1 - p3) - 2 * torch.cos(t2 - p3 - alpha) - 2 * torch.cos(t3 - p3 + alpha)
    return EJ * (term_phi1 + term_phi2 + term_phi3) / math.sqrt(3.0)

# ---------------------------------------------------------
# Strict constraint: sum = S_target (NO MOD)
# Parameterize p3 = S_target - (t1+t2+t3+p1+p2)
# while enforcing all angles in [-pi,pi] (esp p3)
# ---------------------------------------------------------
def build_strict_constrained_angles(raw5, S_target: float):
    """
    raw5: (t1r,t2r,t3r,p1r,p2r) unconstrained reals
    S_target: desired strict sum of all 6 angles
    Returns:
      t1,t2,t3,p1,p2 wrapped in (-pi,pi]
      p3_strict (unwrapped, satisfies strict sum exactly)
      p3_wrapped (for display / clustering)
      pen (box constraint violation for p3_strict in [-pi,pi])
    """
    t1r, t2r, t3r, p1r, p2r = raw5
    t1 = wrap_centered(t1r)
    t2 = wrap_centered(t2r)
    t3 = wrap_centered(t3r)
    p1 = wrap_centered(p1r)
    p2 = wrap_centered(p2r)

    # strict equality
    S = torch.tensor(S_target, device=t1.device, dtype=t1.dtype)
    p3_strict = S - (t1 + t2 + t3 + p1 + p2)

    # enforce p3_strict in [-pi,pi]
    pen = clamp_penalty_interval(p3_strict, -PI, PI)

    # for printing/comparison only
    p3_wrapped = wrap_centered(p3_strict)

    return t1, t2, t3, p1, p2, p3_strict, p3_wrapped, pen

@torch.no_grad()
def sample_feasible_starts(n, S_target: float, max_tries=80):
    """
    Sample (t1,t2,t3,p1,p2) uniformly in [-pi,pi],
    accept only if p3_strict = S_target - sum(others) lies in [-pi,pi].
    """
    out = []
    need = n
    tries = 0
    S = torch.tensor(S_target, device=device, dtype=dtype)

    while need > 0 and tries < max_tries:
        tries += 1
        m = int(max(need * 2, 1024))

        t1 = (torch.rand(m, device=device, dtype=dtype) * TWO_PI - PI)
        t2 = (torch.rand(m, device=device, dtype=dtype) * TWO_PI - PI)
        t3 = (torch.rand(m, device=device, dtype=dtype) * TWO_PI - PI)
        p1 = (torch.rand(m, device=device, dtype=dtype) * TWO_PI - PI)
        p2 = (torch.rand(m, device=device, dtype=dtype) * TWO_PI - PI)

        p3 = S - (t1 + t2 + t3 + p1 + p2)  # strict
        mask = (p3 >= -PI) & (p3 <= PI)
        idx = torch.nonzero(mask, as_tuple=False).flatten()

        if idx.numel() > 0:
            take = min(need, idx.numel())
            sel = idx[:take]
            out.append(torch.stack([t1[sel], t2[sel], t3[sel], p1[sel], p2[sel]], dim=-1))
            need -= take

    if need > 0:
        raise RuntimeError(
            f"Could not sample enough feasible starts (missing {need}). "
            f"Likely S_target is too large in magnitude, making p3 rarely in [-pi,pi]."
        )

    return torch.cat(out, dim=0)  # (n,5)

def greedy_cluster_minima(angles6: torch.Tensor, tol_angle=2e-3):
    if angles6.numel() == 0:
        return []
    K = angles6.shape[0]
    used = torch.zeros((K,), device=angles6.device, dtype=torch.bool)
    reps = []
    for i in range(K):
        if used[i]:
            continue
        reps.append(i)
        d = circ_dist_centered(angles6, angles6[i:i+1, :])
        d2 = torch.sum(d * d, dim=-1)
        used |= (d2 <= (tol_angle * tol_angle * 6.0))
    return reps

def find_all_global_minima_strict_sum(
    EJ=80.0,
    S_target=math.pi/2,        # <-- center of mass (SUM) target, adjustable
    n_starts=30000,
    steps=2500,
    lr=0.05,
    seed=0,
    print_every=250,
    lam_box=5e3,
    tol_box=1e-6,
    tol_energy=1e-8,
    tol_angle=2e-3,
    track_best_curve=True,
):
    torch.manual_seed(seed)

    init5 = sample_feasible_starts(n_starts, S_target=S_target)
    t1 = init5[:, 0].clone().requires_grad_(True)
    t2 = init5[:, 1].clone().requires_grad_(True)
    t3 = init5[:, 2].clone().requires_grad_(True)
    p1 = init5[:, 3].clone().requires_grad_(True)
    p2 = init5[:, 4].clone().requires_grad_(True)

    opt = torch.optim.Adam([t1, t2, t3, p1, p2], lr=lr)
    best_curve = []

    for it in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)

        wt1, wt2, wt3, wp1, wp2, p3_strict, p3_wrapped, pen = build_strict_constrained_angles(
            (t1, t2, t3, p1, p2), S_target=S_target
        )
        V = potential6D(wt1, wt2, wt3, wp1, wp2, p3_strict, EJ=EJ)

        loss = V.mean() + lam_box * pen.mean()
        loss.backward()
        opt.step()

        if track_best_curve:
            with torch.no_grad():
                feas = (pen <= tol_box) & (p3_strict >= -PI) & (p3_strict <= PI)
                if feas.any():
                    best_curve.append(float(V[feas].min().cpu()))
                else:
                    best_curve.append(float(V.min().cpu()))

        if (it == 1) or (it % print_every == 0) or (it == steps):
            with torch.no_grad():
                feas = (pen <= tol_box) & (p3_strict >= -PI) & (p3_strict <= PI)
                if feas.any():
                    print(f"[it={it:4d}] feasible={int(feas.sum())}  Vmin_feas={float(V[feas].min()):.12f}  Vmean_feas={float(V[feas].mean()):.12f}")
                else:
                    print(f"[it={it:4d}] feasible=0  Vmin_all={float(V.min()):.12f}")

    with torch.no_grad():
        wt1, wt2, wt3, wp1, wp2, p3_strict, p3_wrapped, pen = build_strict_constrained_angles(
            (t1, t2, t3, p1, p2), S_target=S_target
        )
        V = potential6D(wt1, wt2, wt3, wp1, wp2, p3_strict, EJ=EJ)

        feas = (pen <= tol_box) & (p3_strict >= -PI) & (p3_strict <= PI)
        if not feas.any():
            raise RuntimeError("No feasible points found at the end. Try larger lam_box / more steps / smaller lr.")

        V_feas = V[feas]
        vmin_global = V_feas.min()

        pick = feas.clone()
        pick[feas] = (V_feas <= vmin_global + tol_energy)
        idxs = torch.nonzero(pick, as_tuple=False).flatten()

        # angles shown in (-pi,pi]
        angles6_disp = torch.stack([wt1, wt2, wt3, wp1, wp2, p3_wrapped], dim=-1)
        angles_sel = angles6_disp[idxs]
        V_sel = V[idxs]

        reps = greedy_cluster_minima(angles_sel, tol_angle=tol_angle)
        uniq_angles = angles_sel[reps].cpu()
        uniq_V = V_sel[reps].cpu()

        order = torch.argsort(uniq_V)
        uniq_V = uniq_V[order]
        uniq_angles = uniq_angles[order]

    return float(vmin_global.cpu()), uniq_V, uniq_angles, best_curve

def main():
    EJ = 80.0

    # ====== set your center-of-mass (SUM) target here ======
    S_target = 2*math.pi/3
    # 例如：S_target = 0.4
    # 注意：如果 |S_target| 太大，会让可行域变小（因为要 p3 ∈ [-pi,pi]）。

    vmin_global, uniq_V, uniq_angles, curve = find_all_global_minima_strict_sum(
        EJ=EJ,
        S_target=S_target,
        n_starts=300000,
        steps=2500,
        lr=0.05,
        seed=0,
        print_every=250,
        lam_box=5e3,
        tol_box=1e-6,
        tol_energy=1e-8,
        tol_angle=2e-3,
        track_best_curve=True,
    )

    print("\n==== Strict constraint: t1+t2+t3+p1+p2+p3 = S_target ====")
    print(f"S_target = {S_target:.12f}")
    print(f"Global min (feasible) = {vmin_global:.12f}")

    print("\n==== Unique global-min points (angles in (-pi,pi]) ====")
    print(f"Found {uniq_angles.shape[0]} unique minima (within tolerances).")

    for k in range(uniq_angles.shape[0]):
        a = uniq_angles[k].numpy().tolist()
        Vk = float(uniq_V[k].item())
        t1,t2,t3,p1,p2,p3 = a
        print(f"\n[{k:02d}] V = {Vk:.12f}")
        print(f"     (t1,t2,t3,p1,p2,p3) = "
              f"({t1:.6f}, {t2:.6f}, {t3:.6f}, {p1:.6f}, {p2:.6f}, {p3:.6f})")

    if curve is not None and len(curve) > 0:
        plt.figure()
        plt.plot(curve)
        plt.xlabel("iteration")
        plt.ylabel("best feasible V_min among starts")
        plt.grid(True)
        plt.title(f"Convergence (strict sum = {S_target:.3f})")
        plt.show()

if __name__ == "__main__":
    main()