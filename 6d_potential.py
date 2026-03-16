import math
import torch
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
print("device:", device)

TWO_PI = 2.0 * math.pi

def wrap_pi_torch(x: torch.Tensor) -> torch.Tensor:
    # maps to [-pi, pi)
    return x - TWO_PI * torch.floor((x + math.pi) / TWO_PI)

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
# Multi-start gradient minimization on 6D torus (angles)
# ---------------------------------------------------------
@torch.no_grad()
def random_angles(n):
    return (torch.rand(n, device=device, dtype=dtype) * TWO_PI - math.pi)

def minimize_potential_multistart(
    EJ=80.0,
    n_starts=1024,      # number of random initial points
    steps=800,          # gradient steps
    lr=0.05,            # Adam learning rate
    print_every=100,
    seed=0,
    track_best_curve=True,
):
    torch.manual_seed(seed)

    # We optimize "raw" variables and wrap them inside the objective.
    # shape: (n_starts,)
    t1 = random_angles(n_starts).requires_grad_(True)
    t2 = random_angles(n_starts).requires_grad_(True)
    t3 = random_angles(n_starts).requires_grad_(True)
    p1 = random_angles(n_starts).requires_grad_(True)
    p2 = random_angles(n_starts).requires_grad_(True)
    p3 = random_angles(n_starts).requires_grad_(True)

    opt = torch.optim.Adam([t1, t2, t3, p1, p2, p3], lr=lr)

    best_curve = []

    for it in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)

        wt1 = wrap_pi_torch(t1)
        wt2 = wrap_pi_torch(t2)
        wt3 = wrap_pi_torch(t3)
        wp1 = wrap_pi_torch(p1)
        wp2 = wrap_pi_torch(p2)
        wp3 = wrap_pi_torch(p3)

        V = potential6D(wt1, wt2, wt3, wp1, wp2, wp3, EJ=EJ)  # (n_starts,)
        loss = V.mean()
        loss.backward()
        opt.step()

        if track_best_curve:
            with torch.no_grad():
                best_curve.append(float(V.min().detach().cpu()))

        if (it == 1) or (it % print_every == 0) or (it == steps):
            with torch.no_grad():
                vmin = float(V.min().detach().cpu())
                vmean = float(V.mean().detach().cpu())
                print(f"[it={it:4d}] V_min={vmin:.10f}   V_mean={vmean:.10f}")

    # pick best among starts
    with torch.no_grad():
        wt1 = wrap_pi_torch(t1); wt2 = wrap_pi_torch(t2); wt3 = wrap_pi_torch(t3)
        wp1 = wrap_pi_torch(p1); wp2 = wrap_pi_torch(p2); wp3 = wrap_pi_torch(p3)
        V = potential6D(wt1, wt2, wt3, wp1, wp2, wp3, EJ=EJ)
        idx = torch.argmin(V)
        v_best = float(V[idx].cpu())
        angles_best = (
            float(wt1[idx].cpu()), float(wt2[idx].cpu()), float(wt3[idx].cpu()),
            float(wp1[idx].cpu()), float(wp2[idx].cpu()), float(wp3[idx].cpu()),
        )

    return v_best, angles_best, best_curve

def main():
    EJ = 80.0
    vmin, ang, curve = minimize_potential_multistart(
        EJ=EJ,
        n_starts=2048,   # increase for better global search
        steps=1200,
        lr=0.05,
        print_every=200,
        seed=0,
        track_best_curve=True
    )

    t1, t2, t3, p1, p2, p3 = ang
    print("\n==== Best found classical minimum ====")
    print(f"V_min = {vmin:.12f}")
    print(f"(t1,t2,t3,p1,p2,p3) = ({t1:.6f}, {t2:.6f}, {t3:.6f}, {p1:.6f}, {p2:.6f}, {p3:.6f})")

    if curve is not None and len(curve) > 0:
        plt.figure()
        plt.plot(curve)
        plt.xlabel("iteration")
        plt.ylabel("best V_min among starts")
        plt.grid(True)
        plt.title("Convergence of best-found minimum")
        plt.show()

if __name__ == "__main__":
    main()

# import math
# import torch
# import matplotlib.pyplot as plt

# ============================================================
# Device / dtype
# ============================================================
# device = "cuda" if torch.cuda.is_available() else "cpu"
# dtype = torch.float64
# print("device:", device)
#
# TWO_PI = 2.0 * math.pi
#
# def wrap_pi(x: torch.Tensor) -> torch.Tensor:
#     # map to [-pi, pi)
#     return x - TWO_PI * torch.floor((x + math.pi) / TWO_PI)
#
# def angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#     # shortest difference on circle, in [-pi, pi)
#     return wrap_pi(a - b)
#
# def torus_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     # x,y: (...,6)
#     d = angle_diff(x, y)
#     return torch.sqrt((d * d).sum(dim=-1))
#
# # ============================================================
# # 6D classical potential (same as your code)
# # ============================================================
# def potential6D_vec(x: torch.Tensor, EJ: float = 80.0) -> torch.Tensor:
#     """
#     x: (N,6) angles in radians (not necessarily wrapped)
#     returns V: (N,)
#     """
#     t1, t2, t3, p1, p2, p3 = x.T
#     alpha = 2 * torch.pi / 3
#     term_phi1 = -2 * torch.cos(t1 - p1) - 2 * torch.cos(t2 - p1) - 2 * torch.cos(t3 - p1)
#     term_phi2 = -2 * torch.cos(t1 - p2) - 2 * torch.cos(t2 - p2 + alpha) - 2 * torch.cos(t3 - p2 - alpha)
#     term_phi3 = -2 * torch.cos(t1 - p3) - 2 * torch.cos(t2 - p3 - alpha) - 2 * torch.cos(t3 - p3 + alpha)
#     return EJ * (term_phi1 + term_phi2 + term_phi3) / math.sqrt(3.0)
#
# @torch.no_grad()
# def random_angles(n: int) -> torch.Tensor:
#     return (torch.rand(n, 6, device=device, dtype=dtype) * TWO_PI - math.pi)
#
# # ============================================================
# # 1) Multi-start optimize V: min or max
# # ============================================================
# def optimize_V(
#     n_starts: int = 4096,
#     steps: int = 1200,
#     lr: float = 0.05,
#     EJ: float = 80.0,
#     maximize: bool = False,
#     seed: int = 0,
# ):
#     torch.manual_seed(seed)
#     x = random_angles(n_starts).requires_grad_(True)
#     opt = torch.optim.Adam([x], lr=lr)
#
#     for _ in range(steps):
#         opt.zero_grad(set_to_none=True)
#         xw = wrap_pi(x)
#         V = potential6D_vec(xw, EJ=EJ)  # (N,)
#         loss = (-V.mean()) if maximize else (V.mean())
#         loss.backward()
#         opt.step()
#
#     with torch.no_grad():
#         xw = wrap_pi(x)
#         V = potential6D_vec(xw, EJ=EJ)
#         idx = torch.argmax(V) if maximize else torch.argmin(V)
#         return float(V[idx].cpu()), xw[idx].detach()
#
# # ============================================================
# # 2) Stationary points by minimizing ||grad V||^2
# # ============================================================
# def grad_norm_sq(xw: torch.Tensor, EJ: float = 80.0, create_graph: bool = True):
#     """
#     xw: (N,6) requires_grad=True
#     returns: gn2 (N,), grads (N,6), V (N,)
#     """
#     V = potential6D_vec(xw, EJ=EJ)  # (N,)
#     grads = torch.autograd.grad(V.sum(), xw, create_graph=create_graph)[0]  # (N,6)
#     gn2 = (grads * grads).sum(dim=1)
#     return gn2, grads, V
#
# def find_stationary_points(
#     n_starts_grad2: int = 8192,
#     steps: int = 1500,
#     lr: float = 0.03,
#     EJ: float = 80.0,
#     seed: int = 2,
#     keep_top: int = 400,
# ):
#     """
#     Returns candidates (K,6), their V (K,), and gn2 (K,)
#     by minimizing ||grad V||^2 from many random starts.
#     """
#     torch.manual_seed(seed)
#
#     x = random_angles(n_starts_grad2).requires_grad_(True)
#     opt = torch.optim.Adam([x], lr=lr)
#
#     for it in range(1, steps + 1):
#         opt.zero_grad(set_to_none=True)
#         xw = wrap_pi(x)
#         gn2, _, _ = grad_norm_sq(xw, EJ=EJ, create_graph=True)
#         loss = gn2.mean()
#         loss.backward()
#         opt.step()
#
#         if it == 1 or it % 300 == 0 or it == steps:
#             with torch.no_grad():
#                 print(f"[grad2 it={it}] mean||grad||^2 = {float(loss.detach().cpu()):.6e}")
#
#     # After optimization, we want to rank by ||grad||^2.
#     # IMPORTANT: we must compute grads with x requiring grad.
#     with torch.no_grad():
#         xw = wrap_pi(x).detach()
#
#     with torch.enable_grad():
#         xw2 = xw.clone().requires_grad_(True)
#         gn2, _, V = grad_norm_sq(xw2, EJ=EJ, create_graph=False)  # no need for 2nd-order graph here
#
#     with torch.no_grad():
#         k = min(keep_top, gn2.numel())
#         vals, idx = torch.topk(gn2.detach(), k=k, largest=False)
#         return xw[idx].detach(), V.detach()[idx].detach(), vals.detach()
#
# # ============================================================
# # 3) Deduplicate points on torus
# # ============================================================
# @torch.no_grad()
# def dedup_points(points: torch.Tensor, values: torch.Tensor, dist_tol: float = 5e-2, max_keep: int = 80, prefer: str = "lowV"):
#     """
#     points: (M,6)
#     values: (M,)
#     prefer: "lowV" or "highV"
#     returns list of kept indices
#     """
#     if points.numel() == 0:
#         return []
#
#     order = torch.argsort(values) if prefer == "lowV" else torch.argsort(values, descending=True)
#     kept = []
#
#     for idx in order.tolist():
#         p = points[idx]
#         ok = True
#         for j in kept:
#             if float(torus_dist(p, points[j]).cpu()) < dist_tol:
#                 ok = False
#                 break
#         if ok:
#             kept.append(idx)
#         if len(kept) >= max_keep:
#             break
#     return kept
#
# # ============================================================
# # 4) Hessian eigs and classification
# # ============================================================
# def hessian_eigs_at_point(x6: torch.Tensor, EJ: float = 80.0):
#     """
#     x6: (6,) wrapped angles
#     returns: V (float), grad (6,), eigvals (6,)
#     """
#     x = x6.clone().detach().to(device=device, dtype=dtype).requires_grad_(True)
#     V = potential6D_vec(x.view(1, 6), EJ=EJ)[0]
#     g = torch.autograd.grad(V, x, create_graph=True)[0]  # (6,)
#
#     H_rows = []
#     for i in range(6):
#         Hi = torch.autograd.grad(g[i], x, retain_graph=True)[0]  # (6,)
#         H_rows.append(Hi)
#     H = torch.stack(H_rows, dim=0)
#     H = 0.5 * (H + H.T)
#
#     eigs = torch.linalg.eigvalsh(H)  # sorted
#     return float(V.detach().cpu()), g.detach(), eigs.detach()
#
# def classify_eigs(eigs: torch.Tensor, tol: float = 1e-6) -> str:
#     pos = int((eigs > tol).sum().item())
#     neg = int((eigs < -tol).sum().item())
#     zero = eigs.numel() - pos - neg
#     if neg == 0 and zero == 0:
#         return "min"
#     if pos == 0 and zero == 0:
#         return "max"
#     if neg > 0 and pos > 0:
#         return f"saddle(index={neg})"
#     return f"degenerate(pos={pos}, neg={neg}, zero~={zero})"
#
# # ============================================================
# # Main
# # ============================================================
# def main():
#     EJ = 80.0
#
#     # ---- Min / Max
#     Vmin, xmin = optimize_V(n_starts=4096, steps=1200, lr=0.05, EJ=EJ, maximize=False, seed=0)
#     Vmax, xmax = optimize_V(n_starts=4096, steps=1200, lr=0.05, EJ=EJ, maximize=True,  seed=1)
#
#     print("\n=== Best-found minimum / maximum (multi-start Adam) ===")
#     print(f"V_min ≈ {Vmin:.12f} at x={xmin.detach().cpu().numpy()}")
#     print(f"V_max ≈ {Vmax:.12f} at x={xmax.detach().cpu().numpy()}")
#
#     # ---- Stationary point candidates
#     cand_x, cand_V, cand_gn2 = find_stationary_points(
#         n_starts_grad2=8192,
#         steps=1500,
#         lr=0.03,
#         EJ=EJ,
#         seed=2,
#         keep_top=400,
#     )
#
#     # ---- Dedup
#     keep_idx = dedup_points(cand_x, cand_V, dist_tol=5e-2, max_keep=100, prefer="lowV")
#     cand_x = cand_x[keep_idx]
#     cand_V = cand_V[keep_idx]
#     cand_gn2 = cand_gn2[keep_idx]
#
#     print(f"\nDeduped stationary candidates: {cand_x.shape[0]}")
#
#     # ---- Classify by Hessian
#     results = []
#     for i in range(cand_x.shape[0]):
#         x6 = cand_x[i]
#         V, g, eigs = hessian_eigs_at_point(x6, EJ=EJ)
#         gn = float(torch.linalg.norm(g).detach().cpu())
#         typ = classify_eigs(eigs, tol=1e-6)
#         results.append((V, gn, typ, x6.detach().cpu(), eigs.detach().cpu()))
#
#     results.sort(key=lambda t: t[0])
#
#     # ---- Print blocks
#     def print_block(block, title):
#         print(f"\n--- {title} ---")
#         for (V, gn, typ, x6, eigs) in block:
#             print(f"V={V: .12f}  |grad|={gn:.3e}  type={typ}")
#             print(f"  x={x6.numpy()}")
#             print(f"  eigs={eigs.numpy()}")
#
#     lowest = results[:10]
#     highest = results[-10:] if len(results) >= 10 else results
#     saddles = [r for r in results if "saddle" in r[2]]
#     mid_saddles = saddles[len(saddles)//2 : len(saddles)//2 + 10] if len(saddles) > 0 else []
#
#     print_block(lowest, "Lowest V stationary points")
#     if len(mid_saddles) > 0:
#         print_block(mid_saddles, "Some mid-energy saddles")
#     print_block(highest, "Highest V stationary points")
#
#     nmin = sum(1 for r in results if r[2] == "min")
#     nmax = sum(1 for r in results if r[2] == "max")
#     nsad = sum(1 for r in results if "saddle" in r[2])
#     ndeg = len(results) - nmin - nmax - nsad
#
#     print("\n=== Counts (in deduped set) ===")
#     print(f"min: {nmin}, max: {nmax}, saddle: {nsad}, degenerate/flat: {ndeg}")
#     print("(If many are 'degenerate', try increasing tol in classify_eigs, e.g. 1e-5.)")
#
# if __name__ == "__main__":
#     main()