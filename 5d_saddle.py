import math
import torch

# ============================================================
# Device / dtype
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
print("device:", device)

PI = math.pi
TWO_PI = 2.0 * PI

def wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + PI, TWO_PI) - PI

def circ_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = torch.abs(a - b)
    return torch.minimum(d, TWO_PI - d)

# ============================================================
# Potential
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
#   Phi = Phi_tilde @ U -> (t1,t2,t3,p1,p2,p3)
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

# ============================================================
# Shape safety helpers (fixes your dim errors)
# ============================================================
def ensure_2d_x5(x5: torch.Tensor) -> torch.Tensor:
    # want shape (N,5)
    if x5.ndim == 1:
        return x5.view(1, 5)
    if x5.ndim == 2:
        return x5
    return x5.view(-1, 5)

def x5_to_phi6(x5: torch.Tensor, x0: float = 0.0, wrap=True) -> torch.Tensor:
    x5 = ensure_2d_x5(x5)  # (N,5)
    N = x5.shape[0]
    x0t = torch.full((N, 1), float(x0), device=x5.device, dtype=x5.dtype)
    Phi_tilde = torch.cat([x0t, x5], dim=1)     # (N,6)
    Phi = Phi_tilde @ U                         # (N,6)
    if wrap:
        Phi = wrap_pi(Phi)
    return Phi

def potential5D(x5: torch.Tensor, x0: float = 0.0) -> torch.Tensor:
    x5 = ensure_2d_x5(x5)                       # (N,5)
    # DO NOT wrap for energy/grad; cos is periodic and smooth
    Phi = x5_to_phi6(x5, x0=x0, wrap=False)     # (N,6)
    t1, t2, t3, p1, p2, p3 = Phi.unbind(dim=1)  # safe
    return potential6D(t1, t2, t3, p1, p2, p3, EJ=EJ)

# ============================================================
# Fast clustering in wrapped Phi space
# ============================================================
def greedy_cluster_phi(Phi_wrapped: torch.Tensor, tol_angle: float = 2e-3):
    Phi_wrapped = ensure_2d_x5(Phi_wrapped) if Phi_wrapped.shape[-1] == 5 else Phi_wrapped
    K = Phi_wrapped.shape[0]
    if K == 0:
        return []
    used = torch.zeros(K, device=Phi_wrapped.device, dtype=torch.bool)
    reps = []
    thr = (tol_angle * tol_angle) * 6.0
    for i in range(K):
        if used[i]:
            continue
        reps.append(i)
        d = circ_dist(Phi_wrapped, Phi_wrapped[i:i+1])
        d2 = (d*d).sum(dim=1)
        used |= (d2 <= thr)
    return reps

# ============================================================
# Hessian / classify ONLY for selected points (Top-K)
# ============================================================
def V_scalar(x5_1: torch.Tensor, x0: float) -> torch.Tensor:
    # x5_1 can be (5,) or (1,5); always return scalar
    V = potential5D(x5_1, x0=x0)   # (N,)
    return V.view(-1)[0]

def hessian_V(x5_1: torch.Tensor, x0: float) -> torch.Tensor:
    x = x5_1.detach().clone().view(5).requires_grad_(True)
    V = V_scalar(x, x0)
    g = torch.autograd.grad(V, x, create_graph=True)[0]  # (5,)
    H = []
    for i in range(5):
        Hi = torch.autograd.grad(g[i], x, retain_graph=True)[0]
        H.append(Hi)
    H = torch.stack(H, dim=0)
    H = 0.5 * (H + H.T)
    return H

def classify(x5_1: torch.Tensor, x0: float, eps_eig=1e-9):
    x = x5_1.detach().clone().view(5).requires_grad_(True)
    V = V_scalar(x, x0)
    g = torch.autograd.grad(V, x, create_graph=False)[0]
    gnorm = float(torch.linalg.norm(g).detach().cpu())
    H = hessian_V(x, x0).detach()
    evals = torch.linalg.eigvalsh(H).detach().cpu()
    nneg = int((evals < -eps_eig).sum().item())
    return float(V.detach().cpu()), gnorm, evals, nneg

# ============================================================
# (A) Minima: quick multi-start Adam
# ============================================================
@torch.no_grad()
def collect_global_minima(x_all, V_all, x0=0.0, tol_energy=1e-8, tol_angle=2e-3):
    vmin = V_all.min()
    idx = torch.nonzero(V_all <= vmin + tol_energy, as_tuple=False).flatten()
    x_sel = x_all[idx]
    Phi_sel = x5_to_phi6(x_sel, x0=x0, wrap=True)  # (K,6)
    reps = greedy_cluster_phi(Phi_sel, tol_angle=tol_angle)
    return x_sel[reps].cpu(), Phi_sel[reps].cpu(), float(vmin.cpu())

def find_minima_fast(x0=0.0, n_starts=20000, steps=1200, lr=0.05, seed=0):
    torch.manual_seed(seed)
    x = (torch.randn(n_starts, 5, device=device, dtype=dtype) * (4*PI)).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)

    for it in range(steps):
        opt.zero_grad(set_to_none=True)
        V = potential5D(x, x0=x0)
        loss = V.mean()
        loss.backward()
        opt.step()
        if it % 200 == 0:
            print(f"[min it={it:4d}] Vmin={float(V.min()):.6f}")

    with torch.no_grad():
        V = potential5D(x, x0=x0)
        x_u, Phi_u, vmin = collect_global_minima(x.detach(), V.detach(), x0=x0)
    return x_u, Phi_u, vmin

# ============================================================
# (B) Saddles: minimize F = 1/2 ||grad V||^2 (fast), then classify Top-K
# ============================================================
def find_saddles_fast(
    x0=0.0,
    n_starts=20000,
    steps=2500,
    lr=0.03,
    seed=123,
    topK=300,          # only classify topK best (smallest grad)
    tol_angle=2e-3,
    eps_eig=1e-9,
):
    torch.manual_seed(seed)
    x = (torch.randn(n_starts, 5, device=device, dtype=dtype) * (PI)).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)

    for it in range(steps):
        opt.zero_grad(set_to_none=True)
        V = potential5D(x, x0=x0)  # (N,)
        g = torch.autograd.grad(V.sum(), x, create_graph=True)[0]  # (N,5)
        F = 0.5 * (g*g).sum(dim=1)  # (N,)
        loss = F.mean()
        loss.backward()
        opt.step()
        if it % 500 == 0:
            with torch.no_grad():
                gnorm_min = float(torch.linalg.norm(g.detach(), dim=1).min().cpu())
                print(f"[sad it={it:4d}] min||grad||={gnorm_min:.3e}  Fmin={float(F.min()):.3e}")

    # pick Top-K by grad norm
    with torch.enable_grad():
        V = potential5D(x, x0=x0)
        g = torch.autograd.grad(V.sum(), x, create_graph=False)[0]
    with torch.no_grad():
        gnorm = torch.linalg.norm(g, dim=1)
        topK = min(topK, x.shape[0])
        idx = torch.topk(-gnorm, k=topK).indices  # smallest gnorm
        x_top = x.detach()[idx].cpu()            # (topK,5)

    # classify ONLY Top-K points
    saddles = []
    saddles_phi = []
    for i in range(x_top.shape[0]):
        x_i = x_top[i].to(device=device, dtype=dtype).view(5)
        Vv, gnv, evals, nneg = classify(x_i, x0=x0, eps_eig=eps_eig)
        if nneg == 1:  # index-1 saddle
            saddles.append((x_i.detach().cpu(), Vv, gnv, evals))
            saddles_phi.append(x5_to_phi6(x_i, x0=x0, wrap=True).view(6).detach().cpu())

    if len(saddles) == 0:
        return [], []

    PhiS = torch.stack(saddles_phi, dim=0).to(device=device, dtype=dtype)  # (M,6)
    reps = greedy_cluster_phi(PhiS, tol_angle=tol_angle)

    uniq = [saddles[i] for i in reps]
    uniq_phi = [saddles_phi[i] for i in reps]
    uniq = sorted(zip(uniq, uniq_phi), key=lambda t: t[0][1])  # sort by V
    return [u for (u, p) in uniq], [p for (u, p) in uniq]

# ============================================================
# Main
# ============================================================
def main():
    x0 = 0.0  # gauge value (should not change energies)

    # --- minima ---
    x_u, Phi_u, vmin = find_minima_fast(x0=x0, n_starts=20000, steps=1200, lr=0.05, seed=0)
    print(f"\n[Minima] unique global minima = {len(x_u)}, Vmin = {vmin:.12f}")

    # --- saddles ---
    saddles, saddles_phi = find_saddles_fast(
        x0=x0,
        n_starts=100000,
        steps=2500,
        lr=0.03,
        seed=123,
        topK=3000,
        tol_angle=2e-3,
        eps_eig=1e-9,
    )

    print(f"\n[Saddles] unique index-1 saddles = {len(saddles)}")
    for k, ((x5, Vv, gnv, evals), ph) in enumerate(zip(saddles, saddles_phi)):
        t1, t2, t3, p1, p2, p3 = [float(v) for v in ph]
        print(f"\n[S{k:02d}] V={Vv:.12f}  ||grad||={gnv:.3e}")
        print(f"  Phi=(t1..p3)=({t1:.6f},{t2:.6f},{t3:.6f},{p1:.6f},{p2:.6f},{p3:.6f})")
        print(f"  Hess evals={evals.numpy()}")

if __name__ == "__main__":
    main()