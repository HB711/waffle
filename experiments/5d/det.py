import math
import torch
import torch.nn as nn
from contextlib import contextmanager

# ============================================================
# Device / dtype
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
print("device:", device)

TWO_PI = 2.0 * math.pi

def wrap_centered_1d(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    # (c-pi, c+pi]
    return c + (torch.remainder(x - c + math.pi, TWO_PI) - math.pi)

# ============================================================
# Hyperparameters (NES-VMC, MH local box)
# ============================================================
N_states = 2

N_walkers = 2000
N_burn_in = 80
Sigma0 = 0.20
P_GLOBAL = 0.02
Delta = 1.0

Hidden_Dim = 256
Seed = 0
torch.manual_seed(Seed)

N_steps  = 8000
lr       = 1e-3
GRAD_CLIP_NORM = 10.0

BETA_IMAG = 0.0

DET_EPS = 1e-10
LOGDET_EPS = 1e-12

PRINT_EVERY = 10

# ============================================================
# Physics constants / capacitance matrix / kinetic prefactors
# ============================================================
h = 6.626
e = 1.60
UNIT = 100.0 * (4.0 * e**2) / (2.0 * h)

Cth = 10.00
Cph = 10.00
Cv  = 30.00
Cp  = 47.00
EJ  = 80.00

ALPHA = -2.0 * math.pi / 3.0

# harmonic estimate (optional)
w1 = math.sqrt(UNIT * 2 * EJ / (7 * Cth))
w2 = w1
w3 = math.sqrt(UNIT * 2 * 3 * EJ / (7 * Cth))
w4 = w3
w5 = math.sqrt(UNIT * 2 * 4 * EJ / (7 * Cth))
E_th = -6 * EJ + 0.5 * (w1 + w2 + w3 + w4 + w5)
print("Eth=", E_th)

C = torch.tensor(
    [[Cth + 3*Cv + 2*Cp, -Cp, -Cp, -Cv, -Cv, -Cv],
     [-Cp, Cth + 3*Cv + 2*Cp, -Cp, -Cv, -Cv, -Cv],
     [-Cp, -Cp, Cth + 3*Cv + 2*Cp, -Cv, -Cv, -Cv],
     [-Cv, -Cv, -Cv, Cph + 3*Cv + 2*Cp, -Cp, -Cp],
     [-Cv, -Cv, -Cv, -Cp, Cph + 3*Cv + 2*Cp, -Cp],
     [-Cv, -Cv, -Cv, -Cp, -Cp, Cph + 3*Cv + 2*Cp]],
    device=device, dtype=dtype
)
Cinv = torch.linalg.inv(C)
E_mat = UNIT * Cinv
E_mat = 0.5 * (E_mat + E_mat.T)

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

E_tilde = U @ E_mat @ U.T
E_tilde = 0.5 * (E_tilde + E_tilde.T)
E5 = torch.diag(E_tilde)

# ============================================================
# Choose ONE 6D minimum -> 5D x_loc
# ============================================================
theta_phi_min_6 = torch.tensor(
    [0.0, 0.0, 2.0*math.pi/3.0,  math.pi/6.0, math.pi/6.0, -math.pi/2.0],
    device=device, dtype=dtype
)

@torch.no_grad()
def theta_phi_to_xloc(Phi6: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    Phi6 = Phi6.clone()
    Phi6 = Phi6 - Phi6.mean()
    Phi_tilde = Phi6 @ U.T
    return Phi_tilde[1:].contiguous()

x_loc = theta_phi_to_xloc(theta_phi_min_6, U)
print("x_loc (x2..x6) =", x_loc.detach().cpu().numpy())

# ============================================================
# Potential
# ============================================================
def potential6D(t1, t2, t3, p1, p2, p3):
    a = ALPHA
    term_phi1 = -2 * torch.cos(t1 - p1) - 2 * torch.cos(t2 - p1) - 2 * torch.cos(t3 - p1)
    term_phi2 = -2 * torch.cos(t1 - p2) - 2 * torch.cos(t2 - p2 - a) - 2 * torch.cos(t3 - p2 + a)
    term_phi3 = -2 * torch.cos(t1 - p3) - 2 * torch.cos(t2 - p3 + a) - 2 * torch.cos(t3 - p3 - a)
    return EJ * (term_phi1 + term_phi2 + term_phi3) / math.sqrt(3.0)

def potential5D(x2, x3, x4, x5, x6):
    Phi_tilde = torch.stack([torch.zeros_like(x2), x2, x3, x4, x5, x6], dim=1)
    Phi = Phi_tilde @ U
    t1, t2, t3, p1, p2, p3 = Phi.T
    return potential6D(t1, t2, t3, p1, p2, p3)

# ============================================================
# Periodic embedding
# ============================================================
def periodic_emb(x2, x3, x4, x5, x6):
    feats = [
        torch.sin(x2), torch.cos(x2),
        torch.sin(x3), torch.cos(x3),
        torch.sin(x4), torch.cos(x4),
        torch.sin(x5), torch.cos(x5),
        torch.sin(x6), torch.cos(x6),
        torch.sin(2*x2), torch.cos(2*x2),
        torch.sin(2*x3), torch.cos(2*x3),
        torch.sin(2*x4), torch.cos(2*x4),
        torch.sin(2*x5), torch.cos(2*x5),
        torch.sin(2*x6), torch.cos(2*x6),
    ]
    return torch.stack(feats, dim=-1)  # (N,20)

# ============================================================
# Model
# ============================================================
class NN_Waffle_Complex_5D(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=Hidden_Dim):
        super().__init__()
        self.backbone_f = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )
        self.out_f = nn.Linear(hidden_dim, 1)

        self.backbone_g = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )
        self.out_g = nn.Linear(hidden_dim, 1)

        nn.init.zeros_(self.out_g.weight)
        nn.init.zeros_(self.out_g.bias)

    def forward(self, x2, x3, x4, x5, x6):
        x = periodic_emb(x2, x3, x4, x5, x6)
        f = self.out_f(self.backbone_f(x)).squeeze(-1)
        g = self.out_g(self.backbone_g(x)).squeeze(-1)
        return f, g

# ============================================================
# Utilities
# ============================================================
@contextmanager
def freeze_params(model: nn.Module):
    """Temporarily set all model params requires_grad=False (restore afterwards)."""
    req = [p.requires_grad for p in model.parameters()]
    try:
        for p in model.parameters():
            p.requires_grad_(False)
        yield
    finally:
        for p, r in zip(model.parameters(), req):
            p.requires_grad_(r)

def psi_row_scaled(f_row: torch.Tensor, g_row: torch.Tensor):
    # f_row,g_row: (N,2) at same x (one row of M)
    m = torch.max(f_row, dim=-1).values  # (N,)
    psi = torch.exp((f_row - m[:, None]).to(dtype)).to(cdtype) * torch.exp(1j * g_row.to(cdtype))
    return psi

def det_abs2_from_M(M: torch.Tensor) -> torch.Tensor:
    detM = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
    return (detM.conj() * detM).real.clamp_min(1e-300)

# ============================================================
# Local energy wrt x ONLY (params frozen outside)
#   Need create_graph=True for FIRST derivative so that SECOND derivative exists.
# ============================================================
def local_energy_5D_xonly(x2, x3, x4, x5, x6, model: nn.Module):
    xs = [
        x2.detach().requires_grad_(True),
        x3.detach().requires_grad_(True),
        x4.detach().requires_grad_(True),
        x5.detach().requires_grad_(True),
        x6.detach().requires_grad_(True),
    ]
    f, g = model(xs[0], xs[1], xs[2], xs[3], xs[4])
    ones = torch.ones_like(f)

    # IMPORTANT: create_graph=True so dfs keeps graph wrt x for second derivative
    dfs = torch.autograd.grad(f, xs, grad_outputs=ones, create_graph=True, retain_graph=True)
    dgs = torch.autograd.grad(g, xs, grad_outputs=ones, create_graph=True, retain_graph=True)

    d2fs, d2gs = [], []
    for i in range(5):
        d2fi = torch.autograd.grad(dfs[i], xs[i], grad_outputs=torch.ones_like(dfs[i]),
                                   create_graph=False, retain_graph=True)[0]
        d2gi = torch.autograd.grad(dgs[i], xs[i], grad_outputs=torch.ones_like(dgs[i]),
                                   create_graph=False, retain_graph=True)[0]
        d2fs.append(d2fi)
        d2gs.append(d2gi)

    E_real = 0.0
    E_imag = 0.0
    for i in range(5):
        Ek = E5[i + 1]
        df, dg = dfs[i], dgs[i]
        d2f, d2g = d2fs[i], d2gs[i]
        E_real = E_real + (-Ek) * (d2f + df**2 - dg**2)
        E_imag = E_imag + (-Ek) * (d2g + 2.0 * df * dg)

    V = potential5D(xs[0], xs[1], xs[2], xs[3], xs[4])
    E_real = E_real + V

    # detach everything (we only use as estimator)
    return E_real.detach(), E_imag.detach(), f.detach(), g.detach()

# ============================================================
# Build M for gradient (uses f,g with grad to params)
# ============================================================
def build_M_for_grad(models, x1, x2):
    # row 0: x1
    f0_1, g0_1 = models[0](*x1)
    f1_1, g1_1 = models[1](*x1)
    f_row1 = torch.stack([f0_1, f1_1], dim=-1)
    g_row1 = torch.stack([g0_1, g1_1], dim=-1)
    psi_row1 = psi_row_scaled(f_row1, g_row1)

    # row 1: x2
    f0_2, g0_2 = models[0](*x2)
    f1_2, g1_2 = models[1](*x2)
    f_row2 = torch.stack([f0_2, f1_2], dim=-1)
    g_row2 = torch.stack([g0_2, g1_2], dim=-1)
    psi_row2 = psi_row_scaled(f_row2, g_row2)

    M = torch.stack([
        torch.stack([psi_row1[:, 0], psi_row1[:, 1]], dim=-1),
        torch.stack([psi_row2[:, 0], psi_row2[:, 1]], dim=-1),
    ], dim=-2)  # (N,2,2)
    return M

# ============================================================
# Trace(E) estimator (NO param-grad):
#   compute M,B, solve, return trE (detached) using x-only derivatives
# ============================================================
def traceE_estimator_detached(models, x1, x2, good_mask: torch.Tensor):
    # We compute row-scaled psi with detached f,g as well (scaling cancels in solve)
    with torch.enable_grad():
        # row 0 at x1
        with freeze_params(models[0]):
            E0r_1, E0i_1, f0_1, g0_1 = local_energy_5D_xonly(*x1, models[0])
        with freeze_params(models[1]):
            E1r_1, E1i_1, f1_1, g1_1 = local_energy_5D_xonly(*x1, models[1])

        f_row1 = torch.stack([f0_1, f1_1], dim=-1)
        g_row1 = torch.stack([g0_1, g1_1], dim=-1)
        psi_row1 = psi_row_scaled(f_row1, g_row1)  # uses detached f,g -> detached psi

        Eloc0_1 = E0r_1.to(cdtype) + 1j * E0i_1.to(cdtype)
        Eloc1_1 = E1r_1.to(cdtype) + 1j * E1i_1.to(cdtype)

        # row 1 at x2
        with freeze_params(models[0]):
            E0r_2, E0i_2, f0_2, g0_2 = local_energy_5D_xonly(*x2, models[0])
        with freeze_params(models[1]):
            E1r_2, E1i_2, f1_2, g1_2 = local_energy_5D_xonly(*x2, models[1])

        f_row2 = torch.stack([f0_2, f1_2], dim=-1)
        g_row2 = torch.stack([g0_2, g1_2], dim=-1)
        psi_row2 = psi_row_scaled(f_row2, g_row2)

        Eloc0_2 = E0r_2.to(cdtype) + 1j * E0i_2.to(cdtype)
        Eloc1_2 = E1r_2.to(cdtype) + 1j * E1i_2.to(cdtype)

        M = torch.stack([
            torch.stack([psi_row1[:, 0], psi_row1[:, 1]], dim=-1),
            torch.stack([psi_row2[:, 0], psi_row2[:, 1]], dim=-1),
        ], dim=-2)

        B = torch.stack([
            torch.stack([psi_row1[:, 0] * Eloc0_1, psi_row1[:, 1] * Eloc1_1], dim=-1),
            torch.stack([psi_row2[:, 0] * Eloc0_2, psi_row2[:, 1] * Eloc1_2], dim=-1),
        ], dim=-2)

        # solve only on good
        Mg = M[good_mask]
        Bg = B[good_mask]
        Eg = torch.linalg.solve(Mg, Bg)
        trE = Eg[:, 0, 0] + Eg[:, 1, 1]
        return trE.detach(), Eg.detach()

# ============================================================
# MH on extended (x1,x2) with target ∝ |det(M)|^2
#   uses M-only, no_grad
# ============================================================
@torch.no_grad()
def build_M_only_nograd(models, x1, x2):
    return build_M_for_grad(models, x1, x2).detach()

@torch.no_grad()
def init_pair_local(N: int, x_loc: torch.Tensor, delta: float):
    u1 = (torch.rand(N, 5, device=device, dtype=dtype) * 2.0 - 1.0) * delta
    u2 = (torch.rand(N, 5, device=device, dtype=dtype) * 2.0 - 1.0) * delta
    X1 = x_loc[None, :] + u1
    X2 = x_loc[None, :] + u2
    X1 = torch.stack([wrap_centered_1d(X1[:, k], x_loc[k]) for k in range(5)], dim=1)
    X2 = torch.stack([wrap_centered_1d(X2[:, k], x_loc[k]) for k in range(5)], dim=1)
    x1 = (X1[:, 0], X1[:, 1], X1[:, 2], X1[:, 3], X1[:, 4])
    x2 = (X2[:, 0], X2[:, 1], X2[:, 2], X2[:, 3], X2[:, 4])
    return (x1, x2)

def _in_box(xpack, x_loc, delta):
    x2, x3, x4, x5, x6 = xpack
    return (
        ((x2 - x_loc[0]).abs() <= delta) &
        ((x3 - x_loc[1]).abs() <= delta) &
        ((x4 - x_loc[2]).abs() <= delta) &
        ((x5 - x_loc[3]).abs() <= delta) &
        ((x6 - x_loc[4]).abs() <= delta)
    )

@torch.no_grad()
def mh_chain_pair_local(models, pair_state, x_loc, sigma, Nb, p_global, delta):
    (x1, x2) = pair_state
    N = x1[0].shape[0]
    acc = 0

    M_cur = build_M_only_nograd(models, x1, x2)
    w = det_abs2_from_M(M_cur)

    def step_pack(xp):
        return (
            xp[0] + torch.randn_like(xp[0]) * sigma,
            xp[1] + torch.randn_like(xp[1]) * sigma,
            xp[2] + torch.randn_like(xp[2]) * sigma,
            xp[3] + torch.randn_like(xp[3]) * sigma,
            xp[4] + torch.randn_like(xp[4]) * sigma,
        )

    for _ in range(Nb):
        if torch.rand((), device=device) < p_global:
            u1 = (torch.rand(N, 5, device=device, dtype=dtype) * 2.0 - 1.0) * delta
            u2 = (torch.rand(N, 5, device=device, dtype=dtype) * 2.0 - 1.0) * delta
            X1p = torch.stack([wrap_centered_1d(x_loc[k] + u1[:, k], x_loc[k]) for k in range(5)], dim=1)
            X2p = torch.stack([wrap_centered_1d(x_loc[k] + u2[:, k], x_loc[k]) for k in range(5)], dim=1)
            x1p = (X1p[:, 0], X1p[:, 1], X1p[:, 2], X1p[:, 3], X1p[:, 4])
            x2p = (X2p[:, 0], X2p[:, 1], X2p[:, 2], X2p[:, 3], X2p[:, 4])
        else:
            x1p = step_pack(x1)
            x2p = step_pack(x2)

        inb = _in_box(x1p, x_loc, delta) & _in_box(x2p, x_loc, delta)

        M_p = build_M_only_nograd(models, x1p, x2p)
        wp = det_abs2_from_M(M_p)

        log_a = torch.log(wp) - torch.log(w)
        accept = inb & (torch.log(torch.rand_like(log_a)) < log_a)

        def where_pack(a, newp, oldp):
            return tuple(torch.where(a, newp[k], oldp[k]) for k in range(5))

        x1 = where_pack(accept, x1p, x1)
        x2 = where_pack(accept, x2p, x2)
        w = torch.where(accept, wp, w)
        acc += int(accept.sum())

    return (x1, x2), acc / (Nb * N)

# ============================================================
# Training (stable NES-VMC)
# ============================================================
def train_nes_vmc_mh_local(steps=N_steps, eta=lr):
    models = [NN_Waffle_Complex_5D().to(device=device, dtype=dtype) for _ in range(N_states)]
    opts   = [torch.optim.Adam(m.parameters(), lr=eta) for m in models]

    pair_state = init_pair_local(N_walkers, x_loc=x_loc, delta=Delta)
    sigma = float(Sigma0)

    for it in range(steps):
        # ---- sample from |det M|^2 ----
        pair_state, acc = mh_chain_pair_local(
            models, pair_state,
            x_loc=x_loc, sigma=sigma, Nb=N_burn_in,
            p_global=P_GLOBAL, delta=Delta
        )
        (x1, x2) = pair_state

        # ---- M for gradient ----
        M_grad = build_M_for_grad(models, x1, x2)
        detM = M_grad[:, 0, 0] * M_grad[:, 1, 1] - M_grad[:, 0, 1] * M_grad[:, 1, 0]
        good = detM.abs() > DET_EPS
        Ng = int(good.sum())
        if Ng < max(64, N_walkers // 20):
            if it % PRINT_EVERY == 0:
                print(f"[it={it}] WARNING: too few good det samples: {Ng}/{N_walkers}. Skipping update.")
            continue

        # ---- energy estimator (detached) ----
        trE_det, E_mat_det = traceE_estimator_detached(models, x1, x2, good)

        Er = trE_det.real
        Ebar = Er.mean()

        # ---- stable VMC-style gradient: grads only through log|detM| ----
        logabsdet = torch.log(detM[good].abs() + LOGDET_EPS)
        loss = 2.0 * ((Er - Ebar) * logabsdet).mean()

        if BETA_IMAG != 0.0:
            loss = loss + float(BETA_IMAG) * (trE_det.imag.pow(2).mean())

        # ---- optimize ----
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(models[0].parameters(), GRAD_CLIP_NORM)
        torch.nn.utils.clip_grad_norm_(models[1].parameters(), GRAD_CLIP_NORM)

        for opt in opts:
            opt.step()

        # ---- sigma adapt ----
        if acc < 0.35:
            sigma *= 0.8
        elif acc > 0.65:
            sigma *= 1.2
        sigma = float(max(0.02, min(0.8, sigma)))

        # ---- logging ----
        if it == 0 or (it % PRINT_EVERY) == 0:
            with torch.no_grad():
                Ebar_mat = E_mat_det.mean(dim=0)  # (2,2) complex
                evals = torch.linalg.eigvals(Ebar_mat)
                evals_real_sorted, _ = torch.sort(evals.real)

                tr_mu = float(Er.mean().cpu())
                tr_std = float(Er.std(unbiased=False).cpu())
                tr_sem = tr_std / math.sqrt(Er.numel() + 1e-30)

                frac_good = float(good.float().mean().cpu())
                det_med = float(detM.abs()[good].median().cpu())

                f0a, _ = models[0](*x1)
                f1a, _ = models[1](*x1)
                fmax = float(torch.max(torch.abs(torch.stack([f0a, f1a], dim=-1))).cpu())

            print(
                f"[it={it}] TrE={tr_mu:.6f}±{tr_sem:.2e} "
                f"E0~{evals_real_sorted[0].item():.6f}, E1~{evals_real_sorted[1].item():.6f} "
                f"acc={acc:.2f} sig={sigma:.3f} good={frac_good*100:.1f}% det_med={det_med:.2e} fmax~{fmax:.2f}"
            )

    return models

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    _ = train_nes_vmc_mh_local()