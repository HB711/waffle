import math
import torch
import torch.nn as nn

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
# Hyperparameters (UNIFORM sampling; direct GD on stabilized loss)
# ============================================================
# Sampling
N_walkers = 2000
Delta = 1.0  # local box half-width around x_loc

# Training
N_states = 2
N_steps  = 8400
lr       = 5e-4   # <<< MUCH smaller to avoid blow-up

Hidden_Dim = 256
Seed = 0
torch.manual_seed(Seed)

# Warmup then adaptive alpha (block updates)
WARMUP = 1050
BLOCK_STEPS = 50
OVLP_SUBSAMPLE = 2000
EPS_OVLP = 1e-12

# adaptive rule
S_TRIGGER    = 5e-2
ALPHA_INIT   = 100.0
ALPHA_GROWTH = 1.0
ALPHA_DECAY  = 0.9
ALPHA_MAX    = 5e3
ALPHA_MIN    = 0.0

# Logging
PRINT_EVERY = 10

# Overlap stability
CLIP_DF = 40.0  # clamp df=f1-f0 when computing overlap weights only

# Stabilizers (IMPORTANT)
GRAD_REG_GAMMA = 1e-3     # penalize |∇f|^2 + |∇g|^2 to prevent kinetic blow-up
CLIP_GRAD_NORM = 1.0      # global grad norm clip

# Optional imaginary penalty (usually keep 0 for now)
BETA_IMAG_L2 = 0.0

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
# Choose ONE 6D minimum -> 5D x_loc (x2..x6)
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
# Potential (6D -> 5D embedding)
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

    def f_only(self, x2, x3, x4, x5, x6):
        x = periodic_emb(x2, x3, x4, x5, x6)
        return self.out_f(self.backbone_f(x)).squeeze(-1)

# ============================================================
# UNIFORM sampling in local box
# ============================================================
@torch.no_grad()
def uniform_box_samples(N: int, x_loc: torch.Tensor, delta: float):
    u = (torch.rand(N, 5, device=device, dtype=dtype) * 2.0 - 1.0) * delta
    X = x_loc[None, :] + u
    X = torch.stack([wrap_centered_1d(X[:, k], x_loc[k]) for k in range(5)], dim=1)
    return (X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4])

# ============================================================
# Local energy (5D) + gradients (dfs,dgs) for regularization
# ============================================================
def local_energy_5D_with_grads(x2, x3, x4, x5, x6, model: nn.Module):
    xs = [
        x2.detach().requires_grad_(True),
        x3.detach().requires_grad_(True),
        x4.detach().requires_grad_(True),
        x5.detach().requires_grad_(True),
        x6.detach().requires_grad_(True),
    ]
    f, g = model(xs[0], xs[1], xs[2], xs[3], xs[4])
    ones = torch.ones_like(f)

    dfs = torch.autograd.grad(f, xs, grad_outputs=ones, create_graph=True)
    dgs = torch.autograd.grad(g, xs, grad_outputs=ones, create_graph=True)

    d2fs, d2gs = [], []
    for i in range(5):
        d2fi = torch.autograd.grad(dfs[i], xs[i], grad_outputs=torch.ones_like(dfs[i]), create_graph=True)[0]
        d2gi = torch.autograd.grad(dgs[i], xs[i], grad_outputs=torch.ones_like(dgs[i]), create_graph=True)[0]
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
    return E_real, E_imag, f, g, dfs, dgs

# ============================================================
# Stabilized loss: variance-only + grad regularization + optional imag penalty
# ============================================================
def loss_stable_var_plus_reg(E_real, E_imag, dfs, dgs, gamma=GRAD_REG_GAMMA, beta_imag=BETA_IMAG_L2):
    Er = E_real
    mu = Er.mean().detach()
    var = ((Er - mu) ** 2).mean()

    reg = 0.0
    for i in range(5):
        reg = reg + (dfs[i]**2).mean() + (dgs[i]**2).mean()
    reg = float(gamma) * reg

    loss = var + reg
    if beta_imag != 0.0:
        loss = loss + float(beta_imag) * (E_imag**2).mean()
    return loss, var.detach(), reg.detach()

def mean_var_sem(x: torch.Tensor):
    xd = x.detach()
    mu = xd.mean()
    var = (xd - mu).pow(2).mean()
    sem = torch.sqrt(var / (xd.numel() + 1e-30))
    return float(mu), float(var), float(sem)

# ============================================================
# Overlap penalty (explicit/stable one-way |S|^2 grad estimator)
# ============================================================
@torch.no_grad()
def _stable_mean_exp_real(a: torch.Tensor) -> torch.Tensor:
    amax = a.max()
    return torch.exp(a - amax).mean() * torch.exp(amax)

@torch.no_grad()
def _stable_mean_exp_complex(df: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    amax = df.max()
    z = torch.exp(df - amax).to(cdtype) * torch.exp(1j * phase.to(cdtype))
    return z.mean() * torch.exp(amax).to(cdtype)

def penalty_obj_absS2_oneway_explicit(
    model_i,  # trainable (state1)
    model_j,  # frozen   (state0)
    x_i,      # samples from i  (for norm term)
    x_j,      # samples from j  (for cross term)
    alpha: float,
    eps: float = EPS_OVLP,
):
    if alpha <= 0.0:
        z = torch.zeros((), device=device, dtype=dtype)
        return z, z, z

    x2i, x3i, x4i, x5i, x6i = x_i
    x2j, x3j, x4j, x5j, x6j = x_j

    f_i_j, g_i_j = model_i(x2j, x3j, x4j, x5j, x6j)
    f_i_i, g_i_i = model_i(x2i, x3i, x4i, x5i, x6i)

    with torch.no_grad():
        f_j_j, g_j_j = model_j(x2j, x3j, x4j, x5j, x6j)
        f_j_i, g_j_i = model_j(x2i, x3i, x4i, x5i, x6i)

    with torch.no_grad():
        df = (f_i_j.detach() - f_j_j).clamp(-CLIP_DF, CLIP_DF)
        dg = (g_i_j.detach() - g_j_j)

        r_mean = _stable_mean_exp_complex(df, phase=(-dg))
        denom2 = _stable_mean_exp_real(2.0 * df)
        denom = torch.sqrt(denom2 + eps).to(dtype)

        S = (r_mean / denom.to(cdtype))
        S_abs2 = (S.real**2 + S.imag**2).to(dtype)
        S_abs = torch.sqrt(S_abs2 + eps).to(dtype)

        amax = df.max()
        z = torch.exp(df - amax).to(cdtype) * torch.exp((-1j) * dg.to(cdtype))
        scale = (torch.exp(amax).to(dtype) / denom).to(dtype)
        q = S.conj() * z

        w_f = (scale * q.real).to(dtype)
        w_g = (scale * q.imag).to(dtype)

        norm_coeff = (-S_abs2).to(dtype)

    cross_obj = (w_f * f_i_j + w_g * g_i_j).mean()
    norm_obj = norm_coeff * f_i_i.mean()
    pen_obj = 2.0 * float(alpha) * (cross_obj + norm_obj)

    return pen_obj, S_abs2.detach(), S_abs.detach()

# ============================================================
# alpha adaptive update
# ============================================================
def clamp_alpha(a: float) -> float:
    return float(max(ALPHA_MIN, min(ALPHA_MAX, a)))

def update_alpha(alpha: float, S01_abs: float) -> float:
    if S01_abs >= S_TRIGGER:
        alpha = alpha * ALPHA_GROWTH
    else:
        alpha = alpha * ALPHA_DECAY
    return clamp_alpha(alpha)

# ============================================================
# Grad clip + SGD step
# ============================================================
def clip_grads(grads, max_norm=1.0):
    total = None
    for g in grads:
        if g is None:
            continue
        s = (g.detach()**2).sum()
        total = s if total is None else (total + s)
    if total is None:
        return grads
    total = torch.sqrt(total + 1e-30)
    scale = min(1.0, float(max_norm) / float(total.cpu()))
    if scale >= 1.0:
        return grads
    return tuple(None if g is None else (g * scale) for g in grads)

def sgd_step(params, grads, lr: float):
    with torch.no_grad():
        for p, g in zip(params, grads):
            if g is None:
                continue
            p.add_(g, alpha=-lr)

# ============================================================
# Training: UNIFORM sampling + stabilized loss
#   - state0: var(E) + gamma*|∇|^2
#   - state1: same + overlap penalty after warmup
# ============================================================
def train_uniform_gd_stable(steps=N_steps, eta=lr):
    assert N_states == 2, "This trainer is for N_states=2."

    models = [NN_Waffle_Complex_5D().to(device=device, dtype=dtype) for _ in range(N_states)]

    alpha = 0.0
    last_S01 = float("nan")
    last_S01_2 = float("nan")

    def subsample_pack(xpack, n=OVLP_SUBSAMPLE):
        if (n is None) or (xpack[0].shape[0] <= n):
            return xpack
        return tuple(t[:n] for t in xpack)

    for it in range(steps):

        if it == WARMUP:
            alpha = clamp_alpha(ALPHA_INIT)
        do_block_update = (it >= WARMUP) and ((it - WARMUP) % BLOCK_STEPS == 0)

        # -------- uniform samples --------
        xs_pack = [uniform_box_samples(N_walkers, x_loc=x_loc, delta=Delta) for _ in range(N_states)]

        # -------- overlap magnitude for alpha update (use state0 samples) --------
        x0s = subsample_pack(xs_pack[0], OVLP_SUBSAMPLE)
        x1s = subsample_pack(xs_pack[1], OVLP_SUBSAMPLE)

        with torch.no_grad():
            x2j, x3j, x4j, x5j, x6j = x0s
            f1_j, g1_j = models[1](x2j, x3j, x4j, x5j, x6j)
            f0_j, g0_j = models[0](x2j, x3j, x4j, x5j, x6j)
            df = (f1_j - f0_j).clamp(-CLIP_DF, CLIP_DF)
            dg = (g1_j - g0_j)

            r_mean = _stable_mean_exp_complex(df, phase=(-dg))
            denom2 = _stable_mean_exp_real(2.0 * df)
            denom = math.sqrt(float(denom2.cpu()) + EPS_OVLP)
            S_complex = r_mean / (torch.tensor(denom, device=device, dtype=dtype).to(cdtype))
            S_abs2 = float((S_complex.real**2 + S_complex.imag**2).cpu())
            S_abs = math.sqrt(max(S_abs2, 0.0))
            last_S01 = S_abs
            last_S01_2 = S_abs2

        if do_block_update and it >= WARMUP:
            alpha = update_alpha(alpha, last_S01)

        # -------- state0 loss --------
        x2, x3, x4, x5, x6 = xs_pack[0]
        E_real0, E_imag0, f0, g0, dfs0, dgs0 = local_energy_5D_with_grads(x2, x3, x4, x5, x6, models[0])
        loss0, var0, reg0 = loss_stable_var_plus_reg(E_real0, E_imag0, dfs0, dgs0)

        params0 = [p for p in models[0].parameters() if p.requires_grad]
        grads0 = torch.autograd.grad(loss0, params0, retain_graph=False, create_graph=False, allow_unused=True)
        grads0 = clip_grads(grads0, CLIP_GRAD_NORM)

        # -------- state1 loss (+ penalty) --------
        x2, x3, x4, x5, x6 = xs_pack[1]
        E_real1, E_imag1, f1, g1, dfs1, dgs1 = local_energy_5D_with_grads(x2, x3, x4, x5, x6, models[1])
        loss1, var1, reg1 = loss_stable_var_plus_reg(E_real1, E_imag1, dfs1, dgs1)

        pen_obj = torch.zeros((), device=device, dtype=dtype)
        if (it >= WARMUP) and (alpha > 0.0):
            pen_obj, _Sabs2_t, _Sabs_t = penalty_obj_absS2_oneway_explicit(
                model_i=models[1],
                model_j=models[0],
                x_i=x1s,
                x_j=x0s,
                alpha=alpha,
            )

        loss1_total = loss1 + pen_obj

        params1 = [p for p in models[1].parameters() if p.requires_grad]
        grads1 = torch.autograd.grad(loss1_total, params1, retain_graph=False, create_graph=False, allow_unused=True)
        grads1 = clip_grads(grads1, CLIP_GRAD_NORM)

        # -------- GD step --------
        sgd_step(params0, grads0, eta)
        sgd_step(params1, grads1, eta)

        # -------- logging --------
        if it == 0 or (it % PRINT_EVERY) == 0:
            phase = "warmup" if it < WARMUP else ("block-update" if do_block_update else "block-fixed")

            e0m, e0v, e0se = mean_var_sem(E_real0)
            i0m, i0v, i0se = mean_var_sem(E_imag0)
            e1m, e1v, e1se = mean_var_sem(E_real1)
            i1m, i1v, i1se = mean_var_sem(E_imag1)

            print(
                f"[it={it}] phase={phase} alpha={alpha:.2e} |S01|={last_S01:.3e} |S01|^2={last_S01_2:.3e} "
                f"Delta={Delta:.2f} lr={eta:.1e} gamma={GRAD_REG_GAMMA:.1e} clip={CLIP_GRAD_NORM:.1f} | "
                f"S0:E={e0m:.4f},VarE={e0v:.3e},SEM={e0se:.1e},Im={i0m:+.2e},VarI={i0v:.3e},SEMI={i0se:.1e},var_loss={float(var0.cpu()):.3e},reg={float(reg0.cpu()):.3e} | "
                f"S1:E={e1m:.4f},VarE={e1v:.3e},SEM={e1se:.1e},Im={i1m:+.2e},VarI={i1v:.3e},SEMI={i1se:.1e},var_loss={float(var1.cpu()):.3e},reg={float(reg1.cpu()):.3e} | "
                f"loss0={float(loss0.detach().cpu()):+.3e} loss1={float(loss1.detach().cpu()):+.3e} pen_obj={float(pen_obj.detach().cpu()):+.3e}"
            )

    return models

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    _ = train_uniform_gd_stable()