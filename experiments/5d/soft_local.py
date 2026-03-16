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

def periodic_delta(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Minimal periodic displacement in (-pi, pi], relative to center c.
    """
    return torch.remainder(x - c + math.pi, TWO_PI) - math.pi

# ============================================================
# Hyperparameters (MH only, LOCAL but SOFT anchor penalty)
# ============================================================
# Sampling / MH
N_walkers = 4000
N_burn_in = 100
Sigma0 = 0.30
P_GLOBAL = 0.02      # global jump inside a wide window around x_loc
Delta = 0.60         # used for init + global jump scale (not a hard constraint anymore)

# Soft anchor strength (Gaussian well around x_loc):
# target pi(x) ∝ exp(2 f(x)) * exp(-kappa * ||x-x_loc||_per^2)
KAPPA = 3.0

# Model / Opt
Hidden_Dim = 256
Seed = 0
torch.manual_seed(Seed)

N_states = 2
N_steps  = 8400
lr       = 1e-2

# Warmup then adaptive alpha (block updates)
WARMUP = 50
BLOCK_STEPS = 50
OVLP_SUBSAMPLE = 4000
EPS_OVLP = 1e-8

# adaptive alpha rule (by overlap)
S_TRIGGER    = 5e-2
ALPHA_INIT   = 1.0
ALPHA_GROWTH = 1.5
ALPHA_DECAY  = 0.9
ALPHA_MAX    = 1e4
ALPHA_MIN    = 0.0

# Optional: drive phase g via imag local energy
BETA_IMAG = 1.0   # set 0.0 to disable

# Logging
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

# (optional) harmonic estimate print
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
# Soft anchor energy: d2 = sum_k (delta_per)^2
# ============================================================
@torch.no_grad()
def anchor_d2(x2, x3, x4, x5, x6, x_loc: torch.Tensor) -> torch.Tensor:
    d2 = periodic_delta(x2, x_loc[0])**2
    d2 = d2 + periodic_delta(x3, x_loc[1])**2
    d2 = d2 + periodic_delta(x4, x_loc[2])**2
    d2 = d2 + periodic_delta(x5, x_loc[3])**2
    d2 = d2 + periodic_delta(x6, x_loc[4])**2
    return d2

# ============================================================
# MH sampling (target ∝ exp(2f - kappa*d2))  [NO hard box]
# ============================================================
@torch.no_grad()
def initial_local(model: nn.Module, N: int, x_loc: torch.Tensor, delta: float):
    u = (torch.rand(N, 5, device=device, dtype=dtype) * 2.0 - 1.0) * delta
    X = x_loc[None, :] + u
    # map into the centered branch around x_loc (purely for numerical neatness)
    X = torch.stack([wrap_centered_1d(X[:, k], x_loc[k]) for k in range(5)], dim=1)
    x2, x3, x4, x5, x6 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    f = model.f_only(x2, x3, x4, x5, x6)
    d2 = anchor_d2(x2, x3, x4, x5, x6, x_loc)
    return (x2, x3, x4, x5, x6, f, d2)

@torch.no_grad()
def mh_chain_local_soft_anchor(
    model: nn.Module,
    state,
    x_loc: torch.Tensor,
    sigma: float,
    Nb: int,
    p_global: float,
    delta: float,
    kappa: float
):
    """
    Target log density: log pi(x) = 2 f(x) - kappa * d2(x)
    where d2 uses periodic minimal displacement from x_loc.
    """
    x2, x3, x4, x5, x6, f, d2 = state
    f = model.f_only(x2, x3, x4, x5, x6)
    d2 = anchor_d2(x2, x3, x4, x5, x6, x_loc)

    N = x2.shape[0]
    acc = 0

    # keep centered branch around x_loc (not a constraint)
    x2 = wrap_centered_1d(x2, x_loc[0])
    x3 = wrap_centered_1d(x3, x_loc[1])
    x4 = wrap_centered_1d(x4, x_loc[2])
    x5 = wrap_centered_1d(x5, x_loc[3])
    x6 = wrap_centered_1d(x6, x_loc[4])

    for _ in range(Nb):
        if (torch.rand((), device=device) < p_global):
            u = (torch.rand(N, 5, device=device, dtype=dtype) * 2.0 - 1.0) * delta
            Xp = x_loc[None, :] + u
            Xp = torch.stack([wrap_centered_1d(Xp[:, k], x_loc[k]) for k in range(5)], dim=1)
            nx2, nx3, nx4, nx5, nx6 = Xp[:, 0], Xp[:, 1], Xp[:, 2], Xp[:, 3], Xp[:, 4]
        else:
            nx2 = x2 + torch.randn_like(x2) * sigma
            nx3 = x3 + torch.randn_like(x3) * sigma
            nx4 = x4 + torch.randn_like(x4) * sigma
            nx5 = x5 + torch.randn_like(x5) * sigma
            nx6 = x6 + torch.randn_like(x6) * sigma

        # keep within the centered branch (optional)
        nx2 = wrap_centered_1d(nx2, x_loc[0])
        nx3 = wrap_centered_1d(nx3, x_loc[1])
        nx4 = wrap_centered_1d(nx4, x_loc[2])
        nx5 = wrap_centered_1d(nx5, x_loc[3])
        nx6 = wrap_centered_1d(nx6, x_loc[4])

        nf = model.f_only(nx2, nx3, nx4, nx5, nx6)
        nd2 = anchor_d2(nx2, nx3, nx4, nx5, nx6, x_loc)

        log_a = 2.0 * (nf - f) - float(kappa) * (nd2 - d2)
        accept = (torch.log(torch.rand_like(log_a)) < log_a)

        x2 = torch.where(accept, nx2, x2)
        x3 = torch.where(accept, nx3, x3)
        x4 = torch.where(accept, nx4, x4)
        x5 = torch.where(accept, nx5, x5)
        x6 = torch.where(accept, nx6, x6)
        f  = torch.where(accept, nf, f)
        d2 = torch.where(accept, nd2, d2)

        acc += int(accept.sum())

    return (x2, x3, x4, x5, x6, f, d2), acc / (Nb * N)

# ============================================================
# Local energy (5D)  [NOTE: still physical H; anchor only biases sampling]
# ============================================================
def local_energy_5D(x2, x3, x4, x5, x6, model: nn.Module):
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
    return E_real, E_imag, f, g

# ============================================================
# Standard VMC losses (NO reweight)
# ============================================================
def loss_real_standard(E_real: torch.Tensor, f: torch.Tensor):
    Er = E_real.detach()
    Ebar = Er.mean()
    return 2.0 * ((Er - Ebar) * f).mean()

def loss_imag_standard(E_imag: torch.Tensor, g: torch.Tensor):
    Ei = E_imag.detach()
    Ibar = Ei.mean()
    return 2.0 * ((Ei - Ibar) * g).mean()

def mean_var_sem(x: torch.Tensor):
    xd = x.detach()
    mu = xd.mean()
    var = (xd - mu).pow(2).mean()
    sem = torch.sqrt(var / (xd.numel() + 1e-30))
    return float(mu), float(var), float(sem)

# ============================================================
# Overlap estimator (NO reweight), gradient ONLY to model_i
# ============================================================
def _stable_complex_mean_exp(df: torch.Tensor, dg: torch.Tensor) -> torch.Tensor:
    amax = df.max().detach()
    z = torch.exp(df - amax).to(cdtype) * torch.exp(1j * dg.to(cdtype))
    return z.mean() * torch.exp(amax).to(cdtype)

def overlap_abs_symmetric_stable_oneway(model_i, model_j, x_i, x_j, eps=EPS_OVLP):
    x2i, x3i, x4i, x5i, x6i = x_i
    x2j, x3j, x4j, x5j, x6j = x_j

    # A: samples from i
    f_i_i, g_i_i = model_i(x2i, x3i, x4i, x5i, x6i)  # grad
    with torch.no_grad():
        f_j_i, g_j_i = model_j(x2i, x3i, x4i, x5i, x6i)
    A = _stable_complex_mean_exp((f_j_i - f_i_i).to(dtype), (g_j_i - g_i_i).to(dtype))

    # B: samples from j
    f_i_j, g_i_j = model_i(x2j, x3j, x4j, x5j, x6j)  # grad
    with torch.no_grad():
        f_j_j, g_j_j = model_j(x2j, x3j, x4j, x5j, x6j)
    B = _stable_complex_mean_exp((f_i_j - f_j_j).to(dtype), (g_i_j - g_j_j).to(dtype))

    Sabs = torch.sqrt(torch.abs(A) * torch.abs(B)).to(dtype)
    Sclamp = torch.clamp(Sabs, max=(1.0 - eps))
    pen = (1.0 / (1.0 - Sclamp)) - 1.0
    return Sabs, pen

# ============================================================
# alpha adaptive update (by overlap) - block wise
# ============================================================
def clamp_alpha(a: float) -> float:
    return float(max(ALPHA_MIN, min(ALPHA_MAX, a)))

def update_alpha(alpha: float, S01: float) -> float:
    if S01 >= S_TRIGGER:
        alpha = alpha * ALPHA_GROWTH
    else:
        alpha = alpha * ALPHA_DECAY
    return clamp_alpha(alpha)

# ============================================================
# Training (MH soft-anchored sampling; warmup then block alpha adapt)
# ============================================================
def train_vmc_mh_soft_anchor_adapt_alpha(steps=N_steps, eta=lr):
    assert N_states == 2, "This trainer is for N_states=2."

    models = [NN_Waffle_Complex_5D().to(device=device, dtype=dtype) for _ in range(N_states)]
    opts   = [torch.optim.Adam(m.parameters(), lr=eta) for m in models]

    mh_states = [initial_local(models[i], N_walkers, x_loc=x_loc, delta=Delta) for i in range(N_states)]
    sigmas    = [float(Sigma0) for _ in range(N_states)]

    alpha = 0.0
    last_S01 = float("nan")
    last_pen = float("nan")
    last_d2_0 = float("nan")
    last_d2_1 = float("nan")

    def subsample_pack(xpack, n=OVLP_SUBSAMPLE):
        if (n is None) or (xpack[0].shape[0] <= n):
            return xpack
        return tuple(t[:n] for t in xpack)

    for it in range(steps):
        if it == WARMUP:
            alpha = clamp_alpha(ALPHA_INIT)

        do_block_update = (it >= WARMUP) and ((it - WARMUP) % BLOCK_STEPS == 0)

        # --- sample (SOFT anchor) ---
        xs_pack = []
        accs = []
        d2_means = []

        for i in range(N_states):
            mh_states[i], acc = mh_chain_local_soft_anchor(
                models[i], mh_states[i],
                x_loc=x_loc, sigma=sigmas[i], Nb=N_burn_in,
                p_global=P_GLOBAL, delta=Delta,
                kappa=KAPPA
            )
            accs.append(float(acc))
            x2, x3, x4, x5, x6, _, d2 = mh_states[i]
            xs_pack.append((x2, x3, x4, x5, x6))
            d2_means.append(float(d2.detach().mean().cpu()))

        last_d2_0, last_d2_1 = d2_means[0], d2_means[1]

        # --- overlap + alpha update at block boundaries ---
        x0s = subsample_pack(xs_pack[0], OVLP_SUBSAMPLE)
        x1s = subsample_pack(xs_pack[1], OVLP_SUBSAMPLE)

        S01_t, pen01_t = overlap_abs_symmetric_stable_oneway(models[1], models[0], x1s, x0s)
        last_S01 = float(S01_t.detach().cpu())
        last_pen = float(pen01_t.detach().cpu())

        if do_block_update:
            alpha = update_alpha(alpha, last_S01)

        # --- losses + stats ---
        total_loss = torch.zeros((), device=device, dtype=dtype)
        stats = []

        for i in range(N_states):
            x2, x3, x4, x5, x6 = xs_pack[i]
            E_real, E_imag, f, g = local_energy_5D(x2, x3, x4, x5, x6, models[i])

            em, ev, es = mean_var_sem(E_real)
            im, iv, isem = mean_var_sem(E_imag)

            li = loss_real_standard(E_real, f)
            if BETA_IMAG != 0.0:
                li = li + float(BETA_IMAG) * loss_imag_standard(E_imag, g)

            # penalty only for state1 after warmup
            if (i == 1) and (it >= WARMUP) and (alpha > 0.0):
                li = li + float(alpha) * pen01_t

            total_loss = total_loss + li
            stats.append((em, ev, es, im, iv, isem))

        # --- optimize ---
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        total_loss.backward()
        for opt in opts:
            opt.step()

        # --- sigma adapt ---
        for i in range(N_states):
            if accs[i] < 0.4:
                sigmas[i] *= 0.8
            elif accs[i] > 0.6:
                sigmas[i] *= 1.2
            sigmas[i] = float(max(0.02, min(0.8, sigmas[i])))

        # --- logging ---
        if it == 0 or (it % PRINT_EVERY) == 0:
            phase = "warmup" if it < WARMUP else ("block-update" if do_block_update else "block-fixed")
            (e0, v0, se0, im0, iv0, ise0) = stats[0]
            (e1, v1, se1, im1, iv1, ise1) = stats[1]
            print(
                f"[it={it}] phase={phase} alpha={alpha:.2e} |S01|={last_S01:.3e} pen={last_pen:.3e} "
                f"KAPPA={KAPPA:.2f} d2mean=[{last_d2_0:.2e},{last_d2_1:.2e}] "
                f"p_global={P_GLOBAL:.2f} Delta={Delta:.2f} beta_imag={BETA_IMAG:.1e} | "
                f"S0:E={e0:.4f},Var={v0:.3e},SEM={se0:.1e},Im={im0:+.2e},VarI={iv0:.3e},SEMI={ise0:.1e},acc={accs[0]:.2f},sig={sigmas[0]:.3f} | "
                f"S1:E={e1:.4f},Var={v1:.3e},SEM={se1:.1e},Im={im1:+.2e},VarI={iv1:.3e},SEMI={ise1:.1e},acc={accs[1]:.2f},sig={sigmas[1]:.3f}"
            )

    return models

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    _ = train_vmc_mh_soft_anchor_adapt_alpha()