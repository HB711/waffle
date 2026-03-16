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
# Hyperparameters
# ============================================================
# Sampling / MH
N_walkers = 3000
N_burn_in = 100
Sigma0 = 0.30
P_GLOBAL = 0.02
Delta = 0.8 # local box half-width

# Model
Hidden_Dim = 256
Seed = 0
torch.manual_seed(Seed)

# Explicit GD
lr = 1e-2

# Stage A: train state0 only
STEPS_STATE0 = 250
EARLYSTOP_WIN = 200          # moving average window
EARLYSTOP_TOL = 2e-4         # stop if |E_ma - E_ma_prev| < tol
EARLYSTOP_PATIENCE = 5       # require patience times

# Stage B: train state1 with orthogonality penalty vs frozen state0
STEPS_STATE1 = 28400
WARMUP_STATE1 = 100          # stage1 warmup before enabling alpha
BLOCK_STEPS = 50
OVLP_SUBSAMPLE = 3000
EPS_OVLP = 1e-12

# adaptive rule for alpha
S_TRIGGER    = 5e-1
ALPHA_INIT   = 1.0
ALPHA_GROWTH = 1.2
ALPHA_DECAY  = 1.0
ALPHA_MAX    = 5e3
ALPHA_MIN    = 0.0

# Optional: drive phase g via imag local energy
BETA_IMAG = 1.0   # set 0.0 to disable

# Logging
PRINT_EVERY = 10

# Overlap stability
CLIP_DF = 40.0  # clamp df=f1-f0 when computing overlap weights only

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
# MH sampling (target ∝ exp(2f)) but LOCAL BOX
# ============================================================
@torch.no_grad()
def initial_local(model: nn.Module, N: int, x_loc: torch.Tensor, delta: float):
    u = (torch.rand(N, 5, device=device, dtype=dtype) * 2.0 - 1.0) * delta
    X = x_loc[None, :] + u
    X = torch.stack([wrap_centered_1d(X[:, k], x_loc[k]) for k in range(5)], dim=1)
    x2, x3, x4, x5, x6 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    f = model.f_only(x2, x3, x4, x5, x6)
    return (x2, x3, x4, x5, x6, f)

@torch.no_grad()
def mh_chain_local(model: nn.Module, state, x_loc: torch.Tensor, sigma: float, Nb: int, p_global: float, delta: float):
    x2, x3, x4, x5, x6, f = state
    f = model.f_only(x2, x3, x4, x5, x6)
    N = x2.shape[0]
    acc = 0

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

        in_box = (
            ((nx2 - x_loc[0]).abs() <= delta) &
            ((nx3 - x_loc[1]).abs() <= delta) &
            ((nx4 - x_loc[2]).abs() <= delta) &
            ((nx5 - x_loc[3]).abs() <= delta) &
            ((nx6 - x_loc[4]).abs() <= delta)
        )

        nf = model.f_only(nx2, nx3, nx4, nx5, nx6)
        log_a = 2.0 * (nf - f)
        accept = in_box & (torch.log(torch.rand_like(log_a)) < log_a)

        x2 = torch.where(accept, nx2, x2)
        x3 = torch.where(accept, nx3, x3)
        x4 = torch.where(accept, nx4, x4)
        x5 = torch.where(accept, nx5, x5)
        x6 = torch.where(accept, nx6, x6)
        f  = torch.where(accept, nf, f)

        acc += int(accept.sum())

    return (x2, x3, x4, x5, x6, f), acc / (Nb * N)

# ============================================================
# Local energy (5D)
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
# Explicit-gradient VMC energy objectives (linear in f,g)
# ============================================================
def energy_obj_real(E_real: torch.Tensor, f: torch.Tensor):
    Er = E_real.detach()
    Ebar = Er.mean()
    return 2.0 * ((Er - Ebar) * f).mean()

def energy_obj_imag(E_imag: torch.Tensor, g: torch.Tensor):
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
# Overlap helpers (stable means)
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

# ============================================================
# Penalty objective: alpha * |S|^2, explicit gradient, one-way (model0 frozen)
#   - Returns pen_obj whose grad == grad(alpha*|S|^2) w.r.t model1 params
# ============================================================
def penalty_obj_absS2_oneway_explicit(
    model_i,  # trainable (state1)
    model_j,  # frozen   (state0)
    x_i,      # samples from i (p_i) for norm term
    x_j,      # samples from j (p_j) for cross term
    alpha: float,
    eps: float = EPS_OVLP,
):
    if alpha <= 0.0:
        z = torch.zeros((), device=device, dtype=dtype)
        return z, z, z

    x2i, x3i, x4i, x5i, x6i = x_i  # from p_i
    x2j, x3j, x4j, x5j, x6j = x_j  # from p_j

    # Trainable model_i evaluations (need grad)
    f_i_j, g_i_j = model_i(x2j, x3j, x4j, x5j, x6j)  # on x~p_j
    f_i_i, g_i_i = model_i(x2i, x3i, x4i, x5i, x6i)  # on x~p_i

    # Frozen model_j evaluations
    with torch.no_grad():
        f_j_j, g_j_j = model_j(x2j, x3j, x4j, x5j, x6j)
        # (f_j_i,g_j_i) not needed in this one-way explicit formula

    # Build STOP-GRAD weights
    with torch.no_grad():
        # For x~p_j (here j=state0): O_ij^* = exp(df) * exp(-i dg)
        df = (f_i_j.detach() - f_j_j).clamp(-CLIP_DF, CLIP_DF)
        dg = (g_i_j.detach() - g_j_j)

        # S estimated using p_j only: S = E_j[O_ij^*] / sqrt(E_j[|O_ij|^2])
        r_mean = _stable_mean_exp_complex(df, phase=(-dg))
        denom2 = _stable_mean_exp_real(2.0 * df)
        denom = torch.sqrt(denom2 + eps).to(dtype)
        S = (r_mean / denom.to(cdtype))  # complex

        S_abs2 = (S.real**2 + S.imag**2).to(dtype)
        S_abs = torch.sqrt(S_abs2 + eps).to(dtype)

        # Per-sample stable r_s and weights:
        amax = df.max()
        z = torch.exp(df - amax).to(cdtype) * torch.exp((-1j) * dg.to(cdtype))  # r_s / exp(amax)
        scale = (torch.exp(amax).to(dtype) / denom).to(dtype)                  # real
        q = S.conj() * z                                                      # complex, per-sample (missing scale)

        # weights for f_i(x_j), g_i(x_j)
        w_f = (scale * q.real).to(dtype)
        w_g = (scale * q.imag).to(dtype)

        # normalization coefficient: -|S|^2 multiplying E_i[∇f]  -> gradient of mean(f_i_i)
        norm_coeff = (-S_abs2).to(dtype)

    # Linear objective with correct explicit gradient:
    cross_obj = (w_f * f_i_j + w_g * g_i_j).mean()
    norm_obj  = norm_coeff * f_i_i.mean()
    pen_obj = 2.0 * float(alpha) * (cross_obj + norm_obj)

    return pen_obj, S_abs2.detach(), S_abs.detach()

# ============================================================
# alpha adaptive update (by overlap |S|)
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
# Explicit GD update helper
# ============================================================
def sgd_step(params, grads, lr: float):
    with torch.no_grad():
        for p, g in zip(params, grads):
            if g is None:
                continue
            p.add_(g, alpha=-lr)

# ============================================================
# Utilities
# ============================================================
def _subsample_pack(xpack, n=OVLP_SUBSAMPLE):
    if (n is None) or (xpack[0].shape[0] <= n):
        return xpack
    return tuple(t[:n] for t in xpack)

@torch.no_grad()
def estimate_S_abs_only(model1, model0, x0_pack, eps=EPS_OVLP):
    # estimate |S10| using x~p0 only: S = E0[O10^*]/sqrt(E0[|O10|^2])
    x2, x3, x4, x5, x6 = x0_pack
    f1, g1 = model1(x2, x3, x4, x5, x6)
    f0, g0 = model0(x2, x3, x4, x5, x6)
    df = (f1 - f0).clamp(-CLIP_DF, CLIP_DF)
    dg = (g1 - g0)
    r_mean = _stable_mean_exp_complex(df, phase=(-dg))
    denom2 = _stable_mean_exp_real(2.0 * df)
    denom = torch.sqrt(denom2 + eps).to(dtype)
    S = r_mean / denom.to(cdtype)
    S_abs2 = float((S.real**2 + S.imag**2).cpu())
    S_abs = math.sqrt(max(S_abs2, 0.0))
    return S_abs, S_abs2

# ============================================================
# Stage A: train state0 only (energy)
# ============================================================
def train_state0_energy_only(model0: nn.Module, mh_state0, steps=STEPS_STATE0, eta=lr):
    sigma0 = float(Sigma0)

    # early stop trackers
    E_hist = []
    ma_prev = None
    patience = 0

    for it in range(steps):
        mh_state0, acc = mh_chain_local(
            model0, mh_state0,
            x_loc=x_loc, sigma=sigma0, Nb=N_burn_in,
            p_global=P_GLOBAL, delta=Delta
        )
        x2, x3, x4, x5, x6, _ = mh_state0

        E_real, E_imag, f, g = local_energy_5D(x2, x3, x4, x5, x6, model0)

        obj = energy_obj_real(E_real, f)
        if BETA_IMAG != 0.0:
            obj = obj + float(BETA_IMAG) * energy_obj_imag(E_imag, g)

        params0 = [p for p in model0.parameters() if p.requires_grad]
        grads0 = torch.autograd.grad(obj, params0, retain_graph=False, create_graph=False, allow_unused=True)
        sgd_step(params0, grads0, eta)

        # sigma adapt
        if acc < 0.4:
            sigma0 *= 0.8
        elif acc > 0.6:
            sigma0 *= 1.2
        sigma0 = float(max(0.02, min(0.8, sigma0)))

        # stats
        e0m, e0v, e0se = mean_var_sem(E_real)
        i0m, i0v, i0se = mean_var_sem(E_imag)
        E_hist.append(e0m)

        # early stop check (moving average)
        if len(E_hist) >= 2 * EARLYSTOP_WIN:
            ma = sum(E_hist[-EARLYSTOP_WIN:]) / EARLYSTOP_WIN
            ma_old = sum(E_hist[-2*EARLYSTOP_WIN:-EARLYSTOP_WIN]) / EARLYSTOP_WIN
            if abs(ma - ma_old) < EARLYSTOP_TOL:
                patience += 1
            else:
                patience = 0
            if patience >= EARLYSTOP_PATIENCE:
                print(f"[STATE0 early-stop] it={it} | ΔE_ma={abs(ma-ma_old):.3e} < {EARLYSTOP_TOL} (patience={patience})")
                break

        if it == 0 or (it % PRINT_EVERY) == 0:
            print(
                f"[STATE0 it={it}] "
                f"E={e0m:.6f},Var={e0v:.3e},SEM={e0se:.1e},Im={i0m:+.2e},VarI={i0v:.3e},SEMI={i0se:.1e} "
                f"acc={acc:.2f},sig={sigma0:.3f} obj={float(obj.detach().cpu()):+.3e}"
            )

    return model0, mh_state0, sigma0

# ============================================================
# Stage B: train state1 with orthogonality penalty vs frozen state0
# ============================================================
def train_state1_with_orth(model0: nn.Module, mh_state0, sigma0: float, steps=STEPS_STATE1, eta=lr):
    # freeze model0
    for p in model0.parameters():
        p.requires_grad_(False)
    model0.eval()

    model1 = NN_Waffle_Complex_5D().to(device=device, dtype=dtype)
    sigma1 = float(Sigma0)
    mh_state1 = initial_local(model1, N_walkers, x_loc=x_loc, delta=Delta)

    alpha = 0.0
    last_S = float("nan")
    last_S2 = float("nan")

    for it in range(steps):
        # alpha schedule
        if it == WARMUP_STATE1:
            alpha = clamp_alpha(ALPHA_INIT)
        do_block_update = (it >= WARMUP_STATE1) and ((it - WARMUP_STATE1) % BLOCK_STEPS == 0)

        # sample state0 and state1
        mh_state0, acc0 = mh_chain_local(
            model0, mh_state0,
            x_loc=x_loc, sigma=sigma0, Nb=N_burn_in,
            p_global=P_GLOBAL, delta=Delta
        )
        mh_state1, acc1 = mh_chain_local(
            model1, mh_state1,
            x_loc=x_loc, sigma=sigma1, Nb=N_burn_in,
            p_global=P_GLOBAL, delta=Delta
        )

        x2_0, x3_0, x4_0, x5_0, x6_0, _ = mh_state0
        x2_1, x3_1, x4_1, x5_1, x6_1, _ = mh_state1
        x0_pack = (x2_0, x3_0, x4_0, x5_0, x6_0)
        x1_pack = (x2_1, x3_1, x4_1, x5_1, x6_1)

        x0s = _subsample_pack(x0_pack, OVLP_SUBSAMPLE)
        x1s = _subsample_pack(x1_pack, OVLP_SUBSAMPLE)

        # estimate |S| for logging + alpha update
        with torch.no_grad():
            last_S, last_S2 = estimate_S_abs_only(model1, model0, x0s)

        if do_block_update and it >= WARMUP_STATE1:
            alpha = update_alpha(alpha, last_S)

        # energy objective for state1
        E_real1, E_imag1, f1, g1 = local_energy_5D(x2_1, x3_1, x4_1, x5_1, x6_1, model1)
        obj1 = energy_obj_real(E_real1, f1)
        if BETA_IMAG != 0.0:
            obj1 = obj1 + float(BETA_IMAG) * energy_obj_imag(E_imag1, g1)

        # penalty objective (only after warmup)
        pen_obj = torch.zeros((), device=device, dtype=dtype)
        if (it >= WARMUP_STATE1) and (alpha > 0.0):
            pen_obj, Sabs2_t, Sabs_t = penalty_obj_absS2_oneway_explicit(
                model_i=model1,
                model_j=model0,
                x_i=x1s,   # p1 samples (norm term)
                x_j=x0s,   # p0 samples (cross term)
                alpha=alpha,
            )

        total_obj = obj1 + pen_obj

        params1 = [p for p in model1.parameters() if p.requires_grad]
        grads1 = torch.autograd.grad(total_obj, params1, retain_graph=False, create_graph=False, allow_unused=True)
        sgd_step(params1, grads1, eta)

        # sigma adapt
        if acc1 < 0.4:
            sigma1 *= 0.8
        elif acc1 > 0.6:
            sigma1 *= 1.2
        sigma1 = float(max(0.02, min(0.8, sigma1)))

        # (optional) keep sigma0 adapting too (even though frozen, helps sampling)
        if acc0 < 0.4:
            sigma0 *= 0.8
        elif acc0 > 0.6:
            sigma0 *= 1.2
        sigma0 = float(max(0.02, min(0.8, sigma0)))

        if it == 0 or (it % PRINT_EVERY) == 0:
            phase = "warmup" if it < WARMUP_STATE1 else ("block-update" if do_block_update else "block-fixed")
            e1m, e1v, e1se = mean_var_sem(E_real1)
            i1m, i1v, i1se = mean_var_sem(E_imag1)
            print(
                f"[STATE1 it={it}] phase={phase} alpha={alpha:.2e} |S01|={last_S:.3e} |S01|^2={last_S2:.3e} "
                f"E1={e1m:.6f},Var={e1v:.3e},SEM={e1se:.1e},Im={i1m:+.2e},VarI={i1v:.3e},SEMI={i1se:.1e} "
                f"acc0={acc0:.2f},sig0={sigma0:.3f} acc1={acc1:.2f},sig1={sigma1:.3f} "
                f"objE={float(obj1.detach().cpu()):+.3e} objPen={float(pen_obj.detach().cpu()):+.3e}"
            )

    return model0, model1

# ============================================================
# Main: Stage A then Stage B
# ============================================================
if __name__ == "__main__":
    # init state0
    model0 = NN_Waffle_Complex_5D().to(device=device, dtype=dtype)
    mh_state0 = initial_local(model0, N_walkers, x_loc=x_loc, delta=Delta)

    print("\n===== Stage A: train state0 (energy only) =====")
    model0, mh_state0, sigma0 = train_state0_energy_only(model0, mh_state0, steps=STEPS_STATE0, eta=lr)

    print("\n===== Stage B: train state1 (energy + alpha*|S01|^2 vs frozen state0) =====")
    model0, model1 = train_state1_with_orth(model0, mh_state0, sigma0, steps=STEPS_STATE1, eta=lr)