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

def wrap_pi_torch(x: torch.Tensor) -> torch.Tensor:
    # (-pi, pi]
    return x - TWO_PI * torch.floor((x + math.pi) / TWO_PI)

def wrap_centered_1d(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    # (c-pi, c+pi]
    return c + (torch.remainder(x - c + math.pi, TWO_PI) - math.pi)

# ============================================================
# Hyperparameters (5D, tempered MH = scheme A, local sampling)
# ============================================================
# Sampling / MH
N_walkers = 2000
N_burn_in = 100
Sigma0 = 0.30

# Tempering: target pi_beta ∝ |psi|^(2 beta)
BETA_START = 0.70
BETA_END   = 1.00
BETA_ANNEAL_STEPS = 0     # 0 => fixed at BETA_START

# IMPORTANT: "global jump" now means "global within the local box"
P_GLOBAL = 0.02

CLIP_LOGW = 40.0

# Local box half-width around x_loc
Delta = 0.40

# Model / Opt
Hidden_Dim = 256
Seed = 0
torch.manual_seed(Seed)

N_states = 4
N_steps  = 8400
lr       = 1e-2

# Orthogonality alpha_ij schedule: warmup -> ramp -> adaptive
WARMUP = 150
RAMP_TARGET = 200.0
RAMP_STEPS  = 600
RAMP_EVERY  = 20

S_TRIGGER   = 1e-3
ALPHA_GROWTH = 1.5
ALPHA_DECAY  = 0.9
ALPHA_MAX    = 1e8
ALPHA_MIN    = 0.0

# Overlap / printing
OVLP_EVERY = 20
OVLP_SUBSAMPLE = 2000
EPS_OVLP = 1e-8

# Optional: drive phase g via imag local energy
BETA_IMAG = 1.0   # set 0.0 to disable; try 0.1 if unstable

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

# FIX: alpha = -2pi/3 and use SAME sign structure as your earlier 6D code
ALPHA = -2.0 * math.pi / 3.0

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
print("E5[0] =", float(E5[0]))

# ============================================================
# Choose ONE 6D minimum (theta,phi), map -> 5D x_loc (x2..x6)
# ============================================================
theta_phi_min_6 = torch.tensor(
    [0.0, 0.0, 2.0*math.pi/3.0,  math.pi/6.0, math.pi/6.0, -math.pi/2.0],
    device=device, dtype=dtype
)

@torch.no_grad()
def theta_phi_to_xloc(Phi6: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    # gauge-fix: remove common shift (potential depends only on differences)
    Phi6 = Phi6.clone()
    Phi6 = Phi6 - Phi6.mean()
    # Convention: Phi = Phi_tilde @ U  => Phi_tilde = Phi @ U.T (U orthogonal)
    Phi_tilde = Phi6 @ U.T
    return Phi_tilde[1:].contiguous()

x_loc = theta_phi_to_xloc(theta_phi_min_6, U)  # (5,)
print("x_loc (x2..x6) =", x_loc.detach().cpu().numpy())

# ============================================================
# Potential (6D -> 5D embedding)
# ============================================================
def potential6D(t1, t2, t3, p1, p2, p3):
    alpha = ALPHA
    term_phi1 = -2 * torch.cos(t1 - p1) - 2 * torch.cos(t2 - p1) - 2 * torch.cos(t3 - p1)
    term_phi2 = -2 * torch.cos(t1 - p2) - 2 * torch.cos(t2 - p2 - alpha) - 2 * torch.cos(t3 - p2 + alpha)
    term_phi3 = -2 * torch.cos(t1 - p3) - 2 * torch.cos(t2 - p3 + alpha) - 2 * torch.cos(t3 - p3 - alpha)
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

        hf = self.backbone_f(x)
        f = self.out_f(hf).squeeze(-1)

        hg = self.backbone_g(x)
        g = self.out_g(hg).squeeze(-1)

        return f, g

    def f_only(self, x2, x3, x4, x5, x6):
        x = periodic_emb(x2, x3, x4, x5, x6)
        return self.out_f(self.backbone_f(x)).squeeze(-1)

# ============================================================
# Tempered MH sampling but LOCAL (scheme A):
# - Init only in local box around x_loc
# - Proposals are RW; accept only if stay in box
# - "global" proposals jump uniformly inside the same box
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
def mh_chain_tempered_local(
    model: nn.Module,
    state,
    x_loc: torch.Tensor,
    sigma: float,
    Nb: int,
    beta: float,
    p_global: float,
    delta: float
):
    x2, x3, x4, x5, x6, f = state
    f = model.f_only(x2, x3, x4, x5, x6)
    N = x2.shape[0]
    acc = 0

    # represent current coords in the centered branch around x_loc
    x2 = wrap_centered_1d(x2, x_loc[0])
    x3 = wrap_centered_1d(x3, x_loc[1])
    x4 = wrap_centered_1d(x4, x_loc[2])
    x5 = wrap_centered_1d(x5, x_loc[3])
    x6 = wrap_centered_1d(x6, x_loc[4])

    for _ in range(Nb):
        # proposal
        if (torch.rand((), device=device) < p_global):
            # global jump (BUT ONLY within the local box)
            u = (torch.rand(N, 5, device=device, dtype=dtype) * 2.0 - 1.0) * delta
            Xp = x_loc[None, :] + u
            # keep centered branch
            Xp = torch.stack([wrap_centered_1d(Xp[:, k], x_loc[k]) for k in range(5)], dim=1)
            nx2, nx3, nx4, nx5, nx6 = Xp[:, 0], Xp[:, 1], Xp[:, 2], Xp[:, 3], Xp[:, 4]
        else:
            nx2 = x2 + torch.randn_like(x2) * sigma
            nx3 = x3 + torch.randn_like(x3) * sigma
            nx4 = x4 + torch.randn_like(x4) * sigma
            nx5 = x5 + torch.randn_like(x5) * sigma
            nx6 = x6 + torch.randn_like(x6) * sigma

        # hard local box constraint
        in_box = (
            ((nx2 - x_loc[0]).abs() <= delta) &
            ((nx3 - x_loc[1]).abs() <= delta) &
            ((nx4 - x_loc[2]).abs() <= delta) &
            ((nx5 - x_loc[3]).abs() <= delta) &
            ((nx6 - x_loc[4]).abs() <= delta)
        )

        nf = model.f_only(nx2, nx3, nx4, nx5, nx6)

        log_alpha = 2.0 * beta * (nf - f)
        accept = in_box & (torch.log(torch.rand_like(log_alpha)) < log_alpha)

        x2 = torch.where(accept, nx2, x2)
        x3 = torch.where(accept, nx3, x3)
        x4 = torch.where(accept, nx4, x4)
        x5 = torch.where(accept, nx5, x5)
        x6 = torch.where(accept, nx6, x6)
        f  = torch.where(accept, nf, f)

        acc += int(accept.sum())

    return (x2, x3, x4, x5, x6, f), acc / (Nb * N)

def beta_schedule(it: int):
    if BETA_ANNEAL_STEPS <= 0:
        return float(BETA_START)
    t = min(1.0, max(0.0, it / float(BETA_ANNEAL_STEPS)))
    return float(BETA_START + (BETA_END - BETA_START) * t)

# ============================================================
# Local energy (5D) -> returns (E_real, E_imag, f, g)
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
# Reweight (pi/pi_beta) and weighted stats/losses
# ============================================================
def reweight_from_beta(f: torch.Tensor, beta: float):
    logw = torch.clamp(2.0 * (1.0 - beta) * f.detach(), -CLIP_LOGW, CLIP_LOGW)
    w = torch.exp(logw)
    w_mean = w.mean()
    return w, w_mean

def weighted_mean(x: torch.Tensor, w: torch.Tensor, w_mean: torch.Tensor):
    return (w * x).mean() / (w_mean + 1e-30)

def mean_sem_weighted(x: torch.Tensor, w: torch.Tensor):
    xd = x.detach()
    w = w.detach()
    wsum = w.sum() + 1e-30
    mu = (w * xd).sum() / wsum
    var = (w * (xd - mu)**2).sum() / wsum
    ess = (wsum**2) / (w.pow(2).sum() + 1e-30)
    sem = torch.sqrt(var / (ess + 1e-30))
    return float(mu), float(sem), float(ess)

def loss_real_tempered(E_real: torch.Tensor, f: torch.Tensor, beta: float):
    Er = E_real.detach()
    w, w_mean = reweight_from_beta(f, beta)
    Ebar = weighted_mean(Er, w, w_mean)
    return 2.0 * ((w * (Er - Ebar) * f).mean() / (w_mean + 1e-30))

def loss_imag_tempered(E_imag: torch.Tensor, g: torch.Tensor, f_for_w: torch.Tensor, beta: float):
    Ei = E_imag.detach()
    w, w_mean = reweight_from_beta(f_for_w, beta)
    Ibar = weighted_mean(Ei, w, w_mean)
    return 2.0 * ((w * (Ei - Ibar) * g).mean() / (w_mean + 1e-30))

# ============================================================
# Stable overlap estimator (symmetric), WITH reweight for pi
# ============================================================
def _stable_weighted_complex_mean_exp(df: torch.Tensor, dg: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    logw = torch.log(torch.clamp(w, min=1e-300))
    a = df + logw
    amax = a.max().detach()
    z = torch.exp(a - amax).to(cdtype) * torch.exp(1j * dg.to(cdtype))
    return z.mean() * torch.exp(amax).to(cdtype)

def overlap_abs_symmetric_stable_tempered(model_i, model_j, x_i, x_j, beta_i: float, beta_j: float, eps=EPS_OVLP):
    x2i, x3i, x4i, x5i, x6i = x_i
    x2j, x3j, x4j, x5j, x6j = x_j

    # ---- A: samples from i ----
    f_i_i, g_i_i = model_i(x2i, x3i, x4i, x5i, x6i)  # grad
    with torch.no_grad():
        f_j_i, g_j_i = model_j(x2i, x3i, x4i, x5i, x6i)

    w_i, w_i_mean = reweight_from_beta(f_i_i, beta_i)
    dfA = (f_j_i - f_i_i).to(dtype)
    dgA = (g_j_i - g_i_i).to(dtype)
    numA = _stable_weighted_complex_mean_exp(dfA, dgA, w_i)
    A = numA / (w_i_mean.to(cdtype) + 1e-30)

    # ---- B: samples from j ----
    f_i_j, g_i_j = model_i(x2j, x3j, x4j, x5j, x6j)  # grad
    with torch.no_grad():
        f_j_j, g_j_j = model_j(x2j, x3j, x4j, x5j, x6j)

    w_j, w_j_mean = reweight_from_beta(f_j_j, beta_j)
    dfB = (f_i_j - f_j_j).to(dtype)
    dgB = (g_i_j - g_j_j).to(dtype)
    numB = _stable_weighted_complex_mean_exp(dfB, dgB, w_j)
    B = numB / (w_j_mean.to(cdtype) + 1e-30)

    Sabs = torch.sqrt(torch.abs(A) * torch.abs(B)).to(dtype)
    Sclamp = torch.clamp(Sabs, max=(1.0 - eps))
    pen = (1.0 / (1.0 - Sclamp)) - 1.0
    return Sabs, pen

# ============================================================
# Overlap matrix printing (subsample)
# ============================================================
@torch.no_grad()
def compute_overlap_matrix(models, xs_pack, betas, subsample=OVLP_SUBSAMPLE):
    n = len(models)
    xs_use = []
    for x in xs_pack:
        if subsample is None:
            xs_use.append(x)
        else:
            xs_use.append(tuple(t[:subsample] for t in x))

    S = [[0.0]*n for _ in range(n)]
    for i in range(n):
        S[i][i] = 1.0
    for i in range(n):
        for j in range(i):
            s, _ = overlap_abs_symmetric_stable_tempered(
                models[i], models[j], xs_use[i], xs_use[j], betas[i], betas[j]
            )
            S[i][j] = float(s.detach().cpu())
            S[j][i] = S[i][j]
    return S

def print_matrix(mat, title, fmt="{:0.3e}"):
    n = len(mat)
    print(title)
    header = "      " + " ".join([f"{j:>10d}" for j in range(n)])
    print(header)
    for i in range(n):
        row = " ".join([fmt.format(mat[i][j]) for j in range(n)])
        print(f"{i:>4d}  {row}")

# ============================================================
# alpha_ij matrix scheduling (two stages)
# ============================================================
def init_alpha_mat(n, init=0.0):
    A = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i):
            A[i][j] = float(init)
    return A

def clamp_alpha(a):
    return float(max(ALPHA_MIN, min(ALPHA_MAX, a)))

def ramp_alpha_all(alpha_mat, delta):
    n = len(alpha_mat)
    for i in range(n):
        for j in range(i):
            alpha_mat[i][j] = clamp_alpha(alpha_mat[i][j] + delta)
    return alpha_mat

def adapt_alpha_pairwise(alpha_mat, S_mat):
    n = len(alpha_mat)
    for i in range(n):
        for j in range(i):
            s = S_mat[i][j]
            a = alpha_mat[i][j]
            if s >= S_TRIGGER:
                a = a * ALPHA_GROWTH
            else:
                a = a * ALPHA_DECAY
            alpha_mat[i][j] = clamp_alpha(a)
    return alpha_mat

RAMP_UPDATES = max(1, RAMP_STEPS // RAMP_EVERY)
RAMP_DELTA = RAMP_TARGET / RAMP_UPDATES

# ============================================================
# Training (LOCAL sampling)
# ============================================================
def train_vmc_tempered_5d_local(n_states=N_states, steps=N_steps, eta=lr):
    models = [NN_Waffle_Complex_5D().to(device=device, dtype=dtype) for _ in range(n_states)]
    opts   = [torch.optim.Adam(m.parameters(), lr=eta) for m in models]

    # IMPORTANT: local init for each state (same x_loc center)
    mh_states = [initial_local(models[i], N_walkers, x_loc=x_loc, delta=Delta) for i in range(n_states)]
    sigmas    = [Sigma0 for _ in range(n_states)]

    alpha_ij  = init_alpha_mat(n_states, init=0.0)

    for it in range(steps):
        beta_it = beta_schedule(it)
        betas = [beta_it for _ in range(n_states)]

        # --- sample each state with tempered MH (LOCAL) ---
        accs = []
        xs_pack = []
        for i in range(n_states):
            mh_states[i], acc = mh_chain_tempered_local(
                models[i], mh_states[i],
                x_loc=x_loc, sigma=sigmas[i], Nb=N_burn_in,
                beta=betas[i], p_global=P_GLOBAL,
                delta=Delta
            )
            accs.append(acc)
            x2,x3,x4,x5,x6,_ = mh_states[i]
            xs_pack.append((x2,x3,x4,x5,x6))

        # --- overlaps + alpha updates ---
        S_mat = None
        if (it % OVLP_EVERY) == 0:
            S_mat = compute_overlap_matrix(models, xs_pack, betas, subsample=OVLP_SUBSAMPLE)

            if it < WARMUP:
                pass
            elif it <= (WARMUP + RAMP_STEPS):
                if (it % RAMP_EVERY) == 0:
                    alpha_ij = ramp_alpha_all(alpha_ij, RAMP_DELTA)
            else:
                alpha_ij = adapt_alpha_pairwise(alpha_ij, S_mat)

        # --- build total loss ---
        total_loss = torch.zeros((), device=device, dtype=dtype)
        E_means, E_sems, ESSs = [], [], []
        I_means, I_sems, I_ESS = [], [], []

        for i in range(n_states):
            x2,x3,x4,x5,x6 = xs_pack[i]
            E_real, E_imag, f, g = local_energy_5D(x2,x3,x4,x5,x6, models[i])

            w, _ = reweight_from_beta(f, betas[i])
            em, es, ess = mean_sem_weighted(E_real, w)
            im, isem, iess = mean_sem_weighted(E_imag, w)
            E_means.append(em); E_sems.append(es); ESSs.append(ess)
            I_means.append(im); I_sems.append(isem); I_ESS.append(iess)

            li = loss_real_tempered(E_real, f, beta=betas[i])
            if BETA_IMAG != 0.0:
                li = li + float(BETA_IMAG) * loss_imag_tempered(E_imag, g, f_for_w=f, beta=betas[i])

            if it >= WARMUP:
                for j in range(i):
                    aij = alpha_ij[i][j]
                    if aij <= 0.0:
                        continue
                    _, pen = overlap_abs_symmetric_stable_tempered(
                        models[i], models[j], xs_pack[i], xs_pack[j], betas[i], betas[j]
                    )
                    li = li + aij * pen

            total_loss = total_loss + li

        # --- optimize ---
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        total_loss.backward()
        for opt in opts:
            opt.step()

        # --- sigma adapt ---
        for i in range(n_states):
            if accs[i] < 0.4:
                sigmas[i] *= 0.8
            elif accs[i] > 0.6:
                sigmas[i] *= 1.2
            sigmas[i] = float(max(0.02, min(0.8, sigmas[i])))

        # --- logging ---
        if it == 0 or (it % PRINT_EVERY) == 0:
            phase = "pre-warmup" if it < WARMUP else ("ramp" if it <= (WARMUP + RAMP_STEPS) else "adaptive")
            parts = []
            for i in range(n_states):
                parts.append(
                    f"S{i}:E={E_means[i]:.4f}±{E_sems[i]:.1e},ESS={ESSs[i]:.0f},"
                    f"Im={I_means[i]:+.2e}±{I_sems[i]:.1e},"
                    f"acc={accs[i]:.2f},sig={sigmas[i]:.3f}"
                )
            print(f"[it={it}] phase={phase} beta={beta_it:.3f} p_global(local)={P_GLOBAL:.2f} "
                  f"Delta={Delta:.2f} beta_imag={BETA_IMAG:.1e} | " + " | ".join(parts))

        if S_mat is not None and (it % OVLP_EVERY) == 0:
            print_matrix(S_mat, title=f"Overlap |S_ij| (reweighted, sub={OVLP_SUBSAMPLE}) at it={it}", fmt="{:0.3e}")
            Ashow = [[0.0]*n_states for _ in range(n_states)]
            for i in range(n_states):
                for j in range(i):
                    Ashow[i][j] = alpha_ij[i][j]
            print_matrix(Ashow, title=f"alpha_ij (lower triangle) at it={it}", fmt="{:0.2e}")

    return models, alpha_ij

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    models, alpha_ij = train_vmc_tempered_5d_local()