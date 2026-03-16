import os
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
    return x - TWO_PI * torch.floor((x + math.pi) / TWO_PI)

# ============================================================
# Hyperparameters (Uniform sampling experiment)
# ============================================================
# Uniform sampling
N_uni = 12000          # number of uniform points per state per iteration
SUB_OVLP = 5000        # subsample for overlap matrix (speed)
CLIP_LOGW = 40.0       # clamp for 2f to avoid exp overflow

# NN / Opt
Hidden_Dim = 256
Seed = 0
torch.manual_seed(Seed)

# Train
N_states = 8
N_steps  = 4000
lr       = 1e-2

# Orth schedule (pairwise alpha_ij like 5D)
WARMUP = 300
S_TRIGGER = 1e-3       # if S_ij >= 0.001 => increase alpha_ij
ALPHA_INIT = 1.0
ALPHA_UP   = 2.0
ALPHA_DOWN = 0.9
ALPHA_MAX  = 1e7
ALPHA_MIN  = 0.0

# Imag-loss weight (optional, drives g using E_imag)
BETA_IMAG = 1.0        # try 0.1 if unstable

# Logging
PRINT_EVERY = 10
OVLP_EVERY  = 50

# ============================================================
# Physics constants / capacitance matrix / kinetic prefactors
# ============================================================
h = 6.626
e = 1.60
UNIT = 100.0 * (4.0 * e**2) / (2.0 * h)

Cth =10.00
Cph = 10.00
Cv = 30.00
Cp = 47.00
EJ  = 80.00

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
# Potential (5D)
# ============================================================
def potential6D(t1, t2, t3, p1, p2, p3):
    alpha = 2 * torch.pi / 3
    term_phi1 = -2 * torch.cos(t1 - p1) - 2 * torch.cos(t2 - p1) - 2 * torch.cos(t3 - p1)
    term_phi2 = -2 * torch.cos(t1 - p2) - 2 * torch.cos(t2 - p2 + alpha) - 2 * torch.cos(t3 - p2 - alpha)
    term_phi3 = -2 * torch.cos(t1 - p3) - 2 * torch.cos(t2 - p3 - alpha) - 2 * torch.cos(t3 - p3 + alpha)
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
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )
        self.f_out = nn.Linear(hidden_dim, 1)
        self.g_out = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.g_out.weight)
        nn.init.zeros_(self.g_out.bias)

    def forward(self, x2, x3, x4, x5, x6):
        X = periodic_emb(x2, x3, x4, x5, x6)
        h = self.MLP(X)
        f = self.f_out(h).squeeze(-1)
        g = self.g_out(h).squeeze(-1)
        return f, g

# ============================================================
# Uniform sampling
# ============================================================
@torch.no_grad()
def uniform_sample_5d(N: int):
    x2 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    x3 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    x4 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    x5 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    x6 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    return (x2, x3, x4, x5, x6)

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
# Weighted energy + weighted score-function losses (uniform q)
# ============================================================
def _weights_from_f(f: torch.Tensor):
    # w = exp(2f) but clamp for stability
    logw = torch.clamp(2.0 * f, -CLIP_LOGW, CLIP_LOGW)
    w = torch.exp(logw)
    w_mean = w.mean()
    return w, w_mean

def weighted_energy_mean(E: torch.Tensor, w: torch.Tensor, w_mean: torch.Tensor):
    # mean_q[w E] / mean_q[w]
    return (w * E).mean() / (w_mean + 1e-30)

def loss_real_uniform(E_real: torch.Tensor, f: torch.Tensor):
    # 2 * E_q[ w (E - <E>_w) f ] / E_q[w]
    Er = E_real.detach()
    w, w_mean = _weights_from_f(f.detach())
    Ebar = weighted_energy_mean(Er, w, w_mean)
    return 2.0 * ((w * (Er - Ebar) * f).mean() / (w_mean + 1e-30))

def loss_imag_uniform(E_imag: torch.Tensor, g: torch.Tensor, f_for_w: torch.Tensor):
    # 2 * E_q[ w (Im - <Im>_w) g ] / E_q[w]
    Ei = E_imag.detach()
    w, w_mean = _weights_from_f(f_for_w.detach())
    Ibar = weighted_energy_mean(Ei, w, w_mean)
    return 2.0 * ((w * (Ei - Ibar) * g).mean() / (w_mean + 1e-30))

def mean_sem_weighted(x: torch.Tensor, w: torch.Tensor, w_mean: torch.Tensor):
    # weighted mean (self-normalized), SEM via effective sample size approximation
    xd = x.detach()
    w = w.detach()
    wsum = w.sum() + 1e-30
    mu = (w * xd).sum() / wsum
    # weighted variance (not unbiased)
    var = (w * (xd - mu)**2).sum() / wsum
    # ESS
    ess = (wsum**2) / (w.pow(2).sum() + 1e-30)
    sem = torch.sqrt(var / (ess + 1e-30))
    return float(mu), float(sem), float(ess)

# ============================================================
# Stable overlap under UNIFORM with |psi_i|^2 weights
# A = E_q[w_i * exp(df + i dg)] / E_q[w_i]
# B = E_q[w_j * exp(-df - i dg)] / E_q[w_j]  (implemented via its own df, dg)
# S = sqrt(|A||B|), penalty keeps your 5D form: 1/(1-S)-1
# ============================================================
def _stable_weighted_complex_mean_exp(df: torch.Tensor, dg: torch.Tensor, w: torch.Tensor):
    # compute mean_q[ w * exp(df) * exp(i dg) ] stably
    # use shift on (df + log w)
    logw = torch.log(torch.clamp(w, min=1e-300))
    a = df + logw
    amax = a.max().detach()
    z = torch.exp(a - amax).to(cdtype) * torch.exp(1j * dg.to(cdtype))
    return z.mean() * torch.exp(amax).to(cdtype)

def overlap_abs_symmetric_stable_uniform(model_i, x_i, model_j, x_j, eps=1e-6):
    x2i, x3i, x4i, x5i, x6i = x_i
    x2j, x3j, x4j, x5j, x6j = x_j

    # --- evaluate needed f,g ---
    f_i_i, g_i_i = model_i(x2i, x3i, x4i, x5i, x6i)  # grad (for penalty on i)
    f_i_j, g_i_j = model_i(x2j, x3j, x4j, x5j, x6j)  # grad

    with torch.no_grad():
        f_j_i, g_j_i = model_j(x2i, x3i, x4i, x5i, x6i)
        f_j_j, g_j_j = model_j(x2j, x3j, x4j, x5j, x6j)

    # weights for A, B
    w_i, w_i_mean = _weights_from_f(f_i_i.detach())
    w_j, w_j_mean = _weights_from_f(f_j_j.detach())  # weight uses |psi_j|^2 on x_j

    # A = E_q[ w_i * exp(f_j - f_i) * exp(i (g_j - g_i)) ] / E_q[w_i]
    dfA = (f_j_i - f_i_i).to(dtype)
    dgA = (g_j_i - g_i_i).to(dtype)
    numA = _stable_weighted_complex_mean_exp(dfA, dgA, w_i)
    denA = w_i_mean.to(cdtype)
    A = numA / (denA + 1e-30)

    # B = E_q[ w_j * exp(f_i - f_j) * exp(i (g_i - g_j)) ] / E_q[w_j]
    dfB = (f_i_j - f_j_j).to(dtype)
    dgB = (g_i_j - g_j_j).to(dtype)
    numB = _stable_weighted_complex_mean_exp(dfB, dgB, w_j)
    denB = w_j_mean.to(cdtype)
    B = numB / (denB + 1e-30)

    Sabs = torch.sqrt(torch.abs(A) * torch.abs(B)).to(dtype)
    Sclamp = torch.clamp(Sabs, max=(1.0 - eps))
    pen = (1.0 / (1.0 - Sclamp)) - 1.0
    return Sabs, pen

# ============================================================
# Overlap matrix printing
# ============================================================
@torch.no_grad()
def compute_overlap_matrix(models, xs_pack, subsample=SUB_OVLP):
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
            s, _ = overlap_abs_symmetric_stable_uniform(models[i], xs_use[i], models[j], xs_use[j])
            S[i][j] = float(s.detach().clamp(0.0, 1.0).cpu())
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
# Pairwise alpha_ij adaptation
# ============================================================
def init_alpha_mat(n, init=0.0):
    A = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i):
            A[i][j] = float(init)
    return A

def clamp_alpha(a):
    return float(max(ALPHA_MIN, min(ALPHA_MAX, a)))

def adapt_alpha_pairwise(alpha_mat, S_mat, it):
    if it < WARMUP:
        return alpha_mat
    n = len(alpha_mat)
    for i in range(n):
        for j in range(i):
            s = S_mat[i][j]
            a = alpha_mat[i][j]
            if s >= S_TRIGGER:
                if a <= 0.0:
                    a = ALPHA_INIT
                a = a * ALPHA_UP
            else:
                a = a * ALPHA_DOWN
            alpha_mat[i][j] = clamp_alpha(a)
    return alpha_mat

# ============================================================
# Training loop (uniform sampling)
# ============================================================
def train_uniform(n_states=N_states, steps=N_steps, eta=lr):
    models = [NN_Waffle_Complex_5D().to(device=device, dtype=dtype) for _ in range(n_states)]
    opts   = [torch.optim.Adam(m.parameters(), lr=eta) for m in models]

    alpha_ij = init_alpha_mat(n_states, init=0.0)

    for it in range(steps):
        # sample uniform points per state
        xs_pack = [uniform_sample_5d(N_uni) for _ in range(n_states)]

        # compute overlap matrix occasionally (also used to adapt alpha_ij)
        S_mat = None
        if (it % OVLP_EVERY) == 0:
            S_mat = compute_overlap_matrix(models, xs_pack, subsample=SUB_OVLP)
            alpha_ij = adapt_alpha_pairwise(alpha_ij, S_mat, it)

        total_loss = torch.zeros((), device=device, dtype=dtype)
        E_means, E_sems, ESSs = [], [], []
        I_means, I_sems, I_ESS = [], [], []

        for i in range(n_states):
            x2,x3,x4,x5,x6 = xs_pack[i]
            E_real, E_imag, f, g = local_energy_5D(x2,x3,x4,x5,x6, models[i])

            # weights from f for energy estimation
            w, w_mean = _weights_from_f(f.detach())
            e_mean, e_sem, ess = mean_sem_weighted(E_real, w, w_mean)
            i_mean, i_sem, iess = mean_sem_weighted(E_imag, w, w_mean)

            E_means.append(e_mean); E_sems.append(e_sem); ESSs.append(ess)
            I_means.append(i_mean); I_sems.append(i_sem); I_ESS.append(iess)

            li = loss_real_uniform(E_real, f)
            if BETA_IMAG != 0.0:
                li = li + float(BETA_IMAG) * loss_imag_uniform(E_imag, g, f_for_w=f)

            # orth penalties (after warmup)
            if it >= WARMUP:
                for j in range(i):
                    aij = alpha_ij[i][j]
                    if aij <= 0.0:
                        continue
                    _, pen = overlap_abs_symmetric_stable_uniform(models[i], xs_pack[i], models[j], xs_pack[j])
                    li = li + aij * pen

            total_loss = total_loss + li

        for opt in opts:
            opt.zero_grad(set_to_none=True)
        total_loss.backward()
        for opt in opts:
            opt.step()

        # logging
        if it == 0 or (it % PRINT_EVERY) == 0:
            parts = []
            for k in range(n_states):
                parts.append(
                    f"S{k}:E={E_means[k]:.4f}±{E_sems[k]:.1e},ESS={ESSs[k]:.0f},"
                    f"Im={I_means[k]:+.2e}±{I_sems[k]:.1e},"
                    f"beta_imag={BETA_IMAG:.1e}"
                )
            msg = f"[Uniform it={it}] " + " | ".join(parts)
            if it < WARMUP:
                msg += f" || warmup<{WARMUP}"
            else:
                msg += f" || S_trigger={S_TRIGGER:.1e}"
            print(msg)

        if S_mat is not None:
            print_matrix(S_mat, title=f"Overlap |S_ij| (uniform, sub={SUB_OVLP}) at it={it}", fmt="{:0.3e}")
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
    models, alpha_ij = train_uniform(n_states=N_states, steps=N_steps, eta=lr)
