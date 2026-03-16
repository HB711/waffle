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
# Hyperparameters (5D, 6D-style loop)
# ============================================================
# MH
N_walkers = 2000
N_burn_in = 50
Sigma0 = 0.30

# NN / Opt
Hidden_Dim = 256
Seed = 0
torch.manual_seed(Seed)

# Training
N_states = 8
N_steps  = 4000
lr       = 1e-2

# Warmup (no orth penalties)
WARMUP = 300

# Pairwise alpha_ij adaptation (after warmup)
S_TRIGGER = 1e-3         # if S_ij >= 0.001 => increase alpha_ij
ALPHA_INIT = 1.0         # initial alpha_ij when triggered
ALPHA_UP = 2.0           # multiplicative increase (fast)
ALPHA_DOWN = 0.9        # decay when overlap is small
ALPHA_MAX = 1e7
ALPHA_MIN = 0.0

# Imag-loss weight (drives g via E_imag)
BETA_IMAG = 1.0          # you can try 0.1 if g becomes too noisy

# Overlap / penalty clamp
EPS_OVLP = 1e-6

# Logging
PRINT_EVERY = 10
OVLP_EVERY  = 50
OVLP_SUBSAMPLE = 2000

# ============================================================
# Physics constants / capacitance matrix / kinetic prefactors
# ============================================================
h = 6.626
e = 1.60
UNIT = 100.0 * (4.0 * e**2) / (2.0 * h)

# Cth =10.00
# Cph = 10.00
# Cv = 30.00
# Cp = 47.00

Cth =500.00
Cph = 500.00
Cv = 500.00
Cp = 500.00

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
# Potential
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

    def f_only(self, x2, x3, x4, x5, x6):
        return self.forward(x2, x3, x4, x5, x6)[0]

# ============================================================
# Sampling
# ============================================================
@torch.no_grad()
def initial(model: nn.Module, N: int):
    x2 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    x3 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    x4 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    x5 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    x6 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    f = model.f_only(x2, x3, x4, x5, x6)
    return (x2, x3, x4, x5, x6, f)

@torch.no_grad()
def mh_chain(model: nn.Module, state, sigma: float, Nb: int = N_burn_in):
    x2, x3, x4, x5, x6, f = state
    f = model.f_only(x2, x3, x4, x5, x6)
    N = x2.shape[0]
    acc = 0
    for _ in range(Nb):
        nx2 = wrap_pi_torch(x2 + torch.randn_like(x2) * sigma)
        nx3 = wrap_pi_torch(x3 + torch.randn_like(x3) * sigma)
        nx4 = wrap_pi_torch(x4 + torch.randn_like(x4) * sigma)
        nx5 = wrap_pi_torch(x5 + torch.randn_like(x5) * sigma)
        nx6 = wrap_pi_torch(x6 + torch.randn_like(x6) * sigma)
        nf = model.f_only(nx2, nx3, nx4, nx5, nx6)
        log_alpha = 2.0 * (nf - f)
        accept = (torch.log(torch.rand_like(log_alpha)) < log_alpha)
        x2 = torch.where(accept, nx2, x2)
        x3 = torch.where(accept, nx3, x3)
        x4 = torch.where(accept, nx4, x4)
        x5 = torch.where(accept, nx5, x5)
        x6 = torch.where(accept, nx6, x6)
        f  = torch.where(accept, nf, f)
        acc += int(accept.sum())
    return (x2, x3, x4, x5, x6, f), acc / (Nb * N)

# ============================================================
# Local energy (unchanged)
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
# Energy loss: add imag-driven term for g
# ============================================================
def energy_loss_real(E_real: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    Er = E_real.detach()
    return 2.0 * ((Er - Er.mean()) * f).mean()

def energy_loss_imag(E_imag: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    Ei = E_imag.detach()
    return 2.0 * ((Ei - Ei.mean()) * g).mean()

def mean_sem(x: torch.Tensor):
    x_det = x.detach()
    mean = x_det.mean()
    std  = x_det.std(unbiased=False)
    sem  = std / math.sqrt(x_det.numel())
    return float(mean), float(sem)

# ============================================================
# Stable overlap (keep penalty form EXACTLY your 5D: 1/(1-S)-1)
# ============================================================
def _stable_complex_mean_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    amax = a.max().detach()
    z = torch.exp(a - amax).to(cdtype) * torch.exp(1j * b.to(cdtype))
    return z.mean() * torch.exp(amax).to(cdtype)

def compute_Sabs_and_penalty_for_pair(model_i, x_i, model_j_frozen, x_j, eps=EPS_OVLP):
    x2i, x3i, x4i, x5i, x6i = x_i
    x2j, x3j, x4j, x5j, x6j = x_j

    f_i_i, g_i_i = model_i(x2i, x3i, x4i, x5i, x6i)
    f_i_j, g_i_j = model_i(x2j, x3j, x4j, x5j, x6j)

    with torch.no_grad():
        f_j_i, g_j_i = model_j_frozen(x2i, x3i, x4i, x5i, x6i)
        f_j_j, g_j_j = model_j_frozen(x2j, x3j, x4j, x5j, x6j)

    a  = (f_j_i - f_i_i).to(dtype)
    b  = (g_j_i - g_i_i).to(dtype)
    A_ij = _stable_complex_mean_exp(a, b)

    a2 = (f_i_j - f_j_j).to(dtype)
    b2 = (g_i_j - g_j_j).to(dtype)
    B_ij = _stable_complex_mean_exp(a2, b2)

    Sabs = torch.sqrt(torch.abs(A_ij) * torch.abs(B_ij)).to(dtype)
    Sclamp = torch.clamp(Sabs, max=(1.0 - eps))
    pen = (1.0 / (1.0 - Sclamp)) - 1.0
    return Sabs, pen

# ============================================================
# Overlap matrix printing
# ============================================================
@torch.no_grad()
def compute_overlap_matrix(models, xs_pack, subsample=OVLP_SUBSAMPLE):
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
            Sabs, _ = compute_Sabs_and_penalty_for_pair(models[i], xs_use[i], models[j], xs_use[j])
            s = float(Sabs.detach().clamp(0.0, 1.0).cpu())
            S[i][j] = s
            S[j][i] = s
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
    for i in range(len(alpha_mat)):
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
# Training
# ============================================================
def train_vmc_ex(n_states=N_states, steps=N_steps, eta=lr):
    models = [NN_Waffle_Complex_5D().to(device=device, dtype=dtype) for _ in range(n_states)]
    opts   = [torch.optim.Adam(m.parameters(), lr=eta) for m in models]

    states = [initial(models[i], N_walkers) for i in range(n_states)]
    sigmas = [Sigma0 for _ in range(n_states)]

    alpha_ij = init_alpha_mat(n_states, init=0.0)

    for it in range(steps):
        accs_it = [0.0 for _ in range(n_states)]
        xs_it   = [None for _ in range(n_states)]

        # ---- MH ----
        for k in range(n_states):
            states[k], acc = mh_chain(models[k], states[k], sigma=sigmas[k], Nb=N_burn_in)
            accs_it[k] = acc
            x2,x3,x4,x5,x6,_ = states[k]
            xs_it[k] = (x2,x3,x4,x5,x6)

        # ---- overlap matrix (subsample) for alpha update + printing ----
        S_mat = None
        if (it % OVLP_EVERY) == 0:
            S_mat = compute_overlap_matrix(models, xs_it, subsample=OVLP_SUBSAMPLE)
            alpha_ij = adapt_alpha_pairwise(alpha_ij, S_mat, it)

        # ---- build loss ----
        total_loss = torch.zeros((), device=device, dtype=dtype)

        E_means, E_sems = [], []
        I_means, I_sems = [], []

        for i in range(n_states):
            x2,x3,x4,x5,x6 = xs_it[i]
            E_real, E_imag, f, g = local_energy_5D(x2,x3,x4,x5,x6, models[i])

            # real drives f
            li = energy_loss_real(E_real, f)

            # imag drives g (new)
            if BETA_IMAG != 0.0:
                li = li + float(BETA_IMAG) * energy_loss_imag(E_imag, g)

            e_mean, e_sem = mean_sem(E_real)
            i_mean, i_sem = mean_sem(E_imag)
            E_means.append(e_mean); E_sems.append(e_sem)
            I_means.append(i_mean); I_sems.append(i_sem)

            # warmup: no penalties
            if it >= WARMUP:
                for j in range(i):
                    aij = alpha_ij[i][j]
                    if aij <= 0.0:
                        continue
                    _, pen = compute_Sabs_and_penalty_for_pair(models[i], xs_it[i], models[j], xs_it[j])
                    li = li + aij * pen

            total_loss = total_loss + li

        # ---- step ----
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        total_loss.backward()
        for opt in opts:
            opt.step()

        # ---- sigma adapt ----
        for k in range(n_states):
            if accs_it[k] < 0.4:
                sigmas[k] *= 0.8
            elif accs_it[k] > 0.6:
                sigmas[k] *= 1.2
            sigmas[k] = float(max(0.02, min(0.8, sigmas[k])))

        # ---- logging ----
        if it == 0 or (it % PRINT_EVERY) == 0:
            parts = []
            for k in range(n_states):
                parts.append(
                    f"S{k}:E={E_means[k]:.4f}±{E_sems[k]:.1e},"
                    f"Im={I_means[k]:+.2e}±{I_sems[k]:.1e},"
                    f"acc={accs_it[k]:.2f},sig={sigmas[k]:.3f}"
                )
            msg = f"it={it} | " + " | ".join(parts)
            if it < WARMUP:
                msg += f" || warmup<{WARMUP}"
            else:
                msg += f" || beta_imag={BETA_IMAG:.2e}"
            print(msg)

        if S_mat is not None:
            print_matrix(S_mat, title=f"Overlap |S_ij| (sub={OVLP_SUBSAMPLE}) at it={it}", fmt="{:0.3e}")

            # alpha_ij matrix print
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
    models, alpha_ij = train_vmc_ex(n_states=N_states, steps=N_steps, eta=lr)
