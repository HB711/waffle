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
    # map to (-pi, pi]
    return x - TWO_PI * torch.floor((x + math.pi) / TWO_PI)


# ============================================================
# Hyperparameters (4DOF, plain MH sampling from |psi|^2)
# ============================================================
# Sampling / MH
N_walkers = 5000
N_burn_in = 40
Sigma0 = 0.30
P_GLOBAL = 0.02  # occasional global proposals

# Model / Opt
Hidden_Dim = 256
Seed = 0
torch.manual_seed(Seed)

N_states = 8
N_steps = 8400
lr = 1e-2

# Orthogonality alpha_ij schedule: warmup -> ramp -> adaptive
WARMUP = 200
RAMP_TARGET = 200.0
RAMP_STEPS = 600
RAMP_EVERY = 20

S_TRIGGER = 1e-3
ALPHA_GROWTH = 1.5
ALPHA_DECAY = 0.9
ALPHA_MAX = 1e8
ALPHA_MIN = 0.0

# Overlap / printing
OVLP_EVERY = 20
OVLP_SUBSAMPLE = 5000   # used for printing AND penalty (speed)
EPS_OVLP = 1e-8

# Optional: drive phase g via imag local energy
BETA_IMAG = 0.0  # set e.g. 0.1 if you want; 0.0 to disable

# Logging
PRINT_EVERY = 10

# ============================================================
# Physics constants / 4x4 capacitance matrix / kinetic prefactors
# DOF order: [t1, t2, p1, p2]
# ============================================================
h = 6.626
e = 1.60
UNIT = 100.0 * (4.0 * e**2) / (2.0 * h)

Cth = 1000.00
Cv = 1000.00
Cp = 1000.00
EJ = 80.00

C = torch.tensor(
    [[2 * Cv + Cth + Cp, -Cp, -Cv, -Cv],
     [-Cp, 2 * Cv + Cth + Cp, -Cv, -Cv],
     [-Cv, -Cv, 2 * Cv + Cth + Cp, -Cp],
     [-Cv, -Cv, -Cp, 2 * Cv + Cth + Cp]],
    device=device, dtype=dtype
)
Cinv = torch.linalg.inv(C)
E_mat = UNIT * Cinv
E_mat = 0.5 * (E_mat + E_mat.T)


# ============================================================
# Potential (4DOF)
# V = -2EJ [ cos(t1-p1)+cos(t1-p2)+cos(t2-p1)-cos(t2-p2) ]
# ============================================================
def potential4D(t1, t2, p1, p2):
    return -2.0 * EJ * (
        torch.cos(t1 - p1)
        + torch.cos(t1 - p2)
        + torch.cos(t2 - p1)
        - torch.cos(t2 - p2)
    )


# ============================================================
# Periodic embedding (4 vars -> 16 features)
# ============================================================
def periodic_emb(t1, t2, p1, p2):
    feats = [
        torch.sin(t1), torch.cos(t1), torch.sin(2 * t1), torch.cos(2 * t1),
        torch.sin(t2), torch.cos(t2), torch.sin(2 * t2), torch.cos(2 * t2),
        torch.sin(p1), torch.cos(p1), torch.sin(2 * p1), torch.cos(2 * p1),
        torch.sin(p2), torch.cos(p2), torch.sin(2 * p2), torch.cos(2 * p2),
    ]
    return torch.stack(feats, dim=-1)  # (N,16)


# ============================================================
# Model (separate f / g branches)
# log psi = f + i g
# ============================================================
class NN_Waffle_Complex_4D(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=Hidden_Dim):
        super().__init__()
        # f branch
        self.backbone_f = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )
        self.out_f = nn.Linear(hidden_dim, 1)

        # g branch
        self.backbone_g = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )
        self.out_g = nn.Linear(hidden_dim, 1)

        # keep g initially ~0
        nn.init.zeros_(self.out_g.weight)
        nn.init.zeros_(self.out_g.bias)

    def forward(self, t1, t2, p1, p2):
        x = periodic_emb(t1, t2, p1, p2)

        hf = self.backbone_f(x)
        f = self.out_f(hf).squeeze(-1)

        hg = self.backbone_g(x)
        g = self.out_g(hg).squeeze(-1)
        return f, g

    def f_only(self, t1, t2, p1, p2):
        x = periodic_emb(t1, t2, p1, p2)
        return self.out_f(self.backbone_f(x)).squeeze(-1)


# ============================================================
# Plain MH sampling: target pi ∝ exp(2 f) = |psi|^2
# ============================================================
@torch.no_grad()
def initial(model: nn.Module, N: int):
    t1 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    t2 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    p1 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    p2 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    f = model.f_only(t1, t2, p1, p2)
    return (t1, t2, p1, p2, f)


@torch.no_grad()
def mh_chain(model: nn.Module, state, sigma: float, Nb: int, p_global: float):
    t1, t2, p1, p2, f = state
    f = model.f_only(t1, t2, p1, p2)
    N = t1.shape[0]
    acc = 0

    for _ in range(Nb):
        if (torch.rand((), device=device) < p_global):
            nt1 = torch.rand_like(t1) * TWO_PI - math.pi
            nt2 = torch.rand_like(t2) * TWO_PI - math.pi
            np1 = torch.rand_like(p1) * TWO_PI - math.pi
            np2 = torch.rand_like(p2) * TWO_PI - math.pi
        else:
            nt1 = wrap_pi_torch(t1 + torch.randn_like(t1) * sigma)
            nt2 = wrap_pi_torch(t2 + torch.randn_like(t2) * sigma)
            np1 = wrap_pi_torch(p1 + torch.randn_like(p1) * sigma)
            np2 = wrap_pi_torch(p2 + torch.randn_like(p2) * sigma)

        nf = model.f_only(nt1, nt2, np1, np2)

        log_alpha = 2.0 * (nf - f)  # beta=1
        accept = (torch.log(torch.rand_like(log_alpha)) < log_alpha)

        t1 = torch.where(accept, nt1, t1)
        t2 = torch.where(accept, nt2, t2)
        p1 = torch.where(accept, np1, p1)
        p2 = torch.where(accept, np2, p2)
        f = torch.where(accept, nf, f)

        acc += int(accept.sum())

    return (t1, t2, p1, p2, f), acc / (Nb * N)


# ============================================================
# Local energy (matrix form) -> returns (E_real, E_imag, f, g)
# ============================================================
def local_energy_4D(t1, t2, p1, p2, model: nn.Module):
    x = torch.stack([t1, t2, p1, p2], dim=1)  # (N,4)
    x = x.detach().requires_grad_(True)

    f, g = model(x[:, 0], x[:, 1], x[:, 2], x[:, 3])
    ones = torch.ones_like(f)

    grad_f = torch.autograd.grad(f, x, grad_outputs=ones, create_graph=True)[0]  # (N,4)
    grad_g = torch.autograd.grad(g, x, grad_outputs=ones, create_graph=True)[0]  # (N,4)

    quad_f = torch.einsum("ni,ij,nj->n", grad_f, E_mat, grad_f)  # (N,)
    quad_g = torch.einsum("ni,ij,nj->n", grad_g, E_mat, grad_g)  # (N,)
    cross_fg = torch.einsum("ni,ij,nj->n", grad_f, E_mat, grad_g)  # (N,)

    # Tr(E Hess f), Tr(E Hess g)
    EH_f = torch.zeros_like(f)
    EH_g = torch.zeros_like(f)
    for i in range(4):
        # Hess row via grad(grad[:,i], x) -> (N,4)
        dgrad_f_i = torch.autograd.grad(
            grad_f[:, i], x,
            grad_outputs=torch.ones_like(grad_f[:, i]),
            create_graph=True
        )[0]
        dgrad_g_i = torch.autograd.grad(
            grad_g[:, i], x,
            grad_outputs=torch.ones_like(grad_g[:, i]),
            create_graph=True
        )[0]

        EH_f = EH_f + (dgrad_f_i * E_mat[i]).sum(dim=1)
        EH_g = EH_g + (dgrad_g_i * E_mat[i]).sum(dim=1)

    V = potential4D(x[:, 0], x[:, 1], x[:, 2], x[:, 3])

    E_real = -(EH_f + quad_f - quad_g) + V
    E_imag = -(EH_g + 2.0 * cross_fg)
    return E_real, E_imag, f, g


# ============================================================
# Stable complex mean: mean(exp(df) * exp(i dg))
# ============================================================
def stable_complex_mean(df: torch.Tensor, dg: torch.Tensor) -> torch.Tensor:
    m = df.max().detach()
    z = torch.exp(df - m).to(cdtype) * torch.exp(1j * dg.to(cdtype))
    return z.mean() * torch.exp(m).to(cdtype)


# ============================================================
# Symmetric overlap estimator (no reweight)
# S = sqrt(|A||B|)
# A = E_i[ psi_j/psi_i ], B = E_j[ psi_i/psi_j ]
# penalty barrier on S (clamped)
# ============================================================
def overlap_abs_symmetric_stable(model_i, model_j, x_i, x_j, eps=EPS_OVLP):
    t1i, t2i, p1i, p2i = x_i
    t1j, t2j, p1j, p2j = x_j

    # ---- A: samples from i ----
    f_i_i, g_i_i = model_i(t1i, t2i, p1i, p2i)  # grad flows to model_i
    with torch.no_grad():
        f_j_i, g_j_i = model_j(t1i, t2i, p1i, p2i)

    dfA = (f_j_i - f_i_i).to(dtype)
    dgA = (g_j_i - g_i_i).to(dtype)
    A = stable_complex_mean(dfA, dgA)
    A_abs = torch.abs(A)

    # ---- B: samples from j ----
    f_i_j, g_i_j = model_i(t1j, t2j, p1j, p2j)  # grad flows to model_i
    with torch.no_grad():
        f_j_j, g_j_j = model_j(t1j, t2j, p1j, p2j)

    dfB = (f_i_j - f_j_j).to(dtype)
    dgB = (g_i_j - g_j_j).to(dtype)
    B = stable_complex_mean(dfB, dgB)
    B_abs = torch.abs(B)

    Sabs = torch.sqrt(torch.clamp(A_abs * B_abs, min=0.0)).to(dtype)
    Sclamp = torch.clamp(Sabs, max=(1.0 - eps))
    pen = (1.0 / (1.0 - Sclamp)) - 1.0
    return Sabs, pen


# ============================================================
# Overlap matrix printing (subsample, no grad)
# ============================================================
def _subsample_tuple(x_tuple, n):
    if n is None:
        return x_tuple
    return tuple(t[:n] for t in x_tuple)


@torch.no_grad()
def compute_overlap_matrix(models, xs_pack, subsample=OVLP_SUBSAMPLE):
    n = len(models)
    xs_use = [_subsample_tuple(x, subsample) for x in xs_pack]

    S = [[0.0] * n for _ in range(n)]
    for i in range(n):
        S[i][i] = 1.0
    for i in range(n):
        for j in range(i):
            s, _ = overlap_abs_symmetric_stable(models[i], models[j], xs_use[i], xs_use[j])
            val = float(s.detach().cpu())
            S[i][j] = val
            S[j][i] = val
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
# alpha_ij matrix scheduling
# ============================================================
def init_alpha_mat(n, init=0.0):
    A = [[0.0] * n for _ in range(n)]
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
# Training (plain MH, no reweight)
# ============================================================
def train_vmc_mh_4d(n_states=N_states, steps=N_steps, eta=lr):
    models = [NN_Waffle_Complex_4D().to(device=device, dtype=dtype) for _ in range(n_states)]
    opts = [torch.optim.Adam(m.parameters(), lr=eta) for m in models]

    mh_states = [initial(models[i], N_walkers) for i in range(n_states)]
    sigmas = [Sigma0 for _ in range(n_states)]
    alpha_ij = init_alpha_mat(n_states, init=0.0)

    for it in range(steps):
        # --- sample each state with plain MH ---
        accs = []
        xs_pack = []
        for i in range(n_states):
            mh_states[i], acc = mh_chain(
                models[i], mh_states[i], sigma=sigmas[i], Nb=N_burn_in, p_global=P_GLOBAL
            )
            accs.append(acc)
            t1, t2, p1, p2, _ = mh_states[i]
            xs_pack.append((t1, t2, p1, p2))

        # --- overlaps + alpha updates (no grad) ---
        S_mat = None
        if (it % OVLP_EVERY) == 0:
            S_mat = compute_overlap_matrix(models, xs_pack, subsample=OVLP_SUBSAMPLE)

            if it < WARMUP:
                pass
            elif it <= (WARMUP + RAMP_STEPS):
                if (it % RAMP_EVERY) == 0:
                    alpha_ij = ramp_alpha_all(alpha_ij, RAMP_DELTA)
            else:
                alpha_ij = adapt_alpha_pairwise(alpha_ij, S_mat)

        # --- build total loss ---
        total_loss = torch.zeros((), device=device, dtype=dtype)
        E_means, E_sems = [], []
        I_means, I_sems = [], []

        for i in range(n_states):
            t1, t2, p1, p2 = xs_pack[i]
            E_real, E_imag, f, g = local_energy_4D(t1, t2, p1, p2, models[i])

            Er = E_real.detach()
            em = float(Er.mean())
            es = float(Er.std(unbiased=False) / math.sqrt(Er.numel()))
            E_means.append(em)
            E_sems.append(es)

            Ei = E_imag.detach()
            im = float(Ei.mean())
            isem = float(Ei.std(unbiased=False) / math.sqrt(Ei.numel()))
            I_means.append(im)
            I_sems.append(isem)

            # standard VMC loss (beta=1)
            loss_i = 2.0 * ((Er - Er.mean()) * f).mean()
            if BETA_IMAG != 0.0:
                loss_i = loss_i + float(BETA_IMAG) * (2.0 * ((Ei - Ei.mean()) * g).mean())

            # orthogonality penalties (with grad), after warmup
            if it >= WARMUP:
                xi = _subsample_tuple(xs_pack[i], OVLP_SUBSAMPLE)
                for j in range(i):
                    aij = alpha_ij[i][j]
                    if aij <= 0.0:
                        continue
                    xj = _subsample_tuple(xs_pack[j], OVLP_SUBSAMPLE)
                    _, pen = overlap_abs_symmetric_stable(models[i], models[j], xi, xj)
                    loss_i = loss_i + aij * pen

            total_loss = total_loss + loss_i

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
                    f"S{i}:E={E_means[i]:.4f}±{E_sems[i]:.1e},"
                    f"Im={I_means[i]:+.2e}±{I_sems[i]:.1e},"
                    f"acc={accs[i]:.2f},sig={sigmas[i]:.3f}"
                )
            print(f"[it={it}] phase={phase} p_global={P_GLOBAL:.2f} beta_imag={BETA_IMAG:.1e} | " + " | ".join(parts))

        if S_mat is not None and (it % OVLP_EVERY) == 0:
            print_matrix(S_mat, title=f"Overlap |S_ij| (sub={OVLP_SUBSAMPLE}) at it={it}", fmt="{:0.3e}")
            Ashow = [[0.0] * n_states for _ in range(n_states)]
            for i in range(n_states):
                for j in range(i):
                    Ashow[i][j] = alpha_ij[i][j]
            print_matrix(Ashow, title=f"alpha_ij (lower triangle) at it={it}", fmt="{:0.2e}")

    return models, alpha_ij


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    models, alpha_ij = train_vmc_mh_4d()