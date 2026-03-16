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

# ============================================================
# Physical params
# ============================================================
h = 6.626
e = 1.60
UNIT = 100.0 * (4.0 * e**2) / (2.0 * h)

Cth = 10.00
Cphi = 10.00
Cv = 30.00
Cp = 47.00
EJ = 80.00

# ============================================================
# VMC / MH params
# ============================================================
N_walkers = 4000
N_burn_in = 40
Sigma0 = 0.35

Hidden_dim = 256

N_steps = 8400
Lr = 1e-2

N_states = 8

Seed = 0
torch.manual_seed(Seed)

# ============================================================
# Orthogonality alpha_ij schedule (your request)
# ============================================================
WARMUP = 400

# Phase-1 ramp: after warmup, linearly ramp all alpha_ij to 200 over ~600 steps, update every 20
RAMP_TARGET = 200.0
RAMP_STEPS  = 600
RAMP_EVERY  = 20

# Phase-2 adapt: after ramp, adapt per pair based on S_ij threshold
S_TRIGGER = 1e-3
ALPHA_GROWTH = 1.5   # you approved
ALPHA_DECAY  = 0.9 # you approved
ALPHA_MAX = 1e8
ALPHA_MIN = 0.0

# overlap / printing
EPS_OVLP = 1e-8
OVLP_EVERY = 20
OVLP_SUBSAMPLE = 4000

# Imag-loss weight (drives g via E_imag)
BETA_IMAG = 1.0   # set 0.1 if unstable; 0 to disable

# Logging
PRINT_EVERY = 10

# ============================================================
# Capacitance matrix -> E_mat
# ============================================================
C = torch.tensor(
    [[Cth + 3*Cv + 2*Cp, -Cp, -Cp, -Cv, -Cv, -Cv],
     [-Cp, Cth + 3*Cv + 2*Cp, -Cp, -Cv, -Cv, -Cv],
     [-Cp, -Cp, Cth + 3*Cv + 2*Cp, -Cv, -Cv, -Cv],
     [-Cv, -Cv, -Cv, Cphi + 3*Cv + 2*Cp, -Cp, -Cp],
     [-Cv, -Cv, -Cv, -Cp, Cphi + 3*Cv + 2*Cp, -Cp],
     [-Cv, -Cv, -Cv, -Cp, -Cp, Cphi + 3*Cv + 2*Cp]],
    device=device, dtype=dtype
)
Cinv = torch.linalg.inv(C)
E_mat = UNIT * Cinv
E_mat = 0.5 * (E_mat + E_mat.T)

# ============================================================
# Helpers
# ============================================================
TWO_PI = 2.0 * math.pi

def wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return x - TWO_PI * torch.floor((x + math.pi) / TWO_PI)

def potential_energy(t1, t2, t3, p1, p2, p3):
    alpha = 2 * torch.pi / 3
    term_phi1 = -2 * torch.cos(t1 - p1) - 2 * torch.cos(t2 - p1) - 2 * torch.cos(t3 - p1)
    term_phi2 = -2 * torch.cos(t1 - p2) - 2 * torch.cos(t2 - p2 - alpha) - 2 * torch.cos(t3 - p2 + alpha)
    term_phi3 = -2 * torch.cos(t1 - p3) - 2 * torch.cos(t2 - p3 + alpha) - 2 * torch.cos(t3 - p3 - alpha)
    return EJ * (term_phi1 + term_phi2 + term_phi3) / math.sqrt(3.0)

def periodic_emb(t1, t2, t3, p1, p2, p3):
    feats = [
        torch.sin(t1), torch.sin(2*t1), torch.cos(t1), torch.cos(2*t1),
        torch.sin(t2), torch.sin(2*t2), torch.cos(t2), torch.cos(2*t2),
        torch.sin(t3), torch.sin(2*t3), torch.cos(t3), torch.cos(2*t3),
        torch.sin(p1), torch.sin(2*p1), torch.cos(p1), torch.cos(2*p1),
        torch.sin(p2), torch.sin(2*p2), torch.cos(p2), torch.cos(2*p2),
        torch.sin(p3), torch.sin(2*p3), torch.cos(p3), torch.cos(2*p3),
    ]
    return torch.stack(feats, dim=-1)

def mean_sem(x: torch.Tensor):
    xd = x.detach()
    m = xd.mean()
    s = xd.std(unbiased=False)
    sem = s / math.sqrt(xd.numel())
    return float(m), float(sem)

# ============================================================
# Model
# ============================================================
# class NN_Waffle_Complex(nn.Module):
#     def __init__(self, input_dim=24, hidden_dim=Hidden_dim):
#         super().__init__()
#         self.backbone = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.SiLU(),
#         )
#         self.out_f = nn.Linear(hidden_dim, 1)
#         self.out_g = nn.Linear(hidden_dim, 1)
#         nn.init.zeros_(self.out_g.weight)
#         nn.init.zeros_(self.out_g.bias)
#
#     def forward(self, t1, t2, t3, p1, p2, p3):
#         x = periodic_emb(t1, t2, t3, p1, p2, p3)
#         h = self.backbone(x)
#         f = self.out_f(h).squeeze(-1)
#         g = self.out_g(h).squeeze(-1)
#         return f, g
#
#     def f_only(self, t1, t2, t3, p1, p2, p3):
#         return self.forward(t1, t2, t3, p1, p2, p3)[0]

class NN_Waffle_Complex(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=Hidden_dim):
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

        # keep g initially zero
        nn.init.zeros_(self.out_g.weight)
        nn.init.zeros_(self.out_g.bias)

    def forward(self, t1, t2, t3, p1, p2, p3):
        x = periodic_emb(t1, t2, t3, p1, p2, p3)

        hf = self.backbone_f(x)
        f = self.out_f(hf).squeeze(-1)

        hg = self.backbone_g(x)
        g = self.out_g(hg).squeeze(-1)

        return f, g

    def f_only(self, t1, t2, t3, p1, p2, p3):
        x = periodic_emb(t1, t2, t3, p1, p2, p3)
        return self.out_f(self.backbone_f(x)).squeeze(-1)


# ============================================================
# Initialization + MH chain (samples from |psi|^2 via f-only)
# ============================================================
@torch.no_grad()
def initial(model, N=N_walkers):
    t1 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    t2 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    t3 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    p1 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    p2 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    p3 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    f = model.f_only(t1, t2, t3, p1, p2, p3)
    return (t1, t2, t3, p1, p2, p3, f)

@torch.no_grad()
def mh_chain(model, state, Nb, sigma):
    t1, t2, t3, p1, p2, p3, f = state
    f = model.f_only(t1, t2, t3, p1, p2, p3)
    N = t1.shape[0]

    acc = 0
    for _ in range(Nb):
        nt1 = wrap_pi(t1 + torch.randn_like(t1) * sigma)
        nt2 = wrap_pi(t2 + torch.randn_like(t2) * sigma)
        nt3 = wrap_pi(t3 + torch.randn_like(t3) * sigma)
        np1 = wrap_pi(p1 + torch.randn_like(p1) * sigma)
        np2 = wrap_pi(p2 + torch.randn_like(p2) * sigma)
        np3 = wrap_pi(p3 + torch.randn_like(p3) * sigma)

        nf = model.f_only(nt1, nt2, nt3, np1, np2, np3)
        log_alpha = 2.0 * (nf - f)
        accept = (torch.log(torch.rand_like(log_alpha)) < log_alpha)

        t1 = torch.where(accept, nt1, t1)
        t2 = torch.where(accept, nt2, t2)
        t3 = torch.where(accept, nt3, t3)
        p1 = torch.where(accept, np1, p1)
        p2 = torch.where(accept, np2, p2)
        p3 = torch.where(accept, np3, p3)
        f  = torch.where(accept, nf, f)

        acc += int(accept.sum())

    return (t1, t2, t3, p1, p2, p3, f), acc / (Nb * N)

# ============================================================
# Local energy (REAL + IMAG) with full E_mat
# ============================================================
def local_energy_complex_matrix(t1, t2, t3, p1, p2, p3, model):
    x = torch.stack([t1, t2, t3, p1, p2, p3], dim=1)
    x = x.detach().requires_grad_(True)

    f, g = model(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5])

    ones = torch.ones_like(f)
    grad_f = torch.autograd.grad(f, x, grad_outputs=ones, create_graph=True)[0]
    grad_g = torch.autograd.grad(g, x, grad_outputs=ones, create_graph=True)[0]

    quad_f  = torch.einsum("ni,ij,nj->n", grad_f, E_mat, grad_f)
    quad_g  = torch.einsum("ni,ij,nj->n", grad_g, E_mat, grad_g)
    quad_fg = torch.einsum("ni,ij,nj->n", grad_f, E_mat, grad_g)

    EH_f = torch.zeros_like(f)
    EH_g = torch.zeros_like(g)
    for i in range(6):
        dgrad_f_i = torch.autograd.grad(
            grad_f[:, i], x, grad_outputs=torch.ones_like(grad_f[:, i]), create_graph=True
        )[0]
        dgrad_g_i = torch.autograd.grad(
            grad_g[:, i], x, grad_outputs=torch.ones_like(grad_g[:, i]), create_graph=True
        )[0]
        EH_f = EH_f + (dgrad_f_i * E_mat[i]).sum(dim=1)
        EH_g = EH_g + (dgrad_g_i * E_mat[i]).sum(dim=1)

    V = potential_energy(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5])

    E_real = -(EH_f + quad_f - quad_g) + V
    E_imag = -(EH_g + 2.0 * quad_fg)
    return E_real, E_imag, f, g

# ============================================================
# Loss (detach Eloc to avoid second-order backprop)
# ============================================================
def energy_loss_real(E_real: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    Er = E_real.detach()
    return 2.0 * ((Er - Er.mean()) * f).mean()

def energy_loss_imag(E_imag: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    Ei = E_imag.detach()
    return 2.0 * ((Ei - Ei.mean()) * g).mean()

# ============================================================
# Stable complex mean for overlap
# ============================================================
def stable_complex_mean(df: torch.Tensor, dg: torch.Tensor) -> torch.Tensor:
    m = df.max()
    w = torch.exp(df - m)
    z = w.to(cdtype) * torch.exp(1j * dg.to(cdtype))
    return torch.exp(m).to(cdtype) * z.mean()

# ============================================================
# Symmetric overlap: S = sqrt(|A||B|), penalty: 1/(1-S)-1
# ============================================================
def overlap_abs_symmetric_stable(model_i, model_j, x_i, x_j, eps=EPS_OVLP):
    x1i, x2i, x3i, x4i, x5i, x6i = x_i
    x1j, x2j, x3j, x4j, x5j, x6j = x_j

    # ---- A: samples from i ----
    f_i_i, g_i_i = model_i(x1i, x2i, x3i, x4i, x5i, x6i)
    with torch.no_grad():
        f_j_i, g_j_i = model_j(x1i, x2i, x3i, x4i, x5i, x6i)
    dfA = (f_j_i - f_i_i).to(dtype)
    dgA = (g_j_i - g_i_i).to(dtype)
    A = stable_complex_mean(dfA, dgA)

    # ---- B: samples from j ----
    f_i_j, g_i_j = model_i(x1j, x2j, x3j, x4j, x5j, x6j)
    with torch.no_grad():
        f_j_j, g_j_j = model_j(x1j, x2j, x3j, x4j, x5j, x6j)
    dfB = (f_i_j - f_j_j).to(dtype)
    dgB = (g_i_j - g_j_j).to(dtype)
    B = stable_complex_mean(dfB, dgB)

    Sabs = torch.sqrt(torch.abs(A) * torch.abs(B)).to(dtype)
    Sclamp = torch.clamp(Sabs, max=(1.0 - eps))
    pen = (1.0 / (1.0 - Sclamp)) - 1.0
    return Sabs, pen

# ============================================================
# Overlap matrix printing (subsample)
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
            s, _ = overlap_abs_symmetric_stable(models[i], models[j], xs_use[i], xs_use[j])
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

# precompute ramp step size
RAMP_UPDATES = max(1, RAMP_STEPS // RAMP_EVERY)
RAMP_DELTA = RAMP_TARGET / RAMP_UPDATES  # ~ 200 / 30 = 6.6667

# ============================================================
# Training
# ============================================================
def train_vmc_ex(n_states, steps, eta):
    models = [NN_Waffle_Complex().to(device=device, dtype=dtype) for _ in range(n_states)]
    opts = [torch.optim.Adam(m.parameters(), lr=eta) for m in models]
    states = [initial(models[i], N_walkers) for i in range(n_states)]
    sigmas = [Sigma0 for _ in range(n_states)]

    alpha_ij = init_alpha_mat(n_states, init=0.0)

    for it in range(steps):
        accs_it = [0.0 for _ in range(n_states)]
        xs_it   = [None for _ in range(n_states)]

        # MH sampling
        for k in range(n_states):
            states[k], acc = mh_chain(models[k], states[k], N_burn_in, sigmas[k])
            accs_it[k] = acc
            t1, t2, t3, p1, p2, p3, _ = states[k]
            xs_it[k] = (t1, t2, t3, p1, p2, p3)

        # compute overlap matrix & schedule alpha_ij
        S_mat = None
        if (it % OVLP_EVERY) == 0:
            S_mat = compute_overlap_matrix(models, xs_it, subsample=OVLP_SUBSAMPLE)

            # Stage 0: before warmup -> keep alpha=0
            if it < WARMUP:
                pass

            # Stage 1: ramp all alpha_ij linearly to 200 over ~600 steps, update every 20
            elif it <= (WARMUP + RAMP_STEPS):
                if ((it - WARMUP) % RAMP_EVERY) == 0:
                    alpha_ij = ramp_alpha_all(alpha_ij, RAMP_DELTA)

            # Stage 2: after ramp -> adapt per pair based on S_ij
            else:
                alpha_ij = adapt_alpha_pairwise(alpha_ij, S_mat)

        total_loss = torch.zeros((), device=device, dtype=dtype)

        E_means, E_sems = [], []
        I_means, I_sems = [], []

        for k in range(n_states):
            t1, t2, t3, p1, p2, p3, _ = states[k]
            E_real, E_imag, f, g = local_energy_complex_matrix(t1, t2, t3, p1, p2, p3, models[k])

            loss_k = energy_loss_real(E_real, f)
            if BETA_IMAG != 0.0:
                loss_k = loss_k + float(BETA_IMAG) * energy_loss_imag(E_imag, g)

            em, es = mean_sem(E_real)
            im, isem = mean_sem(E_imag)
            E_means.append(em); E_sems.append(es)
            I_means.append(im); I_sems.append(isem)

            # Orth penalties only after warmup
            if it >= WARMUP:
                for j in range(k):
                    aij = alpha_ij[k][j]
                    if aij <= 0.0:
                        continue
                    _, pen = overlap_abs_symmetric_stable(models[k], models[j], xs_it[k], xs_it[j])
                    loss_k = loss_k + aij * pen

            total_loss = total_loss + loss_k

        for opt in opts:
            opt.zero_grad(set_to_none=True)
        total_loss.backward()
        for opt in opts:
            opt.step()

        # sigma adaptation
        for k in range(n_states):
            if accs_it[k] < 0.4:
                sigmas[k] *= 0.8
            elif accs_it[k] > 0.6:
                sigmas[k] *= 1.2
            sigmas[k] = float(max(0.02, min(0.8, sigmas[k])))

        # logging
        if it == 0 or (it % PRINT_EVERY) == 0:
            phase = "pre-warmup" if it < WARMUP else ("ramp" if it <= (WARMUP + RAMP_STEPS) else "adaptive")
            parts = []
            for k in range(n_states):
                parts.append(
                    f"N{k}:E={E_means[k]:.4f}±{E_sems[k]:.1e},Im={I_means[k]:+.2e}±{I_sems[k]:.1e},"
                    f"acc={accs_it[k]:.2f},sig={sigmas[k]:.3f}"
                )
            print(f"it={it} phase={phase} beta_imag={BETA_IMAG:.1e} | " + " | ".join(parts))

        if S_mat is not None and (it % OVLP_EVERY) == 0:
            print_matrix(S_mat, title=f"Overlap |S_ij| (sub={OVLP_SUBSAMPLE}) at it={it}", fmt="{:0.3e}")
            # show alpha_ij (lower triangle)
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
    models, alpha_ij = train_vmc_ex(n_states=N_states, steps=N_steps, eta=Lr)
