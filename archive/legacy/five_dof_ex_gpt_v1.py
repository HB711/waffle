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
# Hyperparameters (simplified like 6D)
# ============================================================
# MH
N_walkers = 5000
N_burn_in = 100
Sigma0 = 0.30

# NN / Opt
Hidden_Dim = 256
Seed = 0
torch.manual_seed(Seed)

# Training
N_states = 8            # <<< set this as you like
N_steps  = 2000
lr       = 1e-2

# alpha schedule (like 6D: warmup then ramp)
alpha_final  = 200.0   # final strength (you used big alpha_B in PhaseB)
alpha_warmup = 300      # first 300 steps alpha=0
alpha_power  = 2.0      # ramp curvature
alpha_warm   = 0.0

# Logging
PRINT_EVERY = 10
OVLP_EVERY  = 50
OVLP_SUBSAMPLE = 512    # subsample walkers when printing overlap (faster)
EPS_OVLP = 1e-6         # for clamp in penalty

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
# Potential
# ============================================================
def potential6D(t1, t2, t3, p1, p2, p3):
    alpha = 2 * torch.pi / 3
    term_phi1 = -2 * torch.cos(t1 - p1) - 2 * torch.cos(t2 - p1) - 2 * torch.cos(t3 - p1)
    term_phi2 = -2 * torch.cos(t1 - p2) - 2 * torch.cos(t2 - p2 + alpha) - 2 * torch.cos(t3 - p2 - alpha)
    term_phi3 = -2 * torch.cos(t1 - p3) - 2 * torch.cos(t2 - p3 - alpha) - 2 * torch.cos(t3 - p3 + alpha)
    return EJ * (term_phi1 + term_phi2 + term_phi3) / math.sqrt(3.0)

def potential5D(x2, x3, x4, x5, x6):
    # x1 set to 0 in Phi_tilde basis
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
# Local energy (same logic as your 5D code)
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
        Ek = E5[i + 1]  # x2..x6 correspond to indices 1..5 in Phi_tilde
        df, dg = dfs[i], dgs[i]
        d2f, d2g = d2fs[i], d2gs[i]
        E_real = E_real + (-Ek) * (d2f + df**2 - dg**2)
        E_imag = E_imag + (-Ek) * (d2g + 2.0 * df * dg)

    V = potential5D(xs[0], xs[1], xs[2], xs[3], xs[4])
    E_real = E_real + V
    return E_real, E_imag, f, g

def energy_loss_from_real_local_energy(E_real: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    Er_det = E_real.detach()
    return 2.0 * ((Er_det - Er_det.mean()) * f).mean()

def mean_sem(x: torch.Tensor):
    x_det = x.detach()
    mean = x_det.mean()
    std  = x_det.std(unbiased=False)
    sem  = std / math.sqrt(x_det.numel())
    return float(mean), float(sem)

# ============================================================
# alpha schedule (same idea as 6D)
# ============================================================
def alpha_schedule(it, steps, alpha_final, warmup=300, power=2.0, alpha_warm=0.0):
    if it < warmup:
        return float(alpha_warm)
    denom = max(1, steps - warmup - 1)
    t = (it - warmup) / denom
    t = float(max(0.0, min(1.0, t)))
    return float(alpha_final) * (t ** power)

# ============================================================
# Stable overlap estimator (keep YOUR penalty form: 1/(1-S)-1)
# ============================================================
def _stable_complex_mean_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return mean( exp(a) * exp(1j*b) ) stably by shifting a by max(a).
    a,b: real tensors (N,)
    """
    amax = a.max().detach()
    z = torch.exp(a - amax).to(cdtype) * torch.exp(1j * b.to(cdtype))
    return z.mean() * torch.exp(amax).to(cdtype)

def compute_Sabs_and_penalty_for_pair(model_i, x_i, model_j_frozen, x_j, eps=EPS_OVLP):
    """
    Sabs = sqrt(|A_ij|*|B_ij|)
    pen  = 1/(1-clamp(Sabs))-1    <-- keep exactly your 5D penalty style
    gradient flows only to model_i (model_j_frozen evaluated under no_grad).
    """
    x2i, x3i, x4i, x5i, x6i = x_i
    x2j, x3j, x4j, x5j, x6j = x_j

    # model_i with grad
    f_i_i, g_i_i = model_i(x2i, x3i, x4i, x5i, x6i)
    f_i_j, g_i_j = model_i(x2j, x3j, x4j, x5j, x6j)

    # model_j frozen
    with torch.no_grad():
        f_j_i, g_j_i = model_j_frozen(x2i, x3i, x4i, x5i, x6i)
        f_j_j, g_j_j = model_j_frozen(x2j, x3j, x4j, x5j, x6j)

    # A_ij = E_i[ psi_j/psi_i ]
    a  = (f_j_i - f_i_i).to(dtype)
    b  = (g_j_i - g_i_i).to(dtype)
    A_ij = _stable_complex_mean_exp(a, b)

    # B_ij = E_j[ psi_i/psi_j ]
    a2 = (f_i_j - f_j_j).to(dtype)
    b2 = (g_i_j - g_j_j).to(dtype)
    B_ij = _stable_complex_mean_exp(a2, b2)

    Sabs = torch.sqrt(torch.abs(A_ij) * torch.abs(B_ij)).to(dtype)

    # keep YOUR penalty: 1/(1-S)-1
    Sclamp = torch.clamp(Sabs, max=(1.0 - eps))
    pen = (1.0 / (1.0 - Sclamp)) - 1.0
    return Sabs, pen

# ============================================================
# Training (6D-style)
# ============================================================
def train_vmc_ex(n_states=N_states, steps=N_steps, eta=lr):
    models = [NN_Waffle_Complex_5D().to(device=device, dtype=dtype) for _ in range(n_states)]
    opts   = [torch.optim.Adam(m.parameters(), lr=eta) for m in models]

    states = [initial(models[i], N_walkers) for i in range(n_states)]
    sigmas = [Sigma0 for _ in range(n_states)]

    for it in range(steps):
        alpha_it = alpha_schedule(
            it, steps,
            alpha_final=alpha_final,
            warmup=alpha_warmup,
            power=alpha_power,
            alpha_warm=alpha_warm
        )

        # per-iter buffers (like 6D)
        accs_it = [0.0 for _ in range(n_states)]
        xs_it   = [None for _ in range(n_states)]

        # MH for each state
        for k in range(n_states):
            states[k], acc = mh_chain(models[k], states[k], sigma=sigmas[k], Nb=N_burn_in)
            accs_it[k] = acc
            x2,x3,x4,x5,x6,_ = states[k]
            xs_it[k] = (x2,x3,x4,x5,x6)

        # build total loss
        total_loss = torch.zeros((), device=device, dtype=dtype)
        E_means, E_sems = [], []

        # store overlaps for printing
        S_mat = [[None for _ in range(n_states)] for _ in range(n_states)]

        for k in range(n_states):
            x2,x3,x4,x5,x6 = xs_it[k]
            E_real, _, f, _ = local_energy_5D(x2,x3,x4,x5,x6, models[k])
            lk = energy_loss_from_real_local_energy(E_real, f)

            e_mean, e_sem = mean_sem(E_real)
            E_means.append(e_mean)
            E_sems.append(e_sem)

            # orth penalties: only push later states away from earlier ones (j<k)
            if alpha_it > 0.0:
                for j in range(k):
                    Sabs, pen = compute_Sabs_and_penalty_for_pair(models[k], xs_it[k], models[j], xs_it[j])
                    lk = lk + alpha_it * pen
                    S_mat[k][j] = Sabs.detach()

            total_loss = total_loss + lk

        # step
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
            parts = []
            for k in range(n_states):
                parts.append(f"S{k}:E={E_means[k]:.4f}±{E_sems[k]:.1e},acc={accs_it[k]:.2f},sig={sigmas[k]:.3f}")
            msg = f"it={it} alpha={alpha_it:.2e} | " + " | ".join(parts)

            # print pair overlaps (use subsample for speed)
            if (it % OVLP_EVERY) == 0 and n_states > 1:
                s_parts = []
                for i in range(n_states):
                    for j in range(i):
                        xi = xs_it[i]
                        xj = xs_it[j]
                        if OVLP_SUBSAMPLE is not None:
                            xi = tuple(t[:OVLP_SUBSAMPLE] for t in xi)
                            xj = tuple(t[:OVLP_SUBSAMPLE] for t in xj)

                        # if already computed in loss (i>j and alpha>0), reuse; else compute now
                        if S_mat[i][j] is None:
                            with torch.no_grad():
                                Sabs, _ = compute_Sabs_and_penalty_for_pair(models[i], xi, models[j], xj)
                            sval = float(Sabs.detach().clamp(0.0, 1.0).cpu())
                        else:
                            sval = float(S_mat[i][j].detach().clamp(0.0, 1.0).cpu())
                        s_parts.append(f"S({i},{j})={sval:.3e}")

                if len(s_parts) > 0:
                    msg = msg + " || " + " ".join(s_parts)

            print(msg)

    return models

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    models = train_vmc_ex(n_states=N_states, steps=N_steps, eta=lr)
