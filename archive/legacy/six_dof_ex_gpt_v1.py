import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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
UNIT = 100.0 * (4.0 * e**2) / (2.0 * h)  # = 100*4e^2/(2h)

Cth = 10.00
Cphi = 10.00
Cv = 30.00
Cp = 47.00
EJ = 80.00

# ============================================================
# VMC / MH params
# ============================================================
N_walkers = 1000
N_burn_in = 40
Sigma = 0.35

Hidden_dim = 216

N_steps = 2400
Lr = 1e-2

# Orthogonality schedule (your request)
alpha_final = 100.0          # final alpha strength
alpha_warmup = 400         # first 300 steps alpha=0 (or alpha_warm)
alpha_power = 2.0          # ramp curvature
alpha_warm = 0.0           # warmup alpha value (0 or tiny)

N_states = 8

Seed = 0
torch.manual_seed(Seed)

# ============================================================
# Capacitance matrix -> E_mat
# ============================================================
C = torch.tensor(
    [[Cth + 3*Cv + 2*Cp, -Cp, -Cp, -Cv, -Cv, -Cv],
     [-Cp, Cth + 3*Cv + 2*Cp, -Cp, -Cv, -Cv, -Cv],
     [-Cp, -Cp, Cth + 3*Cv + 2*Cp, -Cv, -Cv, -Cv],
     [-Cv, -Cv, -Cv, Cth + 3*Cv + 2*Cp, -Cp, -Cp],
     [-Cv, -Cv, -Cv, -Cp, Cth + 3*Cv + 2*Cp, -Cp],
     [-Cv, -Cv, -Cv, -Cp, -Cp, Cth + 3*Cv + 2*Cp]],
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
    # map to (-pi, pi]
    return x - TWO_PI * torch.floor((x + math.pi) / TWO_PI)

def alpha_schedule(it, steps, alpha_final, warmup=300, power=2.0, alpha_warm=0.0):
    """
    it < warmup: alpha = alpha_warm (default 0)
    it >= warmup: ramp to alpha_final with (t^power)
    """
    if it < warmup:
        return alpha_warm
    denom = max(1, steps - warmup - 1)
    t = (it - warmup) / denom
    t = float(max(0.0, min(1.0, t)))
    return alpha_final * (t ** power)

def potential_energy(t1, t2, t3, p1, p2, p3):
    alpha = 2 * torch.pi / 3
    term_phi1 = -2 * torch.cos(t1 - p1) - 2 * torch.cos(t2 - p1) - 2 * torch.cos(t3 - p1)
    term_phi2 = -2 * torch.cos(t1 - p2) - 2 * torch.cos(t2 - p2 - alpha) - 2 * torch.cos(t3 - p2 + alpha)
    term_phi3 = -2 * torch.cos(t1 - p3) - 2 * torch.cos(t2 - p3 + alpha) - 2 * torch.cos(t3 - p3 - alpha)
    return EJ * (term_phi1 + term_phi2 + term_phi3) / math.sqrt(3)

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

# ============================================================
# Model
# ============================================================
class NN_Waffle_Complex(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=Hidden_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.SiLU(),
        )
        self.out_f = nn.Linear(hidden_dim, 1)  # log|psi|
        self.out_g = nn.Linear(hidden_dim, 1)  # phase

        # tame initial phase scale
        nn.init.zeros_(self.out_g.weight)
        nn.init.zeros_(self.out_g.bias)

    def forward(self, t1, t2, t3, p1, p2, p3):
        x = periodic_emb(t1, t2, t3, p1, p2, p3)
        h = self.backbone(x)
        f = self.out_f(h).squeeze(-1)
        g = self.out_g(h).squeeze(-1)
        return f, g

    def f_only(self, t1, t2, t3, p1, p2, p3):
        return self.forward(t1, t2, t3, p1, p2, p3)[0]

# ============================================================
# Initialization + MH chain (samples from |psi|^2 via f-only)
# ============================================================
@torch.no_grad()
def initial(model, N=N_walkers):
    t1 = torch.rand(N, device=device, dtype=dtype) * 2 * torch.pi - torch.pi
    t2 = torch.rand(N, device=device, dtype=dtype) * 2 * torch.pi - torch.pi
    t3 = torch.rand(N, device=device, dtype=dtype) * 2 * torch.pi - torch.pi
    p1 = torch.rand(N, device=device, dtype=dtype) * 2 * torch.pi - torch.pi
    p2 = torch.rand(N, device=device, dtype=dtype) * 2 * torch.pi - torch.pi
    p3 = torch.rand(N, device=device, dtype=dtype) * 2 * torch.pi - torch.pi
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
# Local energy (real part only, consistent with your current setup)
# ============================================================
def local_energy_complex_matrix(t1, t2, t3, p1, p2, p3, model):
    x = torch.stack([t1, t2, t3, p1, p2, p3], dim=1)  # (N,6)
    x = x.detach().requires_grad_(True)

    f, g = model(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5])  # (N,)

    grad_f = torch.autograd.grad(
        f, x, grad_outputs=torch.ones_like(f), create_graph=True
    )[0]  # (N,6)

    grad_g = torch.autograd.grad(
        g, x, grad_outputs=torch.ones_like(g), create_graph=True
    )[0]  # (N,6)

    quad_f = torch.einsum("ni,ij,nj->n", grad_f, E_mat, grad_f)
    quad_g = torch.einsum("ni,ij,nj->n", grad_g, E_mat, grad_g)

    # Tr(E Hess f)
    EH_f = torch.zeros_like(f)
    for i in range(6):
        dgrad_i = torch.autograd.grad(
            grad_f[:, i], x,
            grad_outputs=torch.ones_like(grad_f[:, i]),
            create_graph=True
        )[0]  # (N,6)
        EH_f = EH_f + (dgrad_i * E_mat[i]).sum(dim=1)

    V = potential_energy(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5])

    # Real local energy
    E_loc_real = -(EH_f + quad_f - quad_g) + V
    return E_loc_real, f

# ============================================================
# Stable complex mean: mean(exp(df) * exp(1j*dg)) without overflow/underflow
# ============================================================
def stable_complex_mean(df: torch.Tensor, dg: torch.Tensor) -> torch.Tensor:
    """
    Compute mean( exp(df) * exp(1j*dg) ) stably by shifting df by its max.
    df, dg are real tensors of shape (N,).
    Returns a complex scalar.
    """
    m = df.max()
    w = torch.exp(df - m)               # in (0,1]
    z = w * torch.exp(1j * dg)          # complex (N,)
    return torch.exp(m) * z.mean()      # complex scalar

# ============================================================
# Symmetric orthogonality score with stable A,B:
# S = sqrt(|A| |B|), A=E_i[psi_j/psi_i], B=E_j[psi_i/psi_j]
# Penalty only backprops to the later state (model_i)
# ============================================================
def overlap_abs_symmetric_stable(model_i, model_j, x_i, x_j, eps=1e-8):
    x1i, x2i, x3i, x4i, x5i, x6i = x_i
    x1j, x2j, x3j, x4j, x5j, x6j = x_j

    # ---- A: samples from i ----
    f_i_i, g_i_i = model_i(x1i, x2i, x3i, x4i, x5i, x6i)  # grad
    with torch.no_grad():
        f_j_i, g_j_i = model_j(x1i, x2i, x3i, x4i, x5i, x6i)

    df_A = (f_j_i - f_i_i).to(dtype)
    dg_A = (g_j_i - g_i_i).to(dtype)
    A = stable_complex_mean(df_A, dg_A)
    A_abs = torch.abs(A)

    # ---- B: samples from j ----
    f_i_j, g_i_j = model_i(x1j, x2j, x3j, x4j, x5j, x6j)  # grad
    with torch.no_grad():
        f_j_j, g_j_j = model_j(x1j, x2j, x3j, x4j, x5j, x6j)

    df_B = (f_i_j - f_j_j).to(dtype)
    dg_B = (g_i_j - g_j_j).to(dtype)
    B = stable_complex_mean(df_B, dg_B)
    B_abs = torch.abs(B)

    # symmetric real score
    S = torch.sqrt(torch.clamp(A_abs * B_abs, min=0.0))

    # barrier penalty on S^2
    S2 = torch.clamp(S**2, 0.0, 1.0 - eps)
    pen = (1.0 / (1.0 - S2)) - 1.0
    return S, pen

# ============================================================
# Training
# ============================================================
def train_vmc_ex(n_states, steps, eta):
    models = [NN_Waffle_Complex().to(device=device, dtype=dtype) for _ in range(n_states)]
    opts = [torch.optim.Adam(m.parameters(), lr=eta) for m in models]
    states = [initial(models[i], N_walkers) for i in range(n_states)]
    sigmas = [Sigma for _ in range(n_states)]

    for it in range(steps):
        alpha_it = alpha_schedule(
            it, steps,
            alpha_final=alpha_final,
            warmup=alpha_warmup,
            power=alpha_power,
            alpha_warm=alpha_warm
        )

        accs_it = [0.0 for _ in range(n_states)]
        xs_it   = [None for _ in range(n_states)]

        # MH sampling
        for k in range(n_states):
            states[k], acc = mh_chain(models[k], states[k], N_burn_in, sigmas[k])
            accs_it[k] = acc
            x1, x2, x3, x4, x5, x6, _ = states[k]
            xs_it[k] = (x1, x2, x3, x4, x5, x6)

        total_loss = torch.zeros((), device=device, dtype=dtype)
        E_means, E_sems = [], []

        # store S_ij for printing (i>j)
        S_mat = [[None for _ in range(n_states)] for _ in range(n_states)]

        for k in range(n_states):
            x1, x2, x3, x4, x5, x6, _ = states[k]
            E_real, f = local_energy_complex_matrix(x1, x2, x3, x4, x5, x6, models[k])

            E_det = E_real.detach()
            loss_k = 2.0 * ((E_det - E_det.mean()) * f).mean()

            e_mean = E_det.mean()
            std = E_det.std(unbiased=False)
            e_sem = std / math.sqrt(E_det.numel())
            E_means.append(e_mean)
            E_sems.append(e_sem)

            # Orthogonality penalty (only affects later states)
            if alpha_it > 0.0:
                for j in range(k):
                    S, pen = overlap_abs_symmetric_stable(models[k], models[j], xs_it[k], xs_it[j])
                    loss_k = loss_k + alpha_it * pen
                    S_mat[k][j] = S.detach()

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

        # logging
        if it == 0 or (it % 10) == 0:
            parts = []
            for k in range(n_states):
                parts.append(
                    f"N{k}:E={E_means[k]:.4f}±{E_sems[k]:.1e},acc={accs_it[k]:.2f},sig={sigmas[k]:.3f}"
                )
            msg = f"it={it} alpha={alpha_it:.2e} | " + " | ".join(parts)

            # print all pair symmetric scores S_ij (i>j)
            s_parts = []
            for i in range(n_states):
                for j in range(i):
                    if S_mat[i][j] is None:
                        with torch.no_grad():
                            S, _ = overlap_abs_symmetric_stable(models[i], models[j], xs_it[i], xs_it[j])
                        s_val = float(S)
                    else:
                        s_val = float(S_mat[i][j])
                    s_parts.append(f"S({i},{j})={s_val:.3e}")

            if len(s_parts) > 0:
                msg = msg + " || " + " ".join(s_parts)

            print(msg)

    return models

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    models = train_vmc_ex(n_states=N_states, steps=N_steps, eta=Lr)
