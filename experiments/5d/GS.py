import math
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ============================================================
# Device / dtype
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
print("device:", device)

TWO_PI = 2.0 * math.pi

def wrap_pi_torch(x: torch.Tensor) -> torch.Tensor:
    return x - TWO_PI * torch.floor((x + math.pi) / TWO_PI)

# ============================================================
# Physics constants / params
# ============================================================
h = 6.626
e = 1.60
UNIT = 100.0 * (4.0 * e**2) / (2.0 * h)

EJ = 80.00  # fixed

# ============================================================
# Hyperparameters (keep your defaults)
# ============================================================
N_walkers  = 2000
N_burn_in  = 200
Sigma      = 0.35

Hidden_dim = 256
N_steps    = 400
Lr         = 1e-2
Seed       = 0

# print every optimization step (set to 1 as you asked)
PRINT_EVERY = 1

torch.manual_seed(Seed)

# ============================================================
# U transform: Phi_tilde -> Phi (6D)
# ============================================================
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

# ============================================================
# Build E5_mat for a given common capacitance C
# ============================================================
def build_E5_mat(Cval: float) -> torch.Tensor:
    Cth = Cphi = Cv = Cp = float(Cval)

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

    E_tilde = U @ E_mat @ U.T
    E_tilde = 0.5 * (E_tilde + E_tilde.T)

    # drop common mode (index 0)
    E5_mat = E_tilde[1:, 1:].contiguous()
    E5_mat = 0.5 * (E5_mat + E5_mat.T)
    return E5_mat

# ============================================================
# Potential
# ============================================================
def potential6D(t1, t2, t3, p1, p2, p3, EJ_: float):
    alpha = 2 * torch.pi / 3
    term_phi1 = -2 * torch.cos(t1 - p1) - 2 * torch.cos(t2 - p1) - 2 * torch.cos(t3 - p1)
    term_phi2 = -2 * torch.cos(t1 - p2) - 2 * torch.cos(t2 - p2 + alpha) - 2 * torch.cos(t3 - p2 - alpha)
    term_phi3 = -2 * torch.cos(t1 - p3) - 2 * torch.cos(t2 - p3 - alpha) - 2 * torch.cos(t3 - p3 + alpha)
    return EJ_ * (term_phi1 + term_phi2 + term_phi3) / math.sqrt(3.0)

def potential5D(x2, x3, x4, x5, x6, EJ_: float):
    # Phi_tilde = [0, x2, x3, x4, x5, x6], then Phi = Phi_tilde @ U
    Phi_tilde = torch.stack(
        [torch.zeros_like(x2), x2, x3, x4, x5, x6],
        dim=1
    )  # (N,6)
    Phi = Phi_tilde @ U  # (N,6)
    t1, t2, t3, p1, p2, p3 = Phi.T
    return potential6D(t1, t2, t3, p1, p2, p3, EJ_)

# ============================================================
# Periodic embedding (5D -> 20)
# ============================================================
def periodic_emb_5d(x2, x3, x4, x5, x6):
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
# NN (complex): f=log|psi|, g=phase
# ============================================================
class NN_Waffle_Complex_5D(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=Hidden_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.out_f = nn.Linear(hidden_dim, 1)
        self.out_g = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.out_g.weight)
        nn.init.zeros_(self.out_g.bias)

    def forward(self, x2, x3, x4, x5, x6):
        x = periodic_emb_5d(x2, x3, x4, x5, x6)
        h = self.backbone(x)
        f = self.out_f(h).squeeze(-1)
        g = self.out_g(h).squeeze(-1)
        return f, g

    def f_only(self, x2, x3, x4, x5, x6):
        return self.forward(x2, x3, x4, x5, x6)[0]

# ============================================================
# Init / MH (5D)
# ============================================================
@torch.no_grad()
def initial(model, N=N_walkers):
    x2 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    x3 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    x4 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    x5 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    x6 = torch.rand(N, device=device, dtype=dtype) * TWO_PI - math.pi
    f  = model.f_only(x2, x3, x4, x5, x6)
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
# Local energy (complex, 5D kinetic + mapped 6D potential)
# E_loc = -(Tr(E5 Hess f) + grad f^T E5 grad f - grad g^T E5 grad g) + V(x)
# ============================================================
def local_energy_complex_5d(x2, x3, x4, x5, x6, model, E5_mat: torch.Tensor, EJ_: float):
    x = torch.stack([x2, x3, x4, x5, x6], dim=1)  # (N,5)
    x = x.detach().requires_grad_(True)

    f, g = model(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4])  # (N,)

    grad_f = torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]  # (N,5)
    grad_g = torch.autograd.grad(g, x, grad_outputs=torch.ones_like(g), create_graph=True)[0]  # (N,5)

    quad_f = torch.einsum("ni,ij,nj->n", grad_f, E5_mat, grad_f)
    quad_g = torch.einsum("ni,ij,nj->n", grad_g, E5_mat, grad_g)

    EH_f = torch.zeros_like(f)
    for i in range(5):
        dgrad_i = torch.autograd.grad(
            grad_f[:, i], x,
            grad_outputs=torch.ones_like(grad_f[:, i]),
            create_graph=True
        )[0]  # (N,5)
        EH_f = EH_f + (dgrad_i * E5_mat[i]).sum(dim=1)

    V = potential5D(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], EJ_)

    E_loc = -(EH_f + quad_f - quad_g) + V
    return E_loc, f

# ============================================================
# Theory E_th(C) and x=(UNIT/C)/EJ
# ============================================================
def E_th_of_C(Cval: float) -> float:
    C = float(Cval)
    w1 = math.sqrt(UNIT * 2 * EJ / (7 * C))
    w2 = w1
    w3 = math.sqrt(UNIT * 2 * 3 * EJ / (7 * C))
    w4 = w3
    w5 = math.sqrt(UNIT * 2 * 4 * EJ / (7 * C))
    return -6 * EJ + 0.5 * (w1 + w2 + w3 + w4 + w5)

def x_axis_of_C(Cval: float) -> float:
    return (UNIT / float(Cval)) / EJ

# ============================================================
# Train VMC for one C (prints every step E_vmc and E_th)
# ============================================================
def train_vmc_for_C(Cval: float, steps=N_steps, eta=Lr):
    # kinetic matrix for this C
    E5_mat = build_E5_mat(Cval)

    # theory value for this C
    Eth = E_th_of_C(Cval)
    xax = x_axis_of_C(Cval)

    model = NN_Waffle_Complex_5D().to(device=device, dtype=dtype)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=eta)

    state = initial(model, N_walkers)
    sigma = float(Sigma)

    E_last = None

    for it in range(1, steps + 1):
        state, acc = mh_chain(model, state, sigma=sigma, Nb=N_burn_in)
        x2, x3, x4, x5, x6, _ = state

        E_loc, f = local_energy_complex_5d(x2, x3, x4, x5, x6, model, E5_mat, EJ)
        E_mean = E_loc.mean()

        loss = 2.0 * ((E_loc.detach() - E_mean.detach()) * f).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # adapt sigma
        if acc < 0.4:
            sigma *= 0.8
        elif acc > 0.6:
            sigma *= 1.2

        E_last = float(E_mean.detach())

        if (it % PRINT_EVERY) == 0:
            var_val = float(((E_loc - E_mean) ** 2).mean().detach())
            sem = math.sqrt(var_val) / math.sqrt(len(x2))
            print(
                f"[C={Cval:.6f} x={(xax):.6e} it={it:04d}] "
                f"E_vmc={E_last:.8f}  E_th={Eth:.8f}  "
                f"Var={var_val:.3e}  SEM={sem:.3e}  acc={acc:.3f}  sigma={sigma:.4f}"
            )

    return E_last, Eth, xax

# ============================================================
# Main sweep and plot
# ============================================================
def main():
    # 50 values in [0.5, 10]
    Cs = torch.linspace(0.5, 10.0, 50).tolist()

    xs = []
    E_th_list = []
    E_vmc_list = []

    for i, Cval in enumerate(Cs, 1):
        # optional: different seed per C to reduce identical trajectories
        torch.manual_seed(Seed + i)

        print("\n" + "="*80)
        print(f"Running C={Cval:.6f}   (#{i}/50)   EJ={EJ}")
        print("="*80)

        Ev, Eth, xax = train_vmc_for_C(Cval, steps=N_steps, eta=Lr)

        xs.append(xax)
        E_th_list.append(Eth)
        E_vmc_list.append(Ev)

        print(f"== DONE C={Cval:.6f}: x={(xax):.6e}  E_th={Eth:.8f}  E_vmc={Ev:.8f}")

    # sort by x for clean plot
    order = sorted(range(len(xs)), key=lambda k: xs[k])
    xs = [xs[k] for k in order]
    E_th_list = [E_th_list[k] for k in order]
    E_vmc_list = [E_vmc_list[k] for k in order]

    plt.figure()
    plt.plot(xs, E_th_list, label="E_th (harmonic est.)")
    plt.plot(xs, E_vmc_list, label="E_VMC (NN-VMC)")
    plt.xlabel(r"(UNIT/C) / EJ")
    plt.ylabel("Ground state energy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("energy_vs_x.png", dpi=200, bbox_inches="tight")
    print("Figure saved to energy_vs_x.png")

if __name__ == "__main__":
    main()