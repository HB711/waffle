import math
import torch
import torch.nn as nn
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
# Physics params
# ============================================================
h = 6.626
e = 1.60
UNIT = 100.0 * (4.0 * e**2) / (2.0 * h)

Cth = 5.00
Cphi = 5.00
Cv = 5.00
Cp = 5.00
EJ = 80.00

# (optional) your harmonic estimate
w1 = math.sqrt(UNIT * 2 * EJ / (7 * Cth))
w2 = w1
w3 = math.sqrt(UNIT * 2 * 3 * EJ / (7 * Cth))
w4 = w3
w5 = math.sqrt(UNIT * 2 * 4 * EJ / (7 * Cth))
E_th = -6 * EJ + 0.5 * (w1 + w2 + w3 + w4 + w5)
print("Eth=", E_th)

# ============================================================
# Hyperparameters
# ============================================================
N_walkers  = 1000
N_burn_in  = 200
Sigma      = 0.35

Hidden_dim = 216
N_steps    = 200
Lr         = 1e-2
Seed       = 0
torch.manual_seed(Seed)

# ============================================================
# 6D C matrix => E_mat
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

# Kinetic in tilde basis
E_tilde = U @ E_mat @ U.T
E_tilde = 0.5 * (E_tilde + E_tilde.T)

# Use full 5x5 block (drop the first coord: common mode)
E5_mat = E_tilde[1:, 1:].contiguous()
E5_mat = 0.5 * (E5_mat + E5_mat.T)  # sym

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
    # Phi_tilde = [0, x2, x3, x4, x5, x6], then Phi = Phi_tilde @ U
    Phi_tilde = torch.stack(
        [torch.zeros_like(x2), x2, x3, x4, x5, x6],
        dim=1
    )  # (N,6)
    Phi = Phi_tilde @ U  # (N,6)
    t1, t2, t3, p1, p2, p3 = Phi.T
    return potential6D(t1, t2, t3, p1, p2, p3)

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
def local_energy_complex_5d(x2, x3, x4, x5, x6, model):
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

    V = potential5D(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4])

    E_loc = -(EH_f + quad_f - quad_g) + V
    return E_loc, f

# ============================================================
# Train VMC
# ============================================================
def train_vmc(steps, eta):
    model = NN_Waffle_Complex_5D().to(device=device, dtype=dtype)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=eta)

    state = initial(model, N_walkers)
    sigma = Sigma

    E_hist, Var_hist, Acc_hist, Sig_hist = [], [], [], []

    for it in range(1, steps + 1):
        state, acc = mh_chain(model, state, sigma=sigma, Nb=N_burn_in)
        x2, x3, x4, x5, x6, _ = state

        E_loc, f = local_energy_complex_5d(x2, x3, x4, x5, x6, model)
        E_mean = E_loc.mean()

        loss = 2.0 * ((E_loc.detach() - E_mean.detach()) * f).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        e_val = float(E_mean.detach())
        var_val = float(((E_loc - E_mean) ** 2).mean().detach())
        sem = math.sqrt(var_val) / math.sqrt(len(x2))

        if acc < 0.4:
            sigma *= 0.8
        elif acc > 0.6:
            sigma *= 1.2

        if it == 1 or it % 10 == 0:
            print(f"[it={it}] E={e_val:.6f}  Var={var_val:.6e}  SEM={sem:.3e}  "
                  f"acc={acc:.3f}  sigma={sigma:.4f}")
            E_hist.append(e_val)
            Var_hist.append(var_val)
            Acc_hist.append(acc)
            Sig_hist.append(sigma)

    return model, state, E_hist, Var_hist, Acc_hist, Sig_hist

# ============================================================
# Main / plots
# ============================================================
def main():
    model, state, E_hist, Var_hist, Acc, Sig = train_vmc(steps=N_steps, eta=Lr)

    # x = list(range(len(E_hist)))
    #
    # plt.figure()
    # plt.plot(x, E_hist, label="E (mean)")
    # plt.xlabel("iteration (x10)")
    # plt.ylabel("Energy")
    # plt.grid(True)
    # plt.legend()
    #
    # plt.figure()
    # plt.plot(x, Var_hist, label="Var(E_loc)")
    # plt.yscale("log")
    # plt.xlabel("iteration (x10)")
    # plt.ylabel("Variance (log scale)")
    # plt.grid(True)
    # plt.legend()
    #
    # plt.figure()
    # plt.plot(x, Acc, label="acceptance")
    # plt.xlabel("iteration (x10)")
    # plt.ylabel("Acceptance")
    # plt.grid(True)
    # plt.legend()
    #
    # x2, x3, x4, x5, x6, _ = state
    # coords = [x2, x3, x4, x5, x6]
    # names  = ["x2", "x3", "x4", "x5", "x6"]
    #
    # for i in range(5):
    #     plt.figure()
    #     plt.scatter(range(len(coords[i])), coords[i].detach().cpu().numpy(), s=3)
    #     plt.xlabel("walker index")
    #     plt.ylabel(names[i])
    #     plt.title(f"Walker distribution: {names[i]}")
    #     plt.grid(True)
    #
    # plt.show()

if __name__ == "__main__":
    main()