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

# ============================================================
# Constants
# ============================================================
h = 6.626
e = 1.60
UNIT = 100.0 * (4.0 * e**2) / (2.0 * h)

# Circuit params
Cth = 10.00
Cphi = 10.00
Cv = 30.00
Cp = 47.00
EJ = 80.00

# FIX alpha here (requirement 1)
ALPHA = -2.0 * math.pi / 3.0

# VMC / MCMC
N_walkers = 4000
N_burn_in = 200          # MH steps per iteration
Sigma = 0.35

# Local sampling window (requirement 2)
Delta = 0.80             # rad, half-width of local box around the global minimum

# NN
Hidden_dim = 256

# Training
N_steps = 600
Lr = 1e-2
Seed = 0
torch.manual_seed(Seed)

TWO_PI = 2.0 * math.pi

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
# Global minimum point (your point)
# theta* = (0,0,2pi/3), phi* = (pi/6, pi/6, -pi/2)
# ============================================================
theta_star = torch.tensor([0.0, 0.0, 2.0*math.pi/3.0], device=device, dtype=dtype)
phi_star   = torch.tensor([math.pi/6.0, math.pi/6.0, -math.pi/2.0], device=device, dtype=dtype)

# ============================================================
# Helpers
# ============================================================
def wrap_centered(x: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """
    Map x to (center - pi, center + pi]
    """
    return center + (torch.remainder(x - center + math.pi, TWO_PI) - math.pi)

def potential_energy(t1, t2, t3, p1, p2, p3):
    """
    Using fixed alpha = -2pi/3
    """
    alpha = ALPHA
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

# ============================================================
# Model
# ============================================================
class NN_Waffle_Complex(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=Hidden_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
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
# Local init: sample uniformly in a box around the minimum
# ============================================================
@torch.no_grad()
def initial_local(model, N=N_walkers, delta=Delta):
    # Uniform in [-delta, +delta] around the minimum
    u = (torch.rand(N, 3, device=device, dtype=dtype) * 2.0 - 1.0) * delta
    v = (torch.rand(N, 3, device=device, dtype=dtype) * 2.0 - 1.0) * delta

    th = theta_star[None, :] + u
    ph = phi_star[None, :] + v

    # keep a consistent branch around the center (optional but nice)
    th = wrap_centered(th, theta_star[None, :])
    ph = wrap_centered(ph, phi_star[None, :])

    t1, t2, t3 = th[:, 0], th[:, 1], th[:, 2]
    p1, p2, p3 = ph[:, 0], ph[:, 1], ph[:, 2]

    f = model.f_only(t1, t2, t3, p1, p2, p3)
    return (t1, t2, t3, p1, p2, p3, f)

# ============================================================
# Local MH: random-walk proposal + hard box constraint
# ============================================================
@torch.no_grad()
def mh_chain_local(model, state, Nb, sigma, delta=Delta):
    t1, t2, t3, p1, p2, p3, f = state
    f = model.f_only(t1, t2, t3, p1, p2, p3)
    N = t1.shape[0]

    # represent all coords in the centered branch near the minimum
    t1 = wrap_centered(t1, theta_star[0]); t2 = wrap_centered(t2, theta_star[1]); t3 = wrap_centered(t3, theta_star[2])
    p1 = wrap_centered(p1, phi_star[0]);   p2 = wrap_centered(p2, phi_star[1]);   p3 = wrap_centered(p3, phi_star[2])

    acc = 0
    for _ in range(Nb):
        nt1 = t1 + torch.randn_like(t1) * sigma
        nt2 = t2 + torch.randn_like(t2) * sigma
        nt3 = t3 + torch.randn_like(t3) * sigma
        np1 = p1 + torch.randn_like(p1) * sigma
        np2 = p2 + torch.randn_like(p2) * sigma
        np3 = p3 + torch.randn_like(p3) * sigma

        # hard box around the minimum
        in_box = (
            ((nt1 - theta_star[0]).abs() <= delta) &
            ((nt2 - theta_star[1]).abs() <= delta) &
            ((nt3 - theta_star[2]).abs() <= delta) &
            ((np1 - phi_star[0]).abs()   <= delta) &
            ((np2 - phi_star[1]).abs()   <= delta) &
            ((np3 - phi_star[2]).abs()   <= delta)
        )

        # only need nf for proposals; ok to compute for all then mask
        nf = model.f_only(nt1, nt2, nt3, np1, np2, np3)
        log_alpha = 2.0 * (nf - f)
        accept = in_box & (torch.log(torch.rand_like(log_alpha)) < log_alpha)

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
# Local energy (real part) using E_mat
# ============================================================
def local_energy_complex_matrix(t1, t2, t3, p1, p2, p3, model):
    x = torch.stack([t1, t2, t3, p1, p2, p3], dim=1)  # (N,6)
    x = x.detach().requires_grad_(True)

    f, g = model(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5])

    grad_f = torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]  # (N,6)
    grad_g = torch.autograd.grad(g, x, grad_outputs=torch.ones_like(g), create_graph=True)[0]  # (N,6)

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
    E_loc = -(EH_f + quad_f - quad_g) + V
    return E_loc, f

# ============================================================
# Train VMC
# ============================================================
def train_vmc(steps, eta):
    model = NN_Waffle_Complex().to(device=device, dtype=dtype)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=eta)

    state = initial_local(model, N_walkers, Delta)  # local init
    sigma = Sigma

    E_hist, Var_hist, Acc_hist, Sig_hist = [], [], [], []

    for it in range(1, steps + 1):
        # local MH
        state, acc = mh_chain_local(model, state, N_burn_in, sigma, Delta)

        T1, T2, T3, P1, P2, P3, _ = state
        E_loc, f = local_energy_complex_matrix(T1, T2, T3, P1, P2, P3, model)
        E_mean = E_loc.mean()

        loss = 2.0 * ((E_loc.detach() - E_mean.detach()) * f).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        e_val = float(E_mean.detach())
        var_val = float(((E_loc - E_mean)**2).mean().detach())
        sem_naive = math.sqrt(var_val) / math.sqrt(len(T1))

        # sigma adaptation (keep acceptance roughly 0.4~0.6)
        if acc < 0.4:
            sigma *= 0.9
        elif acc > 0.6:
            sigma *= 1.1

        if it == 1 or it % 10 == 0:
            print(f"[it={it}] E={e_val:.6f}  Var={var_val:.6e}  SEM={sem_naive:.3e}  "
                  f"acc={acc:.3f}  sigma={sigma:.4f}  Delta={Delta:.3f}")

            E_hist.append(e_val)
            Var_hist.append(var_val)
            Acc_hist.append(acc)
            Sig_hist.append(sigma)

    return model, state, E_hist, Var_hist, Acc_hist, Sig_hist

# ============================================================
# Main
# ============================================================
def main():
    model, state, E_hist, Var_hist, Acc, Sig = train_vmc(steps=N_steps, eta=Lr)

    x = list(range(len(E_hist)))

    plt.figure()
    plt.plot(x, E_hist, label="E (mean)")
    plt.xlabel("iteration (x10)")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(x, Var_hist, label="Var(E_loc)")
    plt.yscale("log")
    plt.xlabel("iteration (x10)")
    plt.ylabel("Variance (log scale)")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(x, Acc, label="acceptance")
    plt.xlabel("iteration (x10)")
    plt.ylabel("Acceptance")
    plt.grid(True)
    plt.legend()

    # Final walker scatter plots (around the minimum)
    T1, T2, T3, P1, P2, P3, _ = state
    coords = [T1, T2, T3, P1, P2, P3]
    names = ["theta1", "theta2", "theta3", "phi1", "phi2", "phi3"]
    centers = [theta_star[0], theta_star[1], theta_star[2], phi_star[0], phi_star[1], phi_star[2]]

    for i in range(6):
        plt.figure()
        plt.scatter(
            range(len(coords[i])),
            coords[i].detach().cpu().numpy(),
            s=3
        )
        plt.axhline(float(centers[i].detach().cpu()), linestyle="--")
        plt.xlabel("walker index")
        plt.ylabel(names[i])
        plt.title(f"Walker distribution (local): {names[i]}")
        plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()