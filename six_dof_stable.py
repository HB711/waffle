import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
print("device:", device)

# Capacitances
Cth = 1000.00
Cphi = 1000.00
Cv = 1000.00
Cp = 1000.00

# Physical constants / unit prefactor
h = 6.626
e = 1.60
UNIT = 100.0 * (4.0 * e**2) / (2.0 * h)  # = 100*4e^2/(2h)

# Josephson energy
EJ = 80.00

# MH
N_walkers = 1000
N_burn_in = 200
s=0.35

# Backprop batch (VERY important for stability/speed with Hessians)
BP_BATCH = N_walkers

# Network
HIDDEN_DIM = 216

# Opt
N_steps = 400
lr = 1e-2
seed = 0
torch.manual_seed(seed)

# =========================
# Build C and E-matrix from C^{-1}
# =========================
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

print(f"UNIT={UNIT:.6f}  EJ={EJ:.4f}")
print("Cinv[0,0], Cinv[4,4], Cinv[0,1], Cinv[0,3] =",
      float(Cinv[0,0]), float(Cinv[4,4]), float(Cinv[0,1]), float(Cinv[0,3]))

# =========================
# Helpers
# =========================
def wrap_pi(x):  # (-pi, pi]
    return (x + math.pi) % (2 * math.pi) - math.pi

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


# =========================
# Complex wavefunction model: psi = exp(f + i g)
# =========================
class NN_Waffle_Complex(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )
        self.out_f = nn.Linear(hidden_dim, 1)  # log|psi|
        self.out_g = nn.Linear(hidden_dim, 1)  # phase

        # optional: tame initial phase scale
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

# =========================
# Init / MH chain (sampling uses |psi|^2 = exp(2f) only)
# =========================
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

    acc= 0
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

    return (t1, t2, t3, p1, p2, p3, f), acc/(Nb*N)




# =========================
# Complex local energy (REAL PART)
# psi = exp(f + i g), T = - sum_ij E_ij ∂i∂j
#
# Re(Tpsi/psi) = - sum_ij E_ij [ ∂i∂j f + (∂i f)(∂j f) - (∂i g)(∂j g) ]
# E_loc = Re(Tpsi/psi) + V
# =========================
def local_energy_complex_matrix(t1, t2, t3, p1, p2, p3, model, E_mat):
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

    # Tr(E Hess f) and Tr(E Hess g) (we only need Hess f for the real part formula)
    EH_f = torch.zeros_like(f)
    for i in range(6):
        # Hessian row i for f: d/dx (grad_f[:, i])
        dgrad_i = torch.autograd.grad(
            grad_f[:, i], x,
            grad_outputs=torch.ones_like(grad_f[:, i]),
            create_graph=True
        )[0]  # (N,6)
        EH_f = EH_f + (dgrad_i * E_mat[i]).sum(dim=1)

    V = potential_energy(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5])

    # Real local energy
    E_loc = -(EH_f + quad_f - quad_g) + V
    return E_loc, f

# =========================
# Train
# =========================
def train_vmc(steps=N_steps, eta=lr):
    model = NN_Waffle_Complex().to(device=device, dtype=dtype)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=eta)

    state = initial(model, N=N_walkers)
    sigma=s

    E_hist = []
    Var_hist = []
    Acc_hist = []
    Sig_hist = []


    # variance weight (optimize E + lambda * Var via VMC-style estimator)
    lambda_var = 0.1

    for it in range(1, steps + 1):
        # refresh walkers by MH
        state, acc = mh_chain(model, state, N_burn_in,sigma)
        T1, T2, T3, P1, P2, P3, _ = state

        # backprop on a subset (critical for Hessian cost)
        if BP_BATCH < N_walkers:
            idx = torch.randint(0, N_walkers, (BP_BATCH,), device=device)
            bT1, bT2, bT3 = T1[idx], T2[idx], T3[idx]
            bP1, bP2, bP3 = P1[idx], P2[idx], P3[idx]
        else:
            bT1, bT2, bT3 = T1, T2, T3
            bP1, bP2, bP3 = P1, P2, P3

        # local energy (real) + log|psi|
        E_loc, f = local_energy_complex_matrix(bT1, bT2, bT3, bP1, bP2, bP3, model, E_mat)
        E_mean = E_loc.mean()

        # ---- VMC estimator for objective: E + lambda Var ----
        # O_tilde is detached, so we don't backprop through E_loc (avoids nasty higher-order terms)
        # E_loc_det = E_loc.detach()
        # E_mean_det = E_mean.detach()
        # var_det = (E_loc_det - E_mean_det).pow(2)

        # O_tilde = E_loc_det + lambda_var * var_det
        # O_mean = O_tilde.mean()
        loss = 2.0 * ((E_loc.detach() - E_mean.detach()) * f).mean()

        # score-function / log-derivative trick: ∇ ~ 2 < (O - <O>) f >
        # loss = 2.0 * ((O_tilde - O_mean) * f).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # stats on this backprop batch
        e_val = float(E_mean.detach())
        var_val = float(((E_loc - E_mean)**2).mean().detach())
        sdev_mean = math.sqrt(var_val) / math.sqrt(len(bT1))  # standard error of mean

        # adapt sigma (clamp!)
        if acc < 0.4:
            sigma*= 0.8
        elif acc > 0.6:
            sigma*= 1.2


        if it == 1 or it % 10 == 0:
            print(f"[it={it}] E={e_val:.6f}  Var={var_val:.6e}  SEM={sdev_mean:.3e}  "
                  f"acc={acc:.3f}  "
                  f"sigma={sigma:.4f}")

            E_hist.append(e_val)
            Var_hist.append(var_val)
            Acc_hist.append(acc)
            Sig_hist.append(sigma)

    return E_hist, Var_hist, Acc_hist, Sig_hist

def main():
    E_hist, Var_hist, Acc, Sig = train_vmc()

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
    plt.plot(x, Acc, label="acc_th")
    plt.xlabel("iteration (x10)")
    plt.ylabel("Acceptance")
    plt.grid(True)
    plt.legend()



    plt.show()

if __name__ == "__main__":
    main()
