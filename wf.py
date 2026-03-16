import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

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
s = 0.35

# Backprop batch (VERY important for stability/speed with Hessians)
BP_BATCH = N_walkers

# Network
HIDDEN_DIM = 216

# Opt
N_steps = 300
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
TWO_PI = 2.0 * math.pi

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
    return torch.stack(feats, dim=-1)  # (..., 24)

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

# =========================
# Uniform diagnostics (NO weighted integral)
# =========================
@torch.no_grad()
def sample_uniform_points(Nu, device, dtype):
    x = torch.rand(Nu, 6, device=device, dtype=dtype) * TWO_PI - math.pi
    return x  # (Nu,6)

@torch.no_grad()
def eval_logpsi2(model, x):
    f = model.f_only(x[:,0], x[:,1], x[:,2], x[:,3], x[:,4], x[:,5])
    return 2.0 * f  # log(|psi|^2)

def binned_mean(x, y, nbins=60, x_min=-math.pi, x_max=math.pi):
    edges = np.linspace(x_min, x_max, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    inds = np.digitize(x, edges) - 1
    inds = np.clip(inds, 0, nbins - 1)

    sums = np.zeros(nbins, dtype=np.float64)
    cnts = np.zeros(nbins, dtype=np.int64)
    np.add.at(sums, inds, y)
    np.add.at(cnts, inds, 1)

    means = np.full(nbins, np.nan, dtype=np.float64)
    mask = cnts > 0
    means[mask] = sums[mask] / cnts[mask]
    return centers, means, cnts

def plot_psi2_marginals(model, state, Nu=200000, nbins=60,
                        uniform_mode="mean_logpsi2"):
    """
    12 plots:
      - Walkers (6): histogram density of each coordinate (walkers ~ |psi|^2)
      - Uniform (6): bin-wise MEAN over uniform samples (NO weighting).
        uniform_mode:
          "mean_logpsi2"  -> plot mean(log|psi|^2) per bin  [most stable]
          "geom_mean_psi2"-> exp(mean(log|psi|^2)) per bin  [stable positive]
          "mean_psi2"     -> mean(|psi|^2) per bin          [may be dominated by outliers]
    """
    names = ["theta1","theta2","theta3","phi1","phi2","phi3"]

    # --- walkers histograms ---
    T1, T2, T3, P1, P2, P3, _ = state
    walkers = [T1, T2, T3, P1, P2, P3]
    walkers_np = [w.detach().cpu().numpy() for w in walkers]

    for i, nm in enumerate(names):
        plt.figure()
        plt.hist(walkers_np[i], bins=nbins, range=(-math.pi, math.pi), density=True)
        plt.xlabel(nm)
        plt.ylabel("density (walkers)")
        plt.title(f"Walkers marginal ~ |psi|^2 : {nm}")
        plt.grid(True)

    # --- uniform bin-wise means (NO weights) ---
    xU = sample_uniform_points(Nu, device=device, dtype=dtype)  # (Nu,6)
    logpsi2 = eval_logpsi2(model, xU).detach().cpu().numpy()    # (Nu,)
    xU_np = xU.detach().cpu().numpy()

    if uniform_mode == "mean_logpsi2":
        y_all = logpsi2
        y_label = "mean log(|psi|^2) in bin"
        title_suffix = "Uniform-bin mean of log(|psi|^2) (NO weights)"
    elif uniform_mode == "geom_mean_psi2":
        y_all = logpsi2
        y_label = "exp(mean log(|psi|^2)) in bin"
        title_suffix = "Uniform-bin geometric mean of |psi|^2 (NO weights)"
    elif uniform_mode == "mean_psi2":
        # NOTE: exp(logpsi2) can overflow; we stabilize by shifting.
        # This rescales all values by a constant exp(-max), changing only vertical scale, not shape.
        shift = float(np.max(logpsi2))
        psi2_stable = np.exp(logpsi2 - shift)
        y_all = psi2_stable
        y_label = "mean(|psi|^2) in bin  (scaled)"
        title_suffix = "Uniform-bin mean of |psi|^2 (NO weights, scaled)"
    else:
        raise ValueError("uniform_mode must be one of: mean_logpsi2, geom_mean_psi2, mean_psi2")

    for i, nm in enumerate(names):
        centers, means, cnts = binned_mean(xU_np[:, i], y_all, nbins=nbins)

        plt.figure()
        if uniform_mode == "geom_mean_psi2":
            # means currently mean(logpsi2); convert
            y_plot = np.exp(means)
        else:
            y_plot = means

        plt.plot(centers, y_plot, marker="o", markersize=3, linewidth=1)
        plt.xlabel(nm)
        plt.ylabel(y_label)
        plt.title(f"{title_suffix} : {nm}\n(Nu={Nu}, bins={nbins})")
        plt.grid(True)

# =========================
# Train
# =========================
def train_vmc(steps=N_steps, eta=lr):
    model = NN_Waffle_Complex().to(device=device, dtype=dtype)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=eta)

    state = initial(model, N=N_walkers)
    sigma = s

    E_hist, Var_hist, Acc_hist, Sig_hist = [], [], [], []

    for it in range(1, steps + 1):
        # refresh walkers by MH
        state, acc = mh_chain(model, state, N_burn_in, sigma)
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

        # VMC gradient estimator (score-function style)
        loss = 2.0 * ((E_loc.detach() - E_mean.detach()) * f).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        e_val = float(E_mean.detach())
        var_val = float(((E_loc - E_mean)**2).mean().detach())
        sem = math.sqrt(var_val) / math.sqrt(len(bT1))

        # adapt sigma (clamp)
        if acc < 0.4:
            sigma *= 0.8
        elif acc > 0.6:
            sigma *= 1.2
        sigma = float(max(1e-4, min(2.0, sigma)))

        if it == 1 or it % 10 == 0:
            print(f"[it={it}] E={e_val:.6f}  Var={var_val:.6e}  SEM={sem:.3e}  acc={acc:.3f}  sigma={sigma:.4f}")
            E_hist.append(e_val)
            Var_hist.append(var_val)
            Acc_hist.append(acc)
            Sig_hist.append(sigma)

    return model, state, E_hist, Var_hist, Acc_hist, Sig_hist

# =========================
# Final energy evaluation (on final walkers)
# =========================
def eval_energy_on_walkers(model, state, E_mat):
    model.eval()
    T1, T2, T3, P1, P2, P3, _ = state

    # 必须开 grad（local_energy_complex_matrix 需要对 x 求导）
    with torch.enable_grad():
        E_loc, _ = local_energy_complex_matrix(T1, T2, T3, P1, P2, P3, model, E_mat)

    E_mean = float(E_loc.mean().detach())
    Var = float(((E_loc - E_loc.mean())**2).mean().detach())
    SEM = math.sqrt(Var) / math.sqrt(len(T1))
    return E_mean, Var, SEM


def main():
    model, state, E_hist, Var_hist, Acc, Sig = train_vmc()

    # --- final energy report ---
    E_mean, Var, SEM = eval_energy_on_walkers(model, state, E_mat)
    print("\n==== Final energy on last walkers ====")
    print(f"E = {E_mean:.6f}, Var = {Var:.6e}, SEM = {SEM:.3e}")

    # --- training curves ---
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

    # --- 12 plots of |psi|^2 diagnostics ---
    # uniform_mode: "mean_logpsi2" (recommended), "geom_mean_psi2", or "mean_psi2"
    plot_psi2_marginals(model, state, Nu=200000, nbins=60, uniform_mode="mean_logpsi2")

    plt.show()

if __name__ == "__main__":
    main()
