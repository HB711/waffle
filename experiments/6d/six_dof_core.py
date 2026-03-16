import math
from dataclasses import dataclass

import torch
import torch.nn as nn


TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class PhysicalParams:
    cth: float = 10.0
    cphi: float = 10.0
    cv: float = 30.0
    cp: float = 47.0
    ej: float = 80.0
    h: float = 6.626
    e: float = 1.60

    @property
    def unit(self) -> float:
        return 100.0 * (4.0 * self.e**2) / (2.0 * self.h)


@dataclass(frozen=True)
class ModelParams:
    hidden_dim: int = 216
    dtype: torch.dtype = torch.float64


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if dtype == torch.float64 else torch.complex64


def build_e_matrix(params: PhysicalParams, device: str, dtype: torch.dtype) -> torch.Tensor:
    c = torch.tensor(
        [
            [params.cth + 3 * params.cv + 2 * params.cp, -params.cp, -params.cp, -params.cv, -params.cv, -params.cv],
            [-params.cp, params.cth + 3 * params.cv + 2 * params.cp, -params.cp, -params.cv, -params.cv, -params.cv],
            [-params.cp, -params.cp, params.cth + 3 * params.cv + 2 * params.cp, -params.cv, -params.cv, -params.cv],
            [-params.cv, -params.cv, -params.cv, params.cphi + 3 * params.cv + 2 * params.cp, -params.cp, -params.cp],
            [-params.cv, -params.cv, -params.cv, -params.cp, params.cphi + 3 * params.cv + 2 * params.cp, -params.cp],
            [-params.cv, -params.cv, -params.cv, -params.cp, -params.cp, params.cphi + 3 * params.cv + 2 * params.cp],
        ],
        device=device,
        dtype=dtype,
    )
    cinv = torch.linalg.inv(c)
    e_mat = params.unit * cinv
    return 0.5 * (e_mat + e_mat.T)


def wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + math.pi, TWO_PI) - math.pi


def potential_energy(
    t1: torch.Tensor,
    t2: torch.Tensor,
    t3: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    ej: float,
) -> torch.Tensor:
    alpha = 2 * torch.pi / 3
    term_phi1 = -2 * torch.cos(t1 - p1) - 2 * torch.cos(t2 - p1) - 2 * torch.cos(t3 - p1)
    term_phi2 = -2 * torch.cos(t1 - p2) - 2 * torch.cos(t2 - p2 - alpha) - 2 * torch.cos(t3 - p2 + alpha)
    term_phi3 = -2 * torch.cos(t1 - p3) - 2 * torch.cos(t2 - p3 + alpha) - 2 * torch.cos(t3 - p3 - alpha)
    return ej * (term_phi1 + term_phi2 + term_phi3) / math.sqrt(3.0)


def periodic_emb(
    t1: torch.Tensor,
    t2: torch.Tensor,
    t3: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
) -> torch.Tensor:
    feats = [
        torch.sin(t1), torch.sin(2 * t1), torch.cos(t1), torch.cos(2 * t1),
        torch.sin(t2), torch.sin(2 * t2), torch.cos(t2), torch.cos(2 * t2),
        torch.sin(t3), torch.sin(2 * t3), torch.cos(t3), torch.cos(2 * t3),
        torch.sin(p1), torch.sin(2 * p1), torch.cos(p1), torch.cos(2 * p1),
        torch.sin(p2), torch.sin(2 * p2), torch.cos(p2), torch.cos(2 * p2),
        torch.sin(p3), torch.sin(2 * p3), torch.cos(p3), torch.cos(2 * p3),
    ]
    return torch.stack(feats, dim=-1)


class NNWaffleComplex6D(nn.Module):
    def __init__(self, hidden_dim: int = 216):
        super().__init__()
        self.backbone_f = nn.Sequential(
            nn.Linear(24, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.out_f = nn.Linear(hidden_dim, 1)

        self.backbone_g = nn.Sequential(
            nn.Linear(24, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.out_g = nn.Linear(hidden_dim, 1)

        nn.init.zeros_(self.out_g.weight)
        nn.init.zeros_(self.out_g.bias)

    def forward(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
        t3: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = periodic_emb(t1, t2, t3, p1, p2, p3)
        f = self.out_f(self.backbone_f(x)).squeeze(-1)
        g = self.out_g(self.backbone_g(x)).squeeze(-1)
        return f, g

    def f_only(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
        t3: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
    ) -> torch.Tensor:
        x = periodic_emb(t1, t2, t3, p1, p2, p3)
        return self.out_f(self.backbone_f(x)).squeeze(-1)


@torch.no_grad()
def initial_state(
    model: NNWaffleComplex6D,
    n_walkers: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    coords = [(torch.rand(n_walkers, device=device, dtype=dtype) * TWO_PI) - math.pi for _ in range(6)]
    f = model.f_only(*coords)
    return (*coords, f)


@torch.no_grad()
def mh_chain(
    model: NNWaffleComplex6D,
    state: tuple[torch.Tensor, ...],
    n_burn_in: int,
    sigma: float,
) -> tuple[tuple[torch.Tensor, ...], float]:
    t1, t2, t3, p1, p2, p3, f = state
    f = model.f_only(t1, t2, t3, p1, p2, p3)
    n_walkers = t1.shape[0]

    acc = 0
    for _ in range(n_burn_in):
        proposals = [wrap_pi(x + torch.randn_like(x) * sigma) for x in (t1, t2, t3, p1, p2, p3)]
        nf = model.f_only(*proposals)
        log_alpha = 2.0 * (nf - f)
        accept = torch.log(torch.rand_like(log_alpha)) < log_alpha

        t1 = torch.where(accept, proposals[0], t1)
        t2 = torch.where(accept, proposals[1], t2)
        t3 = torch.where(accept, proposals[2], t3)
        p1 = torch.where(accept, proposals[3], p1)
        p2 = torch.where(accept, proposals[4], p2)
        p3 = torch.where(accept, proposals[5], p3)
        f = torch.where(accept, nf, f)
        acc += int(accept.sum())

    return (t1, t2, t3, p1, p2, p3, f), acc / (n_burn_in * n_walkers)


def sample_batch(state: tuple[torch.Tensor, ...], batch_size: int) -> tuple[torch.Tensor, ...]:
    t1, t2, t3, p1, p2, p3, _ = state
    n_walkers = t1.shape[0]
    if batch_size >= n_walkers:
        return t1, t2, t3, p1, p2, p3
    idx = torch.randint(0, n_walkers, (batch_size,), device=t1.device)
    return t1[idx], t2[idx], t3[idx], p1[idx], p2[idx], p3[idx]


def local_energy(
    t1: torch.Tensor,
    t2: torch.Tensor,
    t3: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    model: NNWaffleComplex6D,
    e_mat: torch.Tensor,
    ej: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.stack([t1, t2, t3, p1, p2, p3], dim=1).detach().requires_grad_(True)
    f, g = model(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5])

    ones = torch.ones_like(f)
    grad_f = torch.autograd.grad(f, x, grad_outputs=ones, create_graph=True)[0]
    grad_g = torch.autograd.grad(g, x, grad_outputs=ones, create_graph=True)[0]

    quad_f = torch.einsum("ni,ij,nj->n", grad_f, e_mat, grad_f)
    quad_g = torch.einsum("ni,ij,nj->n", grad_g, e_mat, grad_g)
    quad_fg = torch.einsum("ni,ij,nj->n", grad_f, e_mat, grad_g)

    eh_f = torch.zeros_like(f)
    eh_g = torch.zeros_like(g)
    for i in range(6):
        dgrad_f_i = torch.autograd.grad(
            grad_f[:, i],
            x,
            grad_outputs=torch.ones_like(grad_f[:, i]),
            create_graph=True,
        )[0]
        dgrad_g_i = torch.autograd.grad(
            grad_g[:, i],
            x,
            grad_outputs=torch.ones_like(grad_g[:, i]),
            create_graph=True,
        )[0]
        eh_f = eh_f + (dgrad_f_i * e_mat[i]).sum(dim=1)
        eh_g = eh_g + (dgrad_g_i * e_mat[i]).sum(dim=1)

    v = potential_energy(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5], ej)
    e_real = -(eh_f + quad_f - quad_g) + v
    e_imag = -(eh_g + 2.0 * quad_fg)
    return e_real, e_imag, f, g


def energy_loss_real(e_real: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    er = e_real.detach()
    return 2.0 * ((er - er.mean()) * f).mean()


def energy_loss_imag(e_imag: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    ei = e_imag.detach()
    return 2.0 * ((ei - ei.mean()) * g).mean()


def mean_sem(x: torch.Tensor) -> tuple[float, float]:
    xd = x.detach()
    mean = xd.mean()
    std = xd.std(unbiased=False)
    return float(mean), float(std / math.sqrt(xd.numel()))


def standard_error_of_mean(var_val: float, n: int) -> float:
    return math.sqrt(max(var_val, 0.0) / max(n, 1))


def stable_complex_mean(df: torch.Tensor, dg: torch.Tensor, complex_dtype: torch.dtype) -> torch.Tensor:
    shift = df.max().detach()
    w = torch.exp(df - shift).to(complex_dtype)
    z = w * torch.exp(1j * dg.to(complex_dtype))
    return z.mean() * torch.exp(shift).to(complex_dtype)


def overlap_abs_symmetric_stable(
    model_i: NNWaffleComplex6D,
    model_j: NNWaffleComplex6D,
    x_i: tuple[torch.Tensor, ...],
    x_j: tuple[torch.Tensor, ...],
    dtype: torch.dtype,
    complex_dtype: torch.dtype,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    x1i, x2i, x3i, x4i, x5i, x6i = x_i
    x1j, x2j, x3j, x4j, x5j, x6j = x_j

    f_i_i, g_i_i = model_i(x1i, x2i, x3i, x4i, x5i, x6i)
    with torch.no_grad():
        f_j_i, g_j_i = model_j(x1i, x2i, x3i, x4i, x5i, x6i)
    a = stable_complex_mean((f_j_i - f_i_i).to(dtype), (g_j_i - g_i_i).to(dtype), complex_dtype)

    f_i_j, g_i_j = model_i(x1j, x2j, x3j, x4j, x5j, x6j)
    with torch.no_grad():
        f_j_j, g_j_j = model_j(x1j, x2j, x3j, x4j, x5j, x6j)
    b = stable_complex_mean((f_i_j - f_j_j).to(dtype), (g_i_j - g_j_j).to(dtype), complex_dtype)

    sabs = torch.sqrt(torch.abs(a) * torch.abs(b)).to(dtype)
    sclamp = torch.clamp(sabs, max=(1.0 - eps))
    pen = (1.0 / (1.0 - sclamp)) - 1.0
    return sabs, pen


@torch.no_grad()
def compute_overlap_matrix(
    models: list[NNWaffleComplex6D],
    xs_pack: list[tuple[torch.Tensor, ...]],
    dtype: torch.dtype,
    complex_dtype: torch.dtype,
    subsample: int | None = None,
) -> list[list[float]]:
    xs_use = []
    for x in xs_pack:
        if subsample is None:
            xs_use.append(x)
        else:
            xs_use.append(tuple(t[:subsample] for t in x))

    n = len(models)
    s_mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        s_mat[i][i] = 1.0
    for i in range(n):
        for j in range(i):
            sabs, _ = overlap_abs_symmetric_stable(models[i], models[j], xs_use[i], xs_use[j], dtype, complex_dtype)
            s = float(sabs.detach().clamp(0.0, 1.0).cpu())
            s_mat[i][j] = s
            s_mat[j][i] = s
    return s_mat
