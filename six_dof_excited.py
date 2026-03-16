import matplotlib.pyplot as plt
import torch

from six_dof_core import (
    ModelParams,
    NNWaffleComplex6D,
    PhysicalParams,
    build_e_matrix,
    compute_overlap_matrix,
    energy_loss_imag,
    energy_loss_real,
    get_complex_dtype,
    get_device,
    initial_state,
    local_energy,
    mean_sem,
    mh_chain,
    overlap_abs_symmetric_stable,
)


device = get_device()
model_params = ModelParams(hidden_dim=216, dtype=torch.float64)
physics = PhysicalParams(cth=10.0, cphi=10.0, cv=30.0, cp=47.0, ej=80.0)
complex_dtype = get_complex_dtype(model_params.dtype)

N_walkers = 1000
N_burn_in = 40
Sigma0 = 0.35
N_steps = 2400
Lr = 1e-2
N_states = 8
PRINT_EVERY = 10
OVLP_EVERY = 20
OVLP_SUBSAMPLE = 1000
EPS_OVLP = 1e-8
Seed = 0

WARMUP = 400
ALPHA_TARGET = 100.0
ALPHA_POWER = 2.0
BETA_IMAG = 1.0

torch.manual_seed(Seed)
E_mat = build_e_matrix(physics, device=device, dtype=model_params.dtype)


def alpha_schedule(it: int, steps: int) -> float:
    if it < WARMUP:
        return 0.0
    denom = max(1, steps - WARMUP - 1)
    frac = max(0.0, min(1.0, (it - WARMUP) / denom))
    return ALPHA_TARGET * (frac**ALPHA_POWER)


def print_matrix(mat: list[list[float]], title: str, fmt: str = "{:0.3e}"):
    n = len(mat)
    print(title)
    header = "      " + " ".join([f"{j:>10d}" for j in range(n)])
    print(header)
    for i in range(n):
        row = " ".join([fmt.format(mat[i][j]) for j in range(n)])
        print(f"{i:>4d}  {row}")


def train_multistate(steps: int = N_steps, eta: float = Lr):
    models = [
        NNWaffleComplex6D(hidden_dim=model_params.hidden_dim).to(device=device, dtype=model_params.dtype)
        for _ in range(N_states)
    ]
    opts = [torch.optim.Adam(model.parameters(), lr=eta) for model in models]
    states = [initial_state(model, n_walkers=N_walkers, device=device, dtype=model_params.dtype) for model in models]
    sigmas = [float(Sigma0) for _ in range(N_states)]

    logs = {
        "E": [[] for _ in range(N_states)],
        "E_imag": [[] for _ in range(N_states)],
        "Acc": [[] for _ in range(N_states)],
        "Sig": [[] for _ in range(N_states)],
        "Orth": [[] for _ in range(N_states)],
        "Alpha": [],
    }

    for it in range(steps):
        alpha_it = alpha_schedule(it, steps)
        logs["Alpha"].append(alpha_it)

        xs_it = []
        accs_it = []
        total_loss = torch.zeros((), device=device, dtype=model_params.dtype)
        energy_means = []
        imag_means = []
        orth_max = [0.0 for _ in range(N_states)]

        for idx, model in enumerate(models):
            states[idx], acc = mh_chain(model, states[idx], n_burn_in=N_burn_in, sigma=sigmas[idx])
            accs_it.append(acc)
            x1, x2, x3, x4, x5, x6, _ = states[idx]
            xs_it.append((x1, x2, x3, x4, x5, x6))

            e_real, e_imag, f, g = local_energy(x1, x2, x3, x4, x5, x6, model, E_mat, physics.ej)
            loss_i = energy_loss_real(e_real, f)
            if BETA_IMAG != 0.0:
                loss_i = loss_i + BETA_IMAG * energy_loss_imag(e_imag, g)

            energy_means.append(mean_sem(e_real)[0])
            imag_means.append(mean_sem(e_imag)[0])

            if alpha_it > 0.0:
                for j in range(idx):
                    sabs, pen = overlap_abs_symmetric_stable(
                        model,
                        models[j],
                        xs_it[idx],
                        xs_it[j],
                        model_params.dtype,
                        complex_dtype,
                        eps=EPS_OVLP,
                    )
                    loss_i = loss_i + alpha_it * pen
                    orth_max[idx] = max(orth_max[idx], float(sabs.detach()))

            total_loss = total_loss + loss_i

        for opt in opts:
            opt.zero_grad(set_to_none=True)
        total_loss.backward()
        for opt in opts:
            opt.step()

        for idx, acc in enumerate(accs_it):
            if acc < 0.4:
                sigmas[idx] *= 0.8
            elif acc > 0.6:
                sigmas[idx] *= 1.2
            sigmas[idx] = float(max(1e-3, min(sigmas[idx], 2.0)))

        if it == 0 or (it % PRINT_EVERY) == 0:
            parts = []
            for idx in range(N_states):
                parts.append(
                    f"S{idx}:E={energy_means[idx]:.4f},Im={imag_means[idx]:+.2e},"
                    f"orth={orth_max[idx]:.2e},acc={accs_it[idx]:.2f},sig={sigmas[idx]:.3f}"
                )
                logs["E"][idx].append(energy_means[idx])
                logs["E_imag"][idx].append(imag_means[idx])
                logs["Acc"][idx].append(accs_it[idx])
                logs["Sig"][idx].append(sigmas[idx])
                logs["Orth"][idx].append(orth_max[idx])
            print(f"[it={it}] alpha={alpha_it:.3e} | " + " | ".join(parts))

        if it == 0 or (it % OVLP_EVERY) == 0:
            s_mat = compute_overlap_matrix(
                models,
                xs_it,
                dtype=model_params.dtype,
                complex_dtype=complex_dtype,
                subsample=OVLP_SUBSAMPLE,
            )
            print_matrix(s_mat, title=f"Overlap |S_ij| (sub={OVLP_SUBSAMPLE}) at it={it}")

    return models, states, logs


def plot_logs(logs: dict[str, list[list[float]]]):
    x = list(range(len(logs["E"][0])))

    plt.figure()
    for idx in range(N_states):
        plt.plot(x, logs["E"][idx], label=f"E state{idx}")
    plt.xlabel(f"iteration (x{PRINT_EVERY})")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.legend()

    plt.figure()
    for idx in range(1, N_states):
        plt.plot(x, logs["Orth"][idx], label=f"max|S| state{idx}")
    plt.yscale("log")
    plt.xlabel(f"iteration (x{PRINT_EVERY})")
    plt.ylabel("Orthogonality")
    plt.grid(True)
    plt.legend()

    plt.figure()
    for idx in range(N_states):
        plt.plot(x, logs["Acc"][idx], label=f"acc state{idx}")
    plt.xlabel(f"iteration (x{PRINT_EVERY})")
    plt.ylabel("Acceptance")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(range(len(logs["Alpha"])), logs["Alpha"], label="alpha schedule")
    plt.xlabel("iteration")
    plt.ylabel("alpha")
    plt.grid(True)
    plt.legend()

    plt.show()


def main():
    print(f"device: {device}")
    print(f"EJ={physics.ej:.2f} UNIT={physics.unit:.6f}")
    models, states, logs = train_multistate()
    _ = (models, states)
    plot_logs(logs)


if __name__ == "__main__":
    main()
