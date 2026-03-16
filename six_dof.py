import matplotlib.pyplot as plt
import torch

from six_dof_core import (
    ModelParams,
    NNWaffleComplex6D,
    PhysicalParams,
    build_e_matrix,
    energy_loss_real,
    get_device,
    initial_state,
    local_energy,
    mean_sem,
    mh_chain,
)


device = get_device()
model_params = ModelParams(hidden_dim=216, dtype=torch.float64)
physics = PhysicalParams(cth=10.0, cphi=10.0, cv=30.0, cp=47.0, ej=80.0)

N_walkers = 4000
N_burn_in = 200
Sigma0 = 0.35
N_steps = 200
Lr = 1e-2
PRINT_EVERY = 10
Seed = 0

torch.manual_seed(Seed)
E_mat = build_e_matrix(physics, device=device, dtype=model_params.dtype)


def train_vmc(steps: int = N_steps, eta: float = Lr):
    model = NNWaffleComplex6D(hidden_dim=model_params.hidden_dim).to(device=device, dtype=model_params.dtype)
    opt = torch.optim.Adam(model.parameters(), lr=eta)

    state = initial_state(model, n_walkers=N_walkers, device=device, dtype=model_params.dtype)
    sigma = float(Sigma0)

    logs = {"E": [], "Var": [], "SEM": [], "Acc": [], "Sig": []}

    for it in range(1, steps + 1):
        state, acc = mh_chain(model, state, n_burn_in=N_burn_in, sigma=sigma)
        t1, t2, t3, p1, p2, p3, _ = state

        e_real, _, f, _ = local_energy(t1, t2, t3, p1, p2, p3, model, E_mat, physics.ej)
        e_mean = e_real.mean()
        loss = energy_loss_real(e_real, f)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        var_val = float(((e_real - e_mean) ** 2).mean().detach())
        _, sem_val = mean_sem(e_real)

        if acc < 0.4:
            sigma *= 0.9
        elif acc > 0.6:
            sigma *= 1.1
        sigma = float(max(1e-3, min(sigma, 2.0)))

        if it == 1 or (it % PRINT_EVERY) == 0:
            e_val = float(e_mean.detach())
            print(
                f"[it={it}] E={e_val:.6f} Var={var_val:.6e} SEM={sem_val:.3e} "
                f"acc={acc:.3f} sigma={sigma:.4f}"
            )
            logs["E"].append(e_val)
            logs["Var"].append(var_val)
            logs["SEM"].append(sem_val)
            logs["Acc"].append(acc)
            logs["Sig"].append(sigma)

    return model, state, logs


def plot_logs(logs: dict[str, list[float]], state: tuple[torch.Tensor, ...]):
    x = list(range(len(logs["E"])))

    plt.figure()
    plt.plot(x, logs["E"], label="E (mean)")
    plt.xlabel(f"iteration (x{PRINT_EVERY})")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(x, logs["Var"], label="Var(E_loc)")
    plt.yscale("log")
    plt.xlabel(f"iteration (x{PRINT_EVERY})")
    plt.ylabel("Variance")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(x, logs["Acc"], label="Acceptance")
    plt.xlabel(f"iteration (x{PRINT_EVERY})")
    plt.ylabel("Acceptance")
    plt.grid(True)
    plt.legend()

    t1, t2, t3, p1, p2, p3, _ = state
    coords = [t1, t2, t3, p1, p2, p3]
    names = ["theta1", "theta2", "theta3", "phi1", "phi2", "phi3"]

    for name, coord in zip(names, coords):
        plt.figure()
        plt.scatter(range(len(coord)), coord.detach().cpu().numpy(), s=3)
        plt.xlabel("walker index")
        plt.ylabel(name)
        plt.title(f"Walker distribution: {name}")
        plt.grid(True)

    plt.show()


def main():
    print(f"device: {device}")
    print(f"EJ={physics.ej:.2f} UNIT={physics.unit:.6f}")
    model, state, logs = train_vmc()
    _ = model
    plot_logs(logs, state)


if __name__ == "__main__":
    main()
