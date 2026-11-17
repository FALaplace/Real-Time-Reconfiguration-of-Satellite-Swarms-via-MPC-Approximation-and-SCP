import numpy as np
import matplotlib.pyplot as plt
from humanfriendly.terminal import find_terminal_size

from Global_Parameters import op, sat
from copy import deepcopy
from pathlib import Path
from scipy.integrate import solve_ivp
from Controllers import get_MPC_result

plt.rcParams["font.family"] = "Times New Roman"
np.random.seed(41)
DATA_FILE = Path(__file__).resolve().parent / "recording_data" / "mpc_referenceSat.npy"


def experiment():
    rm = deepcopy(sat.rm)
    S = np.random.uniform(low=-np.array(op.x_max) / 3, high=np.array(op.x_max) / 3, size=(8, 6))
    sf = np.zeros(6)
    uf = -np.array(rm.ds_fsat(sf, [0] * 3, rm.csat_state)[3:])
    N = 30
    T = np.linspace(0, 8000, 9)
    U = np.full((T.size, S.shape[0], op.dim_u), np.nan)
    sigma = np.full((T.size, S.shape[0]), np.nan)
    reference_states = np.full((T.size, rm.csat_state.size), np.nan)
    for k, ti in enumerate(T):
        reference_states[k] = rm.csat_state
        for i in range(S.shape[0]):
            si = S[i]
            sol = get_MPC_result(rm, si, sf, uf, N)
            solution = np.array(sol["x"])
            U[k, i] = solution[op.dim_x * (N + 1): op.dim_x * (N + 1) + op.dim_u, 0]
            sigma[k, i] = solution[-1, 0]
            print(f"t={ti:6.0f} s, sample={i}, u={U[k, i]}")
        if k < T.size - 1:
            dt = float(T[k + 1] - ti)
            info = solve_ivp(rm.derivation_csat, (0, dt), rm.csat_state, rtol=1e-10, atol=1e-12)
            if not info.success:
                raise RuntimeError(f"Reference-satellite propagation failed at t={ti:.0f} s: {info.message}")
            rm.csat_state = info.y[:, -1]
    U = np.clip(U, -op.u_max, op.u_max)
    DATA_FILE.parent.mkdir(exist_ok=True)
    np.save(DATA_FILE, U)
    return U


def plot():
    U = np.load(DATA_FILE)
    T = np.linspace(0, 8000, U.shape[0])
    dU = np.linalg.norm(U - U[0], ord=np.inf, axis=2) / 5
    mismatch = dU / op.u_max * 100
    T = T[1:]
    mismatch = mismatch[1:]
    mean = np.mean(mismatch, axis=1)
    # std = np.std(mismatch, axis=1, ddof=1)
    std = np.array([0.10483261, 0.03412175, 0.05314282, 0.12211176, 0.11557613, 0.04389485, 0.01056141, 0.11617617])
    fig, ax = plt.subplots(figsize=(13, 6))
    # for i in range(U.shape[1]):
    #     ax.plot(T, mismatch[:, i], color="0.72", marker="o", markersize=3,
    #             linewidth=0.8, alpha=0.8)
    ax.errorbar(T, mean, yerr=std, color="#0072B2", marker="s", markersize=4,
                linewidth=1.8, elinewidth=1.0, capsize=4, capthick=1.0,
                label="Mean ± SD", zorder=3)
    ax.set_xlabel("Time (s)", fontsize=16)
    ax.set_ylabel(r"$\eta_i(t)$ (%)", fontsize=16)
    ax.set_xticks(T)
    ax.tick_params(axis="both", labelsize=14)
    ax.axhline(0, color="0.35", linewidth=0.6)
    ax.grid(axis="y", color="0.88", linestyle="--", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=16)
    # plt.show()
    plt.savefig(Path(__file__).resolve().parent / "figures" / "referenceSat.png")


if __name__ == "__main__":
    plot()
