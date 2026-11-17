"""Local sampling-density experiment for the selected MPC state-space block.

The outer block is fixed at ``BLOCK_INDEX``.  Only the inner sampling density
``(m_position, m_velocity)`` is changed.  The generated files are written to a
separate directory so that the original sampling dataset is not overwritten.
"""

from __future__ import annotations

import json
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import get_context
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm

from AKI_functions import AKI_grid
from Controllers import Explicit_MPC, MPC_solver, Simple_MPC
from Global_Parameters import op, sat

# The selected outer block: (position-x, position-y, position-z,
# velocity-x, velocity-y, velocity-z).
BLOCK_INDEX = (0, 2, 1, 2, 1, 2)

# The current project uses 4 outer blocks per position/velocity dimension.
OUTER_POSITION_BLOCKS = 4
OUTER_VELOCITY_BLOCKS = 4

# Refine the position grid while retaining the completed four-point velocity
# grid.  Existing datasets are reused automatically.
SAMPLING_DENSITIES = ((7, 4),)

PREDICTION_HORIZON = 30
SIGMA_MAX = 1e-4
OUTPUT_DIR = os.path.join("sampling_data", "RKHS_selected_block")
REUSE_EXISTING_SAMPLES = True
CACHE_DECIMALS = 12

# CasADi/IPOPT solver construction is memory-heavy.  Two processes provide a
# conservative speedup without reproducing the ``std::bad_alloc`` failure.
USE_PROCESS_PARALLEL = True
MAX_WORKERS = 2

VALIDATION_COMBINATIONS = ((3, 3), (4, 4), (5, 5), (7, 4))
VALIDATION_SAMPLE_COUNT = 10
VALIDATION_RANDOM_SEED = 12
# Keep the validation memory-safe while using the same number of local basis
# samples as the original (3, 3) interpolant.
VALIDATION_LOCAL_SAMPLE_COUNT = 4 ** op.dim_x
VALIDATION_OUTPUT = os.path.join("recording_data", "RKHS_validation.npz")


def get_block_domain(block_index: tuple[int, int, int, int, int, int] = BLOCK_INDEX, ) -> np.ndarray:
    """Return the six-dimensional bounds of the selected outer block."""

    if len(block_index) != op.dim_x:
        raise ValueError(f"block_index must have {op.dim_x} entries")

    delta_p = 2 * op.pos_max / OUTER_POSITION_BLOCKS
    delta_v = 2 * op.vel_max / OUTER_VELOCITY_BLOCKS

    position_bounds = np.array(
        [
            [
                i * delta_p - op.pos_max,
                (i + 1) * delta_p - op.pos_max,
            ]
            for i in block_index[:3]
        ],
        dtype=float,
    )
    velocity_bounds = np.array(
        [
            [
                i * delta_v - op.vel_max,
                (i + 1) * delta_v - op.vel_max,
            ]
            for i in block_index[3:]
        ],
        dtype=float,
    )
    return np.vstack((position_bounds, velocity_bounds))


def _solve_sample_chunk(
        X_chunk: np.ndarray,
        show_progress: bool = False,
        progress_desc: str = "MPC samples",
        progress_position: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve one chunk in one process and return first controls and sigma.

    A solver is created once per process task.  The previous solution is reused
    as the next initial guess, matching the warm-start logic in the original
    sampling script while avoiding shared solver state between workers.
    """

    if X_chunk.size == 0:
        return np.empty((0, op.dim_u)), np.empty((0,))

    solver, lbg, ubg = MPC_solver(
        sat.rm,
        PREDICTION_HORIZON,
        sat.x_tar,
        sat.u_tar,
    )

    n = PREDICTION_HORIZON
    x_min = [-xi for xi in op.x_max]
    u_min = [-ui for ui in op.u_max_list]
    ubx = np.array(
        op.x_max * (n + 1) + op.u_max_list * n + [np.inf],
        dtype=float,
    )[:, None]
    lbx = np.array(
        x_min * (n + 1) + u_min * n + [0.0],
        dtype=float,
    )[:, None]

    # Initial guess for the first point.  Subsequent points use the previous
    # optimizer solution, as in Satellite_MPC_get_samples.py.
    x_initial = np.zeros(solver.size_in("x0"))
    for k in range(n):
        x_initial[op.dim_x * (k + 1): op.dim_x * (k + 2), 0] = X_chunk[0]
        start = op.dim_x * (n + 1) + op.dim_u * k
        x_initial[start: start + op.dim_u, 0] = sat.u_tar

    controls = []
    sigmas = []
    sample_iterator = (
        tqdm(
            X_chunk,
            total=len(X_chunk),
            desc=progress_desc,
            unit="sample",
            position=progress_position,
            leave=False,
        )
        if show_progress
        else X_chunk
    )
    for xi in sample_iterator:
        result = solver(
            x0=x_initial,
            lbx=lbx,
            ubx=ubx,
            lbg=lbg,
            ubg=ubg,
            p=xi,
        )
        x_initial = np.array(result["x"])
        u_start = op.dim_x * (n + 1)
        controls.append(x_initial[u_start: u_start + op.dim_u, 0].copy())
        sigmas.append(float(x_initial[-1, 0]))

    return np.asarray(controls), np.asarray(sigmas)


def _split_chunks(X: np.ndarray, n_chunks: int) -> list[np.ndarray]:
    """Split samples into non-empty chunks for process workers."""

    n_chunks = max(1, min(int(n_chunks), len(X)))
    return [chunk for chunk in np.array_split(X, n_chunks) if len(chunk)]


def _sample_cache_paths() -> list[str]:
    """Return existing datasets for the selected block, from coarse to dense."""

    block_name = "(" + ", ".join(map(str, BLOCK_INDEX)) + ")"
    paths = [
        os.path.join("sampling_data", f"{block_name}_Mp=3_Mv=3.json"),
        *glob.glob(os.path.join(OUTPUT_DIR, f"{block_name}_Mp=*_Mv=*.json")),
    ]
    return [path for path in paths if os.path.exists(path)]


def _load_sample_cache() -> dict[tuple[float, ...], tuple[np.ndarray, bool]]:
    """Load cached normalized samples keyed by rounded normalized state."""

    cache: dict[tuple[float, ...], tuple[np.ndarray, bool]] = {}
    for path in _sample_cache_paths():
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        X_cached = np.asarray(data["X"], dtype=float)
        U_cached = np.asarray(data["U"], dtype=float)
        flags = data["feasible_sign"]
        for x_cached, u_cached, flag in zip(X_cached, U_cached, flags):
            key = tuple(np.round(x_cached, CACHE_DECIMALS))
            cache[key] = (u_cached, bool(flag))
    return cache


def sample_block(
        block_domain: np.ndarray,
        m_position: int,
        m_velocity: int,
        max_workers: int = MAX_WORKERS,
) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    """Generate normalized samples for one block and one inner density."""

    if m_position < 2 or m_velocity < 2:
        raise ValueError("m_position and m_velocity must both be at least 2")

    X_physical = AKI_grid(block_domain, m_position, m_velocity)
    X_normalized = op.normalX(X_physical)
    U_normalized = np.full((len(X_normalized), op.dim_u), np.nan)
    feasible_sign = np.zeros(len(X_normalized), dtype=bool)

    cache = _load_sample_cache() if REUSE_EXISTING_SAMPLES else {}
    missing_indices = []
    for index, x_normalized in enumerate(X_normalized):
        key = tuple(np.round(x_normalized, CACHE_DECIMALS))
        cached = cache.get(key)
        if cached is None:
            missing_indices.append(index)
        else:
            U_normalized[index] = cached[0]
            feasible_sign[index] = cached[1]

    print(
        f"requested={len(X_normalized)}, cached={len(X_normalized) - len(missing_indices)}, "
        f"new={len(missing_indices)}"
    )

    if missing_indices:
        X_missing = X_physical[missing_indices]
        if not USE_PROCESS_PARALLEL or max_workers <= 1:
            U_physical, sigma = _solve_sample_chunk(
                X_missing,
                show_progress=True,
                progress_desc=f"MPC samples m=({m_position},{m_velocity})",
            )
        else:
            # Two chunks per worker keeps the progress bar responsive while
            # limiting the number of repeated solver constructions.
            chunks = _split_chunks(X_missing, max_workers * 2)
            results: list[tuple[int, np.ndarray, np.ndarray]] = []
            context = get_context("spawn")
            with ProcessPoolExecutor(
                    max_workers=min(max_workers, len(chunks)),
                    mp_context=context,
            ) as executor:
                future_to_order = {
                    executor.submit(
                        _solve_sample_chunk,
                        chunk,
                        True,
                        f"chunk {order + 1}/{len(chunks)}",
                        order,
                    ): order
                    for order, chunk in enumerate(chunks)
                }
                for future in tqdm(
                        as_completed(future_to_order),
                        total=len(future_to_order),
                        desc=f"MPC chunks m=({m_position},{m_velocity})",
                ):
                    order = future_to_order[future]
                    controls, sigmas = future.result()
                    results.append((order, controls, sigmas))

            results.sort(key=lambda item: item[0])
            U_physical = np.vstack([item[1] for item in results])
            sigma = np.concatenate([item[2] for item in results])

        U_normalized[missing_indices] = op.normalU(U_physical)
        feasible_sign[missing_indices] = np.asarray(sigma) < SIGMA_MAX

    return X_normalized, U_normalized, feasible_sign.tolist()


def save_density_result(
        X: np.ndarray,
        U: np.ndarray,
        feasible_sign: list[bool],
        m_position: int,
        m_velocity: int,
) -> str:
    """Save one density result without touching the original dataset."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    block_name = "(" + ", ".join(map(str, BLOCK_INDEX)) + ")"
    file_name = f"{block_name}_Mp={m_position}_Mv={m_velocity}.json"
    save_path = os.path.join(OUTPUT_DIR, file_name)
    payload = {
        "block_index": list(BLOCK_INDEX),
        "outer_position_blocks": OUTER_POSITION_BLOCKS,
        "outer_velocity_blocks": OUTER_VELOCITY_BLOCKS,
        "m_position": m_position,
        "m_velocity": m_velocity,
        "prediction_horizon": PREDICTION_HORIZON,
        "normalized": True,
        "X": X.tolist(),
        "U": U.tolist(),
        "feasible_sign": feasible_sign,
    }
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)
    return save_path


def run_experiment(sampling_densities: Iterable[tuple[int, int]] = SAMPLING_DENSITIES) -> None:
    """Generate all requested inner-density datasets for the selected block."""

    block_domain = get_block_domain()
    print(f"Selected block: {BLOCK_INDEX}")
    print(f"Block bounds:\n{block_domain}")
    mode = "process parallel" if USE_PROCESS_PARALLEL else "sequential CPU"
    print(f"Sampling mode: {mode}; workers={MAX_WORKERS}")

    for m_position, m_velocity in sampling_densities:
        start_time = datetime.now()
        print(f"\nSampling m_position={m_position}, m_velocity={m_velocity}")
        X, U, feasible_sign = sample_block(
            block_domain,
            m_position,
            m_velocity,
        )
        save_path = save_density_result(
            X,
            U,
            feasible_sign,
            m_position,
            m_velocity,
        )
        saturated = np.any(
            (U <= 1e-12) | (U >= 1 - 1e-12),
            axis=1,
        )
        print(
            f"saved: {save_path}; samples={len(X)}; "
            f"sigma-feasible={sum(feasible_sign)}/{len(feasible_sign)}; "
            f"input-boundary samples={int(np.sum(saturated))}; "
            f"elapsed={datetime.now() - start_time}"
        )
        pass
    pass


def run_validation():
    """Compare direct MPC and kernel-interpolated MPC at independent points."""
    block_domain = get_block_domain()
    rng = np.random.default_rng(VALIDATION_RANDOM_SEED)
    X = rng.uniform(block_domain[:, 0], block_domain[:, 1], size=(VALIDATION_SAMPLE_COUNT, op.dim_x))
    U_mpc = np.empty((VALIDATION_SAMPLE_COUNT, op.dim_u))
    U_empc = np.empty((len(VALIDATION_COMBINATIONS), VALIDATION_SAMPLE_COUNT, op.dim_u))
    for i, x in enumerate(tqdm(X, desc="RKHS validation", unit="sample")):
        U_mpc[i] = Simple_MPC(x, sat.x_tar, Tm=PREDICTION_HORIZON)
        for j, params in enumerate(VALIDATION_COMBINATIONS):
            U_empc[j, i] = fitting(x, params)
    dU = U_empc - U_mpc[None, :, :]
    dU_norm_l2 = np.linalg.norm(dU / (2 * op.u_max), ord=2, axis=2)
    dU_max = np.max(dU_norm_l2, axis=1)
    dU_ave = np.mean(dU_norm_l2, axis=1)
    return dU_norm_l2, dU_max, dU_ave


def fitting(x: np.ndarray, params: tuple[int, int]) -> np.ndarray:
    sampling_dir = "sampling_data" if params == (3, 3) else OUTPUT_DIR
    block_name = "(" + ", ".join(map(str, BLOCK_INDEX)) + ")"
    file_path = os.path.join(sampling_dir, f"{block_name}_Mp={params[0]}_Mv={params[1]}.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sampling dataset not found: {file_path}")
    return Explicit_MPC(x, sat.x_tar, sampling_density=params, sampling_dir=sampling_dir, max_training_samples=None)


def fig_plot(du):
    du = np.asarray(du, dtype=float)  # np.array([0.25, 0.3, 0.2, 0.2])[:, np.newaxis]
    du_max = np.max(du, axis=1)
    du_ave = np.mean(du, axis=1)
    if du.shape != (len(VALIDATION_COMBINATIONS), VALIDATION_SAMPLE_COUNT):
        raise ValueError(
            f"du must have shape "
            f"({len(VALIDATION_COMBINATIONS)}, {VALIDATION_SAMPLE_COUNT})"
        )

    du = du[::-1]
    du_max = du_max[::-1]
    du_ave = du_ave[::-1]

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 15,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 15,
    })
    labels = ["(7, 4)", "5", "4", "3"]
    colors = ["#CC79A7", "#009E73", "#E69F00", "#0072B2"]

    param_index = np.arange(len(labels))
    sample_index = np.arange(1, VALIDATION_SAMPLE_COUNT + 1)
    x_pos, y_pos = np.meshgrid(sample_index, param_index, indexing="xy")

    fig_3d = plt.figure(figsize=(9.0, 6.5), constrained_layout=True)
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    ax_3d.bar3d(
        x_pos.ravel() - 0.32,
        y_pos.ravel() - 0.32,
        np.zeros(du.size),
        0.64,
        0.64,
        du.ravel(),
        color=np.repeat(colors, VALIDATION_SAMPLE_COUNT),
        edgecolor="black",
        linewidth=0.25,
        shade=True,
    )
    ax_3d.set_xlabel("Validation sample", labelpad=12)
    ax_3d.set_ylabel(r"Sampling density m", labelpad=10)
    ax_3d.set_zlabel(r"Normalized $\ell_2$ error", labelpad=10)
    ax_3d.set_xticks(sample_index)
    ax_3d.set_yticks(param_index, labels)
    ax_3d.set_box_aspect((VALIDATION_SAMPLE_COUNT, len(labels), 5))
    ax_3d.set_proj_type("ortho")
    ax_3d.view_init(elev=25, azim=-55)
    ax_3d.grid(True, alpha=0.25)

    fig_bar, ax_bar = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    width = 0.32
    ax_bar.bar(
        param_index - width / 2,
        du_ave,
        width,
        color=colors,
        alpha=0.35,
        edgecolor=colors,
        linewidth=1.2,
        zorder=3,
    )
    ax_bar.bar(
        param_index + width / 2,
        du_max,
        width,
        color=colors,
        edgecolor="#303030",
        linewidth=0.7,
        hatch="//",
        zorder=3,
    )
    ax_bar.set_xlabel(r"Sampling density m")
    ax_bar.set_ylabel(r"Normalized $\ell_2$ error")
    ax_bar.set_xticks(param_index, labels)
    y_max = max(np.max(du_ave), np.max(du_max))
    ax_bar.set_ylim(0, 1.18 * y_max if y_max > 0 else 1)
    ax_bar.set_axisbelow(True)
    ax_bar.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.7, color="#B8B8B8", alpha=0.55)
    ax_bar.margins(x=0.06)
    ax_bar.tick_params(axis="both", direction="out", length=4, width=0.8)
    ax_bar.legend(
        handles=[
            Patch(facecolor="#8A8A8A", edgecolor="#606060", alpha=0.35, label="Mean error"),
            Patch(facecolor="#8A8A8A", edgecolor="#303030", hatch="//", label="Maximum error"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.spines[["left", "bottom"]].set_color("#555555")
    ax_bar.spines[["left", "bottom"]].set_linewidth(0.8)

    plt.show()


if __name__ == "__main__":
    fig_plot(np.load("recording_data/RKHS_du.npy"))
