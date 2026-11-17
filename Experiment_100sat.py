import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from tqdm import trange
from Global_Parameters import sat, op
from Controllers import Explicit_MPC, Collision_Avoidance
from typing import Dict
from fitting_process import a, b, ellipse_propagate


def sc_main_empc_scp(isStatic=False, isEMPC_only=False, df=0.0):
    """
    Satellite swarm reconfiguration by MPC-approximation and Successive Convex Methods.
    """
    Tp = 600
    N = 100
    static_str = "static" if isStatic else "dynamic"
    empc_str = "EMPC" if isEMPC_only else "EMPC-SCP"
    print(f"{N} satellite swarm experiment: {static_str} configure, method is {empc_str}")
    r_ini = 12
    r_tar = 3.5
    Xt = np.zeros((N, op.dim_x))
    X_CLUSTER: Dict[int, list | np.ndarray] = dict.fromkeys(range(N), [])
    U_CLUSTER: Dict[int, list | np.ndarray] = dict.fromkeys(range(N), [])
    n = [np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), np.sin(b)]
    u = np.array([-n[2], 0, n[0]]) / np.linalg.norm(np.array([-n[2], 0, n[0]]))
    v = np.array([0, -n[2], n[1]]) / np.linalg.norm(np.array([0, -n[2], n[1]]))
    for i in range(N):
        xi, fi = np.zeros(op.dim_x), i * np.pi * 2 / N
        xi[:3] = r_ini * (u * np.cos(fi) + v * np.sin(fi))
        X_CLUSTER[i] = [xi.copy()]
        U_CLUSTER[i] = [np.zeros(3)]
        if isStatic:
            Xt[i] = r_tar * np.array([np.cos(fi + df) / 2, np.sin(fi + df), 0, 0, 0, 0])
        else:
            Xt[i] = r_tar * np.array([np.cos(fi + df) / 2, np.sin(fi + df), 0,
                                      sat.omega * np.sin(fi + df) / 2, -sat.omega * np.cos(fi + df), 0])

    """setting obstacles with multi-intention & record dynamic obstacles"""
    obs = np.array([[0.00000000, 7.0000000, 0.20000000, 0.0000000, 0.0000000, 0.000000],
                    [0.00000000, -7.000000, 0.70000000, 0.0000000, 0.0000000, 0.000000],
                    [1.60000000, 10.000000, -0.5000000, 0.0000000, -2.443930e-3, 0],
                    [3.06417777, -5.142300, 9.61093238e-2, -2.6182141e-3, -6.2405321e-3, -8.2121471e-5],
                    [0.69459271, -7.878462, 2.17862150e-2, -4.0113367e-3, -1.4146137e-3, -1.2581739e-4],
                    [-2.0000000, -6.928203, -6.2730905e-2, -3.5275103e-3, 4.07321807e-3, -1.1064195e-4],
                    [-3.7587705, -2.736161, -1.1789553e-1, -1.3931226e-3, 7.65514593e-3, -4.3695922e-5],
                    [-3.7587705, 2.7361611, -1.1789553e-1, 1.39312263e-3, 7.65514593e-3, 4.36959223e-5],
                    [-2.0000000, 6.9282032, -6.2730905e-2, 3.52751033e-3, 4.07321807e-3, 1.10641959e-4],
                    [0.69459271, 7.8784620, 2.17862150e-2, 4.01133674e-3, -1.4146138e-3, 1.25817394e-4],
                    [3.06417777, 5.1423008, 9.61093238e-2, 2.61821411e-3, -6.2405321e-3, 8.21214715e-5]])
    rho = 0.6
    X_OBS = [obs]
    """EMPC-SCP real-time satellite cluster reconfiguration"""
    X_SCP = {i: np.array([]) for i in range(N)}
    U_SCP = {i: np.array([]) for i in range(N)}
    N_OBS = {i: 0 for i in range(N)}
    dt = sat.rm.get_sample_time()
    K_max = 7500 // dt
    cur_sign = 0
    for k in trange(K_max):
        Xk = {i: X_CLUSTER[i][-1] for i in range(N)}
        # for i in [TEST_SAT_ID]:
        for i in range(N):
            xk = Xk[i]
            if isEMPC_only:
                ok_i = []
            else:
                ok_i = obs[[oi for oi in range(obs.shape[0]) if np.linalg.norm(obs[oi, :3] - xk[:3]) <= sat.view_rge]]
            if len(ok_i) and X_SCP[i].size == 0:
                Xi_SCP, Ui_SCP, cur_sign = Collision_Avoidance(xk, Xt[i], Tp, ok_i, rho, isStatic)
                X_SCP[i] = Xi_SCP[1:]
                x_next = X_SCP[i][0]
                ui = np.append(Ui_SCP[0], cur_sign)
                X_SCP[i] = X_SCP[i][1:]
                U_SCP[i] = Ui_SCP[1:]
            elif len(ok_i):
                if len(ok_i) > N_OBS[i]:
                    Xi_SCP, Ui_SCP, cur_sign = Collision_Avoidance(xk, Xt[i], Tp, ok_i, rho, isStatic)
                    X_SCP[i] = Xi_SCP[1:]
                    U_SCP[i] = Ui_SCP
                x_next = X_SCP[i][0]
                ui = np.append(U_SCP[i][0], cur_sign)
                X_SCP[i] = X_SCP[i][1:]
                U_SCP[i] = U_SCP[i][1:]
            else:
                ui = Explicit_MPC(xk, Xt[i])
                x_next = sat.rm.update(xk, ui)
                ui = np.append(ui, 0)
                if len(X_SCP[i]):
                    X_SCP[i] = np.array([])
            X_CLUSTER[i].append(x_next)
            U_CLUSTER[i].append(ui)
            N_OBS[i] = len(ok_i)
        obs = sat.rm.update(obs.T, [0, 0, 0]).T
        if not isStatic:
            Xt = ellipse_propagate(Xt, dt)
        X_OBS.append(obs)
    X_CLUSTER_NPY = np.array(list(X_CLUSTER.values()))
    U_CLUSTER_NPY = np.array(list(U_CLUSTER.values()))
    np.save(f'recording_data/cluster_trajectory_{N}_{static_str}_{empc_str}.npy', X_CLUSTER_NPY)
    np.save(f'recording_data/cluster_ctrlseq_{N}_{static_str}_{empc_str}.npy', U_CLUSTER_NPY)


def distance_to_target_ellipse(position, r_tar=3.5):
    x, y, z = position[:3]
    semi_x, semi_y = r_tar / 2, r_tar

    def distance_squared(theta):
        return ((x - semi_x * np.cos(theta)) ** 2
                + (y - semi_y * np.sin(theta)) ** 2)

    theta_grid = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    theta_0 = theta_grid[np.argmin(distance_squared(theta_grid))]
    dtheta = 2 * np.pi / theta_grid.size
    result = minimize_scalar(distance_squared,
                             bounds=(theta_0 - dtheta, theta_0 + dtheta),
                             method="bounded")
    return np.sqrt(result.fun + z ** 2)


def performance_results(X_CLUSTER, U_CLUSTER, isStatic=False):
    OBS = np.load("recording_data/obstacle_trajectory.npy")[:X_CLUSTER.shape[1]]
    df = np.pi / 3
    r_tar = 3.5
    Xt = np.zeros((100, 6))
    dis_obs = np.zeros((100, 11))
    dis_sat = np.empty((100, 100))
    for i in range(100):
        Xi = X_CLUSTER[i]
        fi = i * np.pi * 2 / 100
        Xt[i] = r_tar * np.array([np.cos(fi + df) / 2, np.sin(fi + df), 0, 0, 0, 0])
        for o in range(11):
            Oi = OBS[:, o, :]
            dis_obs[i, o] = np.min(np.linalg.norm(Oi[:, :3] - Xi[:, :3], axis=1))
        for j in range(100):
            dis_sat[i, j] = np.min(np.linalg.norm(X_CLUSTER[j, :, :3] - Xi[:, :3], axis=1))
        dis_sat[i, i] = np.inf
    min_dis_obs = np.min(dis_obs, axis=1)
    min_dis_sat = np.min(dis_sat, axis=1)
    X_end = X_CLUSTER[:, -1, :]
    if isStatic:
        e_end = np.linalg.norm(X_end[:, :3] - Xt[:, :3], axis=1)
    else:
        e_end = np.array([distance_to_target_ellipse(xi, r_tar) for xi in X_end])
    u_sum = np.zeros(100)
    for i in range(100):
        index_i = [i + j * 100 for j in range(0, 750)]
        Ui = U_CLUSTER[index_i]
        u_sum[i] = np.sum(np.linalg.norm(Ui[:, :3], axis=1, ord=1))
    return e_end, min_dis_sat, min_dis_obs, u_sum


def plot_config_error_dynamic(e_arr, r_tar):
    N = e_arr.shape[0]
    sate = np.arange(1, N + 1)
    e_mean = np.mean(e_arr)
    r_tube = r_tar / 10

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 15,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "mathtext.fontset": "stix",
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sate, e_arr, color="#0072B2", linestyle="-", marker="o",
            markersize=5, linewidth=1.8, label="Distance to target ellipse")
    ax.axhline(e_mean, color="#009E73", linestyle="--", linewidth=2.0,
               label=fr"Mean = {e_mean:.4f} km")
    ax.axhline(r_tube, color="#D55E00", linestyle="-.", linewidth=2.5,
               label=fr"Tube radius = {r_tube:.2f} km")
    y_max = max(np.max(e_arr), r_tube) * 1.15
    ax.fill_between(sate, r_tube, 0, color="#D55E00", alpha=0.1)
    ax.set_xlabel("Satellite No.")
    ax.set_ylabel("Terminal Distance to Target Ellipse (km)")
    ax.set_xlim(0.95, N + 0.05)
    ax.set_ylim(0, y_max)
    ax.grid(True, linestyle=":", color="gray", alpha=0.6)
    # ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_config_error(e_arr):
    N = e_arr.shape[0]
    sate = np.arange(1, N + 1)
    e_mean = np.mean(e_arr)

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 15,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "mathtext.fontset": "stix",
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sate, e_arr, color="#0072B2", linestyle="-", marker="o",
            markersize=5, linewidth=1.8, label="Terminal error")
    ax.axhline(e_mean, color="#D55E00", linestyle="--", linewidth=2.0,
               label=fr"Mean = {e_mean*1000:.4f} m")
    ax.set_xlabel("Satellite No.")
    ax.set_ylabel("Terminal Position Error (km)")
    ax.set_xlim(0.95, N + 0.05)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=":", color="gray", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_consumption(u_static, u_dynamic):
    N = u_static.shape[0]
    sate = np.arange(1, N + 1)
    mean_static = np.mean(u_static)
    mean_dynamic = np.mean(u_dynamic)

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 15,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "mathtext.fontset": "stix",
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sate, u_static, color="#0072B2", linestyle="-", marker="o",
            markersize=5, linewidth=1.8,
            label=fr"Static Configuration (mean = {mean_static:.4f})")
    ax.plot(sate, u_dynamic, color="#D55E00", linestyle="--", marker="s",
            markersize=4.5, linewidth=1.8,
            label=fr"Dynamic Configuration (mean = {mean_dynamic:.4f})")
    ax.set_xlabel("Satellite No.")
    ax.set_ylabel(r"Control Consumption $\sum_k\|\mathbf{u}_{i,k}\|_1$")
    ax.set_xlim(0.95, N + 0.05)
    ax.set_ylim(0, 0.03)
    ax.grid(True, linestyle=":", color="gray", alpha=0.6)
    # ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_distance(d_static, d_dynamic, ro=0.6, flag="sat"):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 16

    N = len(d_static)
    sate = np.arange(1, N + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sate, d_static, color="#0072B2", linestyle="-", marker="o", markersize=5,
            linewidth=1.8, label=fr"Static Configuration")
    ax.plot(sate, d_dynamic, color="#009E73", linestyle="--", marker="s", markersize=4.5,
            linewidth=1.8, label=fr"Dynamic Configuration")
    safe_symbol = "r_o" if flag == "obs" else "r_s"
    ax.axhline(y=ro, color="#D55E00", linestyle="-.", linewidth=2.5,
               label=fr"Safety distance (${safe_symbol}={ro}$ km)")
    ax.fill_between(sate, 0, ro, color="#D55E00", alpha=0.1)
    ax.set_xlabel('Satellite No.')
    y_label = ('Minimum Satellite--Obstacle Distance (km)' if flag == "obs"
               else 'Minimum Inter-Satellite Distance (km)')
    ax.set_ylabel(y_label)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    y_max = max(np.max(d_static), np.max(d_dynamic), ro, 0.2)
    ax.set_ylim(bottom=0, top=y_max * 1.1)
    ax.set_xlim(0.95, N + 0.05)
    ax.grid(True, linestyle=':', color='gray', alpha=0.6)
    # ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def main():
    X_CLUSTER = np.load("recording_data/cluster_trajectory_100_static_EMPC-SCP.npy")
    U_CLUSTER = np.load("recording_data/cluster_ctrlseq_100_static_EMPC-SCP.npy")[0]
    X_CLUSTER_DYN = np.load("recording_data/cluster_trajectory_100_dynamic_EMPC-SCP.npy")
    U_CLUSTER_DYN = np.load("recording_data/cluster_ctrlseq_100_dynamic_EMPC-SCP.npy")[0]
    e, ds, do, u = performance_results(X_CLUSTER, U_CLUSTER, isStatic=True)
    e_, ds_, do_, u_ = performance_results(X_CLUSTER_DYN, U_CLUSTER_DYN)
    plot_config_error(e)
    plot_config_error_dynamic(e_, r_tar=3.5)
    plot_distance(ds, ds_, ro=0.08, flag="sat")
    plot_distance(do, do_, ro=0.6, flag="obs")
    plot_consumption(u, u_)


if __name__ == "__main__":
    main()
    pass
