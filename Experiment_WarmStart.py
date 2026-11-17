from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from tqdm import trange
from Global_Parameters import sat, op
from Controllers import Explicit_MPC, Collision_Avoidance, Collision_Avoidance_no_Warm_Start
from typing import Dict
from fitting_process import a, b, ellipse_propagate

TEST_SAT_ID = 4


def sc_main_empc(N=10, Tp=600, if_static_config=False, df=0.0, warm_start=True):
    """
    Args:
        warm_start: if EMPC warm start
        N: member satellite account in the satellite swarm
        Tp: Max SCP horizon
        if_static_config: static/dynamic configuration flag
        df: deviation of the phase angle between initial point and target point
    Returns:
    """
    r_ini = 12
    r_tar = 2
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
        if if_static_config:
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
    K_max = 8000 // dt
    cur_sign = 0
    t1 = datetime.now()
    for k in trange(K_max):
        Xk = {i: X_CLUSTER[i][-1] for i in range(N)}
        # for i in [TEST_SAT_ID]:
        for i in range(N):
            xk = Xk[i]
            ok_i = obs[[oi for oi in range(obs.shape[0]) if np.linalg.norm(obs[oi, :3] - xk[:3]) <= sat.view_rge]]
            if len(ok_i) and X_SCP[i].size == 0:
                if warm_start:
                    Xi_SCP, Ui_SCP, cur_sign = Collision_Avoidance(xk, Xt[i], Tp, ok_i, rho, isStatic=if_static_config)
                else:
                    Xi_SCP, Ui_SCP, cur_sign = Collision_Avoidance_no_Warm_Start(xk, Xt[i], Tp, ok_i, rho,
                                                                                 isStatic=if_static_config)
                X_SCP[i] = Xi_SCP[1:]
                x_next = X_SCP[i][0]
                ui = np.append(Ui_SCP[0], cur_sign)
                X_SCP[i] = X_SCP[i][1:]
                U_SCP[i] = Ui_SCP[1:]
            elif len(ok_i):
                if len(ok_i) > N_OBS[i]:
                    if warm_start:
                        Xi_SCP, Ui_SCP, cur_sign = Collision_Avoidance(xk, Xt[i], Tp, ok_i, rho,
                                                                       isStatic=if_static_config)
                    else:
                        Xi_SCP, Ui_SCP, cur_sign = Collision_Avoidance_no_Warm_Start(xk, Xt[i], Tp, ok_i, rho,
                                                                                     isStatic=if_static_config)
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
        if not if_static_config:
            Xt = ellipse_propagate(Xt, dt)
        X_OBS.append(obs)
    # # for i in [TEST_SAT_ID]:
    t2 = datetime.now()
    return (t2 - t1).total_seconds()


def fig_plot():
    scp_call_warm_start = [0] * 4 + [2, 0, 2, 4, 2, 6, 10, 8, 12, 13, 12, 9, 13, 12, 14]
    iter_ave_warm_start = [0, 0, 0, 0, 5.5, 0, 6.0, 5.25, 6.5, 5.1667, 5.1, 5.125, 5.25, 5.0769, 4.9167, 5.4444, 5.3077,
                           5.25, 5.1429]
    iter_max_warm_start = [0] * 4 + [6, 0, 6, 6, 8, 7, 7, 7, 7, 10, 10, 10, 8, 9, 10]
    run_time_warm_start = [8.839747, 17.075748, 24.909608, 41.440843, 50.69401, 51.029557, 71.695287, 90.318317,
                           148.812769, 183.02213, 295.864311, 555.291124, 700.040673, 608.280436, 758.581179,
                           849.458432, 876.730919, 1031.991171, 1161.23485]
    scp_call_cold_start = [3, 5, 9, 13, 16, 25, 24, 34, 40, 50, 136, 214, 289, 357, 489, 567, 705, 885, 1076]
    iter_ave_cold_start = [6.66, 7.25, 7.77, 8.46, 10.0, 6.64, 11.0, 10.2941, 9.525, 10.22, 10.22, 10.02, 9.63, 10.99,
                           9.96, 9.48, 9.78, 10.69, 10.88]
    iter_max_cold_start = [15, 20, 20, 20, 20, 15, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    run_time_cold_start = [40.42, 72.10, 114.4, 180.1, 274.65, 261.25, 349.32, 516.67, 468.92, 623.9, 1693.73,
                           2771.96, 3800.56, 5900.3, 7900.74, 9062.06, 12958.39, 20897.81, 29252.21]

    n_list = np.array(list(range(1, 10)) + list(range(10, 101, 10)))
    warm_average = np.asarray(iter_ave_warm_start)
    cold_average = np.asarray(iter_ave_cold_start)
    warm_maximum = np.asarray(iter_max_warm_start)
    cold_maximum = np.asarray(iter_max_cold_start)

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 15,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 17,
        "mathtext.fontset": "stix",
    })

    x_ticks = [1, 2, 3, 5, 10, 20, 30, 50, 100]

    def format_x_axis(ax):
        ax.set_xscale("log")
        ax.set_xlabel("Number of member satellites")
        ax.set_xlim(0.9, 110)
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.grid(axis="both", color="0.88", linestyle="--", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    #
    # fig_iter, ax_iter = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    # warm_average_nonzero = warm_average > 0
    # warm_maximum_nonzero = warm_maximum > 0
    # ax_iter.plot(n_list[warm_average_nonzero], warm_average[warm_average_nonzero],
    #              color="#0072B2", marker="o", linestyle="-", linewidth=1.8,
    #              markersize=5, label="Warm start: average")
    # ax_iter.plot(n_list[warm_maximum_nonzero], warm_maximum[warm_maximum_nonzero],
    #              color="#0072B2", marker="^", linestyle="--", linewidth=1.6,
    #              markersize=5, label="Warm start: maximum")
    # ax_iter.scatter(n_list[~warm_average_nonzero], warm_average[~warm_average_nonzero],
    #                 color="#0072B2", marker="o", s=38)
    # ax_iter.scatter(n_list[~warm_maximum_nonzero], warm_maximum[~warm_maximum_nonzero],
    #                 color="#0072B2", marker="^", s=42)
    # ax_iter.plot(n_list, cold_average, color="#D55E00", marker="s", linestyle="-",
    #              linewidth=1.8, markersize=5, label="Cold start: average")
    # ax_iter.plot(n_list, cold_maximum, color="#D55E00", marker="D", linestyle="--",
    #              linewidth=1.6, markersize=4.5, label="Cold start: maximum")
    # ax_iter.axhline(20, color="0.35", linestyle=":", linewidth=1.0)
    # ax_iter.text(100, 19.65, "Iteration limit", color="0.35", fontsize=15, ha="right", va="top")
    # ax_iter.set_ylabel("Number of SCP iterations")
    # ax_iter.set_ylim(-0.75, 21)
    # ax_iter.set_yticks([0, 5, 10, 15, 20])
    # format_x_axis(ax_iter)
    # ax_iter.grid(axis="x", color="0.88", linestyle="--", linewidth=0.7)
    # ax_iter.legend(frameon=True, ncol=1, loc="upper right", bbox_to_anchor=(1.0, 0.9))

    fig_time, ax_time = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax_time.plot(n_list, run_time_warm_start, color="#0072B2", marker="o", linestyle="-",
                 linewidth=1.8, markersize=5, label="EMPC warm start")
    ax_time.plot(n_list, run_time_cold_start, color="#D55E00", marker="s", linestyle="--",
                 linewidth=1.8, markersize=5, label="Zero-input cold start")
    ax_time.set_ylabel("Total runtime (s)")
    ax_time.set_yscale("log")
    format_x_axis(ax_time)
    ax_time.legend(loc="upper left", fontsize=15)
    plt.show()


if __name__ == "__main__":
    # # N_list = list(range(1, 10)) + list(range(10, 101, 10))
    # for n in N_list:
    #     print("=========== " + f"For n = {n} " + "=============")
    #     dt = sc_main_empc(N=n, if_static_config=True, df=np.pi / 3, warm_start=False)
    #     print(f"the computation time is {dt}")
    #     print("=============================================")
    #     pass
    fig_plot()
