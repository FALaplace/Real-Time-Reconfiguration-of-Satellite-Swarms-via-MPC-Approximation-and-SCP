import numpy as np
from tqdm import trange
from Global_Parameters import sat, op
from Controllers import Explicit_MPC, Collision_Avoidance, Simple_MPC, xf_update, Decentrized_MPC
from typing import Dict
import matplotlib.pyplot as plt
import scipy as sp

TEST_SAT_ID = 8
a = 1.5765996259384667
b = 1.2855999197531711


def ellipse_propagate(X, dt):
    w = sat.omega
    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [3 * w ** 2, 0, 0, 0, 2 * w, 0],
        [0, 0, 0, -2 * w, 0, 0],
        [0, 0, -w ** 2, 0, 0, 0]])
    return (sp.linalg.expm(A * dt) @ X.T).T


def sc_main_empc(N=10, Tp=600, if_static_config=False, df=0.0):
    """
    Satellite swarm reconfiguration by MPC-approximation and Successive Convex Methods.
    """
    r_ini = 12
    r_tar = 2
    Xt = np.zeros((N, op.dim_x))
    X_CLUSTER: Dict[int, list | np.ndarray] = dict.fromkeys(range(N))
    U_CLUSTER: Dict[int, list | np.ndarray] = {i: [] for i in range(N)}
    # a = np.random.uniform(0, np.pi * 2)
    # b = np.random.uniform(4 * np.pi / 10, 6 * np.pi / 10)
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
    for k in trange(K_max):
        Xk = {i: X_CLUSTER[i][-1] for i in range(N)}
        for i in [TEST_SAT_ID]:
            # for i in range(N):
            xk = Xk[i]
            x_next = xk
            # ok_i = []
            ok_i = obs[[oi for oi in range(obs.shape[0]) if np.linalg.norm(obs[oi, :3] - xk[:3]) <= sat.view_rge]]
            if len(ok_i) and X_SCP[i].size == 0:
                Xi_SCP, Ui_SCP, cur_sign = Collision_Avoidance(xk, Xt[i], Tp, ok_i, rho)
                X_SCP[i] = Xi_SCP[1:]
                x_next = X_SCP[i][0]
                ui = np.append(Ui_SCP[0], cur_sign)
                X_SCP[i] = X_SCP[i][1:]
                U_SCP[i] = Ui_SCP[1:]
            elif len(ok_i):
                if len(ok_i) > N_OBS[i]:
                    Xi_SCP, Ui_SCP, cur_sign = Collision_Avoidance(xk, Xt[i], Tp, ok_i, rho)
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
    for i in [TEST_SAT_ID]:
        # X_CLUSTER[i] = np.array(X_CLUSTER[i])
        U_CLUSTER[i] = np.array(U_CLUSTER[i])
        np.save(f"recording_data/sat_{i}_control_sequence.npy", U_CLUSTER[i])
        # if if_static_config:
        #     np.save(f"AlgorithmComparisonData/sat_{i}_trajectory_EMPC_static.npy", X_CLUSTER[i])
        # else:
        #     np.save(f"AlgorithmComparisonData/sat_{i}_trajectory_EMPC_dynamic.npy", X_CLUSTER[i])
        # print(f"file sat_{i}_trajectory_EMPC.npy has been recorded!")


def sc_main_mpc(N, if_static_config=False, df=0.0):
    """satellite swarm reconfiguration scenario setting"""
    r_ini = 12
    r_tar = 2
    Xt = np.zeros((N, op.dim_x))
    X_CLUSTER: Dict[int, list | np.ndarray] = dict.fromkeys(range(N))
    # a = np.random.uniform(0, np.pi * 2)
    # b = np.random.uniform(4 * np.pi / 10, 6 * np.pi / 10)
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

    """satellite cluster reconfiguration by MPC"""
    dt = sat.rm.get_sample_time()
    K_max = 8000 // dt
    U = 0
    for k in trange(K_max):
        Xk = {i: X_CLUSTER[i][-1] for i in range(N)}
        # for i in range(N):
        for i in [TEST_SAT_ID]:
            xk = Xk[i]
            ui = Simple_MPC(xk, Xt[i])
            x_next = sat.rm.update(xk, ui)
            X_CLUSTER[i].append(x_next)
            U += np.linalg.norm(ui, ord=np.inf)
        if not if_static_config:
            Xt = ellipse_propagate(Xt, dt)
    for i in [TEST_SAT_ID]:
        X_CLUSTER[i] = np.array(X_CLUSTER[i])
        if if_static_config:
            np.save(f"AlgorithmComparisonData/sat_{i}_trajectory_MPC_static.npy", X_CLUSTER[i])
        else:
            np.save(f"AlgorithmComparisonData/sat_{i}_trajectory_MPC_dynamic.npy", X_CLUSTER[i])
        print(f"file sat_{i}_trajectory_MPC.npy has been recorded!")
    print(f"MPC trajectory energy consumption is {U}.")


def sc_main_dmpc(N, T_mpc, if_static_config=False, df=0.0):
    r_ini = 12
    r_tar = 2
    Xt = np.zeros((N, op.dim_x))
    X_CLUSTER: Dict[int, list | np.ndarray] = dict.fromkeys(range(N))
    # a = np.random.uniform(0, np.pi * 2)
    # b = np.random.uniform(4 * np.pi / 10, 6 * np.pi / 10)
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

    """DMPC real-time satellite cluster reconfiguration"""
    dt = sat.rm.get_sample_time()
    K_max = 8000 // dt
    U = 0
    for k in trange(K_max):
        Xk = {i: X_CLUSTER[i][-1] for i in range(N)}
        # if k >= 186:
        #     print("time up")
        #     print(Xk)
        for i in [TEST_SAT_ID]:
            xk = Xk[i]
            # ok_i = np.array([])
            ok_i = obs[[oi for oi in range(obs.shape[0]) if np.linalg.norm(obs[oi, :3] - xk[:3]) <= sat.view_rge]]
            _, ui = Decentrized_MPC(xk, Xt[i], T_mpc, ok_i, rho, if_static_config)
            x_next = sat.rm.update(xk, ui)
            X_CLUSTER[i].append(x_next)
            U += np.linalg.norm(ui, ord=np.inf)
        obs = sat.rm.update(obs.T, [0, 0, 0]).T
        if not if_static_config:
            Xt = ellipse_propagate(Xt, dt)
        X_OBS.append(obs)
    for i in [TEST_SAT_ID]:
        X_CLUSTER[i] = np.array(X_CLUSTER[i])
        if if_static_config:
            np.save(f"AlgorithmComparisonData/sat_{i}_trajectory_DMPC_static.npy", X_CLUSTER[i])
        else:
            np.save(f"AlgorithmComparisonData/sat_{i}_trajectory_DMPC_dynamic.npy", X_CLUSTER[i])
        print(f"file sat_{i}_trajectory_DMPC.npy has been recorded!")
    print(f"DMPC trajectory energy consumption is {U}.")


if __name__ == "__main__":
    sc_main_empc()
    # sc_main_empc(N=10, if_static_config=True, df=np.pi / 3)
    # sc_main_mpc(N=10)
    # sc_main_mpc(N=10, if_static_config=True, df=np.pi / 3)
    # sc_main_dmpc(N=10, T_mpc=20)
    # sc_main_dmpc(N=10, T_mpc=20, if_static_config=True, df=np.pi / 3)
    # h = sat.rm.get_sample_time()
    # sc_main_empc(Tp=60 * h)
    # sc_main_empc(Tp=80 * h)
    # sc_main_empc(Tp=40 * h)
    # sc_main_empc(if_static_config=True, df=np.pi / 3)
