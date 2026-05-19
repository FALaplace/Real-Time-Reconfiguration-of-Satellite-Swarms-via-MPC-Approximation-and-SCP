import casadi as ca
import numpy as np
from MotionModel import RelativeMotionModel_J2
from scipy.linalg import solve_discrete_are, expm
from Global_Parameters import op, sat, gd
from Kernel_Interpolation_Class import Kernel_Interpolation_GPR_based
from AKI_functions import find_local_samples
import json
from typing import Dict
import os
import cvxpy as cp
from copy import copy
from visualization import plot_satellite_trajectory
from IntentionScenario import obstacle_motion
import matplotlib.pyplot as plt

data_has_read = {}
A = np.array([
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [3 * sat.omega ** 2, 0, 0, 0, 2 * sat.omega, 0],
    [0, 0, 0, -2 * sat.omega, 0, 0],
    [0, 0, -sat.omega ** 2, 0, 0, 0]
])

xf_update = lambda x: expm(A * sat.rm.get_sample_time()) @ x


def cal_terminal_weight(rm: RelativeMotionModel_J2, x_tar: np.ndarray, u_tar):
    """
    Args:
        rm: relative motion model of the spacecraft
        x_tar: the target state of the spacecraft
        u_tar: the control vector at x_tar
    """
    x = ca.MX.sym('x', 6)
    u = ca.MX.sym('u', 3)
    f = rm.discrete_dynamic(x, u)
    A_cas = ca.Function('A', [x, u], [ca.jacobian(f, x)])
    B_cas = ca.Function('B', [x, u], [ca.jacobian(f, u)])
    A = np.array(A_cas(x_tar, u_tar))
    B = np.array(B_cas(x_tar, u_tar))
    P = solve_discrete_are(A, B, op.Q, op.R)
    K = np.linalg.inv(B.T @ P @ B + op.R) @ (B.T @ P @ A)
    rho = min(
        [(op.u_max - u_tar[i]) ** 2 / (K[i] @ np.linalg.inv(P) @ K[i].T) for i in range(op.dim_u)])
    return P, K, rho


def quad_fun(x, A):
    return ca.mtimes([x.T, A, x])


def MPC_solver(rm: RelativeMotionModel_J2, N, x_tar, u_tar):
    w_sigma = 1e3

    x = ca.MX.sym('x', op.dim_x * (N + 1))
    u = ca.MX.sym('u', op.dim_u * N)
    sigma = ca.MX.sym('sigma')

    x0 = ca.MX.sym("x0", op.dim_x)

    def x_k(x_var, k):
        return x_var[k * op.dim_x:(k + 1) * op.dim_x]

    def u_k(u_var, k):
        return u_var[k * op.dim_u:(k + 1) * op.dim_u]

    cost = 0
    P, K, rho = cal_terminal_weight(rm, x_tar, u_tar)
    g = []
    lbg = []
    ubg = []
    for k in range(N):
        ex = x_k(x, k) - x_tar
        eu = u_k(u, k) - u_tar
        cost += quad_fun(ex, op.Q) + quad_fun(eu, op.R)
        g.append((x_k(x, k + 1) - rm.discrete_dynamic(x_k(x, k), u_k(u, k))) * 1e4)
        lbg += [0] * op.dim_x
        ubg += [0] * op.dim_x

    # initial state constraints
    g.append((x_k(x, 0) - x0) * 1e6)
    lbg += [0] * op.dim_x
    ubg += [0] * op.dim_x
    # terminal state cost and constraints
    xN = x_k(x, N)
    xN_ = rm.discrete_dynamic(x_k(x, N), K @ x_k(x, N))
    eN = xN - x_tar
    eN_ = xN_ - x_tar
    cost += quad_fun(eN, P) + w_sigma * sigma
    g.append(quad_fun(eN, P) - rho + sigma)
    g.append(quad_fun(eN_, P) - rho + sigma)
    lbg += [-ca.inf, -ca.inf]
    ubg += [0.0, 0.0]
    g = ca.vertcat(*g)
    X = ca.vertcat(x, u, sigma)
    nlp = {
        'x': X,
        'f': cost,
        'g': g,
        'p': x0
    }
    opts = {
        'ipopt.print_level': 0,
        'print_time': False,
        'verbose': False,
        'ipopt.sb': 'yes',
        'ipopt.max_iter': 300,
        'ipopt.tol': 1e-6,
        'ipopt.acceptable_tol': 1e-6,
        'ipopt.linear_solver': 'mumps',
        'ipopt.hessian_approximation': 'limited-memory',
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.mu_strategy': 'adaptive',
        'ipopt.jacobian_approximation': 'exact',
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    return solver, lbg, ubg


def get_MPC_result(rm, xk, x_tar, u_tar, N=10) -> dict:
    """
    Args:
        rm: motion model; dynamics
        xk: states
        x_tar: target state
        u_tar: target control vector
        N: prediction step
    Returns:
        the solution
    """
    x_min, u_min = [-xi for xi in op.x_max], [-ui for ui in op.u_max_list]
    "the optimized variable: [X, U, sigma]"
    solver, lbg, ubg = MPC_solver(rm, N, x_tar, u_tar)
    ubx = np.array(op.x_max * (N + 1) + op.u_max_list * N + [np.inf])[:, None]
    lbx = np.array(x_min * (N + 1) + u_min * N + [0.0])[:, None]
    x_inigau = np.zeros(solver.size_in("x0"))  # initial guess
    x_inigau[:6, 0] = xk
    for k in range(N):
        x_inigau[6 * (k + 1):6 * (k + 2), 0] = xk
        x_inigau[6 * (N + 1) + 3 * k: 6 * (N + 1) + 3 * (k + 1), 0] = u_tar
    lbx[0:6, 0] = xk
    ubx[0:6, 0] = xk
    sol: dict = solver(x0=x_inigau, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    return sol


def Simple_MPC(xk, xf, Tm=30):
    uf = -np.array(sat.rm.ds_fsat(xf, [0] * 3, sat.rm.csat_state)[3:])
    sol = get_MPC_result(sat.rm, xk, xf, uf, N=Tm)
    uk = np.array(sol["x"])[op.dim_x * (Tm + 1): op.dim_x * (Tm + 1) + op.dim_u, 0]
    return uk


def Explicit_MPC(xk, xf) -> np.ndarray:
    uf = -np.array(sat.rm.ds_fsat(xf, [0] * 3, sat.rm.csat_state)[3:])
    idx = find_local_samples(xk - xf + sat.x_tar)
    file_path = os.path.join("sampling_data", f"{idx}_Mp=3_Mv=3.json")
    if idx in data_has_read:
        X_sam, U_sam, flag = data_has_read[idx]
    else:
        if not os.path.exists(file_path):
            return np.zeros_like(uf)
        with open(file_path, "r") as f:
            DATA: Dict[str, list] = json.load(f)
        X_sam = np.array(DATA["X"])
        U_sam = np.array(DATA["U"])
        flag = DATA["feasible_sign"]
        data_has_read[idx] = (X_sam, U_sam, flag)
    KI_model = Kernel_Interpolation_GPR_based(kernel=gd.kernel)
    coffs = KI_model.fit(X_sam, U_sam)
    xk_nor = op.normalX(xk - xf + sat.x_tar)
    uk_nor = gd.kernel(xk_nor.reshape(1, 6), X_sam) @ coffs
    uk = np.clip(op.inverseU(uk_nor.flatten()) - sat.u_tar + uf, -op.u_max, op.u_max)
    return uk


def Decentrized_MPC(xk: np.ndarray, xf: np.ndarray, Tn: int, obs_pos: np.ndarray, obs_rad, is_static):
    n_obs = len(obs_pos)
    rm = sat.rm
    xi = xk.copy()
    Xr = [xi]
    Ur = []
    X_obs = [obs_pos.copy()]
    xfk = xf.copy()
    Xf = [xfk]
    """reference trajectory generation by EMPC"""
    for ti in range(Tn):
        cur_obs = X_obs[-1]
        ui = Explicit_MPC(xi, xfk)
        xi = rm.update(xi, ui)
        Xr.append(xi)
        Ur.append(ui)
        if not is_static:
            xfk = xf_update(xfk)
        if n_obs:
            X_obs.append(rm.update(cur_obs.T, [0, 0, 0]).T)
        else:
            X_obs.append(cur_obs)
        Xf.append(xfk)
    Xr, X_obs, Xf = np.array(Xr), np.array(X_obs), np.array(Xf)
    Ur = np.array(Ur)
    obj = np.inf
    for _ in range(10):
        Xr_, Ur_, obj_ = Successive_convex_programming(Xr, Xf, rm, X_obs, obs_rad, tau=op.tau, c_pun=op.pun)
        Xr = Xr_
        Ur = Ur_
        if abs(obj_ - obj) <= 0.5:
            break
        obj = obj_
    return Xr[1], Ur[0]


def Collision_Avoidance(xk: np.ndarray, xf: np.ndarray, T, obs_pos: np.ndarray | list, obs_rad):
    """
    Args:
        xk: the state at time k
        xf: the final position
        T: total predicted time
        obs_pos: the position of obstacles
        obs_rad: the safe radius of obstacles
    Returns:
    """
    n_obs = len(obs_pos)
    rm = sat.rm
    ti = 0
    xi = xk.copy()
    Xr = [xi]
    Ur = []
    X_obs = [obs_pos.copy()]
    min_dis = np.inf
    xfk = xf.copy()
    while ti <= T:
        cur_obs = X_obs[-1]
        min_cur_dis = np.inf
        for i in range(n_obs):
            po = cur_obs[i, :3]
            cur_dis = np.linalg.norm(xi[:3] - po)
            min_dis = min(min_dis, cur_dis)
            min_cur_dis = min(min_cur_dis, cur_dis)
        ui = Explicit_MPC(xi, xfk)
        xi = rm.update(xi, ui)
        Xr.append(xi)
        Ur.append(ui)
        xfk = xf_update(xfk)
        ti += rm.get_sample_time()
        X_obs.append(rm.update(cur_obs.T, [0, 0, 0]).T)
        if min_cur_dis >= sat.view_rge:
            break
    Xr, X_obs = np.array(Xr), np.array(X_obs)
    Ur = np.array(Ur)
    if min_dis >= obs_rad:
        print(f"Relying solely on MPC Approximation can avoid collisions, without need for SCP")
        return Xr, Ur, 0
    print(f"the minimum distance between satellite and obstacles is {np.round(min_dis, 4)}, need SCP!")
    xf = Xr[-1].copy()
    obj = np.inf
    for _ in range(10):
        print(f"The {_}-th Iteration: ", end=" ")
        Xr_, Ur_, obj_ = Successive_convex_programming(Xr, xf, rm, X_obs, obs_rad, tau=op.tau, c_pun=op.pun)
        print(f"the objective value is {np.round(obj_, 3)}, the minimum distance between satellite and obstacles is ",
              end="")
        min_dis = np.inf
        for o in range(n_obs):
            min_dis = min(min_dis, min([np.linalg.norm(Xr_[k, :3] - X_obs[k][o, :3]) for k in range(Xr.shape[0])]))
            print(np.round(min_dis, 6), "/", obs_rad, end="; ")
        print("")
        Xr = Xr_
        Ur = Ur_
        if abs(obj_ - obj) <= 0.5:
            break
        obj = obj_
    print("Iteration finished !")
    return Xr, Ur, 1


def Successive_convex_programming(Xr, xf, rm: RelativeMotionModel_J2, obs_pos, obs_rad, **kwargs):
    obs_pos = np.array(obs_pos)
    tau = kwargs["tau"]
    c_pun = kwargs["c_pun"]
    dt = rm.get_sample_time() * 1.2
    N = Xr.shape[0] - 1
    U = cp.Variable(N * op.dim_u)
    X = dict.fromkeys(range(N + 1))
    X[0] = Xr[0]
    constraints = []
    objective = 0.0
    for k in range(N):
        Jac = rm.Jacboi_matrix(Xr[k])
        err = np.array(rm.ds_fsat(Xr[k], [0, 0, 0], rm.csat_state)) - Jac @ Xr[k]
        Ad = expm(dt * Jac)
        Bd = dt * (expm(dt * Jac) @ rm.B + 4 * expm(dt * Jac / 2) @ rm.B + rm.B) / 6
        erd = dt * (expm(dt * Jac) @ err + 4 * expm(dt * Jac / 2) @ err + err) / 6
        X[k + 1] = Ad @ X[k] + Bd @ U[op.dim_u * k: op.dim_u * (k + 1)] + erd
        objective += cp.quad_form(U[op.dim_u * k: op.dim_u * (k + 1)], op.R, assume_PSD=True)
        if np.ndim(xf) == 2:
            objective += cp.quad_form(X[k + 1] - xf[k + 1], op.Q, assume_PSD=True)
        constraints.append(cp.norm(X[k + 1] - Xr[k + 1], p="inf") <= tau)
        """convexification for non-convex constraints: soft constraints"""
        Ok_ = obs_pos[k + 1]
        for o in range(obs_pos.shape[1]):
            po = Ok_[o, :3]
            Df = Xr[k + 1, :3] - po
            # constraints.append(Df @ (X[k + 1][:3] - po) / np.linalg.norm(Df) >= ro * rho)
            objective += cp.pos(obs_rad - Df @ (X[k + 1][:3] - po) / np.linalg.norm(Df)) * c_pun
    constraints.append(U <= op.u_max)
    constraints.append(-U <= op.u_max)
    opt = cp.Minimize(objective)
    prob = cp.Problem(opt, constraints)
    prob.solve(solver=cp.MOSEK)
    if "infeasible" in prob.status:
        print("the problem is infeasible!")
        return None, None, None
    X_value = np.array([Xr[0]] + [X[k].value for k in range(1, N + 1)])
    U_value = U.value.reshape(N, op.dim_u)
    return X_value, U_value, prob.value
