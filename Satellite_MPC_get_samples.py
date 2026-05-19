import numpy as np
from Global_Parameters import op, sat
# from Satellite_MPC import MPC_solver
from datetime import datetime

EDGE_value = {}


def Satellite_MPC_get_samples(X: np.ndarray):
    N = 30
    U = []
    SIGMA = []
    sol, lbg, ubg = MPC_solver(sat.rm, N, sat.x_tar, sat.u_tar)
    x_min, u_min = [-xi for xi in op.x_max], [-ui for ui in op.u_max_list]
    ubx = np.array(op.x_max * (N + 1) + op.u_max_list * N + [np.inf])[:, None]
    lbx = np.array(x_min * (N + 1) + u_min * N + [0.0])[:, None]
    x_inigau = np.zeros(sol.size_in("x0"))
    for k in range(N):
        x_inigau[6 * (k + 1):6 * (k + 2), 0] = X[0, :]
        x_inigau[6 * (N + 1) + 3 * k: 6 * (N + 1) + 3 * (k + 1), 0] = sat.u_tar
    for i in range(X.shape[0]):
        xi = X[i, :]
        if tuple(xi) in EDGE_value:
            ui, sigma = EDGE_value[tuple(xi)]
        else:
            res: dict = sol(x0=x_inigau, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=xi)
            x_inigau = np.array(res["x"])
            ui = x_inigau[op.dim_x * (N + 1): op.dim_x * (N + 1) + op.dim_u, 0]
            sigma = x_inigau[-1, 0]
            EDGE_value[tuple(xi)] = (ui, sigma)
        U.append(ui)
        SIGMA.append(float(sigma))
    return np.array(U), SIGMA

