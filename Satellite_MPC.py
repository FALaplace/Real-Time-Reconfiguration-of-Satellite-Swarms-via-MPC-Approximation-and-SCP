import numpy as np
from Controllers import get_MPC_result
from Global_Parameters import sat, op


if __name__ == "__main__":
    x0 = np.array([-10, -10, -10, 0, 0, 0])
    xk = x0.copy()
    xt = np.array([0, 0, 0, 0, 0, 0])
    ut = -np.array(sat.rm.ds_fsat(xt, [0] * 3, sat.rm.csat_state)[3:])
    N = 30
    X_seq = [xk.copy()]
    U_seq = []
    for _ in range(50):
        sol = get_MPC_result(sat.rm, xk, xt, ut, N=N)
        u_opt = np.array(sol["x"])[op.dim_x * (N + 1): op.dim_x * (N + 1) + op.dim_u, 0]
        U_seq.append(u_opt)
        xk = sat.rm.update(xk, u_opt)
        X_seq.append(xk.copy())
        print(sol["f"])
        pass
    X_seq = np.array(X_seq)
    U_seq = np.array(U_seq)
    print(X_seq.shape)

    # xt_ = np.array([-10, -10, -10, 0, 0, 0])
    # ut_ = -np.array(sat.rm.ds_fsat(xt_, [0] * 3, sat.rm.csat_state)[3:])
    # x0_ = x0 + (xt_ - xt)
    # xk_ = x0_.copy()
    # X_seq_ = [xk_.copy()]
    # for k_ in range(U_seq.shape[0]):
    #     xk_ = sat.rm.update(xk_, U_seq[k_] - ut + ut_)
    #     X_seq_.append(xk_)
    # print(X_seq_[-1], xt_)
    # X_seq_ = np.array(X_seq_)

    # plot_trajectory(X_seq, tar=xt)
    # plot_trajectory(X_seq_, tar=xt_)
