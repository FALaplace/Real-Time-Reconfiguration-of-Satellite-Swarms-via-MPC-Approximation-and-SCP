import numpy as np
from Controllers import Simple_MPC, Explicit_MPC
from Global_Parameters import op, sat
from fitting_process import ellipse_propagate

a = 1.5765996259384667
b = 1.2855999197531711
n = [np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), np.sin(b)]
u = np.array([-n[2], 0, n[0]]) / np.linalg.norm(np.array([-n[2], 0, n[0]]))
v = np.array([0, -n[2], n[1]]) / np.linalg.norm(np.array([0, -n[2], n[1]]))
N = 10

Xt = np.zeros((N, op.dim_x))
r_tar = 2.0
for i in range(N):
    fi = i * np.pi * 2 / N
    Xt[i] = r_tar * np.array([np.cos(fi) / 2, np.sin(fi), 0, sat.omega * np.sin(fi) / 2, -sat.omega * np.cos(fi), 0])
    pass

if __name__ == "__main__":
    sat_id = 1
    x = np.load(f"recording_data/sat_{sat_id}_trajectory_EMPC-SCP_Tp=600_dynamic.npy")
    for i in range(x.shape[0]):
        u_mpc = Simple_MPC(x[i], Xt[sat_id])
        u_empc = Explicit_MPC(x[i], Xt[sat_id])
        Xt = ellipse_propagate(Xt, sat.rm.get_sample_time())
        print(f"at time {i}:", u_empc, u_mpc)

