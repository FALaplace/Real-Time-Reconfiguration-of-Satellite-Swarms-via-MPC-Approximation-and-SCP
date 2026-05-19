import numpy as np
import matplotlib.pyplot as plt
from visualization import plot_trajectory_SCP, plot_cluster_trajectories, plot_distance_with_obstacle, \
    plot_distance_between_sat, plot_diff_algo_tra, plot_consumption, plot_control_sequence

if __name__ == "__main__":
    OBS = np.load("recording_data/obstacle_trajectory.npy")
    # u_seq = np.load("recording_data/sat_5_control_sequence.npy")
    # plot_control_sequence(u_seq)
    eta_scp = 0
    for i in [5]:
        ui = np.load(f"recording_data/sat_{i}_control_sequence.npy")
        eta_scp += np.sum(ui[:, -1])
    print(eta_scp / 800)

    # EMPC = np.load("recording_data/sat_5_trajectory_EMPC_dynamic.npy")
    # EMPC_SCP = np.load("recording_data/sat_5_trajectory_EMPC-SCP_Tp=600_dynamic.npy")
    #
    # plot_trajectory_SCP(EMPC, EMPC_SCP, OBS, rad=0.6, rt=2.0)
    # plt.show()
    # EMPC_dy = np.load("AlgorithmComparisonData/sat_8_trajectory_EMPC_dynamic.npy")
    # MPC_dy = np.load("AlgorithmComparisonData/sat_8_trajectory_MPC_dynamic.npy")
    # DMPC_dy = np.load("AlgorithmComparisonData/sat_8_trajectory_DMPC_dynamic.npy")
    # EMPC_st = np.load("AlgorithmComparisonData/sat_8_trajectory_EMPC_static.npy")
    # MPC_st = np.load("AlgorithmComparisonData/sat_8_trajectory_MPC_static.npy")
    # DMPC_st = np.load("AlgorithmComparisonData/sat_8_trajectory_DMPC_static.npy")
    # rt = np.array([0.9781476, -0.41582338, 0, 0, 0, 0])
    # print(np.linalg.norm(EMPC_st[-1, :3] - rt[:3]))
    # print(np.linalg.norm(DMPC_st[-1, :3] - rt[:3]))
    # print(np.linalg.norm(MPC_st[-1, :3] - rt[:3]))

    # U_st = [0.013015353661484156, 0.008884428038197627, 0.010493229175166870]
    # U_dy = [0.009395268739288552, 0.009034138495369104, 0.009370855910490179]
    # plot_consumption()

    # plot_diff_algo_tra(EMPC_dy, DMPC_dy, MPC_dy, OBS, rad=0.6, rt=2.0)
    # plot_cluster_trajectories(MPC, rt=2.0, obs_pos=OBS, obs_rad=0.6, stat_config=True)
    # plt.show()
    # min_dis = []
    # min_dis_ = []
    # for o in range(OBS.shape[1]):
    #     obs_i = OBS[:, o, :]
    #     md_i, md_i_ = [], []
    #     for i in range(10):
    #         md_i.append(min(np.linalg.norm(obs_i[:, :3] - EMPC_dy[i][:, :3], axis=1)))
    #         md_i_.append(min(np.linalg.norm(obs_i[:, :3] - EMPC_SCP_dy[i][:, :3], axis=1)))
    #     min_dis.append(min(md_i)), min_dis_.append(min(md_i_))
    # plot_distance_with_obstacle(min_dis, min_dis_)
    # MIN_DIS = np.ones((10, 10)) * np.inf
    # MIN_DIS_ = np.ones((10, 10)) * np.inf
    # for i in range(10):
    #     xi, xi_ = EMPC_dy[i], EMPC_SCP_dy[i]
    #     for j in range(10):
    #         if j == i:
    #             continue
    #         xj, xj_ = EMPC_dy[j], EMPC_SCP_dy[j]
    #         MIN_DIS[i][j] = min(np.linalg.norm(xi[:, :3] - xj[:, :3], axis=1))
    #         MIN_DIS_[i][j] = min(np.linalg.norm(xi_[:, :3] - xj_[:, :3], axis=1))
    # # print(MIN_DIS)
    # d1 = [0.57745766, 0.57575897, 0.12248756, 0.12248756, 0.58023361, 0.64654955, 0.57575897, 0.52871762, 0.52871762, 0.57745766]
    # d2 = [0.51146963, 0.57575897, 0.20992814, 0.20992814, 0.58023361, 0.54762395, 0.54762395, 0.51146963, 0.52871762, 0.57745766]
    # plot_distance_between_sat(np.min(MIN_DIS, axis=0), np.min(MIN_DIS_, axis=0))
    # plot_distance_between_sat(d1, d2)
