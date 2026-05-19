import numpy as np
from Global_Parameters import op, sat
from itertools import product
from typing import Dict
from Satellite_MPC_get_samples import Satellite_MPC_get_samples
from AKI_functions import AKI_grid
import json
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sigma_max = 1e-4
folder_path = os.path.join(".", "sampling_data")
os.makedirs(folder_path, exist_ok=True)
parallel = False   # parallel calculation flag


def sampling_process(D, M1=2, M2=2):
    X = AKI_grid(D, M1, M2)
    U, Sigma = Satellite_MPC_get_samples(X)
    feasible_list = [x < sigma_max for x in Sigma]
    return op.normalX(X), op.normalU(U), feasible_list


def process_domain_item(idx, Di, m1, m2):
    save_path = os.path.join(folder_path, f"{idx}_Mp={m1}_Mv={m2}.json")
    if os.path.exists(save_path):
        return f"file {save_path} has existed!"
    time1 = datetime.now()
    Xi, Ui, fea_i = sampling_process(Di, m1, m2)
    data_to_save = {
        "X": Xi.tolist(),
        "U": Ui.tolist(),
        "feasible_sign": fea_i
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4)
    elapsed = datetime.now() - time1
    return f"file {save_path} saved! Time = {elapsed}"


def data_normalization(DOMAIN: dict):
    for idd, Di in tqdm(DOMAIN.items(), desc="data normalizing..."):
        file_path = os.path.join(folder_path, f"{idd}_Mp=3_Mv=3.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding="utf-8") as f:
                origin_data: dict = json.load(f)
            X = np.array(origin_data["X"])
            U = np.array(origin_data["U"])
            X_nor = np.clip((X + np.array(op.x_max)) / (2 * np.array(op.x_max)), 0, 1)
            U_nor = np.clip((U + op.u_max) / (2 * op.u_max), 0, 1)
            normalized_data = {
                "X": X_nor.tolist(),
                "U": U_nor.tolist(),
                "feasible_sign": origin_data["feasible_sign"]
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(normalized_data, f, indent=4)


if __name__ == "__main__":
    # domain grid
    Mp, Mv = 4, 4
    total_domain = Mp ** (op.dim_x // 2) * Mv ** (op.dim_x // 2)
    DOMAIN: Dict[tuple, np.ndarray] = {}
    delta_p = 2 * op.pos_max / Mp
    delta_v = 2 * op.vel_max / Mv
    p_idx = range(Mp)
    v_idx = range(Mv)
    for idx in product(p_idx, p_idx, p_idx, v_idx, v_idx, v_idx):
        i1, i2, i3, i4, i5, i6 = idx
        pos_bounds = np.array([[i * delta_p - op.pos_max, (i + 1) * delta_p - op.pos_max] for i in [i1, i2, i3]])
        vel_bounds = np.array([[i * delta_v - op.vel_max, (i + 1) * delta_v - op.vel_max] for i in [i4, i5, i6]])
        DOMAIN[idx] = np.vstack((pos_bounds, vel_bounds))
    m1, m2 = 3, 3
    if parallel:
        with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            futures = [executor.submit(process_domain_item, idx, Di, m1, m2) for idx, Di in DOMAIN.items()]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Sampling in progress"):
                try:
                    msg = f.result()
                    print(msg)
                except Exception as e:
                    print(f"Error in a task: {e}")
    else:
        for idx, Di in tqdm(DOMAIN.items(), desc="Sampling in progress"):
            log = process_domain_item(idx, Di, m1, m2)
            print(log)
