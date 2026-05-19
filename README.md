# Explicit MPC with SCP for Satellite Swarm Reconfiguration

本项目实现了基于**核插值显式MPC（EMPC）**结合**序列凸规划（SCP）**的卫星集群重构控制算法，在保证碰撞规避的前提下实现高效实时控制。

---

## 系统概述

控制对象为多颗跟随卫星，以 Hill 坐标系下的相对运动描述，状态为：

$$x = [x, y, z, v_x, v_y, v_z]^\top \in \mathbb{R}^6$$

控制输入为推力加速度 $u = [a_x, a_y, a_z]^\top \in \mathbb{R}^3$。

**整体流程分为三阶段：**

```
离线阶段: MPC求解 → 采样数据存储
                        ↓
在线阶段: 核插值查询 → EMPC快速输出
                        ↓
碰撞规避: SCP迭代精化轨迹
```

---

## 动力学模型（`MotionModel.py`）

### J2摄动下的相对运动方程

跟随卫星相对于参考卫星的运动方程由 `RelativeMotionModel_J2.ds_fsat()` 实现，包含完整的 J2 摄动项：

```
dvx = 2·vy·wz - x·(ηf - wz²) + y·az - z·wx·wz - (ξf - ξ)·sin(i)·sin(θ) - r·(ηf - η) + u₀
dvy = -2·vx·wz + 2·vz·wx - x·az - y·(ηf - wz² - wx²) + z·ax - (ξf - ξ)·sin(i)·cos(θ) + u₁
dvz = -2·vy·wx - x·wx·wz - y·ax - z·(ηf - wx²) - (ξf - ξ)·cos(i) + u₂
```

其中 $w_x, w_z$ 为参考轨道角速度分量，$\eta, \xi$ 为 J2 引力梯度系数，$\eta_f, \xi_f$ 为跟随卫星处的对应值。

### 离散化

采用四阶 Runge-Kutta 积分（步长 $h = 10$ s）进行离散化，`discrete_dynamic()` 为 CasADi 符号版本（用于 NLP 求解），`update()` 为 NumPy 数值版本（用于仿真传播）。

### 参考卫星轨道参数（`Global_Parameters.py`）

| 参数 | 值 |
|------|-----|
| 轨道高度 | 900 km（$a = 7271$ km）|
| 轨道倾角 | 60° |
| RAAN | 50.2° |
| 近地点幅角 | 45° |
| J2 系数 | $1.08264 \times 10^{-3}$ |
| 采样时间 $h$ | 10 s |

---

## MPC Ground Truth 构建（`Controllers.py`）

### 优化问题

MPC 以有限时域最优控制问题形式构建，通过 CasADi + IPOPT 求解 NLP：

$$\min_{X, U, \sigma} \sum_{k=0}^{N-1} \left[ (x_k - x_\text{tar})^\top Q (x_k - x_\text{tar}) + (u_k - u_\text{tar})^\top R (u_k - u_\text{tar}) \right] + (x_N - x_\text{tar})^\top P (x_N - x_\text{tar}) + w_\sigma \sigma$$

**约束：**
- 动力学约束：$x_{k+1} = f(x_k, u_k)$（J2 摄动 RK4 离散化）
- 控制约束：$|u_k| \leq 3 \times 10^{-5}$ km/s²
- 状态约束：$|p| \leq 20$ km，$|v| \leq 0.02$ km/s
- 终端集约束（软约束松弛）：$(x_N - x_\text{tar})^\top P (x_N - x_\text{tar}) \leq \rho - \sigma$，$\sigma \geq 0$

### 终端权重矩阵 P 的计算

终端代价矩阵 $P$ 由**离散代数 Riccati 方程（DARE）**求解，确保终端集内的稳定性：

```python
# cal_terminal_weight() 中：
A = Jacobian of f w.r.t. x at (x_tar, u_tar)   # CasADi 自动微分
B = Jacobian of f w.r.t. u at (x_tar, u_tar)
P = solve_discrete_are(A, B, Q, R)              # scipy
K = (B'PB + R)^{-1} B'PA                       # LQR 增益
ρ = min_i { (u_max - u_tar_i)² / (K_i P^{-1} K_i') }  # 终端集半径
```

### 代价权重

| 矩阵 | 值 |
|------|-----|
| $Q$ | $\text{diag}([6,6,6,100,100,100])^2$ |
| $R$ | $\text{diag}([5\times10^6, 5\times10^6, 5\times10^6])$ |
| $w_\sigma$ | $10^3$ |

速度方向权重远大于位置（$100^2$ vs $6^2$），优先惩罚速度偏差；控制代价 $R$ 极大，抑制推力消耗。

### 目标状态

参考卫星处（$x_\text{tar} = 0$）所需的维持推力 $u_\text{tar}$ 由非线性动力学反解：

```python
u_tar = -rm.ds_fsat(x_tar=0, u=[0,0,0], csat_state)[3:]
```

即抵消 J2 摄动在目标点处产生的加速度偏差。

### IPOPT 配置

- 最大迭代次数：300，收敛精度：$10^{-6}$
- 线性求解器：MUMPS，Hessian 近似：L-BFGS
- 启用热启动（`warm_start_init_point`），前一采样点的解作为下一点初始猜测

---

## 离线采样（`sampling_process_paraell.py`）

EMPC 的核心思想是离线预计算 MPC 策略，存储为查找表，在线仅做函数评估。

### 状态空间分区

将 6 维状态空间均匀划分为 $4^3 \times 4^3 = 4096$ 个局部域（`DOMAIN`）：

- 位置各维：$[-20, 20]$ km 均分为 4 段，段宽 $\Delta p = 10$ km
- 速度各维：$[-0.02, 0.02]$ km/s 均分为 4 段，段宽 $\Delta v = 0.01$ km/s

每个域由 6-tuple 索引 $(i_1, i_2, i_3, i_4, i_5, i_6)$ 唯一标识，边界为：

```python
pos_bounds[j] = [i_j * Δp - 20,  (i_j+1) * Δp - 20]   # j = 1,2,3
vel_bounds[j] = [i_j * Δv - 0.02, (i_j+1) * Δv - 0.02]  # j = 4,5,6
```

### 每个域内的采样网格

在每个局部域 $D_i$ 内，用 `AKI_grid(D, Mp=3, Mv=3)` 构造均匀张量积网格：

- 位置各维：$M_p = 3$ 个点（均匀分布在域边界内）
- 速度各维：$M_v = 3$ 个点
- 每个域共 $3^3 \times 3^3 = 729$ 个采样点

```python
# AKI_grid 核心逻辑：
p_grid = [linspace(D[i,0], D[i,1], Mp) for i in range(3)]  # 位置网格
v_grid = [linspace(D[i,0], D[i,1], Mv) for i in range(3)]  # 速度网格
X = meshgrid(*p_grid, *v_grid).reshape(-1, 6)               # 729×6
```

### 对每个采样点求解 MPC（`Satellite_MPC_get_samples.py`）

对域内每个采样点 $x_i$，调用 `MPC_solver`（目标点固定为 $x_\text{tar} = 0$，预测步长 $N = 30$），提取最优首步控制 $u_0^*$ 和松弛变量 $\sigma$：

```python
for xi in X:
    sol = solver(x0=x_inigau, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=xi)
    u_i  = sol["x"][6*(N+1) : 6*(N+1)+3]   # 首步最优控制
    sigma = sol["x"][-1]                     # 终端约束松弛量
    # 当前解作为下一个点的热启动初始猜测
    x_inigau = sol["x"]
```

### 可行性过滤

以松弛变量 $\sigma$ 作为可行性指标：

$$\text{feasible} = \mathbb{1}[\sigma < \sigma_{\max}], \quad \sigma_{\max} = 10^{-4}$$

$\sigma = 0$ 表示终端约束严格满足；$\sigma$ 较大说明该初始状态距目标过远或动力学不允许在 $N$ 步内到达终端集。

### 数据归一化与存储

采样结果归一化至 $[0, 1]$ 后存入 JSON：

$$\bar{x} = \frac{x + x_{\max}}{2 x_{\max}}, \quad \bar{u} = \frac{u + u_{\max}}{2 u_{\max}}$$

```
sampling_data/
├── (0,0,0,0,0,0)_Mp=3_Mv=3.json   # 域索引 → 729条 {X, U, feasible_sign}
├── (0,0,0,0,0,1)_Mp=3_Mv=3.json
├── ...
└── (3,3,3,3,3,3)_Mp=3_Mv=3.json   # 共4096个文件
```

### 并行采样

通过 `parallel = True` 开启多进程并行（`ProcessPoolExecutor`，使用 CPU 核数 - 1 个进程），已存在的文件自动跳过：

```python
# sampling_process_paraell.py
python sampling_process_paraell.py
```

---

## 在线推理：Explicit MPC（`Controllers.py` → `Explicit_MPC()`）

在线阶段无需求解 NLP，仅做核函数评估：

1. **定位域**：根据当前误差状态 $x_k - x_f$ 查找所属域索引（`find_local_samples`）
2. **加载数据**：读取对应 JSON（缓存于全局 `data_has_read` dict 避免重复 IO）
3. **核插值拟合**：用 Matérn 核（$\nu = 3/2$，长度尺度 $\ell = 0.2$）在该域 729 个点上拟合控制策略
4. **查询输出**：

$$\bar{u}^* = k(x_k^*, X_{\text{sam}}) \cdot (K + \lambda I)^{-1} \bar{U}_{\text{sam}}$$

5. **反归一化**：$u = \text{clip}(\text{inverseU}(\bar{u}^*) - u_\text{tar} + u_f,\ -u_{\max},\ u_{\max})$

---

## 碰撞规避：SCP（`Controllers.py` → `Successive_convex_programming()`）

当 EMPC 轨迹与障碍物距离小于安全距离时，激活 SCP 精化：

- **凸化**：将非凸碰撞约束线性化为切平面不等式
- **信赖域**：$\|x_k - \bar{x}_k\|_\infty \leq \tau = 2.0$ km，限制每次迭代偏移
- **软约束**：$\max(0,\ \rho_{\text{safe}} - n^\top(x_k - p_o)) \times 10^5$ 作为惩罚项加入目标
- 求解器：CVXPY + MOSEK，最多迭代 10 次，目标值变化 $\leq 0.5$ 时提前收敛

---

## 算法对比与仿真（`fitting_process.py`）

`fitting_process.py` 运行 6 组场景比较三种算法：

| 算法 | 函数 | 说明 |
|------|------|------|
| EMPC-SCP | `Collision_Avoidance()` | 本文方法 |
| Simple MPC | `Simple_MPC()` | 基线：每步在线求解 NLP |
| Decentralized MPC | `Decentrized_MPC()` | 分散式：各卫星独立规划 |

场景分为动态目标（椭圆传播）和静态目标，每种算法均在两类场景下测试。
轨迹保存为 `.npy` 至 `AlgorithmComparisonData/`。

---

## 可视化（`plot_figures.py`，`visualization.py`）

```python
python plot_figures.py   # 加载 .npy 文件，生成轨迹对比图与能耗分析图
```

`visualization.py` 提供 3D 轨迹绘图、障碍物可视化和多算法比较曲线。

---

## 运行流程

```bash
# 第一步：离线采样（耗时较长，建议开启并行）
python sampling_process_paraell.py

# 第二步：运行仿真对比
python fitting_process.py

# 第三步：生成图表
python plot_figures.py

# 单条轨迹快速测试
python Satellite_MPC.py
```
