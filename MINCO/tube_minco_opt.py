import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
import os

# 添加 tubeRRTstar 路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'tubeRRTstar'))

from tube_rrt_star_2D import (
    SimpleOccupancyMap2D,
    plan_tube_rrt_star_2d,
)

# ==========================================
# 1. 从MINCO/gen_map_tube/tube_corridor.npz 中加载地图和走廊
# ==========================================

setting = {
    "TubeRadius": 0.5,
    "useIntVol": True,
    "costIntVol": 1.0,
    "GoalBias": 3.0,
    "ContinueAfterGoalReached": True,
    "MaxNumTreeNodes": 3000,
    "MaxIterations": 1000,
    "MaxConnectionDistance": 3.0,
    "xLim": [0.0, 20.0],
    "yLim": [0.0, 20.0],
}

# 首先尝试从生成脚本的输出 npz 中加载走廊（优先），请根据实际路径调整
npz_path_candidates = [
    os.path.join(os.path.dirname(__file__), '..','MINCO', 'gen_map_tube','tube_corridor.npz'),
]
found_npz = None
for p in npz_path_candidates:
    if os.path.isfile(p):
        found_npz = p
        break

if found_npz is not None:
    # 严格假定 npz 中包含所需字段：grid, resolution, origin, oc_list, r_list, path
    print(f"Loading corridor (strict) from {found_npz}")
    data = np.load(found_npz)
    # 地图栅格信息
    grid = np.asarray(data['grid'])
    resolution = float(data['resolution'])
    origin = np.asarray(data['origin'], dtype=float)
    # 兼容 origin 为标量或长度1的情况
    if origin.ndim == 0 or origin.size == 1:
        origin = np.array([float(origin), 0.0])
    map2d = SimpleOccupancyMap2D(grid.astype(bool), resolution, origin)
    print(f"Loaded occupancy grid from {found_npz} (shape={grid.shape}, res={resolution})")

    # 走廊数据
    path = np.asarray(data['path'])
    oc_list = np.asarray(data['oc_list'])
    r_list = np.asarray(data['r_list'])

    # endpoints 
    p0 = path[0, :2]
    pf = path[-1, :2]

    n = len(oc_list) + 1
    print(f"Loaded corridor: {len(oc_list)} intermediate balls, n={n}")
    tree = None
else:
    # 没找到npz文件，请用gen_map_tube.py生成
    tried = ', '.join([os.path.abspath(p) for p in npz_path_candidates])
    raise RuntimeError(f"No corridor .npz found. Tried: {tried}")

# Initial velocities/accelerations
v0 = np.array([0.0, 0.0])
a0 = np.array([0.0, 0.0])
vf = np.array([0.0, 0.0])
af = np.array([0.0, 0.0])

print(f"Optimization with n={n} segments.")


# -------------------------------------------------------------------
#  球映射：无约束 ξ -> 球内中间点 q
# -------------------------------------------------------------------
def map_to_ball(xi):
    """
    xi: 扁平化向量，长度应为 2*(n-1)，表示每个中间点的无约束参数
    返回 shape=(n-1,2) 的中间点数组 q
    映射规则：q_i = oc_list[i] + 2*r_list[i] * xi_i / (||xi_i||^2 + 1)
    """
    xi2 = np.asarray(xi).reshape(-1, 2)
    m = xi2.shape[0]
    qs = np.zeros((m, 2))
    for i in range(m):
        v = xi2[i]
        norm2 = np.dot(v, v)
        qs[i] = oc_list[i] + 2.0 * r_list[i] * v / (norm2 + 1.0)
    return qs


# -------------------------------------------------------------------
#  时间映射：τ -> T（保证 T>0）
# -------------------------------------------------------------------
def map_time(tau):
    return np.exp(tau)


# -------------------------------------------------------------------
#  构造 n 段 1D quintic 系数（6n x 6n 线性系统）
# -------------------------------------------------------------------
def compute_n_segment_coeff_1d(p0, v0, a0, q_list, pf, vf, af, T_list):
    """
    通用 n 段系数求解（1D）。
    - q_list: 长度 n-1 的中间点列表（每个是一个标量）
    - T_list: 长度 n 的每段时间
    返回每段的系数数组 shape=(n,6)
    """
    n = len(T_list)
    M = np.zeros((6 * n, 6 * n))
    b = np.zeros(6 * n)

    def row_p(t):
        return np.array([1, t, t**2, t**3, t**4, t**5])

    def row_dp(t):
        return np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4])

    def row_ddp(t):
        return np.array([0, 0, 2, 6*t, 12*t**2, 20*t**3])

    def row_d3p(t):
        return np.array([0, 0, 0, 6, 24*t, 60*t**2])

    def row_d4p(t):
        return np.array([0, 0, 0, 0, 24, 120*t])

    row = 0
    # 起点约束（段0 at t=0）
    M[row, 0:6] = row_p(0.0); b[row] = p0; row += 1
    M[row, 0:6] = row_dp(0.0); b[row] = v0; row += 1
    M[row, 0:6] = row_ddp(0.0); b[row] = a0; row += 1

    # 对每个内部接合点添加约束
    for j in range(n - 1):
        Tj = T_list[j]
        qj = q_list[j]
        # p_j(Tj) = qj
        M[row, j * 6:(j + 1) * 6] = row_p(Tj); b[row] = qj; row += 1
        # p_{j+1}(0) = qj
        M[row, (j + 1) * 6:(j + 2) * 6] = row_p(0.0); b[row] = qj; row += 1
        # continuity of derivatives up to 4th
        M[row, j * 6:(j + 1) * 6] = row_dp(Tj); M[row, (j + 1) * 6:(j + 2) * 6] = -row_dp(0.0); b[row] = 0.0; row += 1
        M[row, j * 6:(j + 1) * 6] = row_ddp(Tj); M[row, (j + 1) * 6:(j + 2) * 6] = -row_ddp(0.0); b[row] = 0.0; row += 1
        M[row, j * 6:(j + 1) * 6] = row_d3p(Tj); M[row, (j + 1) * 6:(j + 2) * 6] = -row_d3p(0.0); b[row] = 0.0; row += 1
        M[row, j * 6:(j + 1) * 6] = row_d4p(Tj); M[row, (j + 1) * 6:(j + 2) * 6] = -row_d4p(0.0); b[row] = 0.0; row += 1

    # 终点约束（段 n-1 at t=Tn）
    Tn = T_list[-1]
    M[row, (n - 1) * 6:n * 6] = row_p(Tn); b[row] = pf; row += 1
    M[row, (n - 1) * 6:n * 6] = row_dp(Tn); b[row] = vf; row += 1
    M[row, (n - 1) * 6:n * 6] = row_ddp(Tn); b[row] = af; row += 1

    # 解决线性系统
    coeff = np.linalg.solve(M, b)
    coeffs = coeff.reshape(n, 6)
    return coeffs


def compute_n_segment_coeff_2d(p0, v0, a0, q_list, pf, vf, af, T_list):
    # 对 x、y 分别求解
    qx = [q[0] for q in q_list]
    qy = [q[1] for q in q_list]
    coeffs_x = compute_n_segment_coeff_1d(p0[0], v0[0], a0[0], qx, pf[0], vf[0], af[0], T_list)
    coeffs_y = compute_n_segment_coeff_1d(p0[1], v0[1], a0[1], qy, pf[1], vf[1], af[1], T_list)
    # 返回每段系数 shape (n,6) for x and y
    coeffs = np.zeros((len(T_list), 2, 6))
    coeffs[:, 0, :] = coeffs_x
    coeffs[:, 1, :] = coeffs_y
    # reshape to list of 2x6 per segment when used
    return coeffs


# -------------------------------------------------------------------
#  jerk 能量：∫ ||p'''(t)||^2 dt
# -------------------------------------------------------------------
def jerk_energy(coeff, T, N=80):
    a3 = coeff[:, 3]
    a4 = coeff[:, 4]
    a5 = coeff[:, 5]

    ts = np.linspace(0.0, T, N)
    dt = T / (N - 1)

    J = 0.0
    for t in ts:
        jerk = 6*a3 + 24*a4*t + 60*a5*(t**2)
        J += np.dot(jerk, jerk) * dt
    return J


# -------------------------------------------------------------------
#  代价函数：F(ξ, τ) = 两段 jerk 能量之和
# -------------------------------------------------------------------
def minco_cost(x):
    # x = [xi (2*(n-1)), tau (n)]
    xi = x[:2 * (n - 1)]
    tau = x[2 * (n - 1):]

    q_list = map_to_ball(xi)  # shape (n-1,2)
    T_list = map_time(tau)    # shape (n,)

    # 求 n 段系数（C^4 连续）
    coeffs = compute_n_segment_coeff_2d(
        p0, v0, a0,
        q_list, pf, vf, af,
        T_list
    )

    # coeffs shape: (n, 2, 6)
    J = 0.0
    for i in range(len(T_list)):
        # coeffs[i] has shape (2,6) for 2D
        J += jerk_energy(coeffs[i], T_list[i])

    # 时间惩罚
    time_penalty = 1.0 * np.sum(T_list)
    return J + time_penalty


# -----------------------------
#  优化 (L-BFGS-B)
# -----------------------------

x0 = np.zeros(2 * (n - 1) + n)  # xi (2*(n-1)), tau (n) -> initial zeros -> xi=0->center, tau=0->T=1

# -----------------------------
#  在优化过程中可视化每次迭代的轨迹（callback）
# -----------------------------

def traj_from_x(x):
    """给定优化变量 x，返回 n 段拼接后的轨迹点数组（N×2）"""
    xi = x[:2 * (n - 1)]
    tau = x[2 * (n - 1):]
    q_list = map_to_ball(xi)
    T_list = map_time(tau)
    coeffs = compute_n_segment_coeff_2d(p0, v0, a0, q_list, pf, vf, af, T_list)
    segs = []
    for i in range(len(T_list)):
        segs.append(sample_segment(coeffs[i], T_list[i]))
    return np.vstack(segs)


# 采样轨迹用于可视化（提前定义，供 callback 使用）
def sample_segment(coeff, T, n=100):
    ts = np.linspace(0.0, T, n)
    pts = []
    for t in ts:
      px = (coeff[0,0] + coeff[0,1]*t + coeff[0,2]*t**2 +
          coeff[0,3]*t**3 + coeff[0,4]*t**4 + coeff[0,5]*t**5)
      py = (coeff[1,0] + coeff[1,1]*t + coeff[1,2]*t**2 +
          coeff[1,3]*t**3 + coeff[1,4]*t**4 + coeff[1,5]*t**5)
      pts.append([px, py])
    return np.array(pts)


# 先创建图，用于在 callback 中实时绘制（绘制静态元素：球、起/终点）
fig, ax = plt.subplots(figsize=(8, 8))

# 绘制地图障碍物
if 'map2d' in globals() and isinstance(map2d, SimpleOccupancyMap2D):
    grid = map2d.grid
    nx, ny = grid.shape
    res = map2d.resolution
    ox, oy = map2d.origin
    xmin, xmax = ox, ox + nx * res
    ymin, ymax = oy, oy + ny * res
    ax.imshow(grid.T, origin="lower", extent=(xmin, xmax, ymin, ymax), cmap="gray_r", alpha=0.5)

# 画球（背景）：对每个 oc_list/r_list 循环绘制
theta = np.linspace(0, 2 * np.pi, 200)
for i_oc, rr in zip(oc_list, r_list):
    xc = i_oc[0] + rr * np.cos(theta)
    yc = i_oc[1] + rr * np.sin(theta)
    ax.fill(xc, yc, color='orange', alpha=0.3)
    ax.plot(xc, yc, color='orange')

# 起点/终点
ax.scatter(p0[0], p0[1], color='green', s=80)
ax.text(p0[0] + 0.1, p0[1] + 0.1, "start")
ax.scatter(pf[0], pf[1], color='red', s=80)
ax.text(pf[0] + 0.1, pf[1] + 0.1, "goal")

# 历史轨迹列表（每次迭代的轨迹）
traj_history = []
# 记录每次迭代的能量与总时间
energy_history = []
time_history = []
# 保存动态绘制的 Line2D 对象，便于后续移除
dynamic_lines = []
# 保存每次迭代的关节点（q）的绘图对象
dynamic_points = []
# 保存每次迭代的 q 值（用于重绘）
q_history = []
# 记录 callback 被调用的次数（用于迭代编号）
iter_count = 0


def opt_callback(xk):
    """scipy minimize 的 callback：接收当前变量向量 xk，每次迭代调用一次。
    我们把每次迭代得到的轨迹都保存到 traj_history，然后重新在同一张图上绘制所有历史轨迹，
    并根据迭代序号调整 alpha（早期迭代更不透明，后期迭代更透明）。"""
    try:
        trajk = traj_from_x(xk)
    except Exception:
        # 若当前 xk 导致系数求解失败（数值问题），则跳过绘制
        return

    traj_history.append(trajk)

    # 注意：opt_callback 也可能在调用前手动调用一次（用于绘制初始轨迹）
    try:
        qv = map_to_ball(xk[:2 * (n - 1)])
        T_list_print = map_time(xk[2 * (n - 1):])
    except Exception:
        qv = None
        T_list_print = None
    # 保存 q 值用于绘制小点（qv shape = (n-1,2)）
    if qv is not None:
        q_history.append(qv)
        # 计算当前迭代的 cost 分解：总 jerk 能量和总时间
        try:
            # 复用 compute_n_segment_coeff_2d 与 jerk_energy
            coeffs_iter = compute_n_segment_coeff_2d(p0, v0, a0, qv, pf, vf, af, T_list_print)
            J_iter = 0.0
            for ii in range(len(T_list_print)):
                J_iter += jerk_energy(coeffs_iter[ii], T_list_print[ii])
            energy_history.append(J_iter)
            time_history.append(np.sum(T_list_print))
        except Exception:
            # 若数值问题导致无法计算能量，则记录 NaN
            energy_history.append(np.nan)
            time_history.append(np.sum(T_list_print) if T_list_print is not None else np.nan)
    # 打印信息（显示所有段时间）
    # print(f"Iter {len(q_history)-1}: q={qv}, T={T_list_print}")
    # 增加迭代计数
    globals()['iter_count'] = len(q_history)
    # 轨迹总时间
    sum_T = np.sum(T_list_print)
    print(f"Iter {globals()['iter_count']-1}: total_time={sum_T:.3f}")
    # 清除之前绘制的历史轨迹线和历史点（但保留静态元素，如球、起点、终点）
    for ln in dynamic_lines:
        try:
            ln.remove()
        except Exception:
            pass
    dynamic_lines.clear()
    for pt in dynamic_points:
        try:
            pt.remove()
        except Exception:
            pass
    dynamic_points.clear()

    # 重新绘制所有历史轨迹，使用紫色并随着迭代索引增大而降低透明度
    total = len(traj_history)
    alpha_max = 0.9
    alpha_min = 0.05
    for i, tr in enumerate(traj_history):
        if total == 1:
            alpha = alpha_max
        else:
            # i=0 为最早的迭代（最不透明），i=total-1 为最新的迭代（最透明）
            alpha = alpha_max - (alpha_max - alpha_min) * (i / (total - 1))
        ln, = ax.plot(tr[:, 0], tr[:, 1], color='purple', alpha=alpha, linewidth=1.5)
        dynamic_lines.append(ln)

    # 绘制每次迭代的中间路点 q（小点，无标签）
    # q_history 中维护了每次迭代的 q
    for i, qv in enumerate(q_history):
        if total == 1:
            alpha = alpha_max
        else:
            alpha = alpha_max - (alpha_max - alpha_min) * (i / (total - 1))
        # qv is (n-1,2) array; 绘制所有中间点
        pt = ax.scatter(qv[:, 0], qv[:, 1], color='black', s=12, alpha=alpha, zorder=4)
        dynamic_points.append(pt)

    plt.draw()
    plt.pause(0.0001)

# 先画初始轨迹（可选）
opt_callback(x0)
# 优化器每迭代一次都会调用 opt_callback
res = minimize(minco_cost, x0, method='L-BFGS-B', callback=opt_callback)

print("===============================Final==================================")

print("优化结果 x:", res.x)
# 拆分优化变量
xi_opt = res.x[:2 * (n - 1)]
tau_opt = res.x[2 * (n - 1):]

# 映射到实际 q 与 T
q_opt = map_to_ball(xi_opt)  # shape (n-1,2)
T_opt = map_time(tau_opt)    # shape (n,)

print("优化后 q:", q_opt)
print("优化后 T:", T_opt)

# 计算每段系数并采样拼接轨迹
coeffs_opt = compute_n_segment_coeff_2d(p0, v0, a0, q_opt, pf, vf, af, T_opt)
segs = []
for i in range(len(T_opt)):
    segs.append(sample_segment(coeffs_opt[i], T_opt[i]))
traj = np.vstack(segs)


# ================= 可视化 =======================
# 在回调绘制的图上覆盖最终结果
# 绘制最终优化得到的中间点 q_opt（覆盖在历史轨迹上）
ax.scatter(q_opt[:, 0], q_opt[:, 1], color='blue', s=80, zorder=5)
for idx, qp in enumerate(q_opt):
    ax.text(qp[0] + 0.05, qp[1] + 0.05, f"q{idx}")

# 绘制最终轨迹（加粗，颜色与历史轨迹一致但更明显）
ax.plot(traj[:, 0], traj[:, 1], color='purple', linewidth=3, label='final_traj', zorder=6)

# 确保输出目录存在，保存最终轨迹图与数据（数据格式兼容 tube_corridor.npz 的字段）
out_dir = os.path.join('MINCO', 'minco_opt_data')
os.makedirs(out_dir, exist_ok=True)
try:
    # 保存当前主图（final trajectory）为 PNG
    fig.savefig(os.path.join(out_dir, 'final_traj.png'), dpi=150)
except Exception as e:
    print('Failed to save final_traj.png:', e)

try:
    # 保存为压缩 npz，包含与 tube_corridor.npz 一致的字段以及优化结果
    np.savez_compressed(
        os.path.join(out_dir, 'minco_result.npz'),
        grid=grid,
        resolution=resolution,
        origin=origin,
        oc_list=oc_list,
        r_list=r_list,
        path=path,
        traj=traj,
        q_opt=q_opt,
        T_opt=T_opt,
    )
except Exception as e:
    print('Failed to save minco_result.npz:', e)

# 保证图像外观（若之前已设置过，这些也不会有害）
ax.set_aspect('equal')
ax.grid(True)
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f'{n}-segment MINCO-style trajectory (C^4 at waypoints)')
ax.legend(loc='upper left')
plt.tight_layout()
# 在展示主图之前，先保存迭代历史图（如果有），以防用户在 GUI 中中断
try:
    if len(energy_history) > 0 or len(time_history) > 0:
        fig2, ax1 = plt.subplots(figsize=(8, 4))
        iters = np.arange(len(energy_history))
        ax1.plot(iters, energy_history, color='tab:purple', marker='o', label='Total Jerk Energy')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Total Jerk Energy', color='tab:purple')
        ax1.tick_params(axis='y', labelcolor='tab:purple')

        ax2 = ax1.twinx()
        ax2.plot(iters, time_history, color='tab:green', marker='x', label='Total Flight Time')
        ax2.set_ylabel('Total Flight Time (s)', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
        ax1.set_title('MINCO Iteration History: Jerk Energy & Flight Time')
        plt.tight_layout()
    # 确保输出目录存在，仅保存 PNG 到 MINCO/minco_opt_data（不再保存 PDF）
    out_dir = os.path.join('MINCO', 'minco_opt_data')
    os.makedirs(out_dir, exist_ok=True)
    fig2.savefig(os.path.join(out_dir, 'iter_history.png'), dpi=150)
except Exception as e:
    print("Failed to auto-save iteration history before show:", e)

plt.show()

# 绘制迭代历史：总 jerk 能量 和 总飞行时间（两条曲线在同一张图，右侧为时间轴）
try:
    if len(energy_history) > 0 or len(time_history) > 0:
        fig2, ax1 = plt.subplots(figsize=(8, 4))
        iters = np.arange(len(energy_history))
        ax1.plot(iters, energy_history, color='tab:purple', marker='o', label='Total Jerk Energy')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Total Jerk Energy', color='tab:purple')
        ax1.tick_params(axis='y', labelcolor='tab:purple')

        ax2 = ax1.twinx()
        ax2.plot(iters, time_history, color='tab:green', marker='x', label='Total Flight Time')
        ax2.set_ylabel('Total Flight Time (s)', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        # 合并图例
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
        ax1.set_title('MINCO Iteration History: Jerk Energy & Flight Time')
        plt.tight_layout()
        fig2.savefig('MINCO/minco_opt_data/iter_history.png', dpi=150)
        plt.show()
except Exception as e:
    print("Failed to plot iteration history:", e)
