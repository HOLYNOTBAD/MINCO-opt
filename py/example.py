import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --------------------------
# 给定起点、终点、球走廊
# --------------------------
p0 = np.array([0.0, 0.0])
v0 = np.array([0.0, 0.0])
a0 = np.array([0.0, 0.0])

pf = np.array([4.0, 2.0])
vf = np.array([0.0, 0.0])
af = np.array([0.0, 0.0])

# 球走廊
oc = np.array([2.0, 2.5])
r = 1.0


# -------------------------------------------------------------------
#  球映射：无约束 ξ -> 球内中间点 q
# -------------------------------------------------------------------
def map_to_ball(xi):
    """
    q = o + 2r * xi / (||xi||^2 + 1)
    保证 q 始终在中心 oc、半径 r 的球内
    """
    norm2 = np.dot(xi, xi)
    return oc + 2 * r * xi / (norm2 + 1.0)


# -------------------------------------------------------------------
#  时间映射：τ -> T（保证 T>0）
# -------------------------------------------------------------------
def map_time(tau):
    return np.exp(tau)


# -------------------------------------------------------------------
#  构造 1D 两段 quintic 系数（12x12 线性系统）
# -------------------------------------------------------------------
def compute_two_segment_coeff_1d(p0, v0, a0,
                                 q, pf, vf, af,
                                 T1, T2):
    """
    在 1 维上构造两段 5 次多项式：
      段1: p1(t), t ∈ [0, T1]
      段2: p2(t), t ∈ [0, T2]
    约束：
      1) p1(0)=p0, p1'(0)=v0, p1''(0)=a0
      2) p2(T2)=pf, p2'(T2)=vf, p2''(T2)=af
      3) p1(T1)=q, p2(0)=q
      4) 在 joint 处 p',p'',p''',p'''' 连续
    求解 12 个系数 a10..a15, a20..a25
    """

    # 设置时间点
    t1 = T1
    t2 = T2

    # 12 个未知: [a10..a15, a20..a25]
    M = np.zeros((12, 12))
    b = np.zeros(12)

    # 辅助函数：给定 t，返回各阶导数的系数行
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

    # 1) 起点约束：p1(0)=p0, p1'(0)=v0, p1''(0)=a0
    M[row, 0:6] = row_p(0.0)
    b[row] = p0
    row += 1

    M[row, 0:6] = row_dp(0.0)
    b[row] = v0
    row += 1

    M[row, 0:6] = row_ddp(0.0)
    b[row] = a0
    row += 1

    # 2) 终点约束：p2(T2)=pf, p2'(T2)=vf, p2''(T2)=af
    M[row, 6:12] = row_p(t2)
    b[row] = pf
    row += 1

    M[row, 6:12] = row_dp(t2)
    b[row] = vf
    row += 1

    M[row, 6:12] = row_ddp(t2)
    b[row] = af
    row += 1

    # 3) 中间点位置：p1(T1)=q, p2(0)=q
    M[row, 0:6] = row_p(t1)
    b[row] = q
    row += 1

    M[row, 6:12] = row_p(0.0)
    b[row] = q
    row += 1

    # 4) 连续性：p1'(T1)=p2'(0) -> 左 - 右 = 0
    M[row, 0:6] = row_dp(t1)
    M[row, 6:12] = -row_dp(0.0)
    b[row] = 0.0
    row += 1

    #    p1''(T1)=p2''(0)
    M[row, 0:6] = row_ddp(t1)
    M[row, 6:12] = -row_ddp(0.0)
    b[row] = 0.0
    row += 1

    #    p1'''(T1)=p2'''(0)
    M[row, 0:6] = row_d3p(t1)
    M[row, 6:12] = -row_d3p(0.0)
    b[row] = 0.0
    row += 1

    #    p1''''(T1)=p2''''(0)
    M[row, 0:6] = row_d4p(t1)
    M[row, 6:12] = -row_d4p(0.0)
    b[row] = 0.0
    row += 1

    # 解线性方程组
    coeff = np.linalg.solve(M, b)  # shape (12,)
    c1 = coeff[0:6]
    c2 = coeff[6:12]
    return c1, c2


# -------------------------------------------------------------------
#  在 2D 上求两段系数：对 x、y 维度各做一遍
# -------------------------------------------------------------------
def compute_two_segment_coeff_2d(p0, v0, a0,
                                 q, pf, vf, af,
                                 T1, T2):
    # x 维
    c1x, c2x = compute_two_segment_coeff_1d(p0[0], v0[0], a0[0],
                                            q[0], pf[0], vf[0], af[0],
                                            T1, T2)
    # y 维
    c1y, c2y = compute_two_segment_coeff_1d(p0[1], v0[1], a0[1],
                                            q[1], pf[1], vf[1], af[1],
                                            T1, T2)

    coeff1 = np.vstack([c1x, c1y])  # shape (2,6)
    coeff2 = np.vstack([c2x, c2y])
    return coeff1, coeff2


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
    # x = [xi_x, xi_y, tau1, tau2]
    xi = x[:2]
    tau = x[2:]

    q = map_to_ball(xi)
    T1, T2 = map_time(tau)

    # 求两段系数（C^4 连续）
    coeff1, coeff2 = compute_two_segment_coeff_2d(
        p0, v0, a0,
        q, pf, vf, af,
        T1, T2
    )

    J1 = jerk_energy(coeff1, T1)
    J2 = jerk_energy(coeff2, T2)

    # 可以加一点时间惩罚防止时间无限大
    time_penalty = 0.1 * (T1 + T2)

    return J1 + J2 + time_penalty


# -----------------------------
#  优化 (L-BFGS-B)
# -----------------------------

x0 = np.array([0.0, 0.0, 0.0, 0.0])  # xi=(0,0), tau=(0,0)-> T1=T2=1

# -----------------------------
#  在优化过程中可视化每次迭代的轨迹（callback）
# -----------------------------

def traj_from_x(x):
    """给定优化变量 x，返回两段拼接后的轨迹点数组（N×2）"""
    xi = x[:2]
    tau = x[2:]
    q = map_to_ball(xi)
    T1, T2 = map_time(tau)
    coeff1, coeff2 = compute_two_segment_coeff_2d(
        p0, v0, a0, q, pf, vf, af, T1, T2
    )
    traj1 = sample_segment(coeff1, T1)
    traj2 = sample_segment(coeff2, T2)
    return np.vstack((traj1, traj2))


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
fig, ax = plt.subplots(figsize=(6, 6))
# 画球（背景）
theta = np.linspace(0, 2 * np.pi, 200)
xc = oc[0] + r * np.cos(theta)
yc = oc[1] + r * np.sin(theta)
ax.fill(xc, yc, color='orange', alpha=0.3, label='sphere')
ax.plot(xc, yc, color='orange')
# 起点/终点
ax.scatter(p0[0], p0[1], color='green', s=80)
ax.text(p0[0] + 0.1, p0[1] + 0.1, "start")
ax.scatter(pf[0], pf[1], color='red', s=80)
ax.text(pf[0] + 0.1, pf[1] + 0.1, "goal")

# 历史轨迹列表（每次迭代的轨迹）
traj_history = []
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

    # 记录并打印每一次迭代的关键信息：迭代轮次、q、T1、T2
    # 注意：opt_callback 也可能在调用前手动调用一次（用于绘制初始轨迹）
    try:
        qv = map_to_ball(xk[:2])
        T1_print, T2_print = map_time(xk[2:])
    except Exception:
        qv = None
        T1_print = None
        T2_print = None
    # 保存 q 值用于绘制小点
    if qv is not None:
        q_history.append(qv)
    # 打印信息
    print(f"Iter {len(q_history)-1}: q={qv}, T1={T1_print}, T2={T2_print}")
    # 增加迭代计数
    globals()['iter_count'] = len(q_history)

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
        pt = ax.scatter(qv[0], qv[1], color='black', s=12, alpha=alpha, zorder=4)
        dynamic_points.append(pt)

    plt.draw()
    plt.pause(0.01)

# 先画初始轨迹（可选）
opt_callback(x0)
# 优化器每迭代一次都会调用 opt_callback
res = minimize(minco_cost, x0, method='L-BFGS-B', callback=opt_callback)

print("===============================Final==================================")

print("优化结果 x:", res.x)
xi_opt = res.x[:2]
tau_opt = res.x[2:]

q_opt = map_to_ball(xi_opt)
T1_opt, T2_opt = map_time(tau_opt)

print("优化后 q:", q_opt)
print("优化后 T1,T2:", T1_opt, T2_opt)


coeff1_opt, coeff2_opt = compute_two_segment_coeff_2d(
    p0, v0, a0,
    q_opt, pf, vf, af,
    T1_opt, T2_opt
)

traj1 = sample_segment(coeff1_opt, T1_opt)
traj2 = sample_segment(coeff2_opt, T2_opt)
traj = np.vstack((traj1, traj2))


# ================= 可视化 =======================
# 重用之前创建的 `fig, ax`（回调已在该轴上绘制了历史迭代轨迹）
# 在该图上绘制最终结果，使用更粗的线条突出显示

# 绘制最终优化得到的中间点 q_opt（覆盖在历史轨迹上）
ax.scatter(q_opt[0], q_opt[1], color='blue', s=100, zorder=5)
ax.text(q_opt[0]+0.1, q_opt[1]+0.1, "q_opt")

# 绘制最终轨迹（加粗，颜色与历史轨迹一致但更明显）
ax.plot(traj[:,0], traj[:,1], color='purple', linewidth=3, label='final_traj', zorder=6)

# 保证图像外观（若之前已设置过，这些也不会有害）
ax.set_aspect('equal')
ax.grid(True)
ax.set_xlim(-1, 5.5)
ax.set_ylim(-1, 4)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('2-segment MINCO-style trajectory (C^4 at waypoint)')
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()
