import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
import os

# 添加 tubeRRTstar 路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'tubeRRTstar'))

from tube_rrt_star_2D import SimpleOccupancyMap2D

# ==========================================
#  Utility Functions 
# ==========================================

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


# ==========================================
#  Visualization Class: TubeMincoVisualizer
# ==========================================

class TubeMincoVisualizer:
    def __init__(self, map2d, oc_list, r_list, p0, pf, n):
        self.map2d = map2d
        self.oc_list = oc_list
        self.r_list = r_list
        self.p0 = p0
        self.pf = pf
        self.n = n
        
        self.fig = None
        self.ax = None
        self.dynamic_lines = []
        self.dynamic_points = []

    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        # 绘制地图障碍物
        grid = self.map2d.grid
        nx, ny = grid.shape
        res = self.map2d.resolution
        ox, oy = self.map2d.origin
        xmin, xmax = ox, ox + nx * res
        ymin, ymax = oy, oy + ny * res
        self.ax.imshow(grid.T, origin="lower", extent=(xmin, xmax, ymin, ymax), cmap="gray_r", alpha=0.5)

        # 画球（背景）
        theta = np.linspace(0, 2 * np.pi, 200)
        for i_oc, rr in zip(self.oc_list, self.r_list):
            xc = i_oc[0] + rr * np.cos(theta)
            yc = i_oc[1] + rr * np.sin(theta)
            self.ax.fill(xc, yc, color='orange', alpha=0.3)
            self.ax.plot(xc, yc, color='orange')

        # 起点/终点
        self.ax.scatter(self.p0[0], self.p0[1], color='green', s=80)
        self.ax.text(self.p0[0] + 0.1, self.p0[1] + 0.1, "start")
        self.ax.scatter(self.pf[0], self.pf[1], color='red', s=80)
        self.ax.text(self.pf[0] + 0.1, self.pf[1] + 0.1, "goal")
        
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_xlim(0, 20)
        self.ax.set_ylim(0, 20)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title(f'{self.n}-segment MINCO-style trajectory')

    def update_plot(self, traj_history, q_history, iter_count, total_time):
        # 清除动态元素
        for ln in self.dynamic_lines:
            try: ln.remove()
            except: pass
        self.dynamic_lines.clear()
        for pt in self.dynamic_points:
            try: pt.remove()
            except: pass
        self.dynamic_points.clear()

        print(f"Iter {iter_count}: total_time={total_time:.3f}")

        # 绘制历史轨迹
        total = len(traj_history)
        alpha_max = 0.9
        alpha_min = 0.05
        for i, tr in enumerate(traj_history):
            if total == 1:
                alpha = alpha_max
            else:
                alpha = alpha_max - (alpha_max - alpha_min) * (i / (total - 1))
            ln, = self.ax.plot(tr[:, 0], tr[:, 1], color='purple', alpha=alpha, linewidth=1.5)
            self.dynamic_lines.append(ln)

        # 绘制中间点
        for i, qv in enumerate(q_history):
            if total == 1:
                alpha = alpha_max
            else:
                alpha = alpha_max - (alpha_max - alpha_min) * (i / (total - 1))
            pt = self.ax.scatter(qv[:, 0], qv[:, 1], color='black', s=12, alpha=alpha, zorder=4)
            self.dynamic_points.append(pt)

        plt.draw()
        plt.pause(0.0001)

    def visualize_final(self, q_opt, traj):
        # 绘制最终优化得到的中间点
        self.ax.scatter(q_opt[:, 0], q_opt[:, 1], color='blue', s=80, zorder=5)
        for idx, qp in enumerate(q_opt):
            self.ax.text(qp[0] + 0.05, qp[1] + 0.05, f"q{idx}")

        # 绘制最终轨迹
        self.ax.plot(traj[:, 0], traj[:, 1], color='purple', linewidth=3, label='final_traj', zorder=6)
        self.ax.legend(loc='upper left')
        plt.tight_layout()

    def save_main_fig(self, path):
        try:
            self.fig.savefig(path, dpi=150)
        except Exception as e:
            print('Failed to save final_traj.png:', e)

    def plot_history(self, energy_history, time_history, out_dir):
        if len(energy_history) > 0 or len(time_history) > 0:
            try:
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
                
                fig2.savefig(os.path.join(out_dir, 'iter_history.png'), dpi=150)
            except Exception as e:
                print("Failed to plot/save iteration history:", e)

    def plot_velocity_profile(self, coeffs, T_list, out_dir):
        try:
            times = []
            velocities = []
            current_time = 0.0
            
            for i in range(len(T_list)):
                T = T_list[i]
                c = coeffs[i] # shape (2, 6)
                # Sample points
                ts = np.linspace(0, T, 100)
                for t in ts:
                    # vx = c1 + 2c2*t + 3c3*t^2 + 4c4*t^3 + 5c5*t^4
                    vx = (c[0,1] + 2*c[0,2]*t + 3*c[0,3]*t**2 + 
                          4*c[0,4]*t**3 + 5*c[0,5]*t**4)
                    vy = (c[1,1] + 2*c[1,2]*t + 3*c[1,3]*t**2 + 
                          4*c[1,4]*t**3 + 5*c[1,5]*t**4)
                    v = np.sqrt(vx**2 + vy**2)
                    
                    times.append(current_time + t)
                    velocities.append(v)
                current_time += T
                
            fig_v, ax_v = plt.subplots(figsize=(8, 4))
            ax_v.plot(times, velocities, color='tab:blue', linewidth=2)
            ax_v.set_xlabel('Time (s)')
            ax_v.set_ylabel('Velocity (m/s)')
            ax_v.set_title('Velocity Profile')
            ax_v.grid(True)
            plt.tight_layout()
            
            fig_v.savefig(os.path.join(out_dir, 'velocity_profile.png'), dpi=150)
            print(f"Velocity profile saved to {os.path.join(out_dir, 'velocity_profile.png')}")
            
        except Exception as e:
            print("Failed to plot velocity profile:", e)


# ==========================================
#  Main Class: TubeMincoPlanner
# ==========================================

class TubeMincoPlanner:
    def __init__(self, npz_path):
        self.npz_path = npz_path
        self.load_corridor()
        self.init_conditions()
        
        # Initialize Visualizer
        self.visualizer = TubeMincoVisualizer(
            self.map2d, self.oc_list, self.r_list, self.p0, self.pf, self.n
        )
        
        # Optimization history
        self.traj_history = []
        self.energy_history = []
        self.time_history = []
        self.q_history = []
        self.iter_count = 0

    def load_corridor(self):
        if not os.path.isfile(self.npz_path):
            raise RuntimeError(f"No corridor .npz found at {self.npz_path}")

        print(f"Loading corridor (strict) from {self.npz_path}")
        data = np.load(self.npz_path)
        
        # 地图栅格信息
        self.grid = np.asarray(data['grid'])
        self.resolution = float(data['resolution'])
        origin = np.asarray(data['origin'], dtype=float)
        if origin.ndim == 0 or origin.size == 1:
            origin = np.array([float(origin), 0.0])
        self.origin = origin
        self.map2d = SimpleOccupancyMap2D(self.grid.astype(bool), self.resolution, self.origin)
        print(f"Loaded occupancy grid (shape={self.grid.shape}, res={self.resolution})")

        # 走廊数据
        self.path = np.asarray(data['path'])
        self.oc_list = np.asarray(data['oc_list'])
        self.r_list = np.asarray(data['r_list'])

        # endpoints 
        self.p0 = self.path[0, :2]
        self.pf = self.path[-1, :2]

        self.n = len(self.oc_list) + 1
        print(f"Loaded corridor: {len(self.oc_list)} intermediate balls, n={self.n}")

    def init_conditions(self):
        # Initial velocities/accelerations
        self.v0 = np.array([0.0, 0.0])
        self.a0 = np.array([0.0, 0.0])
        self.vf = np.array([0.0, 0.0])
        self.af = np.array([0.0, 0.0])

    def map_to_ball(self, xi):
        """
        xi: 扁平化向量，长度应为 2*(n-1)
        """
        xi2 = np.asarray(xi).reshape(-1, 2)
        m = xi2.shape[0]
        qs = np.zeros((m, 2))
        for i in range(m):
            v = xi2[i]
            norm2 = np.dot(v, v)
            qs[i] = self.oc_list[i] + 2.0 * self.r_list[i] * v / (norm2 + 1.0)
        return qs

    def map_time(self, tau):
        return np.exp(tau)

    def minco_cost(self, x):
        # x = [xi (2*(n-1)), tau (n)]
        xi = x[:2 * (self.n - 1)]
        tau = x[2 * (self.n - 1):]

        q_list = self.map_to_ball(xi)  # shape (n-1,2)
        T_list = self.map_time(tau)    # shape (n,)

        # 求 n 段系数（C^4 连续）
        coeffs = compute_n_segment_coeff_2d(
            self.p0, self.v0, self.a0,
            q_list, self.pf, self.vf, self.af,
            T_list
        )

        J = 0.0
        for i in range(len(T_list)):
            J += jerk_energy(coeffs[i], T_list[i])

        # 时间惩罚
        time_penalty = 1.0 * np.sum(T_list)
        return J + time_penalty

    def traj_from_x(self, x):
        # 1. 提取优化变量：中间点参数 xi 和时间参数 tau
        xi = x[:2 * (self.n - 1)]
        tau = x[2 * (self.n - 1):]
        
        # 2. 映射回物理量：中间点位置 q_list 和每段时长 T_list
        q_list = self.map_to_ball(xi)
        T_list = self.map_time(tau)
        
        # 3. 计算每段多项式系数
        coeffs = compute_n_segment_coeff_2d(
            self.p0, self.v0, self.a0, 
            q_list, self.pf, self.vf, self.af, 
            T_list
        )
        
        # 4. 对每段轨迹进行采样并拼接
        segs = []
        for i in range(len(T_list)):
            segs.append(sample_segment(coeffs[i], T_list[i]))
        return np.vstack(segs)

    def opt_callback(self, xk):
        # 1. 从当前优化变量 xk 计算轨迹点
        try:
            trajk = self.traj_from_x(xk)
        except Exception:
            return

        ## 下面都是记录和可视化部分 ##

        # 2. 记录轨迹历史
        self.traj_history.append(trajk)

        # 3. 解析中间点 qv 和时间 T
        try:
            qv = self.map_to_ball(xk[:2 * (self.n - 1)])
            T_list_print = self.map_time(xk[2 * (self.n - 1):])
        except Exception:
            qv = None
            T_list_print = None

        if qv is not None:
            # 4. 记录中间点历史
            self.q_history.append(qv)
            try:
                # 5. 计算当前迭代的能量和总时间
                coeffs_iter = compute_n_segment_coeff_2d(
                    self.p0, self.v0, self.a0, 
                    qv, self.pf, self.vf, self.af, 
                    T_list_print
                )
                J_iter = 0.0
                for ii in range(len(T_list_print)):
                    J_iter += jerk_energy(coeffs_iter[ii], T_list_print[ii])
                self.energy_history.append(J_iter)
                self.time_history.append(np.sum(T_list_print))
            except Exception:
                self.energy_history.append(np.nan)
                self.time_history.append(np.sum(T_list_print) if T_list_print is not None else np.nan)

        self.iter_count = len(self.q_history)
        sum_T = np.sum(T_list_print) if T_list_print is not None else 0.0
        
        # 6. 更新可视化
        self.visualizer.update_plot(self.traj_history, self.q_history, self.iter_count-1, sum_T)

    def run(self):
        print(f"Optimization with n={self.n} segments.")
        
        # Initial guess
        x0 = np.zeros(2 * (self.n - 1) + self.n) # 2D xi + tau
        
        # Setup plot
        self.visualizer.setup_plot()
        
        # Initial visualization
        self.opt_callback(x0)
        
        # Optimize
        res = minimize(self.minco_cost, x0, method='L-BFGS-B', callback=self.opt_callback,options={'maxiter': 10} )
        
        print("===============================Final==================================")
        print("优化结果 x:", res.x)
        
        # Process results
        xi_opt = res.x[:2 * (self.n - 1)]
        tau_opt = res.x[2 * (self.n - 1):]
        q_opt = self.map_to_ball(xi_opt)
        T_opt = self.map_time(tau_opt)
        
        print("优化后 q:", q_opt)
        print("优化后 T:", T_opt)
        
        coeffs_opt = compute_n_segment_coeff_2d(
            self.p0, self.v0, self.a0, 
            q_opt, self.pf, self.vf, self.af, 
            T_opt
        )
        segs = []
        for i in range(len(T_opt)):
            segs.append(sample_segment(coeffs_opt[i], T_opt[i]))
        traj = np.vstack(segs)
        
        self.visualizer.visualize_final(q_opt, traj)
        self.save_results(traj, q_opt, T_opt)
        
        out_dir = os.path.join('MINCO', 'minco_opt_data')
        self.visualizer.plot_history(self.energy_history, self.time_history, out_dir)
        self.visualizer.plot_velocity_profile(coeffs_opt, T_opt, out_dir)
        


        
        plt.show()

    def save_results(self, traj, q_opt, T_opt):
        out_dir = os.path.join('MINCO', 'minco_opt_data')
        os.makedirs(out_dir, exist_ok=True)
        
        self.visualizer.save_main_fig(os.path.join(out_dir, 'final_traj.png'))

        try:
            np.savez_compressed(
                os.path.join(out_dir, 'minco_result.npz'),
                grid=self.grid,
                resolution=self.resolution,
                origin=self.origin,
                oc_list=self.oc_list,
                r_list=self.r_list,
                path=self.path,
                traj=traj,
                q_opt=q_opt,
                T_opt=T_opt,
            )
            print(f"Results saved to {os.path.join(out_dir, 'minco_result.npz')}")
        except Exception as e:
            print('Failed to save minco_result.npz:', e)

if __name__ == "__main__":
    # 查找 npz 文件
    npz_path_candidates = [
        os.path.join(os.path.dirname(__file__), '..','MINCO', 'gen_map_tube','tube_corridor.npz'),
        os.path.join(os.path.dirname(__file__), 'gen_map_tube','tube_corridor.npz'), # Fallback
    ]
    found_npz = None
    for p in npz_path_candidates:
        if os.path.isfile(p):
            found_npz = p
            break
            
    if found_npz:
        planner = TubeMincoPlanner(found_npz)
        planner.run()
    else:
        print("Error: tube_corridor.npz not found.")