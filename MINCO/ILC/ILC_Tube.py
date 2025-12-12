import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os
from scipy.interpolate import interp1d

class ILCTubePlanner:
    """
    ILC管道跟踪规划器类，用于运行迭代学习控制以优化UAV在管道中的轨迹。
    """
    
    def __init__(self, path_filename='MINCO/ILC/ref_traj/optimized_traj.npz', 
                 map_filename='MINCO/gen_map_tube/tube_corridor.npz',
                 i_n=50, dt=0.02, v_max=5.0, tau=1.5, kp=3.0, kd=None, 
                 kp_vl=3, kp_law=2, kd_law=0.05):
        """
        初始化ILC规划器。
        
        参数：
        -----------
        path_filename : str
            路径文件路径
        map_filename : str
            地图文件路径
        i_n : int
            迭代次数
        dt : float
            时间步长
        v_max : float
            最大速度
        tau : float
            速度跟踪的时间常数
        kp : float
            垂直控制的比例增益
        kd : float
            垂直控制的微分增益（如果为None，则设为2.0/dt）
        kp_vl : float
            ILC学习增益
        kp_law, kd_law : float
            Sigmoid函数的增益
        """
        self.path_filename = path_filename
        self.map_filename = map_filename
        self.i_n = i_n
        self.dt = dt
        self.v_max = v_max
        self.tau = tau
        self.kp = kp
        self.kd = kd if kd is not None else 2.0 / dt
        self.kp_vl = kp_vl
        self.kp_law = kp_law
        self.kd_law = kd_law
        
        # 加载数据
        self.xd, self.yd, self.bound, self.raw_left_b, self.raw_right_b = self.load_tube_path(self.path_filename)
        self.grid, self.resolution, self.origin = self.load_map(self.map_filename)
        
        if self.xd is None:
            raise ValueError("Failed to load path data.")
        
        self.current_all = len(self.xd)
        print(f"Path loaded with {self.current_all} points. Total length approx {np.sum(np.sqrt(np.diff(self.xd)**2 + np.diff(self.yd)**2)):.2f}m")
        
        # 结果存储
        self.results = None
    
    def load_tube_path(self, filename):
        """
        从 npz 文件加载路径和边界。
        返回插值后的 xd, yd, bound，以及用于绘图的原始边界数据。
        """
        if not os.path.exists(filename):
            print(f"Error: File {filename} not found.")
            return None, None, None, None, None
            
        data = np.load(filename)
        
        if 'traj' in data:
            # optimized_traj.npz 格式
            path = data['traj'] # (N, 2)
            x = path[:, 0]
            y = path[:, 1]
            # 默认半径，因为 optimized_traj.npz 中没有
            r = np.full_like(x, 0.5)
        elif 'path' in data:
            # tube_corridor.npz 格式
            path = data['path'] # (N, 3) -> x, y, radius
            x = path[:, 0]
            y = path[:, 1]
            r = path[:, 2]
        else:
            print("Error: Unknown file format.")
            return None, None, None, None, None
        
        # 用于绘图的原始边界
        raw_left_b = data['left_boundary'] if 'left_boundary' in data else None
        raw_right_b = data['right_boundary'] if 'right_boundary' in data else None
        
        # 插值路径以获得密集的路径点
        # 计算累积距离作为插值参数
        dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        cum_dist = np.concatenate(([0], np.cumsum(dists)))
        total_dist = cum_dist[-1]
        
        # 密集路径的点数（例如，每 0.02m 一个点用于更平滑的控制）
        # ILC.py 为 50m 使用了 1000 个点 -> 0.05m 间距。
        # 我们使用类似的密度。
        n_points = int(total_dist / 0.05)
        if n_points < 100: n_points = 100 # 最少点数
        
        t_new = np.linspace(0, total_dist, n_points)
        
        f_x = interp1d(cum_dist, x, kind='linear')
        f_y = interp1d(cum_dist, y, kind='linear')
        f_r = interp1d(cum_dist, r, kind='linear')
        
        xd = f_x(t_new)
        yd = f_y(t_new)
        bound = f_r(t_new)
        
        # 从距离终点 L 距离开始线性收敛到 0
        L = 3.0  # 从终点开始收敛的距离
        bound_modified = bound.copy()
        idx_start = np.where(t_new >= total_dist - L)[0]
        if len(idx_start) > 0:
            idx_start = idx_start[0]
            for i in range(idx_start, len(bound)):
                ratio = (t_new[i] - (total_dist - L)) / L
                bound_modified[i] = bound[idx_start] * (1 - ratio)
            bound_modified[-1] = 0.0  # 确保终点为 0
        
        return xd, yd, bound_modified, raw_left_b, raw_right_b

    def load_map(self, filename):
        """
        从 npz 文件加载占用栅格。
        """
        if not os.path.exists(filename):
            return None, None, None
            
        data = np.load(filename)
        if 'grid' in data:
            return data['grid'], data['resolution'], data['origin']
        return None, None, None

    @staticmethod
    def getpoint(x_curr, y_curr, xd, yd, current_idx):
        """
        从 current_idx 开始搜索路径上最近的点。
        """
        # 搜索窗口以提高性能
        search_len = 100
        start = current_idx
        end = min(len(xd), current_idx + search_len)
        
        # 如果在终点附近，只检查最后几个点
        if start >= len(xd) - 1:
            start = len(xd) - 1
            end = len(xd)
            
        dists_sq = (xd[start:end] - x_curr)**2 + (yd[start:end] - y_curr)**2
        min_local = np.argmin(dists_sq)
        point_idx = start + min_local
        
        xp = xd[point_idx]
        yp = yd[point_idx]
        
        # 下一点用于方向
        next_idx = min(point_idx + 1, len(xd) - 1)
        xp_next = xd[next_idx]
        yp_next = yd[next_idx]
        
        return yp, xp, yp_next, xp_next, point_idx

    @staticmethod
    def saturate(v_perp, v_para, v_max):
        """
        饱和速度向量。
        """
        v_sum = v_perp + v_para
        v_norm = np.linalg.norm(v_sum)
        
        if v_norm > v_max:
            scale = v_max / v_norm
            return v_perp * scale, v_para * scale
        return v_perp, v_para

    @staticmethod
    def uav_dynamic(pos, vel, acc, dt):
        """
        简单的 UAV 动力学更新。
        """
        new_vel = vel + acc * dt
        new_pos = pos + vel * dt
        return new_pos, new_vel, acc

    def run_ilc_iterations(self):
        """
        运行管道跟踪的 ILC 迭代。
        
        返回：
        --------
        x, y : ndarray
            所有迭代的轨迹位置，形状 (i_n, j_n+1)
        sum_t : ndarray
            每次迭代的总时间，形状 (i_n,)
        uc_parax, uc_paray : ndarray
            平行控制分量，形状 (i_n, current_all+1)
        uc_perpx, uc_perpy : ndarray
            垂直控制分量，形状 (i_n, current_all+1)
        uc_value : ndarray
            控制值，形状 (i_n, current_all+1)
        """
        xd, yd, bound = self.xd, self.yd, self.bound
        i_n, dt, v_max, tau, kp, kd, kp_vl, kp_law, kd_law = self.i_n, self.dt, self.v_max, self.tau, self.kp, self.kd, self.kp_vl, self.kp_law, self.kd_law
        current_all = self.current_all
        
        j_n = current_all * 2  # 最大时间步数（留有余量）
        
        # 存储数组初始化
        y = np.zeros((i_n, j_n + 1))  # 存储所有迭代的y坐标
        x = np.zeros((i_n, j_n + 1))  # 存储所有迭代的x坐标
        
        # 激励阈值（用于Sigmoid函数）
        exc = np.zeros((i_n, current_all))  # 激励阈值数组
        for i in range(i_n):  # 为每次迭代设置阈值
            exc[i, :] = bound / 2.0  # 阈值设为边界的三分之一
            
        # 控制输入数组初始化
        uc_parax = np.zeros((i_n, current_all + 1))  # x方向平行控制
        uc_paray = np.zeros((i_n, current_all + 1))  # y方向平行控制
        uc_perpx = np.zeros((i_n, current_all + 1))  # x方向垂直控制
        uc_perpy = np.zeros((i_n, current_all + 1))  # y方向垂直控制
        
        uc_value = np.zeros((i_n, current_all + 1))  # 控制值初始化为1
        
        sum_t = np.zeros(i_n)  # 存储每次迭代的总时间
        
        u_max = v_max  # 最大控制输入（未使用）
        
        print("Starting ILC Iterations...")  # 开始ILC迭代
        start_time = time.time()  # 记录开始时间
        
        # 迭代学习过程
        for i in range(i_n - 1):  # 迭代 i_n-1 次（最后一次用于评估）
            current = 0  # 当前路径点索引
            j = 0  # 当前时间步索引
            last_e = 0.0  # 上一步的误差
            value_e = 0.0  # 当前误差值
            last_l = 0  # 上一步的路径索引
            
            position = np.array([xd[0], yd[0]])  # 初始位置
            velocity = np.array([0.0, 0.0])  # 初始速度
            
            # 推断初始方向
            dir_init = np.array([xd[1]-xd[0], yd[1]-yd[0]])  # 从前两个路径点计算初始方向
            dir_init = dir_init / np.linalg.norm(dir_init)  # 单位化方向向量
            velocity = dir_init * 1.0  # 以小速度开始
            
            accelerate = np.array([0.0, 0.0])  # 初始加速度
            v_des = velocity.copy()  # 期望速度初始化
            
            while current < current_all and j < j_n:  # current_all 为路径点总数
                x[i, j] = position[0]  # 记录当前位置x
                y[i, j] = position[1]  # 记录当前位置y
                
                # 找到路径上最近的点
                yp, xp, yp_next, xp_next, point = self.getpoint(x[i, j], y[i, j], xd, yd, current)
                
                # 终止条件：接近终点
                dist_end_sq = (xp - xd[-1])**2 + (yp - yd[-1])**2  # 计算到终点的距离平方
                if dist_end_sq < 0.5 and current > current_all - 50 and  np.linalg.norm(velocity) < 1:  # 距离足够近且路径快结束
                    break  # 结束当前轨迹
                    
                current = point  # 更新当前路径点索引
                l = current  # 当前路径点索引
                
                # 误差向量(从当前位置指向参考路径点)
                e_vec = np.array([xp - x[i, j], yp - y[i, j]])  # 计算误差向量
                e_norm = np.linalg.norm(e_vec)  # 计算误差向量范数
                e_value = e_norm - bound[l]  # 计算有界误差（考虑管廊边界）
                
                # 垂直控制分量
                temp_perp = kp * e_vec + kd * (e_norm - last_e) * e_vec / (e_norm + 1e-5)  # PD控制：比例 + 微分项
                
                # 存储垂直控制
                if l >= last_l:  # 确保只在路径索引增加时更新
                    idx_start = last_l  # 更新起始索引
                    idx_end = min(l + 1, current_all)  # 更新结束索引，不能超过路径总长度
                    if idx_end > idx_start:  # 确保有有效的索引范围
                        uc_perpx[i, idx_start : idx_end] = temp_perp[0]  # 存储x方向垂直控制
                        uc_perpy[i, idx_start : idx_end] = temp_perp[1]  # 存储y方向垂直控制
                
                last_e = value_e
                
                # 方向向量
                v_dir = np.array([xp_next - xp, yp_next - yp])  # 计算路径方向向量
                den_para = np.linalg.norm(v_dir) + 1e-10  # 计算方向向量长度（避免除零）
                v_dir = v_dir / den_para  # 单位化方向向量
                
                # 为第一次迭代初始化平行控制
                if i == 0:  # 只有第一次迭代需要初始化
                    if l >= last_l:  # 确保只在路径索引增加时更新
                        idx_start = last_l  # 更新起始索引
                        idx_end = min(l + 1, current_all)  # 更新结束索引
                        if idx_end > idx_start:  # 确保有有效的索引范围
                            uc_parax[i, idx_start : idx_end] = v_dir[0] # uc_parax 存储平行控制分量
                            uc_paray[i, idx_start : idx_end] = v_dir[1] # uc_paray 存储平行控制分量
                
                value = np.dot(velocity, v_dir)  # 计算当前速度在路径方向上的投影
                
                if i == 0:  # 第一次迭代使用较低速度
                    value = 2.0  # 初始速度较慢
                    
                vl = np.array([uc_parax[i, l], uc_paray[i, l]]) * value  # 计算切向控制向量
                
                # 饱和速度
                vcc, vll = self.saturate(temp_perp * value, vl, v_max)  # 速度饱和处理
                v_des = vcc + vll  # 期望速度向量
                
                # 动力学更新
                a = tau * (v_des - velocity)  # 计算加速度（一阶动力学）
                position, velocity, accelerate = self.uav_dynamic(position, velocity, a, dt)  # 更新位置和速度
                
                # ILC 更新
                value_e = np.linalg.norm(e_vec)  # 计算当前误差范数
                de = value_e - last_e  # 计算误差变化率
                
                term = -kp_law * value_e - kd_law * de + exc[i, l]  # 计算Sigmoid函数的自变量
                sigmoid = 1.0 / (1.0 + np.exp(term))  # 计算Sigmoid函数值
                temp = kp_vl * (sigmoid - 0.5)  # 计算学习增益调整量
                
                if l >= last_l:  # 确保只在路径索引增加时更新
                    idx_start = last_l  # 更新起始索引
                    idx_end = min(l + 1, current_all)  # 更新结束索引
                    if idx_end > idx_start:  # 确保有有效的索引范围
                        uc_value[i+1, idx_start : idx_end] = uc_value[i, idx_start : idx_end] - temp  # ILC学习更新
                        # 裁剪
                        uc_value[i+1, idx_start : idx_end] = np.maximum(uc_value[i+1, idx_start : idx_end], 0)  # 确保非负
                        
                        # 更新下一次迭代的切向控制
                        uc_parax[i+1, idx_start : idx_end] = uc_value[i+1, l] * v_dir[0]  # 更新x方向切向控制
                        uc_paray[i+1, idx_start : idx_end] = uc_value[i+1, l] * v_dir[1]  # 更新y方向切向控制
                
                j += 1  # 时间步计数器加1
                last_l = l  # 更新上一步的路径索引
                
            # 计算实际轨迹时间
            valid_x = x[i, :]  # 获取当前迭代的所有x坐标
            non_zeros = np.nonzero(valid_x)[0]  # 找到非零x坐标的索引（有效轨迹点）
            if len(non_zeros) > 0:  # 如果有有效轨迹点
                idx_end = non_zeros[-1]  # 获取最后一个有效点的索引
                t_total = idx_end * dt  # 总时间 = 最后一个点的索引 × 时间步长
            else:  # 如果没有有效轨迹点
                t_total = 0.0  # 时间为0
            
            sum_t[i] = t_total  # 存储当前迭代的总时间
            print(f"Iteration {i+1}/{i_n-1}: Time {t_total:.2f}s, Points {idx_end if len(non_zeros) > 0 else 0}")  # 打印迭代信息
            
        print(f"Total elapsed time: {time.time() - start_time:.2f}s")  # 打印总运行时间
        
        self.results = (x, y, sum_t, uc_parax, uc_paray, uc_perpx, uc_perpy, uc_value)
        return self.results

    def run(self):
        """
        运行完整的ILC规划过程。
        """
        print(f"Loading path from {self.path_filename}...")
        self.run_ilc_iterations()
        plot_ilc_results(self.results, self.xd, self.yd, self.bound, self.grid, self.resolution, self.origin, self.raw_left_b, self.raw_right_b, self.i_n, self.dt, self.tau)

def plot_tube_bound(xd, yd, bound, raw_left_b=None, raw_right_b=None, save_path=None):
    """
    绘制参考路径 xd, yd，以及使用 bound 法向偏移计算的走廊边界，
    以及可选的原始左右边界（如果提供）。
    """
    xd = np.asarray(xd)
    yd = np.asarray(yd)
    bound = np.asarray(bound)

    # 计算切线和法线
    dx = np.gradient(xd)
    dy = np.gradient(yd)
    norms = np.sqrt(dx**2 + dy**2) + 1e-10
    nx = -dy / norms
    ny = dx / norms

    left_x = xd + nx * bound
    left_y = yd + ny * bound
    right_x = xd - nx * bound
    right_y = yd - ny * bound

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.grid(True)

    # 绘制参考（中心）路径
    ax.plot(xd, yd, '--', color='g', linewidth=1.0, label='Reference Path')

    # 起点和终点标记
    try:
        ax.scatter(xd[0], yd[0], color='green', s=60, zorder=5)
        ax.text(xd[0] + 0.02, yd[0] + 0.02, 'start', color='green')
        ax.scatter(xd[-1], yd[-1], color='magenta', s=60, zorder=5)
        ax.text(xd[-1] + 0.02, yd[-1] + 0.02, 'end', color='magenta')
    except Exception:
        pass

    # 绘制计算出的左右边界
    ax.plot(left_x, left_y, color='orange', linewidth=1.2, label='Computed Left Bound')
    ax.plot(right_x, right_y, color='orange', linewidth=1.2, label='Computed Right Bound')

    # 填充走廊
    try:
        ax.fill(np.concatenate([left_x, right_x[::-1]]), np.concatenate([left_y, right_y[::-1]]), color='orange', alpha=0.2)
    except Exception:
        pass

    # 如果提供，绘制原始边界
    if raw_left_b is not None:
        try:
            raw_left_b = np.asarray(raw_left_b)
            ax.plot(raw_left_b[:, 0], raw_left_b[:, 1], 'b.-', linewidth=1.0, label='Raw Left Boundary')
        except Exception:
            pass
    if raw_right_b is not None:
        try:
            raw_right_b = np.asarray(raw_right_b)
            ax.plot(raw_right_b[:, 0], raw_right_b[:, 1], 'b.-', linewidth=1.0, label='Raw Right Boundary')
        except Exception:
            pass

    ax.legend()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return

def plot_ilc_results(results, xd, yd, bound, grid, resolution, origin, raw_left_b, raw_right_b, i_n, dt, tau, save_dir='MINCO/ILC/ref_traj'):
    """
    绘制ILC结果图表。
    """
    if results is None:
        print("No results to plot. Run run_ilc_iterations first.")
        return
    
    x, y, sum_t, uc_parax, uc_paray, uc_perpx, uc_perpy, uc_value = results
    
    # 图1: 时间
    plt.figure(1, figsize=(12, 9))
    iterations = np.arange(len(sum_t))
    plt.plot(iterations, sum_t, linewidth=2, marker='o', markersize=4, label='Trajectory Time')
    plt.title('Trajectory Time vs Iteration', fontname='serif', fontsize=20)
    plt.xlabel("Iteration", fontname='serif', fontsize=20)
    plt.ylabel("Time (s)", fontname='serif', fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 图2: 所有迭代（带地图障碍物）
    plt.figure(2, figsize=(12, 9))
    plt.clf()
    plt.axis('equal')
    plt.grid(True)
    
    # 绘制地图障碍物
    if grid is not None:
        nx, ny = grid.shape
        if origin.size > 1:
            ox, oy = origin[0], origin[1]
        else:
            ox, oy = float(origin), 0.0
        xmin, xmax = ox, ox + nx * resolution
        ymin, ymax = oy, oy + ny * resolution
        plt.imshow(grid.T, origin="lower", extent=(xmin, xmax, ymin, ymax), cmap="gray_r", alpha=0.5)
    
    # 计算边界线
    dx = np.gradient(xd)
    dy = np.gradient(yd)
    norms = np.sqrt(dx**2 + dy**2) + 1e-10
    nx = -dy / norms
    ny = dx / norms
    
    left_x = xd + nx * bound
    left_y = yd + ny * bound
    right_x = xd - nx * bound
    right_y = yd - ny * bound
    
    plt.plot(xd, yd, '--', color='g', linewidth=0.6, label='Reference Path')
    plt.plot(left_x, left_y, color='orange', linewidth=1.2, label='Computed Left Bound')
    plt.plot(right_x, right_y, color='orange', linewidth=1.2, label='Computed Right Bound')
    
    # 填充走廊
    try:
        plt.fill(np.concatenate([left_x, right_x[::-1]]), np.concatenate([left_y, right_y[::-1]]), color='orange', alpha=0.2)
    except Exception:
        pass
    
    if raw_left_b is not None:
        plt.plot(raw_left_b[:, 0], raw_left_b[:, 1], 'b', linewidth=1.0, label='Raw Left Boundary')
    if raw_right_b is not None:
        plt.plot(raw_right_b[:, 0], raw_right_b[:, 1], 'b', linewidth=1.0, label='Raw Right Boundary')
    
    nIter = max(1, i_n - 1)
    cmap = plt.get_cmap('jet')
    colors = [cmap(k) for k in np.linspace(0, 1, nIter)]
    
    for ii in range(nIter):
        valid_x = x[ii, :]
        non_zeros = np.nonzero(valid_x)[0]
        if len(non_zeros) < 2:
            continue
        idx = non_zeros[-1]
        plt.plot(x[ii, :idx+1], y[ii, :idx+1], '-', color=colors[ii], linewidth=1.2)
        
    # 起点和终点
    plt.plot(x[0, 0], y[0, 0], 'go', markerfacecolor='g')
    plt.text(x[0, 0], y[0, 0], '  start', fontsize=10)
    
    last_iter_idx = i_n - 2
    valid_x_last = x[last_iter_idx, :]
    non_zeros_last = np.nonzero(valid_x_last)[0]
    if len(non_zeros_last) > 0:
        idx_last = non_zeros_last[-1]
        plt.plot(x[last_iter_idx, :idx_last+1], y[last_iter_idx, :idx_last+1], 'r', linewidth=2)
        plt.plot(xd[-1], yd[-1], 'mo', markerfacecolor='m')
        plt.text(xd[-1], yd[-1], '  end', fontsize=10)
        
    plt.title(f"tau = {tau}, iterations = {nIter}")
    plt.xlabel("X (m)", fontname='serif', fontsize=20)
    plt.ylabel("Y (m)", fontname='serif', fontsize=20)
    plt.legend(loc='best')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=nIter))
    sm.set_array([])
    plt.colorbar(sm, ticks=np.linspace(1, nIter, int(min(5, nIter))), label='Iteration', ax=plt.gca())
    
    # 图3: 随时间变化的速度
    plt.figure(3, figsize=(12, 9))
    plt.clf()
    
    last_iter_idx = i_n - 2
    valid_x = x[last_iter_idx, :]
    non_zeros = np.nonzero(valid_x)[0]
    if len(non_zeros) > 0:
        idx_last = non_zeros[-1]
        positions = np.column_stack((x[last_iter_idx, :idx_last+1], y[last_iter_idx, :idx_last+1]))
        velocities = np.diff(positions, axis=0) / dt
        speeds = np.linalg.norm(velocities, axis=1)
        times = np.arange(len(speeds)) * dt
        plt.plot(times, speeds, 'b-', linewidth=2, label='Speed')
        
    plt.title('Velocity over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.grid(True)
    plt.legend()
    
    print("Saving figures...")
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(1)
    plt.savefig(f'{save_dir}/ILC_Tube_Figure_1.png', dpi=300, bbox_inches='tight')
    plt.figure(2)
    plt.savefig(f'{save_dir}/ILC_Tube_Figure_2.png', dpi=300, bbox_inches='tight')
    plt.figure(3)
    plt.savefig(f'{save_dir}/ILC_Tube_Figure_3.png', dpi=300, bbox_inches='tight')
    print(f"Figures saved as {save_dir}/ILC_Tube_Figure_*.png")
    
    # 显示figure_2
    plt.figure(2)
    plt.show()

def main():
    # 创建规划器实例
    planner = ILCTubePlanner()
    # 运行规划
    planner.run()

if __name__ == "__main__":
    main()
