import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys
import os
import time

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
    vels = []
    for t in ts:
      px = (coeff[0,0] + coeff[0,1]*t + coeff[0,2]*t**2 +
          coeff[0,3]*t**3 + coeff[0,4]*t**4 + coeff[0,5]*t**5)
      py = (coeff[1,0] + coeff[1,1]*t + coeff[1,2]*t**2 +
          coeff[1,3]*t**3 + coeff[1,4]*t**4 + coeff[1,5]*t**5)
      pts.append([px, py])
      
      vx = (coeff[0,1] + 2*coeff[0,2]*t + 3*coeff[0,3]*t**2 +
          4*coeff[0,4]*t**3 + 5*coeff[0,5]*t**4)
      vy = (coeff[1,1] + 2*coeff[1,2]*t + 3*coeff[1,3]*t**2 +
          4*coeff[1,4]*t**3 + 5*coeff[1,5]*t**4)
      vels.append([vx, vy])
      
    return np.array(pts), np.array(vels)


# ==========================================
#  ILC Planning Class
# ==========================================

class ILCTubePlanner:
    """
    ILC管道跟踪规划器类，用于运行迭代学习控制以优化UAV在管道中的轨迹。
    """
    
    def __init__(self, ref_path=None,
                 map_filename='MINCO/gen_map_tube/tube_corridor.npz',
                 i_n=100, dt=0.02, v_max=5.0, tau=0.4, kp=1.0, kd=None, 
                 kp_vl=5, kp_law=1.0, kd_law=0.5):
        """
        初始化ILC规划器。
        
        参数：
        -----------
        ref_path : np.ndarray
            参考路径
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
        self.ref_path = ref_path
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
        self.grid, self.resolution, self.origin = self.load_map(self.map_filename)
        # 加载路径
        if self.ref_path is not None:
            self.xd, self.yd, self.bound = self.load_tube_path(self.ref_path)
            self.current_all = len(self.xd)
        else:
            self.xd, self.yd, self.bound = None, None, None
            self.current_all = 0

        # 结果存储
        self.results = None

    def load_tube_path(self, ref_path, ref_vel=None):
        """
        传入一条路径，返回插值后的路径点。
        """
        x = ref_path[:, 0]
        y = ref_path[:, 1]
        r = np.full_like(x, 0.5)

        
        # 插值路径以获得密集的路径点
        # 计算累积距离作为插值参数
        dists = np.sqrt(np.diff(ref_path[:,0])**2 + np.diff(ref_path[:,1])**2)
        cum_dist = np.concatenate(([0], np.cumsum(dists)))
        total_dist = cum_dist[-1]
        
        # 密集路径的点数（例如，每 0.02m 一个点用于更平滑的控制）
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
        L = 1.5  # 从终点开始收敛的距离
        bound_modified = bound.copy()
        idx_start = np.where(t_new >= total_dist - L)[0]
        if len(idx_start) > 0:
            idx_start = idx_start[0]
            for i in range(idx_start, len(bound)):
                ratio = (t_new[i] - (total_dist - L)) / L
                bound_modified[i] = bound[idx_start] * (1 - ratio)
            bound_modified[-1] = 0.0  # 确保终点为 0
        
        if ref_vel is not None:
            vx = ref_vel[:, 0]
            vy = ref_vel[:, 1]
            f_vx = interp1d(cum_dist, vx, kind='linear')
            f_vy = interp1d(cum_dist, vy, kind='linear')
            vxd = f_vx(t_new)
            vyd = f_vy(t_new)
            return xd, yd, bound_modified, vxd, vyd

        return xd, yd, bound_modified

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

    def run_ilc_iterations(self, ref_path=None, ref_vel=None):
        """
        运行管道跟踪的 ILC 迭代。
        """
        if ref_path is not None:
            self.ref_path = ref_path
        if self.ref_path is None:
            raise RuntimeError("No reference path provided for ILC.")

        if ref_vel is not None:
            xd, yd, bound, vxd, vyd = self.load_tube_path(self.ref_path, ref_vel)

            # # 画一张二维图展示vd的变化
            #vd=np.linalg.norm(np.array([vxd,vyd]),axis=0)
            # plt.figure(figsize=(10, 5))
            # plt.plot(vd, label='Velocity')
            # plt.title('Velocity Profile')
            # plt.xlabel('Time Step')
            # plt.ylabel('Velocity')
            # plt.legend()
            # plt.grid(True)
            # plt.show()

            self.xd, self.yd, self.bound = xd, yd, bound
            self.vxd, self.vyd = vxd, vyd
        else:
            xd, yd, bound = self.load_tube_path(self.ref_path)
            self.xd, self.yd, self.bound = xd, yd, bound
            self.vxd, self.vyd = None, None
            
        self.current_all = len(xd)

        i_n, dt, v_max, tau, kp, kd, kp_vl, kp_law, kd_law = self.i_n, self.dt, self.v_max, self.tau, self.kp, self.kd, self.kp_vl, self.kp_law, self.kd_law
        current_all = self.current_all
        
        j_n = current_all * 2  # 最大时间步数（留有余量）
        
        # 存储数组初始化
        y = np.zeros((i_n, j_n + 1))  # 存储所有迭代的y坐标
        x = np.zeros((i_n, j_n + 1))  # 存储所有迭代的x坐标
        
        # 激励阈值（用于Sigmoid函数）
        exc = np.zeros((i_n, current_all))  # 激励阈值数组
        for i in range(i_n):  # 为每次迭代设置阈值
            exc[i, :] = bound / 1.0  # 阈值设为边界的三分之一
            

        
        # 控制输入数组初始化
        uc_parax = np.zeros((i_n, current_all + 1))  # x方向平行控制
        uc_paray = np.zeros((i_n, current_all + 1))  # y方向平行控制
        uc_perpx = np.zeros((i_n, current_all + 1))  # x方向垂直控制
        uc_perpy = np.zeros((i_n, current_all + 1))  # y方向垂直控制
        
        uc_value = np.ones((i_n, current_all + 1))  # 控制值初始化为1
        
        sum_t = np.zeros(i_n)  # 存储每次迭代的总时间
        
        u_max = v_max  # 最大控制输入（未使用）
        
        print("Starting ILC Iterations...")  # 开始ILC迭代
        start_time = time.time()  # 记录开始时间
        
        # 迭代学习过程
        for i in range(i_n):  # 迭代 i_n 次
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
            
            while current < current_all and j < j_n:  # current_all 为路径点总数,添加位置与速度限制
                x[i, j] = position[0]  # 记录当前位置x
                y[i, j] = position[1]  # 记录当前位置y
                
                # 找到路径上最近的点
                yp, xp, yp_next, xp_next, point = self.getpoint(x[i, j], y[i, j], xd, yd, current)
                
                # 终止条件：接近终点
                dist_end_sq = (xp - xd[-1])**2 + (yp - yd[-1])**2  # 计算到终点的距离平方
                if dist_end_sq < 0.2 and current > current_all - 50 and  np.linalg.norm(velocity) < 1:  # 距离足够近且路径快结束
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
                            if self.vxd is not None and self.vyd is not None:
                                uc_parax[i, idx_start : idx_end] = self.vxd[l]
                                uc_paray[i, idx_start : idx_end] = self.vyd[l]
                            else:
                                uc_parax[i, idx_start : idx_end] = v_dir[0] # uc_parax 存储平行控制分量
                                uc_paray[i, idx_start : idx_end] = v_dir[1] # uc_paray 存储平行控制分量
                
                value = np.dot(velocity, v_dir)  # 计算当前速度在路径方向上的投影
                
                if i == 0:  # 第一次迭代使用较低速度
                    if self.vxd is not None and self.vyd is not None:
                        value = 1.0
                    else:
                        value = 2.0  # 初始速度较慢
                    
                vl = np.array([uc_parax[i, l], uc_paray[i, l]]) * value  # 计算切向控制向量
                
                # 饱和速度
                vcc, vll = self.saturate(temp_perp * value, vl, v_max)  # 速度饱和处理
                v_des = vcc + vll  # 期望速度向量
                
                # 动力学更新
                a = tau * (v_des - velocity)  # 计算加速度（一阶动力学）
                position, velocity, accelerate = self.uav_dynamic(position, velocity, a, dt)  # 更新位置和速度
                
                if i < i_n - 1:  # 只有在不是最后一次迭代时才进行ILC更新
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
            print(f"Iteration {i+1}/{i_n}: Time {t_total:.2f}s, Points {idx_end if len(non_zeros) > 0 else 0}")  # 打印迭代信息
            
        print(f"Total elapsed time: {time.time() - start_time:.2f}s")  # 打印总运行时间
        
        # 存储ILC迭代结果的元组：i_n: 迭代次数 j_n: 最大时间步数 current_all: 路径点数
        # x: 所有迭代的x坐标数组 (shape: (i_n, j_n + 1))，记录每次迭代中UAV的x位置历史
        # y: 所有迭代的y坐标数组 (shape: (i_n, j_n + 1))，记录每次迭代中UAV的y位置历史
        # sum_t: 每次迭代的总飞行时间数组 (shape: (i_n,))，单位为秒
        # uc_parax: x方向平行控制输入数组 (shape: (i_n, current_all + 1))，沿路径方向的x分量
        # uc_paray: y方向平行控制输入数组 (shape: (i_n, current_all + 1))，沿路径方向的y分量
        # uc_perpx: x方向垂直控制输入数组 (shape: (i_n, current_all + 1))，垂直于路径的x分量（用于纠偏）
        # uc_perpy: y方向垂直控制输入数组 (shape: (i_n, current_all + 1))，垂直于路径的y分量（用于纠偏）
        # uc_value: 控制值数组 (shape: (i_n, current_all + 1))，ILC学习更新的标量值
        # bound: 路径边界半径数组
        self.results = (x, y, sum_t, uc_parax, uc_paray, uc_perpx, uc_perpy, uc_value, bound)
        return self.results

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
        plt.draw()
        plt.pause(0.0001)

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


    
    def plot_time_allocation(self, T_opt, save_path=None, show=False):
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            segments = np.arange(len(T_opt))
            ax.bar(segments, T_opt, color='skyblue', edgecolor='black')
            ax.set_xlabel('Segment Index')
            ax.set_ylabel('Time Duration (s)')
            total_time = np.sum(T_opt)
            ax.set_title(f'Optimized Time Allocation per Segment (Total: {total_time:.2f}s)')
            ax.set_xticks(segments)
            for i, v in enumerate(T_opt):
                ax.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            if save_path:
                # 如果 save_path 是目录或者没有扩展名，则当作目录处理
                if os.path.isdir(save_path) or not os.path.splitext(save_path)[1]:
                     save_path = os.path.join(save_path, 'time_allocation.png')
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, dpi=150)
            
            if show:
                plt.ion()  # 启用交互模式
                plt.draw()  # 绘制图像
                # 图片将保持显示，不阻塞程序运行
                
        except Exception as e:
            print("Failed to plot/save time allocation:", e)

    def plot_traj(self, traj1, traj2=None, bound=None, bound_ref_traj=None):
        """
        绘制地图、管道、障碍物以及传入的轨迹在一张单独的图中。
        traj1: 第一条轨迹，优化后的完整轨迹点序列 (shape: (总采样点数, 2))，每行是一个 (x, y) 坐标
        traj2: 可选的第二条轨迹，同样格式的轨迹点序列 (shape: (总采样点数, 2))，每行是一个 (x, y) 坐标
        bound: 可选的边界半径数组，如果提供，将绘制沿traj1的边界
        bound_ref_traj: 可选的边界参考路径 (shape: (N, 2))，如果bound对应的是重采样后的路径
        """
        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # 绘制地图障碍物
            grid = self.map2d.grid
            nx, ny = grid.shape
            res = self.map2d.resolution
            ox, oy = self.map2d.origin
            xmin, xmax = ox, ox + nx * res
            ymin, ymax = oy, oy + ny * res
            ax.imshow(grid.T, origin="lower", extent=(xmin, xmax, ymin, ymax), cmap="gray_r", alpha=0.5)

            # 绘制管道（球）
            theta = np.linspace(0, 2 * np.pi, 200)
            for i_oc, rr in zip(self.oc_list, self.r_list):
                xc = i_oc[0] + rr * np.cos(theta)
                yc = i_oc[1] + rr * np.sin(theta)
                ax.fill(xc, yc, color='orange', alpha=0.3)
                ax.plot(xc, yc, color='orange')

            # 绘制起点/终点
            ax.scatter(self.p0[0], self.p0[1], color='green', s=80)
            ax.text(self.p0[0] + 0.1, self.p0[1] + 0.1, "start")
            ax.scatter(self.pf[0], self.pf[1], color='red', s=80)
            ax.text(self.pf[0] + 0.1, self.pf[1] + 0.1, "goal")

            # 绘制ILC边界
            if bound is not None:
                center_traj = None
                if bound_ref_traj is not None and len(bound) == len(bound_ref_traj):
                    center_traj = bound_ref_traj
                elif len(bound) == len(traj1):
                    center_traj = traj1
                
                if center_traj is not None:
                    # 计算法向量
                    dx = np.gradient(center_traj[:, 0])
                    dy = np.gradient(center_traj[:, 1])
                    norm = np.sqrt(dx**2 + dy**2) + 1e-6
                    nx = -dy / norm
                    ny = dx / norm
                    
                    # 计算左右边界
                    bx1 = center_traj[:, 0] + bound * nx
                    by1 = center_traj[:, 1] + bound * ny
                    bx2 = center_traj[:, 0] - bound * nx
                    by2 = center_traj[:, 1] - bound * ny
                    
                    ax.plot(bx1, by1, color='cyan', linestyle='--', alpha=0.8, linewidth=1, label='Bound')
                    ax.plot(bx2, by2, color='cyan', linestyle='--', alpha=0.8, linewidth=1)

            # 绘制轨迹
            ax.plot(traj1[:, 0], traj1[:, 1], color='purple', linewidth=3, label='trajectory1')
            if traj2 is not None:
                ax.plot(traj2[:, 0], traj2[:, 1], color='blue', linewidth=3, label='trajectory2')
            ax.legend(loc='upper left')

            ax.set_aspect('equal')
            ax.grid(True)
            ax.set_xlim(0, 20)
            ax.set_ylim(0, 20)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Trajectory Plot')

            plt.tight_layout()
            plt.ion()  # 启用交互模式
            plt.draw()  # 绘制图像
            plt.pause(0.0001)  # 短暂暂停以显示

        except Exception as e:
            print("Failed to plot trajectory:", e)


# ==========================================
#  Main Class: TubeMincoILPlanner
# ==========================================

class TubeMincoILPlanner:
    def __init__(self, npz_path, ilc_iters=50):
        self.npz_path = npz_path
        self.load_corridor()
        self.init_conditions()

        # Optimization history
        self.traj_history = []
        self.energy_history = []
        self.time_history = []
        self.q_history = []
        self.iter_count = 0

        # Initialize Visualizer
        self.visualizer = TubeMincoVisualizer(
            self.map2d, self.oc_list, self.r_list, self.p0, self.pf, self.n
        )
        self.visualizer.setup_plot()

        # Initialize ILC Planner
        self.ilc_planner = ILCTubePlanner(
            ref_path=None,
            map_filename=self.npz_path,
            i_n=ilc_iters, dt=0.02, v_max=5.0,
            tau=3, kp=3.0, kd=None,
            kp_vl=3.0, kp_law=2.0, kd_law=0.05
        )

    def extract_segment_times(self, ilc_x, ilc_y, q_pts, dt):
        """
        Estimate segment times from ILC trajectory.
        """
        # q_pts includes p0, intermediates, pf
        times = []
        last_idx = 0
        
        # Find indices for each point
        indices = [0]
        
        for k in range(1, len(q_pts)):
            q = q_pts[k]
            # Search for closest point in ilc_traj[last_idx:]
            sub_x = ilc_x[last_idx:]
            sub_y = ilc_y[last_idx:]
            
            if len(sub_x) == 0:
                indices.append(last_idx)
                continue

            dists = (sub_x - q[0])**2 + (sub_y - q[1])**2
            idx = np.argmin(dists) + last_idx
            indices.append(idx)
            last_idx = idx
            
        # Calculate times
        T_list = []
        for k in range(len(indices)-1):
            idx_diff = indices[k+1] - indices[k]
            t_seg = idx_diff * dt
            # Ensure non-zero time
            if t_seg < 1e-3: t_seg = 0.1 # Fallback
            T_list.append(t_seg)
            
        return np.array(T_list)

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
        vels = []
        for i in range(len(T_list)):
            pts, vs = sample_segment(coeffs[i], T_list[i])
            segs.append(pts)
            vels.append(vs)
        return np.vstack(segs), np.vstack(vels)

    def opt_callback(self, xk):
        # 1. 从当前优化变量 xk 计算轨迹点
        try:
            trajk, _ = self.traj_from_x(xk)
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

    def run_minco_optimization(self, x0=None, maxiters=50):
        print(f"MINCO Optimization with n={self.n} segments.")

        # Initial guess
        if x0 is None:
            x0 = np.zeros(2 * (self.n - 1) + self.n) # 2D xi + tau
            x0[2 * (self.n - 1):] = np.log(2.0)

        # Initial visualization
        self.opt_callback(x0)

        # Optimize
        res = minimize(self.minco_cost, x0, method='L-BFGS-B', callback=self.opt_callback, options={'maxiter': maxiters} )

        print("===============================Final==================================")

        # Process results
        xi_opt = res.x[:2 * (self.n - 1)]
        tau_opt = res.x[2 * (self.n - 1):]
        q_opt = self.map_to_ball(xi_opt)
        T_opt = self.map_time(tau_opt)

        coeffs_opt = compute_n_segment_coeff_2d(
            self.p0, self.v0, self.a0,
            q_opt, self.pf, self.vf, self.af,
            T_opt
        )
        segs = []
        vels = []
        for i in range(len(T_opt)):
            pts, vs = sample_segment(coeffs_opt[i], T_opt[i])
            segs.append(pts)
            vels.append(vs)
        traj = np.vstack(segs)
        traj_vel = np.vstack(vels)

        # plt.show() # Don't block
        return traj, traj_vel, q_opt, T_opt, res.x
        # traj: 优化后的完整轨迹点序列 (shape: (总采样点数, 2))，每行是一个 (x, y) 坐标
        # traj_vel: 优化后的完整轨迹速度序列 (shape: (总采样点数, 2))
        # q_opt: 优化后的中间点位置 (shape: (n-1, 2))，轨迹的关键拐点
        # T_opt: 优化后的每段轨迹时长 (shape: (n,))，总飞行时间为这些值的和
        # res.x: 优化器的结果向量 (shape: (2*(n-1) + n,))，包含中间点参数 xi 和时间参数 tau


    def run_minco_ilc_iterative_pipeline(self, max_iters=5):
        x0 = None

        for k in range(max_iters):
            print(f"\n=== Outer Iteration {k+1}/{max_iters} ===")

            # 1. Run MINCO
            traj_opt, traj_vel_opt, q_opt, T_opt, x_opt = self.run_minco_optimization(x0,10)
            
            ## check MINCO result ##
            self.visualizer.plot_time_allocation(T_opt, save_path=None, show=True)
            self.visualizer.plot_traj(traj_opt)

            # 2. Run ILC
            # Need to set reference path for ILC
            print(traj_vel_opt)
            ilc_results = self.ilc_planner.run_ilc_iterations(traj_opt, traj_vel_opt)
            ilc_x = ilc_results[0][-1] # Last iteration
            ilc_y = ilc_results[1][-1]
            ilc_bound = ilc_results[-1] # Get bound from results

            # 获取重采样后的参考轨迹
            ref_traj_resampled = np.vstack([self.ilc_planner.xd, self.ilc_planner.yd]).T

            # Extract valid trajectory points (non-zero x coordinates)
            valid_indices = np.nonzero(ilc_x)[0]
            ilc_traj = np.vstack([ilc_x[:valid_indices[-1]+1], ilc_y[:valid_indices[-1]+1]]).T
            
            self.visualizer.plot_traj(traj_opt, ilc_traj, bound=ilc_bound, bound_ref_traj=ref_traj_resampled)

            # 3. Extract times
            q_pts = np.vstack([self.p0, q_opt, self.pf])

            T_new = self.extract_segment_times(ilc_x, ilc_y, q_pts, self.ilc_planner.dt)
            print(f"New times from ILC: {T_new}")

            # 4. Update x0 for next MINCO
            # Keep spatial variables from MINCO result
            xi_next = x_opt[:2 * (self.n - 1)]
            # Update time variables
            # Ensure T_new is valid (positive)
            T_new = np.maximum(T_new, 0.01)
            tau_next = np.log(T_new)

            x0 = np.concatenate([xi_next, tau_next])



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
        # 创建合并的规划器实例
        # ilc_iters: ILC 内部迭代次数
        planner = TubeMincoILPlanner(found_npz, ilc_iters=20)
        # 运行完整的 MINCO-ILC 流水线
        # max_iters: MINCO-ILC 外部循环次数
        planner.run_minco_ilc_iterative_pipeline(max_iters=3)
        #阻塞显示图片
        plt.ioff()
        plt.show()
    else:
        print("Error: tube_corridor.npz not found.")