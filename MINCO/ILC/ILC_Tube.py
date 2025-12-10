import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os
from scipy.interpolate import interp1d

def load_tube_path(filename):
    """
    Load path and boundaries from npz file.
    Returns interpolated xd, yd, bound, and raw boundary data for plotting.
    """
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        return None, None, None, None, None
        
    data = np.load(filename)
    
    if 'traj' in data:
        # optimized_traj.npz format
        path = data['traj'] # (N, 2)
        x = path[:, 0]
        y = path[:, 1]
        # Default radius since it's not in optimized_traj.npz
        r = np.full_like(x, 0.5)
    elif 'path' in data:
        # tube_corridor.npz format
        path = data['path'] # (N, 3) -> x, y, radius
        x = path[:, 0]
        y = path[:, 1]
        r = path[:, 2]
    else:
        print("Error: Unknown file format.")
        return None, None, None, None, None
    
    # Raw boundaries for plotting
    raw_left_b = data['left_boundary'] if 'left_boundary' in data else None
    raw_right_b = data['right_boundary'] if 'right_boundary' in data else None
    
    # Interpolate path to get dense waypoints
    # Calculate cumulative distance to use as interpolation parameter
    dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    cum_dist = np.concatenate(([0], np.cumsum(dists)))
    total_dist = cum_dist[-1]
    
    # Number of points for dense path (e.g., every 0.02m for smoother control)
    # ILC.py used 1000 points for 50m -> 0.05m spacing.
    # Let's use similar density.
    n_points = int(total_dist / 0.05)
    if n_points < 100: n_points = 100 # Minimum points
    
    t_new = np.linspace(0, total_dist, n_points)
    
    f_x = interp1d(cum_dist, x, kind='linear')
    f_y = interp1d(cum_dist, y, kind='linear')
    f_r = interp1d(cum_dist, r, kind='linear')
    
    xd = f_x(t_new)
    yd = f_y(t_new)
    bound = f_r(t_new)
    
    # modify bound to converge linearly to 0 from L distance from end
    L = 10.0  # distance from end to start convergence
    bound_modified = bound.copy()
    idx_start = np.where(t_new >= total_dist - L)[0]
    if len(idx_start) > 0:
        idx_start = idx_start[0]
        for i in range(idx_start, len(bound)):
            ratio = (t_new[i] - (total_dist - L)) / L
            bound_modified[i] = bound[idx_start] * (1 - ratio)
        bound_modified[-1] = 0.0  # ensure end point is 0
    
    return xd, yd, bound_modified, raw_left_b, raw_right_b

def load_map(filename):
    """
    Load occupancy grid from npz file.
    """
    if not os.path.exists(filename):
        return None, None, None
        
    data = np.load(filename)
    if 'grid' in data:
        return data['grid'], data['resolution'], data['origin']
    return None, None, None


def plot_tube_bound(xd, yd, bound, raw_left_b=None, raw_right_b=None, save_path=None):
    """
    Plot reference path xd, yd, the corridor bounds (using normal offsets by `bound`),
    and optional raw left/right boundaries if provided.
    """
    import matplotlib.pyplot as plt
    xd = np.asarray(xd)
    yd = np.asarray(yd)
    bound = np.asarray(bound)

    # compute tangents and normals
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

    # plot reference (center) path
    ax.plot(xd, yd, '--', color='g', linewidth=1.0, label='Reference Path')

    # start and end markers
    try:
        ax.scatter(xd[0], yd[0], color='green', s=60, zorder=5)
        ax.text(xd[0] + 0.02, yd[0] + 0.02, 'start', color='green')
        ax.scatter(xd[-1], yd[-1], color='magenta', s=60, zorder=5)
        ax.text(xd[-1] + 0.02, yd[-1] + 0.02, 'end', color='magenta')
    except Exception:
        pass

    # plot computed left/right bounds
    ax.plot(left_x, left_y, color='orange', linewidth=1.2, label='Computed Left Bound')
    ax.plot(right_x, right_y, color='orange', linewidth=1.2, label='Computed Right Bound')

    # fill corridor
    try:
        ax.fill(np.concatenate([left_x, right_x[::-1]]), np.concatenate([left_y, right_y[::-1]]), color='orange', alpha=0.2)
    except Exception:
        pass

    # plot raw boundaries if provided
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

def getpoint(x_curr, y_curr, xd, yd, current_idx):
    """
    Find the closest point on the path starting search from current_idx.
    """
    # Search window to improve performance
    search_len = 100
    start = current_idx
    end = min(len(xd), current_idx + search_len)
    
    # If we are at the end, just check the last few points
    if start >= len(xd) - 1:
        start = len(xd) - 1
        end = len(xd)
        
    dists_sq = (xd[start:end] - x_curr)**2 + (yd[start:end] - y_curr)**2
    min_local = np.argmin(dists_sq)
    point_idx = start + min_local
    
    xp = xd[point_idx]
    yp = yd[point_idx]
    
    # Next point for direction
    next_idx = min(point_idx + 1, len(xd) - 1)
    xp_next = xd[next_idx]
    yp_next = yd[next_idx]
    
    return yp, xp, yp_next, xp_next, point_idx

def saturate(v_perp, v_para, v_max):
    """
    Saturate velocity vector.
    """
    v_sum = v_perp + v_para
    v_norm = np.linalg.norm(v_sum)
    
    if v_norm > v_max:
        scale = v_max / v_norm
        return v_perp * scale, v_para * scale
    return v_perp, v_para

def uav_dynamic(pos, vel, acc, dt):
    """
    Simple UAV dynamics update.
    """
    new_vel = vel + acc * dt
    new_pos = pos + vel * dt
    return new_pos, new_vel, acc

def main():
    # %% 初始化
    filename = 'MINCO/ILC/ref_traj/optimized_traj.npz'
    print(f"Loading path from {filename}...")
    xd, yd, bound, raw_left_b, raw_right_b = load_tube_path(filename)
    # 绘制并保存管道与边界到文件
    try:
        out_plot = 'MINCO/ILC/ref_traj/combined_bound_plot.png'
        plot_tube_bound(xd, yd, bound, raw_left_b, raw_right_b, save_path=out_plot)
        print(f"Saved combined bound plot to {out_plot}")
    except Exception as e:
        print('Failed to plot combined bounds:', e)
    
    # Load map
    map_filename = 'MINCO/gen_map_tube/tube_corridor.npz'
    grid, resolution, origin = load_map(map_filename)
    
    if xd is None:
        print("Failed to load path.")
        return

    current_all = len(xd)   # current_all为路径点总数
    print(f"Path loaded with {current_all} points. Total length approx {np.sum(np.sqrt(np.diff(xd)**2 + np.diff(yd)**2)):.2f}m")
    
    i_n = 100 # 迭代次数
    j_n = current_all * 2 # 时间步数，预设为路径点数的两倍以确保足够的时间步覆盖路径
    
    # Storage
    y = np.zeros((i_n, j_n + 1))
    x = np.zeros((i_n, j_n + 1))
    
    # exc threshold
    # exc(i, 1:j_n) = bound./3
    # We need to map bound (length current_all) to j (steps).
    # Since we don't know exactly which path point corresponds to which step j beforehand,
    # we will look up bound based on 'l' (closest point index) during the loop.
    # But 'exc' array in original code was (i_n, j_n).
    # And bound was length current_all.
    # And exc was initialized using bound.
    # Wait, original code: exc[i, :] = bound / 3.0
    # This implies j_n == current_all in original code.
    # Here j_n might be larger.
    # We should use bound[l] dynamically or initialize exc with a default and update it?
    # Original code: exc[i, :] = bound / 3.0. This fails if len(bound) != j_n.
    # In original code, j_n = current_all.
    # Here, let's make exc size (i_n, current_all) to match bound size, 
    # and access it using 'l' (path index) instead of 'j' (time step).
    
    exc = np.zeros((i_n, current_all))
    for i in range(i_n):
        exc[i, :] = bound / 3.0
        
    uc_parax = np.zeros((i_n, current_all + 1))
    uc_paray = np.zeros((i_n, current_all + 1))
    uc_perpx = np.zeros((i_n, current_all + 1))
    uc_perpy = np.zeros((i_n, current_all + 1))
    
    uc_value = np.ones((i_n, current_all + 1))
    
    sum_t = np.zeros(i_n)
    
    v_max = 5.0 # Reduced from 20.0 for safety in complex paths
    u_max = v_max
    
    dt = 0.02 # Slightly larger dt
    kp = 1.5
    kd = 1.0
    kp_vl = 0.05
    tau = 0.4
    kp_law = 1.0
    kd_law = 0.5
    
    print("Starting ILC Iterations...")
    start_time = time.time()
    
    # %% 迭代过程
    for i in range(i_n - 1):
        current = 0
        j = 0
        last_e = 0.0
        value_e = 0.0
        last_l = 0
        
        position = np.array([xd[0], yd[0]]) # Start at beginning of path
        velocity = np.array([0.0, 0.0]) # Start from rest? Or initial velocity?
        # Original: velocity = np.array([4.0, 0.0])
        # Let's infer initial direction
        dir_init = np.array([xd[1]-xd[0], yd[1]-yd[0]])
        dir_init = dir_init / np.linalg.norm(dir_init)
        velocity = dir_init * 1.0 # Start with small velocity
        
        accelerate = np.array([0.0, 0.0])
        
        v_des = velocity.copy()
        
        while current < current_all and j < j_n:
            x[i, j] = position[0]   # x[i,j]是第i次迭代，第j个时间步的x坐标
            y[i, j] = position[1]   # y[i,j]是第i次迭代，第j个时间步的y坐标
            
            # 这五个量分别是：路径点坐标yp, xp， 下一个路径点坐标yp_next, xp_next， 当前最近路径点索引point
            yp, xp, yp_next, xp_next, point = getpoint(x[i, j], y[i, j], xd, yd, current)
            
            # Termination condition： close to end point
            dist_end_sq = (xp - xd[-1])**2 + (yp - yd[-1])**2
            if dist_end_sq < 1.0 and current > current_all - 50: # Relaxed condition
                break
                
            current = point
            l = current
            
            # 误差向量：e_vec-当前位置到最近路径点的向量 e_norm-误差大小 e_value-误差值（距离减去管道半径）
            e_vec = np.array([xp - x[i, j], yp - y[i, j]])
            e_norm = np.linalg.norm(e_vec)
            e_value = e_norm - bound[l] # bound[l]是当前位置对应的管道半径
            
            # temp_perp 是垂直控制分量
            temp_perp = kp * e_vec + kd * (e_norm - last_e) * e_vec / (e_norm + 1e-5)
            
            # Store perp control
            if l >= last_l:
                # Handle indices carefully
                idx_start = last_l
                idx_end = min(l + 1, current_all)
                if idx_end > idx_start:
                    uc_perpx[i, idx_start : idx_end] = temp_perp[0]
                    uc_perpy[i, idx_start : idx_end] = temp_perp[1]
            
            last_e = value_e
            
            v_dir = np.array([xp_next - xp, yp_next - yp])
            den_para = np.linalg.norm(v_dir) + 1e-10
            v_dir = v_dir / den_para
            
            if i == 0:
                if l >= last_l:
                    idx_start = last_l
                    idx_end = min(l + 1, current_all)
                    if idx_end > idx_start:
                        uc_parax[i, idx_start : idx_end] = v_dir[0]
                        uc_paray[i, idx_start : idx_end] = v_dir[1]
            
            value = np.dot(velocity, v_dir)
            
            if i == 0:
                value = 2.0 # Slower initial speed
                
            vl = np.array([uc_parax[i, l], uc_paray[i, l]]) * value
            
            # Saturate
            vcc, vll = saturate(temp_perp * value, vl, v_max)
            v_des = vcc + vll
            
            # Dynamics
            a = tau * (v_des - velocity)
            position, velocity, accelerate = uav_dynamic(position, velocity, a, dt)
            
            # ILC Update
            value_e = np.linalg.norm(e_vec)
            de = value_e - last_e 
            
            term = -kp_law * value_e - kd_law * de + exc[i, l]
            sigmoid = 1.0 / (1.0 + np.exp(term))
            temp = kp_vl * (sigmoid - 0.5)
            
            if l >= last_l:
                idx_start = last_l
                idx_end = min(l + 1, current_all)
                if idx_end > idx_start:
                    uc_value[i+1, idx_start : idx_end] = uc_value[i, idx_start : idx_end] - temp
                    # Clip
                    uc_value[i+1, idx_start : idx_end] = np.maximum(uc_value[i+1, idx_start : idx_end], 0)
                    
                    # Update next iteration tangential control
                    uc_parax[i+1, idx_start : idx_end] = uc_value[i+1, l] * v_dir[0]
                    uc_paray[i+1, idx_start : idx_end] = uc_value[i+1, l] * v_dir[1]
            
            j += 1
            last_l = l
            
        t_total = j * dt
        sum_t[i] = t_total
        print(f"Iteration {i+1}/{i_n-1}: Time {t_total:.2f}s, Points {j}")
        
    print(f"Total elapsed time: {time.time() - start_time:.2f}s")
    
    # %% 画图
    
    # Figure 1: Time
    plt.figure(1, figsize=(12, 9))
    plt.plot(range(1, i_n), sum_t[:i_n-1], linewidth=1)
    plt.title('Time', fontname='serif', fontsize=20)
    plt.xlabel("Epoch", fontname='serif', fontsize=20)
    plt.ylabel("Time (s)", fontname='serif', fontsize=20)
    plt.grid(True)
    
    # Figure 2: Original Plot
    plt.figure(2, figsize=(12, 9))
    plt.clf()
    plt.axis('equal')
    plt.grid(True)
    
    # Plot map
    if grid is not None:
        nx, ny = grid.shape
        if origin.size > 1:
            ox, oy = origin[0], origin[1]
        else:
            ox, oy = float(origin), 0.0
        xmin, xmax = ox, ox + nx * resolution
        ymin, ymax = oy, oy + ny * resolution
        plt.imshow(grid.T, origin="lower", extent=(xmin, xmax, ymin, ymax), cmap="gray_r", alpha=0.5)
    
    # Plot boundaries from file
    if raw_left_b is not None:
        plt.plot(raw_left_b[:, 0], raw_left_b[:, 1], 'b.-', linewidth=1.5, label='Left Boundary')
    if raw_right_b is not None:
        plt.plot(raw_right_b[:, 0], raw_right_b[:, 1], 'b.-', linewidth=1.5, label='Right Boundary')
        
    plt.plot(xd, yd, '--', color='g', linewidth=0.6, label='Reference Path')
    
    # Last generation trajectory
    last_iter_idx = i_n - 2
    valid_x = x[last_iter_idx, :]
    non_zeros = np.nonzero(valid_x)[0]
    if len(non_zeros) > 0:
        idx_last = non_zeros[-1]
        plt.plot(x[last_iter_idx, :idx_last+1], y[last_iter_idx, :idx_last+1], 'r', linewidth=2, label='Last Trajectory')
    
    # Start and End points
    plt.plot(xd[0], yd[0], 'go', markerfacecolor='g')
    plt.text(xd[0], yd[0], '  start', fontsize=10)
    plt.plot(xd[-1], yd[-1], 'mo', markerfacecolor='m')
    plt.text(xd[-1], yd[-1], '  end', fontsize=10)
        
    plt.title(f"tau = {tau}")
    plt.xlabel("X (m)", fontname='serif', fontsize=20)
    plt.ylabel("Y (m)", fontname='serif', fontsize=20)
    plt.legend()
    
    # Figure 3: All iterations
    plt.figure(3, figsize=(12, 9))
    plt.clf()
    plt.axis('equal')
    plt.grid(True)
    
    plt.plot(xd, yd, '--', color=[0.6, 0.6, 0.6], linewidth=0.6)
    if raw_left_b is not None:
        plt.plot(raw_left_b[:, 0], raw_left_b[:, 1], 'b', linewidth=1.0)
    if raw_right_b is not None:
        plt.plot(raw_right_b[:, 0], raw_right_b[:, 1], 'b', linewidth=1.0)
    
    nIter = max(1, i_n - 1)
    # Colormap
    cmap = plt.get_cmap('jet')
    colors = [cmap(k) for k in np.linspace(0, 1, nIter)]
    
    for ii in range(nIter):
        valid_x = x[ii, :]
        non_zeros = np.nonzero(valid_x)[0]
        if len(non_zeros) < 2:
            continue
        idx = non_zeros[-1]
        plt.plot(x[ii, :idx+1], y[ii, :idx+1], '-', color=colors[ii], linewidth=1.2)
        
    # Start and End
    plt.plot(x[0, 0], y[0, 0], 'go', markerfacecolor='g')
    plt.text(x[0, 0], y[0, 0], '  start', fontsize=10)
    
    if len(non_zeros) > 0:
        plt.plot(x[last_iter_idx, :idx_last+1], y[last_iter_idx, :idx_last+1], 'r', linewidth=2)
        plt.plot(x[last_iter_idx, idx_last], y[last_iter_idx, idx_last], 'mo', markerfacecolor='m')
        plt.text(x[last_iter_idx, idx_last], y[last_iter_idx, idx_last], '  end', fontsize=10)
        
    plt.title(f"tau = {tau}, iterations = {nIter}")
    plt.xlabel("X (m)", fontname='serif', fontsize=20)
    plt.ylabel("Y (m)", fontname='serif', fontsize=20)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=nIter))
    sm.set_array([])
    plt.colorbar(sm, ticks=np.linspace(1, nIter, int(min(5, nIter))), label='Iteration', ax=plt.gca())
    
    # Figure 4: Velocity over time
    plt.figure(4, figsize=(12, 9))
    plt.clf()
    
    # Calculate velocity for the last iteration
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
    os.makedirs('MINCO/ILC/ref_traj', exist_ok=True)
    plt.figure(1)
    plt.savefig('MINCO/ILC/ref_traj/ILC_Tube_Figure_1.png', dpi=300, bbox_inches='tight')
    plt.figure(2)
    plt.savefig('MINCO/ILC/ref_traj/ILC_Tube_Figure_2.png', dpi=300, bbox_inches='tight')
    plt.figure(3)
    plt.savefig('MINCO/ILC/ref_traj/ILC_Tube_Figure_3.png', dpi=300, bbox_inches='tight')
    plt.figure(4)
    plt.savefig('MINCO/ILC/ref_traj/ILC_Tube_Figure_4.png', dpi=300, bbox_inches='tight')
    print("Figures saved as MINCO/ILC/ref_traj/ILC_Tube_Figure_*.png")

if __name__ == "__main__":
    main()
