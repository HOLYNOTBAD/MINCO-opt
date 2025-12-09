import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os

def road():
    """
    Generate a sample path.
    Returns xd, yd arrays.
    """
    # Generate a sine wave path similar to what might be used in testing
    # 1000 points, length approx 50m
    t = np.linspace(0, 50, 1000)
    xd = t
    yd = 5 * np.sin(t / 5.0)
    return xd, yd

def bound_l(n_points):
    """
    Return tube radius.
    """
    return np.full(n_points, 2.0)

def bound_store(xd, yd, bound):
    """
    Calculate boundary coordinates.
    """
    dx = np.gradient(xd)
    dy = np.gradient(yd)
    norms = np.hypot(dx, dy)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    
    nx = -dy / norms
    ny = dx / norms
    
    xd1 = xd + nx * bound
    yd1 = yd + ny * bound
    xd2 = xd - nx * bound
    yd2 = yd - ny * bound
    
    return xd1, yd1, xd2, yd2

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
    xd, yd = road() # 获取路径
    current_all = len(xd)
    bound = bound_l(current_all) # 获取管道半径 (array)
    xd1, yd1, xd2, yd2 = bound_store(xd, yd, bound) # 获取边界
    
    i_n = 20 # 迭代次数
    j_n = current_all # Max points
    
    # Storage
    # Python 0-based indexing.
    # y, x: (i_n, j_n + 1)
    y = np.zeros((i_n, j_n + 1))
    x = np.zeros((i_n, j_n + 1))
    
    # exc threshold
    # exc(i, 1:j_n) = bound./3
    exc = np.zeros((i_n, j_n))
    for i in range(i_n):
        exc[i, :] = bound / 3.0
        
    uc_parax = np.zeros((i_n, j_n + 1))
    uc_paray = np.zeros((i_n, j_n + 1))
    uc_perpx = np.zeros((i_n, j_n + 1))
    uc_perpy = np.zeros((i_n, j_n + 1))
    
    uc_value = np.ones((i_n, j_n + 1))
    
    sum_t = np.zeros(i_n)
    
    v_max = 20.0
    u_max = v_max
    
    dt = 0.01
    kp = 1.5
    kd = 1.0
    kp_vl = 0.05
    tau = 5.0
    kp_law = 1.0
    kd_law = 0.5
    
    print("Starting ILC Iterations...")
    start_time = time.time()
    
    # %% 迭代过程
    # for i = 1:1:i_n-1 (MATLAB) -> range(i_n - 1) (Python 0 to 18)
    # We will iterate i from 0 to i_n - 2, so that i+1 is valid.
    # Actually MATLAB loop is 1 to i_n-1. So it computes up to i_n-1, and fills i_n (via i+1).
    # So Python range should be range(i_n - 1).
    
    for i in range(i_n - 1):
        current = 0
        j = 0
        last_e = 0.0
        value_e = 0.0
        last_l = 0
        
        position = np.array([xd[0], yd[0]]) # Start at beginning of path
        velocity = np.array([4.0, 0.0])
        accelerate = np.array([0.0, 0.0])
        
        v_des = np.array([4.0, 0.0])
        
        while current < current_all and j < j_n:
            x[i, j] = position[0]
            y[i, j] = position[1]
            
            # getpoint
            yp, xp, yp_next, xp_next, point = getpoint(x[i, j], y[i, j], xd, yd, current)
            
            # Termination condition
            # if (xp)^2+(yp)^2<3 && current>current_all-100
            # Adapted: if distance to end < 3.0 and current is large
            dist_end_sq = (xp - xd[-1])**2 + (yp - yd[-1])**2
            if dist_end_sq < 3.0 and current > current_all - 100:
                break
                
            current = point
            l = current
            
            # Error vector
            e_vec = np.array([xp - x[i, j], yp - y[i, j]])
            e_norm = np.linalg.norm(e_vec)
            e_value = e_norm - bound[l]
            
            # Path convergence term
            # temp_perp = kp.*e_vec + kd.*(norm(e_vec,2)-last_e).*e_vec/(norm(e_vec,2)+0.00001);
            temp_perp = kp * e_vec + kd * (e_norm - last_e) * e_vec / (e_norm + 1e-5)
            
            # Store perp control
            if l >= last_l:
                uc_perpx[i, last_l : l+1] = temp_perp[0]
                uc_perpy[i, last_l : l+1] = temp_perp[1]
            
            last_e = value_e
            
            v_dir = np.array([xp_next - xp, yp_next - yp])
            den_para = np.linalg.norm(v_dir) + 1e-10
            v_dir = v_dir / den_para
            
            if i == 0:
                if l >= last_l:
                    uc_parax[i, last_l : l+1] = v_dir[0]
                    uc_paray[i, last_l : l+1] = v_dir[1]
            
            value = np.dot(velocity, v_dir)
            
            if i == 0:
                value = 4.0
                
            # vl = [uc_parax(i,l),uc_paray(i,l)]*value;
            vl = np.array([uc_parax[i, l], uc_paray[i, l]]) * value
            
            # Saturate
            vcc, vll = saturate(temp_perp * value, vl, v_max)
            v_des = vcc + vll
            
            # Dynamics
            a = tau * (v_des - velocity)
            position, velocity, accelerate = uav_dynamic(position, velocity, a, dt)
            
            # ILC Update
            value_e = np.linalg.norm(e_vec)
            de = value_e - last_e # Note: last_e here is the OLD value_e from previous step?
            # In MATLAB:
            # last_e = value_e; (updates last_e to be the value from START of loop step, which was 0 or prev)
            # ...
            # value_e = norm(e_vec,2); (new error)
            # de = value_e - last_e;
            # So yes, de is (current_error - prev_error).
            
            # temp = kp_vl*(1/(1+exp(-kp_law*value_e-kd_law*de+exc(i,l)))-0.5);
            term = -kp_law * value_e - kd_law * de + exc[i, l]
            sigmoid = 1.0 / (1.0 + np.exp(term))
            temp = kp_vl * (sigmoid - 0.5)
            
            if l >= last_l:
                uc_value[i+1, last_l : l+1] = uc_value[i, last_l : l+1] - temp
                # Clip
                uc_value[i+1, last_l : l+1] = np.maximum(uc_value[i+1, last_l : l+1], 0)
                
                # Update next iteration tangential control
                uc_parax[i+1, last_l : l+1] = uc_value[i+1, l] * v_dir[0]
                uc_paray[i+1, last_l : l+1] = uc_value[i+1, l] * v_dir[1]
            
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
    plt.plot(xd1, yd1, 'b', linewidth=1.5)
    plt.plot(xd, yd, '--', color='g', linewidth=0.6)
    plt.plot(xd2, yd2, 'b', linewidth=1.5)
    
    # Last generation trajectory
    # idx_last = find(x(i_n-1,:)~=0,1,'last');
    # In Python, find last non-zero index
    last_iter_idx = i_n - 2 # The loop ran up to i_n - 2
    # Actually loop ran for i in range(i_n - 1), so last i was i_n - 2.
    # The data is stored in x[i, ...].
    # Let's use the last valid iteration.
    
    valid_x = x[last_iter_idx, :]
    non_zeros = np.nonzero(valid_x)[0]
    if len(non_zeros) > 0:
        idx_last = non_zeros[-1]
        plt.plot(x[last_iter_idx, :idx_last+1], y[last_iter_idx, :idx_last+1], 'r', linewidth=2)
        
    plt.title(f"tau = {tau}")
    plt.xlabel("X (m)", fontname='serif', fontsize=20)
    plt.ylabel("Y (m)", fontname='serif', fontsize=20)
    
    # Figure 3: All iterations
    plt.figure(3, figsize=(12, 9))
    plt.clf()
    plt.axis('equal')
    plt.grid(True)
    
    plt.plot(xd, yd, '--', color=[0.6, 0.6, 0.6], linewidth=0.6)
    plt.plot(xd1, yd1, 'b', linewidth=1.0)
    plt.plot(xd2, yd2, 'b', linewidth=1.0)
    
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
    
    print("Saving figures...")
    os.makedirs('MINCO/ILC/ref_traj', exist_ok=True)
    plt.figure(1)
    plt.savefig('MINCO/ILC/ref_traj/ILC_Figure_1.png', dpi=300, bbox_inches='tight')
    plt.figure(2)
    plt.savefig('MINCO/ILC/ref_traj/ILC_Figure_2.png', dpi=300, bbox_inches='tight')
    plt.figure(3)
    plt.savefig('MINCO/ILC/ref_traj/ILC_Figure_3.png', dpi=300, bbox_inches='tight')
    print("Figures saved as MINCO/ILC/ref_traj/ILC_Figure_1.png, MINCO/ILC/ref_traj/ILC_Figure_2.png, MINCO/ILC/ref_traj/ILC_Figure_3.png")
    
    # plt.show() # Commented out for headless execution

if __name__ == "__main__":
    main()
