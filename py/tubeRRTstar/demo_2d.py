# demo_tube_rrt_star_2d_visual.py
# -*- coding: utf-8 -*-

import numpy as np
from tube_rrt_star_2d import (
    SimpleOccupancyMap2D,
    plan_tube_rrt_star_2d,
    visualize_tube_rrt_result_2d,
)
# 传入参数起始点和终点
def build_test_map(start, goal):
    """
    构造一个简单的 2D 栅格地图:
        - 100x100 的空间
        - 中间放几条障碍墙
    """
    nx, ny = 100, 100
    grid = np.zeros((nx, ny), dtype=bool)

    # 障碍墙密度
    wall_density = 0.01    # 随机放置障碍墙
    for i in range(nx):
        for j in range(ny): 
            if np.random.rand() < wall_density: # 同时在起点和终点附近的10个格子内没有障碍物
                if (np.linalg.norm(np.array([i * 0.2, j * 0.2]) - start) > 3.0 and
                    np.linalg.norm(np.array([i * 0.2, j * 0.2]) - goal) > 3.0):
                    grid[i, j] = True

    resolution = 0.2
    origin = np.array([0.0, 0.0])

    return SimpleOccupancyMap2D(grid, resolution, origin)


def main():


    # 2. 起点终点（注意要在 free 区域内）
    start = np.array([2.0, 2.0])
    goal = np.array([15.0, 15.0])

    # 1. 地图
    map2d = build_test_map(start, goal)



    # 3. RRT* 设置参数（可根据需要调整）
    setting = {
        "useIntVol": True,                 # bool: 是否使用体积/管道体积作为代价项(True 会偏向更安全、远离障碍的路径)
        "costIntVol": 1.0,                 # float: 体积代价的权重，越大越重视避障安全
        "GoalBias": 3.0,                   # float: 目标偏置（采样时偏向目标的强度/频率，越大越快朝目标生长，但探索性下降）
        "ContinueAfterGoalReached": False, # bool: 找到可行路径后是否继续迭代以改进路径(True 继续优化)
        "MaxNumTreeNodes": 3000,           # int: 树节点上限，控制内存与搜索规模
        "MaxIterations": 3000,             # int: 最大迭代次数，限制搜索时间/计算量
        "MaxConnectionDistance": 3.0,      # float: 两点间尝试连接的最大距离，影响局部扩展步长
        "xLim": [0.0, 20.0],               # [xmin, xmax] 采样/可视化的 x 轴边界
        "yLim": [0.0, 20.0],               # [ymin, ymax] 采样/可视化的 y 轴边界
    }

    # 4. 运行 Tube-RRT* 规划
    path, tree = plan_tube_rrt_star_2d(start, goal, map2d, setting)

    if path is None or path.size == 0:
        print("未找到可行路径")
        return

    print("找到路径，节点数:", path.shape[0])

    # 5. 可视化结果
    visualize_tube_rrt_result_2d(
        map2d,
        path,
        tree=tree,
        start_pos=start,
        goal_pos=goal,
        show_tube=True,
        show_tree=True,
        title="Tube-RRT* 2D Demo",
    )


if __name__ == "__main__":
    main()
