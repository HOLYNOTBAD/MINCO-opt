import numpy as np
import matplotlib.pyplot as plt
from tube_rrt_star_2d import (
    SimpleOccupancyMap2D,
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

    # 地图分辨率（每格代表的真实世界长度）
    resolution = 0.2
    origin = np.array([0.0, 0.0])

    # 障碍墙密度
    wall_density = 0.1
    # 为起点/终点保留的安全半径（世界坐标单位）
    safe_radius = 3.0

    # 随机放置障碍墙，但保证 start/goal 周围 safe_radius 内为空
    for i in range(nx):
        for j in range(ny):
            if np.random.rand() < wall_density:
                pos = np.array([i, j]) * resolution + origin
                # 使用平方距离避免每次开方
                if ((pos - start).dot(pos - start) > safe_radius**2 and
                    (pos - goal).dot(pos - goal) > safe_radius**2):
                    grid[i, j] = True

    return SimpleOccupancyMap2D(grid, resolution, origin)
def main():
    # 1. 构造一个简单的 2D 栅格地图
    # 注意：start/goal 使用世界坐标 (meters)，需处于 [0, nx*resolution]
    start = np.array([2.0, 2.0])
    goal = np.array([15.0, 15.0])

    map2d = build_test_map(start, goal)

    # 只展示地图（不调用规划器）——用于检查随机障碍生成是否合理
    grid = map2d.grid
    nx, ny = grid.shape
    res = map2d.resolution
    ox, oy = map2d.origin

    xmin = ox
    xmax = ox + nx * res
    ymin = oy
    ymax = oy + ny * res

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(
        grid.T,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        cmap="gray_r",
        alpha=0.8,
    )
    ax.scatter(start[0], start[1], c="green", s=60, label="start")
    ax.scatter(goal[0], goal[1], c="red", s=60, marker="*", label="goal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", "box")
    ax.set_title("Test occupancy grid (no planning)")
    ax.legend()
    plt.tight_layout()
    # 在桌面/本地显示图像（不要保存为文件）
    plt.show()


if __name__ == "__main__":
    main()
