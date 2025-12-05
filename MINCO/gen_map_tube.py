#!/usr/bin/env python3
"""
生成地图并运行 Tube-RRT*，把生成的球走廊（oc_list, r_list）保存到指定位置。

用法示例:
    python py/gen_map_tube.py --out py/generated/tube_corridor.npz --png py/generated/tube_corridor.png

输出:
- .npz 文件（包含 oc_list, r_list, path）
- 可选的 PNG 可视化图

该脚本只负责地图与 tube-rrt 的生成，后续的 MINCO 优化请在另一个脚本中读取输出并执行。
"""
import argparse
import os
import sys
import numpy as np

# 尝试将本仓库中 MINCO/tubeRRTstar 加入路径，便于导入 tube_rrt_star_2d
_this_dir = os.path.abspath(os.path.dirname(__file__))
_candidates = [
    os.path.join(_this_dir, "tubeRRTstar"),
    os.path.join(_this_dir, "py", "tubeRRTstar"),
    os.path.join(_this_dir, "..", "py", "tubeRRTstar"),
]
for p in _candidates:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
        break

try:
    from tube_rrt_star_2D import SimpleOccupancyMap2D, plan_tube_rrt_star_2d, visualize_tube_rrt_result_2d
except Exception as e:
    raise ImportError("无法导入 tube_rrt_star_2d；请确保 py/tubeRRTstar 在仓库中并可 import") from e

import matplotlib.pyplot as plt


def build_test_map(start, goal, nx=100, ny=100, resolution=0.2, wall_density=0.01, safe_radius=3.0):
    grid = np.zeros((nx, ny), dtype=bool)
    for ix in range(nx):
        for iy in range(ny):
            if np.random.rand() < wall_density:
                x = ix * resolution
                y = iy * resolution
                if (np.linalg.norm(np.array([x, y]) - start) > safe_radius
                        and np.linalg.norm(np.array([x, y]) - goal) > safe_radius):
                    grid[ix, iy] = True
    origin = np.array([0.0, 0.0])
    return SimpleOccupancyMap2D(grid, resolution, origin)


def run_tube_rrt_and_save(start, goal, out_npz, out_png=None, setting=None):
    if setting is None:
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

    print("生成地图...")
    map2d = build_test_map(start, goal, nx=100, ny=100, resolution=0.2, wall_density=0.01, safe_radius=3.0)

    print("运行 Tube-RRT*... 这一步可能需要几秒到几十秒，取决于地图与参数。")
    path, tree = plan_tube_rrt_star_2d(start, goal, map2d, setting)

    if path is None or path.size == 0:
        raise RuntimeError("Tube-RRT* 未能生成路径，请调整参数或地图。")

    # interior nodes as corridor (exclude start and goal)
    if path.shape[0] <= 2:
        oc_list = np.empty((0, 2))
        r_list = np.empty((0,))
    else:
        oc_list = path[1:-1, :2].astype(float)
        r_list = path[1:-1, 2].astype(float)

    # ensure out folder exists
    out_dir = os.path.dirname(out_npz)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"保存走廊数据到 {out_npz}")
    # 同时保存地图信息（grid / resolution / origin），便于下游脚本直接从 npz 恢复 SimpleOccupancyMap2D
    if isinstance(map2d, SimpleOccupancyMap2D):
        grid = map2d.grid
        resolution = map2d.resolution
        origin = np.asarray(map2d.origin)
    else:
        grid = None
        resolution = None
        origin = None
    # 使用压缩格式保存，减小文件体积并包含地图元数据
    np.savez_compressed(out_npz,
                        oc_list=oc_list,
                        r_list=r_list,
                        path=path,
                        grid=grid,
                        resolution=resolution,
                        origin=origin)

    if out_png:
        out_png_dir = os.path.dirname(out_png)
        if out_png_dir and not os.path.exists(out_png_dir):
            os.makedirs(out_png_dir, exist_ok=True)
        print(f"绘制并保存可视化到 {out_png}")
        fig, ax = plt.subplots(figsize=(8, 8))
        # 地图
        if isinstance(map2d, SimpleOccupancyMap2D):# 检查是否为 SimpleOccupancyMap2D 实例
            grid = map2d.grid
            nx, ny = grid.shape
            res = map2d.resolution
            ox, oy = map2d.origin
            xmin, xmax = ox, ox + nx * res
            ymin, ymax = oy, oy + ny * res
            ax.imshow(grid.T, origin='lower', extent=(xmin, xmax, ymin, ymax), cmap='gray_r', alpha=0.5)
        # 画树与路径
        if tree is not None:
            for node in tree.nodes:
                if node.parent_index is None:
                    continue
                px, py = node.x_parent, node.y_parent
                x, y = node.x, node.y
                ax.plot([px, x], [py, y], linewidth=0.5, color='k', alpha=0.3)
        # 画路径中心线
        path_xy = path[:, :2]
        ax.plot(path_xy[:, 0], path_xy[:, 1], '-r', linewidth=2.0)
        # 画每个球
        theta = np.linspace(0, 2*np.pi, 120)
        for (c, rr) in zip(oc_list, r_list):
            xc = c[0] + rr * np.cos(theta)
            yc = c[1] + rr * np.sin(theta)
            ax.plot(xc, yc, '-b', linewidth=1.0)
            ax.fill(xc, yc, color='b', alpha=0.15)
        ax.scatter(start[0], start[1], s=60, marker='o', color='g', label='start')
        ax.scatter(goal[0], goal[1], s=60, marker='*', color='r', label='goal')
        ax.set_aspect('equal')
        ax.set_xlim(setting['xLim'])
        ax.set_ylim(setting['yLim'])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Generated Tube-RRT Corridor')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

    return oc_list, r_list, path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', '-o', help='输出 npz 路径', default='MINCO/gen_map_tube/tube_corridor.npz')
    parser.add_argument('--png', help='可视化 png 路径（可选）', default='MINCO/gen_map_tube/tube_corridor.png')
    parser.add_argument('--start', help='起点 x,y', default='2.0,2.0')
    parser.add_argument('--goal', help='终点 x,y', default='18.0,18.0')
    args = parser.parse_args()

    start = np.fromstring(args.start, sep=',')
    goal = np.fromstring(args.goal, sep=',')
    oc_list, r_list, path = run_tube_rrt_and_save(start, goal, args.out, args.png)
    print('Saved.', args.out, args.png)
