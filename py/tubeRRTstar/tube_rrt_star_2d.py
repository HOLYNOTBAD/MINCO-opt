# tube_rrt_star_2d.py
# -*- coding: utf-8 -*-
"""
Tube-RRT* 2D 规划（Python 版本）

这是原来 3D 版本的完整二维迁移版：
- 所有位置向量从 [x, y, z] 变为 [x, y]
- 地图从 OccupancyMap3D -> OccupancyMap2D
- 射线、最近障碍点搜索、管道几何全部改为 2D 计算
- RRT* 结构（Node / TubeTree）和代价函数形式保持一致（体积公式仍然沿用 4/3*pi*r^3 作为一个权重函数）

依赖：
    numpy
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
import math


# ================================================================
# 地图接口定义 & 2D 栅格实现
# ================================================================

class OccupancyMap2D:
    """
    抽象 2D 地图接口：
    - check_occupancy(point): point 是 [x, y]，True 表示障碍，False 表示空
    - ray_intersection(sensor_pos, directions, max_range, step):
        sensor_pos: (2,) 起点
        directions: (N, 2) 归一化方向
        max_range: float
        step: float

      返回：
        intersection_points: (N, 2)
        is_occupied: (N,) bool
    """

    def check_occupancy(self, point: np.ndarray) -> bool:
        raise NotImplementedError

    def ray_intersection(
        self,
        sensor_pos: np.ndarray,
        directions: np.ndarray,
        max_range: float,
        step: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class SimpleOccupancyMap2D(OccupancyMap2D):
    """
    简单 2D 栅格地图实现示例：

    grid: 2D bool 数组，shape = (nx, ny)
          grid[ix, iy] = True 表示该栅格有障碍
    resolution: 每个栅格的物理尺寸（米）
    origin: 世界坐标下栅格 (0,0) 的物理坐标 [x0, y0]
    """

    def __init__(self, grid: np.ndarray, resolution: float, origin: np.ndarray):
        self.grid = grid.astype(bool)
        self.resolution = float(resolution)
        self.origin = np.asarray(origin, dtype=float)

    def world_to_grid(self, point: np.ndarray) -> Optional[Tuple[int, int]]:
        p = (np.asarray(point, dtype=float) - self.origin) / self.resolution
        idx = np.floor(p).astype(int)
        ix, iy = idx
        if (
            0 <= ix < self.grid.shape[0]
            and 0 <= iy < self.grid.shape[1]
        ):
            return int(ix), int(iy)
        return None

    def check_occupancy(self, point: np.ndarray) -> bool:
        idx = self.world_to_grid(point)
        if idx is None:
            # 地图外直接视为障碍
            return True
        return bool(self.grid[idx])

    def ray_intersection(
        self,
        sensor_pos: np.ndarray,
        directions: np.ndarray,
        max_range: float,
        step: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        2D 步进射线：从 sensor_pos 沿每个方向前进，直到碰到障碍或超过 max_range。
        """
        sensor_pos = np.asarray(sensor_pos, dtype=float)
        directions = np.asarray(directions, dtype=float)
        num_rays = directions.shape[0]

        intersection_points = np.zeros((num_rays, 2), dtype=float)
        is_occupied = np.zeros(num_rays, dtype=bool)

        for i in range(num_rays):
            d = directions[i]
            if np.linalg.norm(d) == 0:
                d = np.array([1.0, 0.0], dtype=float)
            d = d / np.linalg.norm(d)

            t = 0.0
            hit = False
            while t <= max_range:
                p = sensor_pos + t * d
                if self.check_occupancy(p):
                    intersection_points[i] = p
                    is_occupied[i] = True
                    hit = True
                    break
                t += step
            if not hit:
                # 没有击中障碍：返回末端点
                intersection_points[i] = sensor_pos + max_range * d
                is_occupied[i] = False

        return intersection_points, is_occupied


# ================================================================
# Node / Tree 结构（2D）
# ================================================================

@dataclass
class Node:
    x: float
    y: float
    x_parent: float
    y_parent: float
    dist: float          # 起点到该节点沿树的累计距离
    parent_index: Optional[int]   # 根节点为 None
    radius: float
    vol: float           # 累计“体积”代价（实际上是权重）


@dataclass
class TubeTree:
    nodes: List[Node]
    min_dis: float       # 起点-终点直线距离
    max_radius: float
    min_radius: float
    use_int_vol: bool
    cost_int_vol: float  # k_i


# ================================================================
# 球交体积 & 管道几何（二维场景中仍然使用球体积公式作为权重）
# ================================================================

def intersect_volume(
    x_near: np.ndarray,
    x_near_radius: float,
    x_new: np.ndarray,
    x_new_radius: float,
) -> float:
    """
    球交体积近似（形式上与 3D 版本一致，只依赖两个球心距离和半径）；
    用作“交叠程度”权重。
    """
    x_near = np.asarray(x_near, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    d = np.linalg.norm(x_near - x_new)
    r = float(x_near_radius)
    R = float(x_new_radius)

    if d <= 1e-8:
        # 球心重合：取较小半径球的体积
        rr = min(r, R)
        return 4.0 / 3.0 * math.pi * rr ** 3

    cos_alpha = (r ** 2 + d ** 2 - R ** 2) / (2.0 * r * d)
    cos_beta = (R ** 2 + d ** 2 - r ** 2) / (2.0 * R * d)

    cos_alpha = max(min(cos_alpha, 1.0), -1.0)
    cos_beta = max(min(cos_beta, 1.0), -1.0)

    h2 = r - r * cos_alpha
    h1 = R - R * cos_beta

    inter_vol = math.pi * h1 ** 2 * (R - (1.0 / 3.0) * h1) \
        + math.pi * h2 ** 2 * (r - (1.0 / 3.0) * h2)
    return float(inter_vol)


def score_corridor(
    last_pose: np.ndarray,
    last_radius: float,
    now_pose: np.ndarray,
    now_radius: float,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    2D 版 corridor 评分 + 管道几何（只需要方向和法向）：
    返回：
        score: float
        tube: dict 包含
            - direction: (2,)
            - pose: 管道中间点 (2,)
            - radius: 管道截面半径
            - forward_radius: 沿切线方向的长度
            - normal_vect: (2,) 法向
    """
    last_pose = np.asarray(last_pose, dtype=float)
    now_pose = np.asarray(now_pose, dtype=float)
    r = float(last_radius)
    R = float(now_radius)

    V_vol = 4.0 / 3.0 * math.pi * R ** 3
    rho_v = 1.0
    d = np.linalg.norm(now_pose - last_pose)

    if d <= 1e-8:
        # 退化情况
        score = 0.0
        tube = {
            "direction": np.array([1.0, 0.0], dtype=float),
            "pose": last_pose.copy(),
            "radius": r,
            "forward_radius": R,
            "normal_vect": np.array([0.0, 1.0], dtype=float),
        }
        return score, tube

    # 交体积、几何参数
    cos_alpha = (r ** 2 + d ** 2 - R ** 2) / (2.0 * r * d)
    cos_alpha = max(min(cos_alpha, 1.0), -1.0)
    h2 = r - r * cos_alpha
    sin_alpha = math.sqrt(max(0.0, 1.0 - cos_alpha ** 2))

    cos_beta = (R ** 2 + d ** 2 - r ** 2) / (2.0 * R * d)
    cos_beta = max(min(cos_beta, 1.0), -1.0)
    h1 = R - R * cos_beta

    V_int = math.pi * h1 ** 2 * (R - (1.0 / 3.0) * h1) \
        + math.pi * h2 ** 2 * (r - (1.0 / 3.0) * h2)
    rho_i = 1.0

    direction = (now_pose - last_pose) / d
    tube_pose = last_pose + (d - R + h1) * direction
    tube_radius = r * sin_alpha
    tube_forward_radius = r * (1.0 - cos_alpha)

    # 2D 法向：给方向 [dx, dy] 构造 [-dy, dx]
    dx, dy = direction
    normal_vect = np.array([-dy, dx], dtype=float)
    if np.linalg.norm(normal_vect) < 1e-12:
        normal_vect = np.array([0.0, 1.0], dtype=float)
    normal_vect /= np.linalg.norm(normal_vect)

    tube = {
        "direction": direction,
        "pose": tube_pose,
        "radius": tube_radius,
        "forward_radius": tube_forward_radius,
        "normal_vect": normal_vect,
    }

    if d / (r + R) > 0.9:
        score = 0.0
    else:
        score = rho_v * V_vol + rho_i * V_int

    return float(score), tube


def gen_tube_path2(path: np.ndarray) -> Dict[int, np.ndarray]:
    """
    2D 版 genTubePath2
    输入：
        path: (N, 3) 数组，每行为 [x, y, radius]
    输出：
        tube_paths: dict，包含三条曲线：
            tube_paths[0]: 上边界 (M,2)
            tube_paths[1]: 下边界 (M,2)
            tube_paths[2]: 中心线 (M,2)
    """
    path = np.asarray(path, dtype=float)
    n = path.shape[0]
    tube_paths: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}

    if n < 2:
        return {k: np.empty((0, 2)) for k in tube_paths.keys()}

    for k in range(1, n):
        last_pose = path[k - 1, :2]
        last_radius = path[k - 1, 2]
        now_pose = path[k, :2]
        now_radius = path[k, 2]

        _, tube = score_corridor(last_pose, last_radius, now_pose, now_radius)
        p = tube["pose"]
        R_tube = tube["radius"]
        n_vec = tube["normal_vect"]

        # 中心线
        tube_paths[2].append(p)

        # 上下边界
        extreme_up = p + R_tube * n_vec
        extreme_down = p - R_tube * n_vec

        tube_paths[0].append(extreme_up)
        tube_paths[1].append(extreme_down)

    # 转成 numpy
    for k in tube_paths.keys():
        if tube_paths[k]:
            tube_paths[k] = np.vstack(tube_paths[k])
        else:
            tube_paths[k] = np.empty((0, 2))

    return tube_paths


# ================================================================
# RRT* 核心：nearest / near_tube / tube_steer / add_node_tube / ...
# ================================================================

def nearest_node(tree: TubeTree, x_rand: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    找最近节点：返回索引和该节点坐标。
    """
    x_rand = np.asarray(x_rand, dtype=float)
    min_dis = float("inf")
    near_idx = 0
    for idx, node in enumerate(tree.nodes):
        d = np.linalg.norm(np.array([node.x, node.y]) - x_rand)
        if d < min_dis:
            min_dis = d
            near_idx = idx
    near_node = np.array([tree.nodes[near_idx].x, tree.nodes[near_idx].y], dtype=float)
    return near_idx, near_node # idx是节点在树中的索引，near_node是节点的坐标


def collision_checking(
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    map2d: OccupancyMap2D,
    step: float = 0.2,
) -> bool:
    """
    线段碰撞检测：
        True  -> 有碰撞
        False -> 无碰撞
    """
    start_pos = np.asarray(start_pos, dtype=float)
    goal_pos = np.asarray(goal_pos, dtype=float)
    d = np.linalg.norm(goal_pos - start_pos)
    if d <= 1e-8:
        return False

    direction = (goal_pos - start_pos) / d
    t = 0.0
    while t <= d:
        p = start_pos + t * direction
        if map2d.check_occupancy(p):
            return True
        t += step
    return False


def near_tube(
    tree: TubeTree,
    x_new: np.ndarray,
    x_new_radius: float,
    near_idx: int,
    map2d: OccupancyMap2D,
) -> List[int]:
    """
    找到所有与 x_new 球相交且线段无碰撞的邻居节点索引集合。
    """
    x_new = np.asarray(x_new, dtype=float)
    near_indices = [near_idx]

    for idx, node in enumerate(tree.nodes):
        if idx == near_idx:
            continue
        x_near = np.array([node.x, node.y], dtype=float)
        near_radius = node.radius
        d = np.linalg.norm(x_near - x_new)
        if d < (near_radius + x_new_radius) * 0.8:
            if not collision_checking(x_near, x_new, map2d):
                near_indices.append(idx)

    return near_indices


def add_node_tube(
    tree: TubeTree,
    x_new: np.ndarray,
    parent_idx: int,
    radius: float,
) -> int:
    """
    向树中添加新节点（Tube 版），返回新节点索引。
    """
    x_new = np.asarray(x_new, dtype=float)
    parent = tree.nodes[parent_idx]
    near_node = np.array([parent.x, parent.y], dtype=float)

    if tree.use_int_vol:
        k_i = tree.cost_int_vol
    else:
        k_i = 0.0

    inter_vol = intersect_volume(near_node, parent.radius, x_new, radius)
    vol = (
        k_i
        * math.exp(
            -10.0 * inter_vol / (4.0 / 3.0 * math.pi * tree.max_radius ** 3) + 1.0
        )
        + parent.vol
    )

    dist_from_start = parent.dist + np.linalg.norm(x_new - near_node)

    node = Node(
        x=float(x_new[0]),
        y=float(x_new[1]),
        x_parent=parent.x,
        y_parent=parent.y,
        dist=float(dist_from_start),
        parent_index=parent_idx,
        radius=float(radius),
        vol=float(vol),
    )
    tree.nodes.append(node)
    return len(tree.nodes) - 1


def choose_parent_tube(
    tree: TubeTree,
    near_indices: List[int],
    x_new: np.ndarray,
    x_new_radius: float,
) -> Tuple[int, np.ndarray]:
    """
    在邻居集合中选取总代价最小的父节点。
    """
    x_new = np.asarray(x_new, dtype=float)
    min_cost = float("inf")
    best_idx = near_indices[0]

    for idx in near_indices:
        node = tree.nodes[idx]
        nearest = np.array([node.x, node.y], dtype=float)

        # 距离代价
        cost_dis = np.linalg.norm(nearest - x_new) + node.dist

        # 体积代价
        k_v = 0.0
        k_i = tree.cost_int_vol if tree.use_int_vol else 0.0

        inter_vol = intersect_volume(nearest, node.radius, x_new, x_new_radius)

        cost_volume = (
            k_v * math.exp(-4.0 / 3.0 * math.pi * (x_new_radius / tree.max_radius) ** 3)
            + k_i * 1.0 / (inter_vol / (4.0 / 3.0 * math.pi * tree.max_radius ** 3) + 1.0)
            + node.vol
        )

        cost = cost_dis / tree.min_dis + cost_volume

        if cost < min_cost:
            min_cost = cost
            best_idx = idx

    best_node = tree.nodes[best_idx]
    best_pos = np.array([best_node.x, best_node.y], dtype=float)
    return best_idx, best_pos


def rewire_tube(
    tree: TubeTree,
    near_indices: List[int],
    new_idx: int,
    x_new: np.ndarray,
    map2d: OccupancyMap2D,
) -> None:
    """
    RRT* 重连：若通过 new_idx 新路径让邻居代价更小，则更新其父节点。
    """
    x_new = np.asarray(x_new, dtype=float)
    new_node = tree.nodes[new_idx]

    for idx in near_indices:
        node = tree.nodes[idx]
        if node.parent_index == new_idx:
            continue

        pre_cost_dis = node.dist
        pre_cost_volume = node.vol
        pre_cost = pre_cost_dis / tree.min_dis + pre_cost_volume

        near_pos = np.array([node.x, node.y], dtype=float)

        tentative_cost_dis = np.linalg.norm(near_pos - x_new) + new_node.dist

        k_v = 0.0
        k_i = tree.cost_int_vol if tree.use_int_vol else 0.0

        inter_vol = intersect_volume(near_pos, node.radius, x_new, new_node.radius)
        tentative_cost_volume = (
            k_v
            * math.exp(
                -4.0 / 3.0 * math.pi * (node.radius / tree.max_radius) ** 3
            )
            + k_i * 1.0 / (inter_vol / (4.0 / 3.0 * math.pi * tree.max_radius ** 3) + 1.0)
            + new_node.vol
        )

        tentative_cost = tentative_cost_dis / tree.min_dis + tentative_cost_volume

        if not collision_checking(near_pos, x_new, map2d):
            if tentative_cost < pre_cost:
                node.x_parent = float(x_new[0])
                node.y_parent = float(x_new[1])
                node.dist = float(tentative_cost_dis)
                node.parent_index = new_idx
                node.vol = float(tentative_cost_volume)


# ================================================================
# 与地图交互：sample_free / find_nearest_point
# ================================================================

def sample_free(
    map2d: OccupancyMap2D,
    bounds: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """
    在 [offset, offset+bounds] 区域内随机采样空闲点。
    """
    bounds = np.asarray(bounds, dtype=float)
    offset = np.asarray(offset, dtype=float)
    while True:
        x_rand = np.random.rand(2) * bounds + offset
        if not map2d.check_occupancy(x_rand):
            return x_rand


def find_nearest_point(
    map2d: OccupancyMap2D,
    point: np.ndarray,
    num_rays: int = 60,
    max_range: float = 100.0,
) -> Tuple[float, np.ndarray]:
    """
    从 point 出发向各个方向发射射线，找到最近的障碍点。
    """
    point = np.asarray(point, dtype=float)
    angles = np.linspace(-math.pi, math.pi, num_rays, endpoint=False)
    directions = np.vstack([np.cos(angles), np.sin(angles)]).T  # (num_rays, 2)

    inter_points, is_occ = map2d.ray_intersection(point, directions, max_range)
    if not np.any(is_occ):
        return max_range, point + np.array([max_range, 0.0])

    hit_points = inter_points[is_occ]
    dists = np.linalg.norm(hit_points - point, axis=1)
    idx = int(np.argmin(dists))
    return float(dists[idx]), hit_points[idx]


def tube_steer(
    tree: TubeTree,
    near_idx: int,
    x_rand: np.ndarray,
    map2d: OccupancyMap2D,
) -> Tuple[np.ndarray, float]:
    """
    Tube 风格的 steer：在 near -> x_rand 方向上寻找一个点 x_sphere，
    使得以该点为球心的最大无碰撞球与 near 节点球有充分交叠。
    """
    x_rand = np.asarray(x_rand, dtype=float)
    x_near = np.array(
        [tree.nodes[near_idx].x, tree.nodes[near_idx].y], dtype=float
    )
    x_sphere = x_rand.copy()
    x_radius = tree.nodes[near_idx].radius
    dis = np.linalg.norm(x_near - x_rand)

    if dis <= 1e-8:
        new_radius, _ = find_nearest_point(map2d, x_sphere)
        new_radius = min(new_radius, tree.max_radius)
        return x_sphere, new_radius

    direction = (x_rand - x_near) / dis
    find_flag = True

    while find_flag:
        new_radius, _ = find_nearest_point(map2d, x_sphere)
        new_radius = min(new_radius, tree.max_radius)

        if max(new_radius, x_radius) < dis:
            dis = max(new_radius, x_radius)
            x_sphere = x_near + dis * direction
        else:
            find_flag = False

    return x_sphere, new_radius


# ================================================================
# 搜索路径 & 主入口
# ================================================================

def search_path(tree: TubeTree, end_idx: int) -> np.ndarray:
    """
    追溯终点到起点，返回路径 [x, y, radius]。
    """
    path = []
    idx = end_idx
    while True:
        node = tree.nodes[idx]
        if node.parent_index is None:
            break
        path.append([node.x, node.y, node.radius])
        idx = node.parent_index
    path = path[::-1]
    return np.array(path, dtype=float) if path else np.empty((0, 3))


def plan_tube_rrt_star_2d(
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    map2d: OccupancyMap2D,
    setting: dict,
) -> Tuple[Optional[np.ndarray], TubeTree]:
    """
    2D 版 Tube-RRT* 主函数。

    参数：
        start_pos: (2,) 起点 [x,y]
        goal_pos:  (2,) 终点 [x,y]
        map2d: OccupancyMap2D 实例
        setting: dict，字段示例：
            - useIntVol: bool
            - costIntVol: float
            - GoalBias: float
            - ContinueAfterGoalReached: bool
            - MaxNumTreeNodes: int
            - MaxIterations: int
            - MaxConnectionDistance: float
            - xLim: [xmin, xmax]
            - yLim: [ymin, ymax]

    返回：
        path: (N,3) [x,y,radius] 或 None
        tree: TubeTree
    """
    start_pos = np.asarray(start_pos, dtype=float)
    goal_pos = np.asarray(goal_pos, dtype=float)

    # 起点附近最大球
    min_dis, _ = find_nearest_point(map2d, start_pos)
    min_dis = min(min_dis, 2)

    root = Node(
        x=float(start_pos[0]),
        y=float(start_pos[1]),
        x_parent=float(start_pos[0]),
        y_parent=float(start_pos[1]),
        dist=0.0,
        parent_index=None,
        radius=float(min_dis),
        vol=0.0,
    )

    tree = TubeTree(
        nodes=[root],
        min_dis=float(np.linalg.norm(start_pos - goal_pos)),
        max_radius=15.0,
        min_radius=0.05,
        use_int_vol=bool(setting.get("useIntVol", False)),
        cost_int_vol=float(setting.get("costIntVol", 0.0)),
    )

    goal_bias = float(setting.get("GoalBias", 1.0))
    max_iterations = int(setting.get("MaxIterations", 2000))
    continue_after_goal_reached = bool(setting.get("ContinueAfterGoalReached", False))
    max_num_tree_nodes = int(setting.get("MaxNumTreeNodes", 10_000))
    delta = float(setting.get("MaxConnectionDistance", 5.0)) # 最大步长

    x_lim = np.asarray(setting["xLim"], dtype=float)
    y_lim = np.asarray(setting["yLim"], dtype=float)

    bounds = np.array([x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]], dtype=float)
    offset = np.array([x_lim[0], y_lim[0]], dtype=float)

    found = False
    end_idx = 0
    path: Optional[np.ndarray] = None

    for it in range(max_iterations):
        if len(tree.nodes) >= max_num_tree_nodes:
            break

        # 1. 随机采样
        x_rand = sample_free(map2d, bounds, offset)

        # 2. 最近节点
        near_idx, x_nearest = nearest_node(tree, x_rand)

        # 3. Tube-steer 得到新节点及球
        x_new, x_new_radius = tube_steer(tree, near_idx, x_rand, map2d)
        if x_new_radius < tree.min_radius:
            continue

        # 限制最大步长 delta（可选）
        step_dis = np.linalg.norm(x_new - x_nearest)
        if step_dis > delta:
            direction = (x_new - x_nearest) / step_dis
            x_new = x_nearest + delta * direction
            # 重新估一个半径
            x_new_radius, _ = find_nearest_point(map2d, x_new)
            x_new_radius = min(x_new_radius, tree.max_radius)
            if x_new_radius < tree.min_radius:
                continue

        # 碰撞检测
        if collision_checking(x_nearest, x_new, map2d):
            continue

        # 4. 邻域 + 选择父节点 + 添加节点
        near_indices = near_tube(tree, x_new, x_new_radius, near_idx, map2d)
        parent_idx, _ = choose_parent_tube(tree, near_indices, x_new, x_new_radius)
        new_idx = add_node_tube(tree, x_new, parent_idx, x_new_radius)

        # 5. 重连
        rewire_tube(tree, near_indices, new_idx, x_new, map2d)

        # 6. 是否到达目标附近
        d_goal = np.linalg.norm(x_new - goal_pos)
        if not found:
            if d_goal < min(x_new_radius, goal_bias):
                found = True
                end_idx = new_idx
        else:
            path = search_path(tree, end_idx)
            if not continue_after_goal_reached:
                break

    if found and path is None:
        path = search_path(tree, end_idx)

    return path, tree


# 在文件顶部可以加上：
import matplotlib.pyplot as plt

# ================= 可视化函数 =================

def visualize_tube_rrt_result_2d(
    map2d: OccupancyMap2D,
    path: np.ndarray,
    tree: TubeTree = None,
    start_pos: np.ndarray = None,
    goal_pos: np.ndarray = None,
    show_tube: bool = True,
    show_tree: bool = True,
    title: str = "Tube-RRT* 2D Result",
):
    """
    可视化 Tube-RRT* 结果（2D）:
        - 2D 栅格地图障碍物
        - RRT* 搜索树
        - 最终路径
        - Tube 走廊 (上下边界 + 中心线)

    参数:
        map2d: OccupancyMap2D 对象，若是 SimpleOccupancyMap2D 会自动画出栅格
        path: (N,3) 数组，每行 [x, y, radius]
        tree: TubeTree，可选，用于画树
        start_pos, goal_pos: 起点终点 (2,)，可选
        show_tube: 是否画 Tube 走廊
        show_tree: 是否画搜索树
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 1. 如果是 SimpleOccupancyMap2D，画栅格障碍物
    if isinstance(map2d, SimpleOccupancyMap2D):
        grid = map2d.grid  # shape: (nx, ny)
        nx, ny = grid.shape
        res = map2d.resolution
        ox, oy = map2d.origin

        xmin = ox
        xmax = ox + nx * res
        ymin = oy
        ymax = oy + ny * res

        # True = obstacle -> 显示为黑色
        ax.imshow(
            grid.T,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap="gray_r",
            alpha=0.5,
        )

    # # 2. 画搜索树
    # if show_tree and (tree is not None):
    #     for node in tree.nodes:
    #         if node.parent_index is None:
    #             continue
    #         px, py = node.x_parent, node.y_parent
    #         x, y = node.x, node.y
    #         ax.plot([px, x], [py, y], linewidth=0.5)

    # 3. 画最终路径
    if path is not None and path.size > 0:
        path_xy = path[:, :2]
        ax.plot(path_xy[:, 0], path_xy[:, 1], "-r", linewidth=2.0, label="path")
        # 画半径
        if path.shape[1] == 3: # 如果有半径信息
            for i in range(path.shape[0]):
                x, y, r = path[i]
                circle = plt.Circle((x, y), r, color="black", fill=False, linewidth=1.0)
                ax.add_patch(circle)
        #4. 画 Tube 走廊
        if show_tube:
            tube_paths = gen_tube_path2(path)  # {0:上边界, 1:下边界, 2:中心线}
            if tube_paths[2].shape[0] > 0:
                center = tube_paths[2]
                up = tube_paths[0]
                down = tube_paths[1]
                ax.plot(center[:, 0], center[:, 1], "--b", linewidth=1.5, label="tube center")
                ax.plot(up[:, 0], up[:, 1], "-g", linewidth=1.0, label="tube upper")
                ax.plot(down[:, 0], down[:, 1], "-g", linewidth=1.0, label="tube lower")

    # 5. 起点终点
    if start_pos is not None:
        ax.scatter(start_pos[0], start_pos[1], s=60, marker="o", label="start")
    if goal_pos is not None:
        ax.scatter(goal_pos[0], goal_pos[1], s=60, marker="*")

    ax.set_aspect("equal", "box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True)

    plt.show()
