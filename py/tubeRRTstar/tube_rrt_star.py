# tube_rrt_star.py
# -*- coding: utf-8 -*-
"""
Tube-RRT* 3D 规划（Python 版本）

从 MATLAB 代码：
- planTubeRRTStar.m
- AddNodeTube.m
- ChooseParentTube.m
- NearTube.m
- Nearest.m
- tubeSteer.m
- collisionChecking.m
- findNearestPoint.m
- intersectVolume.m
- rewireTube.m
- sampleFree.m
- searchPath.m
- scoreCorridor.m
- genTubePath2.m
尽量保持算法与代价形式一致，实现“效果相同”的 Python 版本。

依赖：
    numpy
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import math
import random


# ================================================================
# 地图接口定义 & 一个简单的3D占据栅格实现
# ================================================================

class OccupancyMap3D:
    """
    抽象地图接口，模仿 MATLAB occupancyMap3D 所需能力：
    - check_occupancy(point): 返回 True 表示占据（有障碍），False 表示空
    - ray_intersection(sensor_pose, directions, max_range):
        sensor_pose: [x, y, z, qx, qy, qz, qw]（这里旋转信息不使用）
        directions: (N, 3) 归一化方向向量
        max_range: float

      返回：
        intersection_points: (N, 3) 数组，若没有击中则返回 sensor_pose 末端点
        is_occupied: (N,) bool 数组，True 表示射线在 max_range 内击中障碍
    """

    def check_occupancy(self, point: np.ndarray) -> bool:
        raise NotImplementedError

    def ray_intersection(
        self,
        sensor_pose: np.ndarray,
        directions: np.ndarray,
        max_range: float,
        step: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class SimpleOccupancyMap3D(OccupancyMap3D):
    """
    一个简单的 3D 栅格地图实现示例：

    grid: 3D numpy bool 数组，True = 占据，False = 空
    resolution: 每个栅格的物理尺寸（米）
    origin: 世界坐标下栅格 (0,0,0) 的物理坐标
    """

    def __init__(self, grid: np.ndarray, resolution: float, origin: np.ndarray):
        self.grid = grid.astype(bool)
        self.resolution = float(resolution)
        self.origin = np.asarray(origin, dtype=float)

    def world_to_grid(self, point: np.ndarray) -> Optional[Tuple[int, int, int]]:
        p = (np.asarray(point, dtype=float) - self.origin) / self.resolution
        idx = np.floor(p).astype(int)
        x, y, z = idx
        if (
            0 <= x < self.grid.shape[0]
            and 0 <= y < self.grid.shape[1]
            and 0 <= z < self.grid.shape[2]
        ):
            return int(x), int(y), int(z)
        return None

    def check_occupancy(self, point: np.ndarray) -> bool:
        idx = self.world_to_grid(point)
        if idx is None:
            # 边界外统一视为占据，防止采样到地图外
            return True
        return bool(self.grid[idx])

    def ray_intersection(
        self,
        sensor_pose: np.ndarray,
        directions: np.ndarray,
        max_range: float,
        step: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用简单的步进算法近似 rayIntersection。
        directions: (N, 3)
        返回：
            intersection_points: (N, 3)
            is_occupied: (N,) bool
        """
        sensor_pos = np.asarray(sensor_pose[:3], dtype=float)
        directions = np.asarray(directions, dtype=float)
        num_rays = directions.shape[0]
        intersection_points = np.zeros((num_rays, 3), dtype=float)
        is_occupied = np.zeros(num_rays, dtype=bool)

        for i in range(num_rays):
            d = directions[i]
            if np.linalg.norm(d) == 0:
                d = np.array([1.0, 0.0, 0.0], dtype=float)
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
                # 没有击中，则给射线末端
                intersection_points[i] = sensor_pos + max_range * d
                is_occupied[i] = False

        return intersection_points, is_occupied


# ================================================================
# Node / Tree 数据结构（对应 MATLAB 里的 T.v）
# ================================================================

@dataclass
class Node:
    x: float
    y: float
    z: float
    x_parent: float
    y_parent: float
    z_parent: float
    dist: float          # 从起点到该节点沿树的累计距离
    parent_index: Optional[int]   # 父节点在 nodes 列表中的索引，根节点为 None
    radius: float
    vol: float           # 累计“体积代价”


@dataclass
class TubeTree:
    nodes: List[Node]
    min_dis: float       # 起点到终点的直线距离
    max_radius: float
    min_radius: float
    use_int_vol: bool
    cost_int_vol: float  # k_i


# ================================================================
# 一些几何 & 体积相关函数（对应 intersectVolume / scoreCorridor）
# ================================================================

def intersect_volume(x_near: np.ndarray,
                     x_near_radius: float,
                     x_new: np.ndarray,
                     x_new_radius: float) -> float:
    """
    对应 intersectVolume.m
    计算两个球的交体积（两球相交的“镜片”体积之和）
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

    # 若完全分离或包含，仍按原公式计算（与 MATLAB 保持形式近似）
    cos_alpha = (r ** 2 + d ** 2 - R ** 2) / (2.0 * r * d)
    cos_beta = (R ** 2 + d ** 2 - r ** 2) / (2.0 * R * d)

    # 数值上做一下截断，避免 acos 域外
    cos_alpha = max(min(cos_alpha, 1.0), -1.0)
    cos_beta = max(min(cos_beta, 1.0), -1.0)

    h2 = r - r * cos_alpha
    h1 = R - R * cos_beta

    inter_vol = math.pi * h1 ** 2 * (R - (1.0 / 3.0) * h1) \
        + math.pi * h2 ** 2 * (r - (1.0 / 3.0) * h2)
    return float(inter_vol)


def score_corridor(last_pose: np.ndarray,
                   last_radius: float,
                   now_pose: np.ndarray,
                   now_radius: float):
    """
    对应 scoreCorridor.m
    输入：
        last.pose, last.radius
        now.pose, now.radius
    返回：
        score, tube（中间管道几何信息）
    """
    last_pose = np.asarray(last_pose, dtype=float)
    now_pose = np.asarray(now_pose, dtype=float)
    r = float(last_radius)
    R = float(now_radius)

    V_vol = 4.0 / 3.0 * math.pi * R ** 3
    rho_v = 1.0
    d = np.linalg.norm(last_pose - now_pose)

    if d <= 1e-8:
        # 两个球心重合，视为退化情况
        score = 0.0
        tube = {
            "direction": np.array([1.0, 0.0, 0.0], dtype=float),
            "pose": last_pose.copy(),
            "radius": r,
            "forward_radius": R,
            "normal_vect": np.array([0.0, 1.0, 0.0]),
            "abnormal_vect": np.array([0.0, 0.0, 1.0]),
        }
        return score, tube

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

    # 构造法向 & 副法向
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    normal_vect = np.cross(direction, z_axis)
    if np.linalg.norm(normal_vect) < 1e-8:
        # 如果方向接近 z 轴，换一个基向量
        z_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        normal_vect = np.cross(direction, z_axis)
    normal_vect = normal_vect / np.linalg.norm(normal_vect)
    abnormal_vect = np.cross(direction, normal_vect)
    abnormal_vect = abnormal_vect / np.linalg.norm(abnormal_vect)

    tube = {
        "direction": direction,
        "pose": tube_pose,
        "radius": tube_radius,
        "forward_radius": tube_forward_radius,
        "normal_vect": normal_vect,
        "abnormal_vect": abnormal_vect,
    }

    # 与 MATLAB 一样，当 d/(r+R) > 0.9 时 score = 0
    if d / (r + R) > 0.9:
        score = 0.0
    else:
        score = rho_v * V_vol + rho_i * V_int

    return float(score), tube


def gen_tube_path2(path: np.ndarray):
    """
    对应 genTubePath2.m 的 Python 版
    输入：
        path: (N, 4) 数组，每行为 [x, y, z, radius]
    输出：
        tube_paths: dict，包含 5 条曲线：
            tube_paths[0..3]: 四条边界曲线 (M_i, 3)
            tube_paths[4]: 管道中心曲线 (M, 3)
    """
    path = np.asarray(path, dtype=float)
    n = path.shape[0]
    tube_paths = {i: [] for i in range(5)}

    if n < 2:
        return {i: np.empty((0, 3)) for i in range(5)}

    for k in range(1, n):  # MATLAB 从2开始，这里用0-based => 1..n-1
        last_pose = path[k - 1, :3]
        last_radius = path[k - 1, 3]
        now_pose = path[k, :3]
        now_radius = path[k, 3]

        # 计算中间管道点
        _, tube = score_corridor(last_pose, last_radius, now_pose, now_radius)

        # 中线
        tube_paths[4].append(tube["pose"])

        # 四条边界曲线上的点
        p = tube["pose"]
        R_tube = tube["radius"]
        dir_vec = tube["direction"]
        n_vec = tube["normal_vect"]
        a_vec = tube["abnormal_vect"]
        forward_R = tube["forward_radius"]

        # 第一条：+ normal
        extreme_1 = p + R_tube * n_vec
        tube_paths[0].append(extreme_1)

        # 第二条：- normal
        extreme_2 = p - R_tube * n_vec
        tube_paths[1].append(extreme_2)

        # 第三条：+ abnormal
        extreme_3 = p + R_tube * a_vec
        tube_paths[2].append(extreme_3)

        # 第四条：沿 direction 推进 forwardRadius
        extreme_4 = p + dir_vec * forward_R
        tube_paths[3].append(extreme_4)

    # 转成 numpy 数组
    for i in range(5):
        if tube_paths[i]:
            tube_paths[i] = np.vstack(tube_paths[i])
        else:
            tube_paths[i] = np.empty((0, 3))

    return tube_paths


# ================================================================
# RRT* 核心：Nearest / NearTube / tubeSteer / AddNodeTube / ...
# ================================================================

def nearest_node(tree: TubeTree, x_rand: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    对应 Nearest.m
    返回：最近节点索引、该节点坐标
    """
    x_rand = np.asarray(x_rand, dtype=float)
    min_dis = float("inf")
    near_idx = 0
    for idx, node in enumerate(tree.nodes):
        d = np.linalg.norm(
            np.array([node.x, node.y, node.z], dtype=float) - x_rand
        )
        if d < min_dis:
            min_dis = d
            near_idx = idx
    near_node = np.array(
        [tree.nodes[near_idx].x, tree.nodes[near_idx].y, tree.nodes[near_idx].z],
        dtype=float,
    )
    return near_idx, near_node


def near_tube(
    tree: TubeTree,
    x_new: np.ndarray,
    x_new_radius: float,
    near_idx: int,
    map3d: OccupancyMap3D,
) -> List[int]:
    """
    对应 NearTube.m
    找到所有与 x_new 的球有交叠且线段无碰撞的节点索引。
    """
    x_new = np.asarray(x_new, dtype=float)
    near_nodes = [near_idx]
    for idx, node in enumerate(tree.nodes):
        if idx == near_idx:
            continue
        x_near = np.array([node.x, node.y, node.z], dtype=float)
        near_radius = node.radius
        d = np.linalg.norm(x_near - x_new)
        if d < (near_radius + x_new_radius) * 0.8:
            # 需要检查线段是否碰撞
            if not collision_checking(x_near, x_new, map3d):
                near_nodes.append(idx)
    return near_nodes


def add_node_tube(
    tree: TubeTree,
    x_new: np.ndarray,
    parent_idx: int,
    radius: float,
) -> int:
    """
    对应 AddNodeTube.m
    把 x_new 节点加入树，父节点为 parent_idx。
    返回新节点的索引。
    """
    x_new = np.asarray(x_new, dtype=float)
    parent = tree.nodes[parent_idx]
    near_node = np.array([parent.x, parent.y, parent.z], dtype=float)

    if tree.use_int_vol:
        k_i = tree.cost_int_vol
    else:
        k_i = 0.0

    inter_vol = intersect_volume(near_node, parent.radius, x_new, radius)
    # MATLAB: 0*exp(-4/3*pi*(radius/T.maxRadius)^3) + k_i*exp(-10*interVol/(4/3*pi*T.maxRadius^3)+1) + parent.vol
    # 第一个项恒为 0
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
        z=float(x_new[2]),
        x_parent=parent.x,
        y_parent=parent.y,
        z_parent=parent.z,
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
    对应 ChooseParentTube.m
    在邻域集合中选取使总代价最小的父节点。
    返回：父节点索引、其坐标。
    """
    x_new = np.asarray(x_new, dtype=float)
    min_cost = float("inf")
    best_idx = near_indices[0]

    for idx in near_indices:
        node = tree.nodes[idx]
        nearest = np.array([node.x, node.y, node.z], dtype=float)

        # 距离代价
        cost_dis = np.linalg.norm(nearest - x_new) + node.dist

        # 体积 & 交体积代价
        k_v = 0.0
        if tree.use_int_vol:
            k_i = tree.cost_int_vol
        else:
            k_i = 0.0

        inter_vol = intersect_volume(
            nearest, node.radius, x_new, x_new_radius
        )

        # MATLAB 版：
        # costVolume = k_v*exp(-4/3*pi*(x_newRadius/T.maxRadius)^3) + ...
        #        k_i*(interVol/(4/3*pi*T.maxRadius^3)+1)^(-1) + ...
        #        T.v(X_nears(i)).vol;
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
    best_pos = np.array([best_node.x, best_node.y, best_node.z], dtype=float)
    return best_idx, best_pos


def rewire_tube(
    tree: TubeTree,
    near_indices: List[int],
    new_idx: int,
    x_new: np.ndarray,
    map3d: OccupancyMap3D,
) -> None:
    """
    对应 rewireTube.m
    如果通过新节点 x_new 能让邻居节点总代价更小，则进行重连。
    """
    x_new = np.asarray(x_new, dtype=float)
    new_node = tree.nodes[new_idx]

    for idx in near_indices:
        node = tree.nodes[idx]

        # 不处理它自己的父节点
        if node.parent_index == new_idx:
            continue

        pre_cost_dis = node.dist
        pre_cost_volume = node.vol
        pre_cost = pre_cost_dis / tree.min_dis + pre_cost_volume

        near_node_pos = np.array([node.x, node.y, node.z], dtype=float)

        tentative_cost_dis = np.linalg.norm(near_node_pos - x_new) + new_node.dist

        k_v = 0.0
        if tree.use_int_vol:
            k_i = tree.cost_int_vol
        else:
            k_i = 0.0

        inter_vol = intersect_volume(
            near_node_pos, node.radius, x_new, new_node.radius
        )

        # MATLAB:
        # tentative_costVolume = k_v * exp(-4/3*pi*(nearRadius/T.maxRadius)^3)
        #       + k_i * 1/(interVol/(4/3*pi*T.maxRadius^3)+1) + T.v(new_idx).vol;
        tentative_cost_volume = (
            k_v
            * math.exp(
                -4.0
                / 3.0
                * math.pi
                * (node.radius / tree.max_radius) ** 3
            )
            + k_i * 1.0 / (inter_vol / (4.0 / 3.0 * math.pi * tree.max_radius ** 3) + 1.0)
            + new_node.vol
        )

        tentative_cost = tentative_cost_dis / tree.min_dis + tentative_cost_volume

        # rewire 过程中也要碰撞检测
        if not collision_checking(near_node_pos, x_new, map3d):
            if tentative_cost < pre_cost:
                # 更新父节点信息
                node.x_parent = float(x_new[0])
                node.y_parent = float(x_new[1])
                node.z_parent = float(x_new[2])
                node.dist = float(tentative_cost_dis)
                node.parent_index = new_idx
                node.vol = float(tentative_cost_volume)


# ================================================================
# 与地图交互：sampleFree / collisionChecking / findNearestPoint
# ================================================================

def sample_free(
    map3d: OccupancyMap3D,
    bounds: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """
    对应 sampleFree.m
    在指定包围盒中随机采样直到落在空闲栅格里：
        x_rand = rand(1,3).*bounds + offset
    """
    bounds = np.asarray(bounds, dtype=float)
    offset = np.asarray(offset, dtype=float)
    while True:
        x_rand = np.random.rand(3) * bounds + offset
        if not map3d.check_occupancy(x_rand):
            return x_rand


def collision_checking(
    start_pose: np.ndarray,
    goal_pose: np.ndarray,
    map3d: OccupancyMap3D,
    step: float = 0.2,
) -> bool:
    """
    对应 collisionChecking.m

    函数返回：
        True  => 有碰撞（不“feasible”）
        False => 无碰撞（可以通过）

    （与 MATLAB 中在其它地方用法保持一致：if ~collisionChecking(...)）
    """
    start_pose = np.asarray(start_pose, dtype=float)
    goal_pose = np.asarray(goal_pose, dtype=float)
    d = np.linalg.norm(goal_pose - start_pose)
    if d <= 1e-8:
        return False

    direction = (goal_pose - start_pose) / d
    t = 0.0
    while t <= d:
        p = start_pose + t * direction
        if map3d.check_occupancy(p):
            return True
        t += step
    return False


def find_nearest_point(
    map3d: OccupancyMap3D,
    point: np.ndarray,
    num_rays: int = 30,
    max_range: float = 100.0,
) -> Tuple[float, np.ndarray]:
    """
    对应 findNearestPoint.m
    通过多方向射线，寻找从 point 出发最靠近的障碍点及其距离。
    """
    point = np.asarray(point, dtype=float)
    angles = np.linspace(-math.pi, math.pi, num_rays)

    intersection_points_list = []

    # 原 MATLAB: for alpha=-pi/2:0.1:pi/2
    alphas = np.arange(-math.pi / 2.0, math.pi / 2.0 + 1e-6, 0.1)
    for alpha in alphas:
        # directions: [cos(alpha)*cos(angles); cos(alpha)*sin(angles); sin(alpha)*ones]^T
        dirs = np.vstack(
            [
                np.cos(alpha) * np.cos(angles),
                np.cos(alpha) * np.sin(angles),
                np.sin(alpha) * np.ones_like(angles),
            ]
        ).T  # (num_rays, 3)

        sensor_pose = np.hstack([point, [1, 0, 0, 0]])  # 旋转忽略
        inter_points, is_occ = map3d.ray_intersection(sensor_pose, dirs, max_range)
        # 只保留真正击中障碍的点
        if np.any(is_occ):
            intersection_points_list.append(inter_points[is_occ])

    if not intersection_points_list:
        # 没有任何击中障碍：视为“无障碍”，返回一个大距离
        return max_range, point + np.array([max_range, 0.0, 0.0])

    intersection_points = np.vstack(intersection_points_list)
    diffs = intersection_points - point
    dists = np.linalg.norm(diffs, axis=1)

    idx = int(np.argmin(dists))
    min_dis = float(dists[idx])
    min_point = intersection_points[idx]
    return min_dis, min_point


def tube_steer(
    tree: TubeTree,
    near_idx: int,
    x_new: np.ndarray,
    map3d: OccupancyMap3D,
) -> Tuple[np.ndarray, float]:
    """
    对应 tubeSteer.m
    沿着 near -> x_new 方向，找到一个点，使得以该点为球心的最大无碰撞球
    与 near 节点的球有足够交叠。
    返回：
        x_sphere: 新节点位置
        x_new_radius: 新节点对应球的半径
    """
    x_new = np.asarray(x_new, dtype=float)
    x_near = np.array(
        [
            tree.nodes[near_idx].x,
            tree.nodes[near_idx].y,
            tree.nodes[near_idx].z,
        ],
        dtype=float,
    )
    x_sphere = x_new.copy()
    x_radius = tree.nodes[near_idx].radius
    dis = np.linalg.norm(x_near - x_new)

    if dis <= 1e-8:
        # 与 near 重合，就直接在此处计算半径
        x_new_radius, _ = find_nearest_point(map3d, x_sphere)
        x_new_radius = min(x_new_radius, tree.max_radius)
        return x_sphere, x_new_radius

    direction = (x_new - x_near) / dis
    find_flag = True

    while find_flag:
        # 求出 x_sphere 处的最大无碰撞球半径
        x_new_radius, _ = find_nearest_point(map3d, x_sphere)
        x_new_radius = min(x_new_radius, tree.max_radius)

        # 如果两个邻接球的半径不足以相交，则把 x_sphere 往 near 靠近
        if max(x_new_radius, x_radius) < dis:
            dis = max(x_new_radius, x_radius)
            x_sphere = x_near + dis * direction
        else:
            find_flag = False

    return x_sphere, x_new_radius


# ================================================================
# 搜索路径 / 主入口：plan_tube_rrt_star
# ================================================================

def search_path(tree: TubeTree, end_idx: int) -> np.ndarray:
    """
    对应 searchPath.m
    从终点节点往回追溯到根，生成 [x, y, z, radius] 的路径。
    """
    path = []
    idx = end_idx
    while True:
        node = tree.nodes[idx]
        if node.parent_index is None:
            break
        path.append([node.x, node.y, node.z, node.radius])
        idx = node.parent_index
    path = path[::-1]  # 反转，使起点在前
    return np.array(path, dtype=float) if path else np.empty((0, 4))


def plan_tube_rrt_star(
    start_pose: np.ndarray,
    goal_pose: np.ndarray,
    map3d: OccupancyMap3D,
    setting: dict,
) -> Tuple[Optional[np.ndarray], TubeTree]:
    """
    Python 版 planTubeRRTStar.m

    参数：
        start_pose: (3,) 起点 [x,y,z]
        goal_pose:  (3,) 终点 [x,y,z]
        map3d: OccupancyMap3D 实例
        setting: dict, 对应 MATLAB 中的 setting 结构体，包含：
            - useIntVol (bool)
            - costIntVol (float)
            - GoalBias (float)
            - ContinueAfterGoalReached (bool)
            - MaxNumTreeNodes (int)  # 目前未严格使用，可用于提前终止
            - MaxConnectionDistance (float)
            - xLim, yLim, zLim: [min, max]

    返回：
        path: (N,4) [x,y,z,radius] 或 None（未找到）
        tree: TubeTree
    """
    start_pose = np.asarray(start_pose, dtype=float)
    goal_pose = np.asarray(goal_pose, dtype=float)

    # 通过 find_nearest_point 计算起点附近最大球半径
    min_dis, _ = find_nearest_point(map3d, start_pose)
    min_dis = min(min_dis, 15.0)  # 与 MATLAB 保持: max radius 限制

    # 初始化树
    # 注意：MATLAB 中 T.v(1).zParent = startPose(2) 应该是个笔误，这里修正为 startPose(3)
    root = Node(
        x=float(start_pose[0]),
        y=float(start_pose[1]),
        z=float(start_pose[2]),
        x_parent=float(start_pose[0]),
        y_parent=float(start_pose[1]),
        z_parent=float(start_pose[2]),
        dist=0.0,
        parent_index=None,
        radius=float(min_dis),
        vol=0.0,  # 0*exp(...) 恒为 0
    )

    tree = TubeTree(
        nodes=[root],
        min_dis=float(np.linalg.norm(start_pose - goal_pose)),
        max_radius=15.0,
        min_radius=3.0,
        use_int_vol=bool(setting.get("useIntVol", False)),
        cost_int_vol=float(setting.get("costIntVol", 0.0)),
    )

    # Planner 参数
    goal_bias = float(setting.get("GoalBias", 1.0))
    max_iterations = int(setting.get("MaxIterations", 2000))
    continue_after_goal_reached = bool(setting.get("ContinueAfterGoalReached", False))
    max_num_tree_nodes = int(setting.get("MaxNumTreeNodes", 10_000))
    delta = float(setting.get("MaxConnectionDistance", 5.0))

    x_lim = np.asarray(setting["xLim"], dtype=float)
    y_lim = np.asarray(setting["yLim"], dtype=float)
    z_lim = np.asarray(setting["zLim"], dtype=float)

    bounds = np.array(
        [
            x_lim[1] - x_lim[0],
            y_lim[1] - y_lim[0],
            z_lim[1] - z_lim[0],
        ],
        dtype=float,
    )
    offset = np.array([x_lim[0], y_lim[0], z_lim[0]], dtype=float)

    found = False
    end_idx = 0
    path = None

    for it in range(max_iterations):
        if len(tree.nodes) >= max_num_tree_nodes:
            break

        # Step 1: sample a free point
        x_rand = sample_free(map3d, bounds, offset)

        # Step 2: find nearest in tree
        near_idx, x_nearest = nearest_node(tree, x_rand)

        # Step 3: tubeSteer 得到新节点及其球半径
        x_new, x_new_radius = tube_steer(tree, near_idx, x_rand, map3d)
        if x_new_radius < tree.min_radius:
            continue

        # 碰撞检测
        if collision_checking(x_nearest, x_new, map3d):
            continue

        # 找到邻域节点集合（与 x_new 的球相交且连线无碰撞）
        near_indices = near_tube(tree, x_new, x_new_radius, near_idx, map3d)

        # 选择父节点
        parent_idx, x_min = choose_parent_tube(tree, near_indices, x_new, x_new_radius)

        # 将新节点加入树
        new_idx = add_node_tube(tree, x_new, parent_idx, x_new_radius)

        # RRT* 重连
        rewire_tube(tree, near_indices, new_idx, x_new, map3d)

        # Step 5: 检查是否到达目标附近
        d_goal = np.linalg.norm(x_new - goal_pose)
        if not found:
            if d_goal < min(x_new_radius, goal_bias):
                found = True
                end_idx = new_idx
        else:
            # 已经找到路径了：更新 path，并视 ContinueAfterGoalReached 决定是否继续
            path = search_path(tree, end_idx)
            if not continue_after_goal_reached:
                break

    if found and path is None:
        path = search_path(tree, end_idx)

    return path, tree
