import numpy as np
from tube_rrt_star import SimpleOccupancyMap3D, plan_tube_rrt_star

# 1. 构造一个简单的 3D 栅格地图（这里只是示例，全空）
grid = np.zeros((50, 50, 20), dtype=bool)  # 50x50x20，全 0 表示无障碍
resolution = 0.5
origin = np.array([0.0, 0.0, 0.0])
map3d = SimpleOccupancyMap3D(grid, resolution, origin)

# 2. 设置起终点
start = np.array([1.0, 1.0, 1.0])
goal  = np.array([20.0, 20.0, 5.0])

# 3. 设置参数（对应 MATLAB setting）
setting = {
    "useIntVol": True,
    "costIntVol": 1.0,
    "GoalBias": 3.0,
    "ContinueAfterGoalReached": False,
    "MaxNumTreeNodes": 2000,
    "MaxIterations": 2000,
    "MaxConnectionDistance": 5.0,
    "xLim": [0.0, 25.0],
    "yLim": [0.0, 25.0],
    "zLim": [0.0, 10.0],
}

path, tree = plan_tube_rrt_star(start, goal, map3d, setting)

if path is not None:
    print("找到路径，节点数：", path.shape[0])
else:
    print("未找到可行路径")
