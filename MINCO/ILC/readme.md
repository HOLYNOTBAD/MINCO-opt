gen_map_tube.py: 生成障碍物地图与tuberrt规划结果，结果在map_manage文件夹下

gen_ref_traj.py: 根据tuberrt规划结果，生成一条顺滑的参考轨迹，保存在ref_traj文件夹下。生成算法使用几轮的MINCO轨迹生成

plot_ref_traj.py: 可以将ref_traj可视化

ILC_Tube.py: 用来进行有Tube时的IL仿真

ILC.py: 核心代码逻辑

* for i in range( i_n - 1 )
  * INITIALIZE
  * while l<l_max 且 j<j_n:
    * xp,yp,l = getpoint()
    * 终止条件：dist_end_sq<3.0 且 l>l_max-100
    * 计算沿切速度 value
    * 计算法向控制量
    * 计算切向控制量
    * 饱和
    * 动力学前向模拟
    * ILC迭代切向速度
    * 整理单轮信息：时间等
