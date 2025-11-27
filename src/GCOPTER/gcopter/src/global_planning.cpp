#include "misc/visualizer.hpp"
#include "gcopter/trajectory.hpp"
#include "gcopter/gcopter.hpp"
#include "gcopter/firi.hpp"
#include "gcopter/flatness.hpp"
#include "gcopter/voxel_map.hpp"
#include "gcopter/sfc_gen.hpp"

#include <ros/ros.h>
#include <ros/console.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <random>

struct Config // 配置各种参数 
{
    std::string mapTopic;
    std::string targetTopic;
    double dilateRadius;
    double voxelWidth;
    std::vector<double> mapBound;
    double timeoutRRT;
    double maxVelMag;
    double maxBdrMag;
    double maxTiltAngle;
    double minThrust;
    double maxThrust;
    double vehicleMass;
    double gravAcc;
    double horizDrag;
    double vertDrag;
    double parasDrag;
    double speedEps;
    double weightT;
    std::vector<double> chiVec;
    double smoothingEps;
    int integralIntervs;
    double relCostTol;

    Config(const ros::NodeHandle &nh_priv)
    {
        nh_priv.getParam("MapTopic", mapTopic);
        nh_priv.getParam("TargetTopic", targetTopic);
        nh_priv.getParam("DilateRadius", dilateRadius);
        nh_priv.getParam("VoxelWidth", voxelWidth);
        nh_priv.getParam("MapBound", mapBound);
        nh_priv.getParam("TimeoutRRT", timeoutRRT);
        nh_priv.getParam("MaxVelMag", maxVelMag);
        nh_priv.getParam("MaxBdrMag", maxBdrMag);
        nh_priv.getParam("MaxTiltAngle", maxTiltAngle);
        nh_priv.getParam("MinThrust", minThrust);
        nh_priv.getParam("MaxThrust", maxThrust);
        nh_priv.getParam("VehicleMass", vehicleMass);
        nh_priv.getParam("GravAcc", gravAcc);
        nh_priv.getParam("HorizDrag", horizDrag);
        nh_priv.getParam("VertDrag", vertDrag);
        nh_priv.getParam("ParasDrag", parasDrag);
        nh_priv.getParam("SpeedEps", speedEps);
        nh_priv.getParam("WeightT", weightT);
        nh_priv.getParam("ChiVec", chiVec);
        nh_priv.getParam("SmoothingEps", smoothingEps);
        nh_priv.getParam("IntegralIntervs", integralIntervs);
        nh_priv.getParam("RelCostTol", relCostTol);
    }
};

class GlobalPlanner
{
private:
    Config config;

    ros::NodeHandle nh;
    ros::Subscriber mapSub;
    ros::Subscriber targetSub;

    bool mapInitialized;
    voxel_map::VoxelMap voxelMap;
    Visualizer visualizer;
    std::vector<Eigen::Vector3d> startGoal;

    Trajectory<5> traj;
    double trajStamp;

public:
    GlobalPlanner(const Config &conf,
                  ros::NodeHandle &nh_)
        : config(conf),
          nh(nh_),
          mapInitialized(false),
          visualizer(nh)
    {
        const Eigen::Vector3i xyz((config.mapBound[1] - config.mapBound[0]) / config.voxelWidth,
                                  (config.mapBound[3] - config.mapBound[2]) / config.voxelWidth,
                                  (config.mapBound[5] - config.mapBound[4]) / config.voxelWidth);

        const Eigen::Vector3d offset(config.mapBound[0], config.mapBound[2], config.mapBound[4]);

        voxelMap = voxel_map::VoxelMap(xyz, offset, config.voxelWidth);

        mapSub = nh.subscribe(config.mapTopic, 1, &GlobalPlanner::mapCallBack, this,
                              ros::TransportHints().tcpNoDelay());

        targetSub = nh.subscribe(config.targetTopic, 1, &GlobalPlanner::targetCallBack, this,
                                 ros::TransportHints().tcpNoDelay());
    }

    inline void mapCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        if (!mapInitialized)
        {
            size_t cur = 0;
            const size_t total = msg->data.size() / msg->point_step;
            float *fdata = (float *)(&msg->data[0]);
            for (size_t i = 0; i < total; i++)
            {
                cur = msg->point_step / sizeof(float) * i;

                if (std::isnan(fdata[cur + 0]) || std::isinf(fdata[cur + 0]) ||
                    std::isnan(fdata[cur + 1]) || std::isinf(fdata[cur + 1]) ||
                    std::isnan(fdata[cur + 2]) || std::isinf(fdata[cur + 2]))
                {
                    continue;
                }
                voxelMap.setOccupied(Eigen::Vector3d(fdata[cur + 0],
                                                     fdata[cur + 1],
                                                     fdata[cur + 2]));
            }

            voxelMap.dilate(std::ceil(config.dilateRadius / voxelMap.getScale()));

            mapInitialized = true;
        }
    }

    inline void plan()
    {
        if (startGoal.size() == 2) // 只有当已有起点和终点时才做规划
        {
            std::vector<Eigen::Vector3d> route; // 存储规划得到的路径点序列
            // 计时：1) 全局路径规划（planPath）
            auto t_plan_start = std::chrono::steady_clock::now();
            sfc_gen::planPath<voxel_map::VoxelMap>(startGoal[0], // 起点
                                                   startGoal[1], // 终点
                                                   voxelMap.getOrigin(), // 地图原点（世界坐标系偏移）
                                                   voxelMap.getCorner(), // 地图角落（尺寸信息）
                                                   &voxelMap, 0.01, // 传入体素地图和规划分辨率/容差
                                                   route); // 输出：route（路径点）
            auto t_plan_end = std::chrono::steady_clock::now();
            double t_plan_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_plan_end - t_plan_start).count();
            std::vector<Eigen::MatrixX4d> hPolys; // 存储生成的凸多面体（每个多面体以若干四元数矩阵表示）
            std::vector<Eigen::Vector3d> pc; // 存放体素地图的表面点云
            voxelMap.getSurf(pc); // 获取体素地图表面点云到 pc

            // 计时：2) 生成飞行安全走廊（convexCover + shortCut）
            auto t_cover_start = std::chrono::steady_clock::now();
            sfc_gen::convexCover(route, // 基于路径点生成凸多面体覆盖（Safe Flight Corridor）
                                 pc, // 利用表面点云作为障碍信息
                                 voxelMap.getOrigin(), // 地图原点
                                 voxelMap.getCorner(), // 地图角落
                                 7.0, // 参数：可以是最大多面体尺寸或搜索半径（库内部定义）
                                 3.0, // 参数：可能是平滑或最小多面体尺寸（库内部定义）
                                 hPolys); // 输出：hPolys（多面体集合）
            sfc_gen::shortCut(hPolys); // 对多面体路径做短路优化，减少不必要的多面体数或路径长度
            auto t_cover_end = std::chrono::steady_clock::now();
            double t_cover_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_cover_end - t_cover_start).count();

            if (route.size() > 1) // 路径至少包含两点才有意义
            {
                visualizer.visualizePolytope(hPolys); // 可视化多面体集合

                Eigen::Matrix3d iniState; // 初始状态矩阵（位置、速度、加速度）
                Eigen::Matrix3d finState; // 目标状态矩阵（位置、速度、加速度）
                iniState << route.front(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(); // 将起点位置放入初始状态，速度/加速度置零
                finState << route.back(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(); // 将终点位置放入目标状态，速度/加速度置零

                gcopter::GCOPTER_PolytopeSFC gcopter; // 创建轨迹优化器实例（基于多面体约束）

                // magnitudeBounds = [v_max, omg_max, theta_max, thrust_min, thrust_max]^T
                // penaltyWeights = [pos_weight, vel_weight, omg_weight, theta_weight, thrust_weight]^T
                // physicalParams = [vehicle_mass, gravitational_acceleration, horitonral_drag_coeff,
                //                   vertical_drag_coeff, parasitic_drag_coeff, speed_smooth_factor]^T
                // initialize some constraint parameters
                Eigen::VectorXd magnitudeBounds(5); // 存放约束上下界
                Eigen::VectorXd penaltyWeights(5); // 存放代价函数的权重
                Eigen::VectorXd physicalParams(6); // 存放物理相关参数
                magnitudeBounds(0) = config.maxVelMag; // 最大速度
                magnitudeBounds(1) = config.maxBdrMag; // 最大角速度/体角速
                magnitudeBounds(2) = config.maxTiltAngle; // 最大倾角
                magnitudeBounds(3) = config.minThrust; // 推力下限
                magnitudeBounds(4) = config.maxThrust; // 推力上限
                penaltyWeights(0) = (config.chiVec)[0]; // 位置权重
                penaltyWeights(1) = (config.chiVec)[1]; // 速度权重
                penaltyWeights(2) = (config.chiVec)[2]; // 角速度权重
                penaltyWeights(3) = (config.chiVec)[3]; // 倾角权重
                penaltyWeights(4) = (config.chiVec)[4]; // 推力权重
                physicalParams(0) = config.vehicleMass; // 质量
                physicalParams(1) = config.gravAcc; // 重力加速度
                physicalParams(2) = config.horizDrag; // 水平阻力系数
                physicalParams(3) = config.vertDrag; // 垂直阻力系数
                physicalParams(4) = config.parasDrag; // 寄生阻力系数
                physicalParams(5) = config.speedEps; // 速度相关的数值平滑项
                const int quadratureRes = config.integralIntervs; // 积分/求积分辨率（用于数值积分）

                traj.clear(); // 清空当前轨迹，准备写入新的结果

                if (!gcopter.setup(config.weightT, // 使用权重参数初始化优化器
                                   iniState, finState, // 初始与目标状态
                                   hPolys, INFINITY, // 传入多面体约束与时间上限（这里为无限制）
                                   config.smoothingEps, // 平滑参数
                                   quadratureRes, // 积分分辨率
                                   magnitudeBounds, // 动力学约束
                                   penaltyWeights, // 代价权重
                                   physicalParams)) // 物理参数
                {
                    return; // setup 失败则退出规划
                }

                // 计时：3) MINCO + 无约束优化（gcopter.optimize）
                auto t_opt_start = std::chrono::steady_clock::now();
                double opt_cost = gcopter.optimize(traj, config.relCostTol); // 返回最终代价（或 INFINITY 表示失败）
                auto t_opt_end = std::chrono::steady_clock::now();
                double t_opt_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_opt_end - t_opt_start).count();
                int opt_iters = gcopter.getLastOptimizeIters(); // 从优化器回调获取迭代次数（若支持）

                if (std::isinf(opt_cost)) // 运行优化器并检查是否返回无穷（失败）
                {
                    ROS_WARN_STREAM("gcopter.optimize failed (cost=INF). planPath(ms): " << t_plan_ms << ", convexCover(ms): " << t_cover_ms << ", optimize(ms): " << t_opt_ms << ", iters: " << opt_iters);
                    return; // 优化失败则退出
                }
                else
                {
                    ROS_INFO_STREAM("Timing (ms) - planPath: " << t_plan_ms << ", convexCover: " << t_cover_ms << ", optimize: " << t_opt_ms << ", optimize_iters: " << opt_iters);
                }

                if (traj.getPieceNum() > 0) // 若优化成功并生成了轨迹片段
                {
                    trajStamp = ros::Time::now().toSec(); // 记录轨迹生成时间戳
                    visualizer.visualize(traj, route); // 可视化轨迹和路径点
                }
            }
        }
    }

    inline void targetCallBack(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        if (mapInitialized)
        {
            if (startGoal.size() >= 2)
            {
                startGoal.clear();
            }
            const double zGoal = config.mapBound[4] + config.dilateRadius +
                                 fabs(msg->pose.orientation.z) *
                                     (config.mapBound[5] - config.mapBound[4] - 2 * config.dilateRadius);
            const Eigen::Vector3d goal(msg->pose.position.x, msg->pose.position.y, zGoal);
            if (voxelMap.query(goal) == 0)
            {
                visualizer.visualizeStartGoal(goal, 0.5, startGoal.size());
                startGoal.emplace_back(goal);
            }
            else
            {
                ROS_WARN("Infeasible Position Selected !!!\n");
            }

            plan();
        }
        return;
    }

    inline void process()
    {
        Eigen::VectorXd physicalParams(6); // 创建长度为6的向量用于存放物理参数
        physicalParams(0) = config.vehicleMass; // 车辆质量
        physicalParams(1) = config.gravAcc; // 重力加速度
        physicalParams(2) = config.horizDrag; // 水平阻力系数
        physicalParams(3) = config.vertDrag; // 垂直阻力系数
        physicalParams(4) = config.parasDrag; // 寄生阻力系数
        physicalParams(5) = config.speedEps; // 速度平滑因子或数值稳健项

        flatness::FlatnessMap flatmap; // 创建平坦化映射对象，用于把轨迹状态映射到控制量
        flatmap.reset(physicalParams(0), physicalParams(1), physicalParams(2), // 使用物理参数初始化平坦化模型
                      physicalParams(3), physicalParams(4), physicalParams(5));

        if (traj.getPieceNum() > 0) // 如果当前存在已生成的轨迹片段
        {
            const double delta = ros::Time::now().toSec() - trajStamp; // 计算从轨迹开始（时间戳）到现在的时间偏移
            if (delta > 0.0 && delta < traj.getTotalDuration()) // 仅在轨迹总时长范围内处理
            {
                double thr; // 推力（输出变量）
                Eigen::Vector4d quat; // 姿态四元数（输出变量）
                Eigen::Vector3d omg; // 机体角速度（输出变量）

                flatmap.forward(traj.getVel(delta), // 将轨迹在当前时间的速度
                                traj.getAcc(delta), // 加速度
                                traj.getJer(delta), // 跳跃/跃度（jerk）
                                0.0, 0.0, // 占位参数（这里传0.0, 0.0）
                                thr, quat, omg); // 输出：推力、四元数和角速度
                double speed = traj.getVel(delta).norm(); // 计算当前速度大小
                double bodyratemag = omg.norm(); // 计算当前机体角速度的模
                double tiltangle = acos(1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2))); // 从四元数计算倾角（近似）
                std_msgs::Float64 speedMsg, thrMsg, tiltMsg, bdrMsg; // 准备发布消息
                speedMsg.data = speed; // 填充速度消息
                thrMsg.data = thr; // 填充推力消息
                tiltMsg.data = tiltangle; // 填充倾角消息
                bdrMsg.data = bodyratemag; // 填充角速度幅值消息
                visualizer.speedPub.publish(speedMsg); // 发布速度到可视化/诊断话题
                visualizer.thrPub.publish(thrMsg); // 发布推力到可视化/诊断话题
                visualizer.tiltPub.publish(tiltMsg); // 发布倾角到可视化/诊断话题
                visualizer.bdrPub.publish(bdrMsg); // 发布机体角速度到可视化/诊断话题

                visualizer.visualizeSphere(traj.getPos(delta), // 在当前轨迹位置画一个球（用于可视化当前位置和安全半径）
                                           config.dilateRadius); // 球的半径为膨胀半径
            }
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "global_planning_node");
    ros::NodeHandle nh_;

    GlobalPlanner global_planner(Config(ros::NodeHandle("~")), nh_);

    ros::Rate lr(1000);// lr的意思是loop rate，即循环频率，这里设置为1000Hz，即1秒循环1000次
    while (ros::ok())
    {
        global_planner.process();
        ros::spinOnce();
        lr.sleep();
    }

    return 0;
}
