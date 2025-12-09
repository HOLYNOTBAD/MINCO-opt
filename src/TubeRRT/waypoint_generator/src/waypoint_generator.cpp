#include <iostream>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Vector3.h>
#include <nav_msgs/Path.h>
#include "sample_waypoints.h"
#include <vector>
#include <deque>
#include <boost/format.hpp>
#include <eigen3/Eigen/Dense>

using namespace std;
using bfmt = boost::format;

// ==============================================================
// waypoint_generator.cpp
// 功能：根据不同模式（manual/circle/eight/point/series/...）生成并发布航点（waypoints）。
// 说明：
//  - 订阅：odom, goal, traj_start_trigger（节点私有命名空间下的 topic）
//  - 发布：waypoints (nav_msgs::Path), waypoints_vis (geometry_msgs::PoseArray)
//  - 支持从参数服务器读取一系列 segment（series 模式），并按时间戳由 odom 驱动逐段发布。
//  - 部分生成函数（circle/point/eight）来自 "sample_waypoints.h"。
// 注意事项（阅读时参考）：
//  - 时间基准：代码中有时使用 ros::Time::now()，有时使用 odom.header.stamp 作为触发基准，可能导致不同步。
//  - is_odom_ready 在文件顶部声明但未显式初始化（下面会有注释提示）。
// ==============================================================

ros::Publisher pub1; // 发布主航点：nav_msgs::Path，topic 名称为 "waypoints"
ros::Publisher pub2; // 发布可视化航点：geometry_msgs::PoseArray，topic 名称为 "waypoints_vis"
ros::Publisher pub3; // 预留 publisher，目前未使用
string waypoint_type = string("manual"); // 模式标识，默认 "manual"
bool is_odom_ready, repeat_flag = false; // 注意：is_odom_ready 未显式初始化，运行时建议初始化为 false
nav_msgs::Odometry odom; // 保存最新的里程计消息（用于位置/朝向基准）
nav_msgs::Path waypoints; // 当前准备发布或展示的航点序列

// series（分段）模式使用的队列：按时间顺序保存多个 Path，等待 odom 驱动发布
std::deque<nav_msgs::Path> waypointSegments;
ros::Time trigged_time; // 触发时间基准（load 时作为 time_base）

// 初始化目标（用于 fix / repeat 模式）
double goal_x, goal_y, goal_z;
int repeat_num = 0;

void load_seg(ros::NodeHandle& nh, int segid, const ros::Time& time_base) {
    // 从参数服务器读取单个 segment 的定义并生成对应的 nav_msgs::Path
    // 参数格式（私有命名空间）示例：
    // seg0/yaw, seg0/time_from_start, seg0/x [array], seg0/y [array], seg0/z [array]
    // time_base: 触发时间基准（通常由 trigged_time 传入），函数会把 time_from_start 加到 time_base 上作为 path.header.stamp
    std::string seg_str = boost::str(bfmt("seg%d/") % segid);
    double yaw;
    double time_from_start;
    ROS_INFO("Getting segment %d", segid);
    ROS_ASSERT(nh.getParam(seg_str + "yaw", yaw));
    ROS_ASSERT_MSG((yaw > -3.1499999) && (yaw < 3.14999999), "yaw=%.3f", yaw);
    ROS_ASSERT(nh.getParam(seg_str + "time_from_start", time_from_start));
    ROS_ASSERT(time_from_start >= 0.0);

    std::vector<double> ptx;
    std::vector<double> pty;
    std::vector<double> ptz;

    ROS_ASSERT(nh.getParam(seg_str + "x", ptx));
    ROS_ASSERT(nh.getParam(seg_str + "y", pty));
    ROS_ASSERT(nh.getParam(seg_str + "z", ptz));

    ROS_ASSERT(ptx.size());
    ROS_ASSERT(ptx.size() == pty.size() && ptx.size() == ptz.size());

    nav_msgs::Path path_msg;

    path_msg.header.stamp = time_base + ros::Duration(time_from_start);

    double baseyaw = tf::getYaw(odom.pose.pose.orientation);
    // baseyaw: 当前里程计的朝向（作为 segment 在世界坐标系中的旋转基准）
    
    for (size_t k = 0; k < ptx.size(); ++k) {
        geometry_msgs::PoseStamped pt;
        pt.pose.orientation = tf::createQuaternionMsgFromYaw(baseyaw + yaw);
        Eigen::Vector2d dp(ptx.at(k), pty.at(k));
        Eigen::Vector2d rdp;
        // 把 segment 中的点（局部坐标 dp）绕 (-baseyaw - yaw) 旋转到世界坐标系中，再加上 odom 的位置偏移
        // 旋转矩阵来源：2D 逆时针旋转的标准变换（这里根据代码顺序写出）
        rdp.x() = std::cos(-baseyaw-yaw)*dp.x() + std::sin(-baseyaw-yaw)*dp.y();
        rdp.y() =-std::sin(-baseyaw-yaw)*dp.x() + std::cos(-baseyaw-yaw)*dp.y();
        // 最终世界坐标 = 旋转后的局部坐标 + odom 的世界位置
        pt.pose.position.x = rdp.x() + odom.pose.pose.position.x;
        pt.pose.position.y = rdp.y() + odom.pose.pose.position.y;
        pt.pose.position.z = ptz.at(k) + odom.pose.pose.position.z;
        path_msg.poses.push_back(pt);
    }

    waypointSegments.push_back(path_msg);
}

void load_waypoints(ros::NodeHandle& nh, const ros::Time& time_base) {
    // 读取 segment_cnt，然后依次调用 load_seg 构造 waypointSegments
    // 要求每个 segment 的 header.stamp 单调递增（函数使用 ROS_ASSERT 做检查）
    int seg_cnt = 0;
    waypointSegments.clear();
    ROS_ASSERT(nh.getParam("segment_cnt", seg_cnt));
    for (int i = 0; i < seg_cnt; ++i) {
        load_seg(nh, i, time_base);
        if (i > 0) {
            ROS_ASSERT(waypointSegments[i - 1].header.stamp < waypointSegments[i].header.stamp);
        }
    }
    ROS_INFO("Overall load %zu segments", waypointSegments.size());
}

void publish_waypoints() {
    // 发布当前的 waypoints（主话题）
    // 注意：原始实现先 publish，然后才把 init_pose 插入到 waypoints 并清空。
    //       这导致发布的消息不包含 init_pose，如果期望包含当前 odom 的初始位姿，
    //       应先插入 init_pose 再 publish（这里保持原逻辑，仅添加说明）。
    waypoints.header.frame_id = std::string("world");
    waypoints.header.stamp = ros::Time::now();
    pub1.publish(waypoints);
    // 将当前 odom 位姿作为 init_pose（未被发布到上面的 pub1，因为插入发生在 publish 之后）
    geometry_msgs::PoseStamped init_pose;
    init_pose.header = odom.header;
    init_pose.pose = odom.pose.pose;
    waypoints.poses.insert(waypoints.poses.begin(), init_pose);
    // 注意：这里清空了 waypoints.poses，调用者如需保存应在 publish 前备份。
    // pub2.publish(waypoints); // 注释掉的历史代码，保留以便未来切换格式
    waypoints.poses.clear();
}

void publish_waypoints_vis() {
    // 生成一个 PoseArray，用于 rviz 可视化：第一个元素为当前 odom 位姿，后续为 waypoints 中的每个点
    nav_msgs::Path wp_vis = waypoints;
    geometry_msgs::PoseArray poseArray;
    poseArray.header.frame_id = std::string("world");
    poseArray.header.stamp = ros::Time::now();

    // 首项为当前位置（odom）—— 便于在可视化中看到起点
    {
        geometry_msgs::Pose init_pose;
        init_pose = odom.pose.pose;
        poseArray.poses.push_back(init_pose);
    }

    // 将 waypoints 中的 PoseStamped 转换为 Pose 并追加
    for (auto it = waypoints.poses.begin(); it != waypoints.poses.end(); ++it) {
        geometry_msgs::Pose p;
        p = it->pose;
        poseArray.poses.push_back(p);
    }
    pub2.publish(poseArray);
}

void odom_callback(const nav_msgs::Odometry::ConstPtr& msg) {
    // 收到里程计消息后更新 odom，并在 series 模式下按时间顺序发布已加载的 segment
    is_odom_ready = true;
    odom = *msg;

    // 如果有预加载的 waypointSegments，则根据每段的 header.stamp 与 odom 时间戳比较决定是否发布
    if (waypointSegments.size()) {
        ros::Time expected_time = waypointSegments.front().header.stamp;
        // 注意：time_base（trigged_time）与 odom.header.stamp 的时间基准应保持一致，否则比较可能失效
        if (odom.header.stamp >= expected_time) {
            waypoints = waypointSegments.front();

            // 输出本次要发送的点，便于调试
            std::stringstream ss;
            ss << bfmt("Series send %.3f from start:\n") % trigged_time.toSec();
            for (auto& pose_stamped : waypoints.poses) {
                ss << bfmt("P[%.2f, %.2f, %.2f] q(%.2f,%.2f,%.2f,%.2f)") %
                          pose_stamped.pose.position.x % pose_stamped.pose.position.y %
                          pose_stamped.pose.position.z % pose_stamped.pose.orientation.w %
                          pose_stamped.pose.orientation.x % pose_stamped.pose.orientation.y %
                          pose_stamped.pose.orientation.z << std::endl;
            }
            ROS_INFO_STREAM(ss.str());

            // 可视化与正式发布
            publish_waypoints_vis();
            publish_waypoints();

            // 弹出已发布的 segment
            waypointSegments.pop_front();
        }
    }
}

void goal_callback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    // 当收到 goal 话题时，根据当前 waypoint_type 处理：
    //  - 如果是预设模式（circle/eight/point/fix），立即生成并发布
    //  - 如果是 series 模式，则调用 load_waypoints，将 segment 加入 waypointSegments（等待 odom 驱动按时间发布）
    //  - manual-lonely-waypoint: 仅接受 z >= 0 的单点并立即发布
    //  - 其它（manual / noyaw 等）：通过 z 字段的不同值进行交互（添加/撤销/结束）
/*    if (!is_odom_ready) {
        ROS_ERROR("[waypoint_generator] No odom!");
        return;
    }*/

    // 触发时间：使用系统时间作为基准（注意：与 odom.header.stamp 的时间基准可能不同）
    trigged_time = ros::Time::now(); //odom.header.stamp;
    //ROS_ASSERT(trigged_time > ros::Time(0));

    ros::NodeHandle n("~");
    n.param("waypoint_type", waypoint_type, string("manual"));
    
    if (waypoint_type == string("circle")) {
        waypoints = circle();
        publish_waypoints_vis();
        publish_waypoints();
    } 
    else if (waypoint_type == string("fix")) {
        // 使用参数中的初始化目标坐标作为单点发布
        waypoints.poses.clear();
        geometry_msgs::PoseStamped pt;
        pt.pose.position.x = goal_x;
        pt.pose.position.y = goal_y;
        pt.pose.position.z = goal_z;
        waypoints.poses.push_back(pt);
        publish_waypoints_vis();
        publish_waypoints();
    }else if (waypoint_type == string("eight")) {
        waypoints = eight();
        publish_waypoints_vis();
        publish_waypoints();
    } else if (waypoint_type == string("point")) {
        waypoints = point();
        publish_waypoints_vis();
        publish_waypoints();
    } else if (waypoint_type == string("series")) {
        // 将参数服务器中定义的多个 segment 加载到 waypointSegments
        load_waypoints(n, trigged_time);
    } else if (waypoint_type == string("manual-lonely-waypoint")) {
        if (msg->pose.position.z >= 0) {
            // if height >= 0, it's a valid goal;
            geometry_msgs::PoseStamped pt = *msg;
            waypoints.poses.clear();
            waypoints.poses.push_back(pt);
            publish_waypoints_vis();
            publish_waypoints();
        } else {
            ROS_WARN("[waypoint_generator] invalid goal in manual-lonely-waypoint mode.");
        }
    } else {
        // 交互式模式：通过 msg.pose.position.z 的不同值来添加、撤销或结束输入
        if (msg->pose.position.z > 0) {
            // if height > 0, it's a normal goal;
            geometry_msgs::PoseStamped pt = *msg;
            if (waypoint_type == string("noyaw")) {
                // 在 noyaw 模式下，使用当前 odom 的朝向覆盖目标的朝向
                double yaw = tf::getYaw(odom.pose.pose.orientation);
                pt.pose.orientation = tf::createQuaternionMsgFromYaw(yaw);
            }
            waypoints.poses.push_back(pt);
            publish_waypoints_vis();
        } else if (msg->pose.position.z > -1.0) {
            // if 0 > height > -1.0, remove last goal;
            if (waypoints.poses.size() >= 1) {
                waypoints.poses.erase(std::prev(waypoints.poses.end()));
            }
            publish_waypoints_vis();
        } else {
            // if -1.0 > height, end of input —— 将当前积累的 waypoints 一并发布
            if (waypoints.poses.size() >= 1) {
                publish_waypoints_vis();
                publish_waypoints();
            }
        }
    }
}

void traj_start_trigger_callback(const geometry_msgs::PoseStamped& msg) {
    // 手动触发轨迹开始（通常由外部控制器发送触发消息）
    if (!is_odom_ready) {
        ROS_ERROR("[waypoint_generator] No odom!");
        return;
    }

    ROS_WARN("[waypoint_generator] Trigger!");
    // 使用 odom 的时间戳作为触发基准，这样后续基于 odom 时间比较的逻辑更可靠
    trigged_time = odom.header.stamp;
    ROS_ASSERT(trigged_time > ros::Time(0));

    ros::NodeHandle n("~");
    n.param("waypoint_type", waypoint_type, string("manual"));

    ROS_ERROR_STREAM("Pattern " << waypoint_type << " generated!");
    if (waypoint_type == string("free")) {
        waypoints = point();
        publish_waypoints_vis();
        publish_waypoints();
    } else if (waypoint_type == string("circle")) {
        waypoints = circle();
        publish_waypoints_vis();
        publish_waypoints();
    } else if (waypoint_type == string("eight")) {
        waypoints = eight();
        publish_waypoints_vis();
        publish_waypoints();
   } else if (waypoint_type == string("point")) {
        waypoints = point();
        publish_waypoints_vis();
        publish_waypoints();
    } else if (waypoint_type == string("series")) {
        // 加载 series，此时会用 trigged_time 作为 time_base
        load_waypoints(n, trigged_time);
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "waypoint_generator");
    ros::NodeHandle n("~");
    n.param("waypoint_type", waypoint_type, string("manual"));
    n.param("init_goal_x", goal_x, 0.0);
    n.param("init_goal_y", goal_y, 0.0);
    n.param("init_goal_z", goal_z, 0.0);
    n.param("repeat", repeat_flag, false);
    ros::Subscriber sub1 = n.subscribe("odom", 10, odom_callback);
    ros::Subscriber sub2 = n.subscribe("goal", 10, goal_callback);
    ros::Subscriber sub3 = n.subscribe("traj_start_trigger", 10, traj_start_trigger_callback);
    pub1 = n.advertise<nav_msgs::Path>("waypoints", 50);
    pub2 = n.advertise<geometry_msgs::PoseArray>("waypoints_vis", 10);

    trigged_time = ros::Time(0);

    if (repeat_flag)
    {
        ros::Rate loop_rate(1);
        while (ros::ok())
        {
            waypoints.poses.clear();
            geometry_msgs::PoseStamped pt;
            pt.pose.position.x = goal_x;
            pt.pose.position.y = goal_y;
            pt.pose.position.z = goal_z;
            waypoints.poses.push_back(pt);
            publish_waypoints();
            // std::cout << "wap_gen: start pub goal point" << waypoints << std::endl;
            ros::spinOnce();
            loop_rate.sleep();
        }
    }
    else
    {
        ros::spin();
    }
   
    return 0;
}
