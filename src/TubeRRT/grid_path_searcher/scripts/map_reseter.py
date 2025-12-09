#!/usr/bin/env python3
"""
map_reseter.py

简单节点：向随机地图生成器的 reset_map 话题发布 std_msgs/Bool
用途：触发 `random_complex_scene` 节点内的地图重置逻辑。

默认话题：/random_complex_scene/reset_map

用法示例：
  # 发布一次 true（触发一次重置）
  rosrun grid_path_searcher map_reseter.py --once

  # 发布一次 true，然后在 1 秒后发布 false（脉冲）
  rosrun grid_path_searcher map_reseter.py --pulse --pulse-duration 1.0

  # 每隔 5 秒发送一次 true（循环触发）直到中断
  rosrun grid_path_searcher map_reseter.py --repeat --interval 5.0
"""
import rospy
from std_msgs.msg import Bool
import argparse
import time


def make_parser():
    p = argparse.ArgumentParser(description='Publish Bool to reset_map to trigger random map generation')
    p.add_argument('--topic', default='/random_complex/reset_map',
                   help='Full topic name to publish to (default: /random_complex_scene/reset_map)')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--once', action='store_true', help='Publish a single True and exit (default)')
    group.add_argument('--pulse', action='store_true', help='Publish True then False after pulse-duration seconds')
    group.add_argument('--repeat', action='store_true', help='Publish True repeatedly every --interval seconds')
    p.add_argument('--pulse-duration', type=float, default=1.0, help='Duration of pulse when using --pulse (seconds)')
    p.add_argument('--interval', type=float, default=5.0, help='Interval for --repeat mode (seconds)')
    return p


def main():
    parser = make_parser()
    args = parser.parse_args()

    rospy.init_node('map_reseter', anonymous=True)
    pub = rospy.Publisher(args.topic, Bool, queue_size=1, latch=True)

    # give publisher time to register
    rospy.sleep(0.2)

    if args.pulse:
        rospy.loginfo('Publishing pulse True -> (wait %.3f s) -> False to %s', args.pulse_duration, args.topic)
        pub.publish(Bool(data=True))
        rospy.sleep(args.pulse_duration)
        pub.publish(Bool(data=False))
        rospy.loginfo('Pulse finished')

    elif args.repeat:
        rospy.loginfo('Publishing True to %s every %.3f seconds. Ctrl-C to stop.', args.topic, args.interval)
        rate = rospy.Rate(1.0 / max(args.interval, 0.0001)) if args.interval > 0 else rospy.Rate(0.2)
        try:
            while not rospy.is_shutdown():
                pub.publish(Bool(data=True))
                rate.sleep()
        except rospy.ROSInterruptException:
            pass

    else:
        # default: once
        rospy.loginfo('Publishing single True to %s', args.topic)
        pub.publish(Bool(data=True))


if __name__ == '__main__':
    main()
