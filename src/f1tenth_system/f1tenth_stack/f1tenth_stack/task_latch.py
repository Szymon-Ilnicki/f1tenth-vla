#!/usr/bin/env python3
# task_latch.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TaskLatch(Node):
    def __init__(self):
        super().__init__('task_latch')
        self.task = "blank task"   # default until overriden
        self.sub = self.create_subscription(
            String, '/task_in', self.cb, 10)
        self.pub = self.create_publisher(String, '/task', 10)
        self.timer = self.create_timer(1.0, self.publish)  # 1 Hz
        self.get_logger().info(f'Task latch ready. Default: "{self.task}"')

    def cb(self, msg):
        if msg.data and msg.data != self.task:
            self.task = msg.data
            self.get_logger().info(f'New task latched: "{self.task}"')

    def publish(self):
        msg = String()
        msg.data = self.task
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = TaskLatch()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()