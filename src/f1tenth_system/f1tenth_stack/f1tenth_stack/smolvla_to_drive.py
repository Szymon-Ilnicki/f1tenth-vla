#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from ackermann_msgs.msg import AckermannDriveStamped


class SmolVLAToDrive(Node):
    def __init__(self):
        super().__init__("smolvla_to_drive")
        self.declare_parameter("action_topic", "/smolvla/action")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("speed_min", 0.0)
        self.declare_parameter("speed_max", 1.0)
        self.declare_parameter("steer_max", 0.4)
        self.declare_parameter("stale_s", 0.5)

        self.speed_min = float(self.get_parameter("speed_min").value)
        self.speed_max = float(self.get_parameter("speed_max").value)
        self.steer_max = float(self.get_parameter("steer_max").value)
        self.stale_s = float(self.get_parameter("stale_s").value)
        self.last_t = self.get_clock().now()
        self.have = False

        self.pub = self.create_publisher(
            AckermannDriveStamped, self.get_parameter("drive_topic").value, 10
        )
        self.create_subscription(
            Float32MultiArray,
            self.get_parameter("action_topic").value,
            self._on_act,
            10,
        )
        self.create_timer(0.05, self._watchdog)
        self.get_logger().info(
            f"adapter {self.get_parameter('action_topic').value} -> "
            f"{self.get_parameter('drive_topic').value}"
        )

    def _clamp(self, speed, steer):
        speed = min(max(speed, self.speed_min), self.speed_max)
        steer = min(max(steer, -self.steer_max), self.steer_max)
        return speed, steer

    def _pub(self, speed, steer):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steer)
        self.pub.publish(msg)

    def _on_act(self, msg: Float32MultiArray):
        if len(msg.data) < 2:
            return
        speed, steer = self._clamp(float(msg.data[0]), float(msg.data[1]))
        self.have = True
        self.last_t = self.get_clock().now()
        self._pub(speed, steer)

    def _watchdog(self):
        if not self.have:
            return
        dt = (self.get_clock().now() - self.last_t).nanoseconds * 1e-9
        if dt > self.stale_s:
            self._pub(0.0, 0.0)


def main():
    rclpy.init()
    node = SmolVLAToDrive()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
