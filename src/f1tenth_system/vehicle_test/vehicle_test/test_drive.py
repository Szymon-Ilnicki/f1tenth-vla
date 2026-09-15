import time

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped


class VehicleTestNode(Node):
    def __init__(self):
        super().__init__("vehicle_test_node")
        self.publisher = self.create_publisher(
            AckermannDriveStamped,
            "/drive",
            10
        )
        self.get_logger().info("Vehicle test node started")

    def send_command(self, speed, steering, duration):
        msg = AckermannDriveStamped()
        msg.drive.speed = speed
        msg.drive.steering_angle = steering

        start = time.time()
        while time.time() - start < duration:
            self.publisher.publish(msg)
            time.sleep(0.05)  # ~20 Hz

    def run_test(self):
        self.get_logger().info("Forward slow")
        self.send_command(0.5, 0.0, 2.0)

        self.get_logger().info("Forward faster")
        self.send_command(1.5, 0.0, 2.0)

        self.get_logger().info("Left turn")
        self.send_command(1.0, 0.3, 2.0)

        self.get_logger().info("Right turn")
        self.send_command(1.0, -0.3, 2.0)

        self.get_logger().info("Stop")
        self.send_command(0.0, 0.0, 1.0)


def main(args=None):
    rclpy.init(args=args)
    node = VehicleTestNode()
    node.run_test()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
