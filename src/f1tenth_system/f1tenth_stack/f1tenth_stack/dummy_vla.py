import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped


class DummyVLA(Node):
    def __init__(self):
        super().__init__('dummy_vla')
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('speed', 0.2)
        self.declare_parameter('steering_angle', 0.0)

        image_topic = self.get_parameter('image_topic').value
        drive_topic = self.get_parameter('drive_topic').value
        self.speed = float(self.get_parameter('speed').value)
        self.steering = float(self.get_parameter('steering_angle').value)

        self.last_image_time = None
        self.pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.create_subscription(Image, image_topic, self._on_image, 10)
        self.create_timer(0.2, self._tick)  # 5 Hz
        self.get_logger().info(f'dummy_vla: {image_topic} -> {drive_topic}')

    def _on_image(self, msg: Image):
        self.last_image_time = self.get_clock().now()

    def _tick(self):
        # no image in the last second → publish nothing (mux will timeout)
        if self.last_image_time is None:
            return
        age = (self.get_clock().now() - self.last_image_time).nanoseconds / 1e9
        if age > 1.0:
            return

        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.drive.speed = self.speed
        out.drive.steering_angle = self.steering
        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(DummyVLA())
    rclpy.shutdown()


if __name__ == '__main__':
    main()