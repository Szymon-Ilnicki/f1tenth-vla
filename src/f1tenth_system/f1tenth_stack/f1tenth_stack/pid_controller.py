#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

class PidTracker(Node):
    def __init__(self):
        super().__init__('pid_tracker')
        self.declare_parameter('kp', 0.8)
        self.declare_parameter('ki', 0.2)
        self.declare_parameter('kd', 0.0)
        self.declare_parameter('speed_max', 0.6)
        self.declare_parameter('steer_rate', 1.0)  # rad/s
        self.declare_parameter('i_limit', 0.4)

        self.kp = self.get_parameter('kp').value
        self.ki = self.get_parameter('ki').value
        self.kd = self.get_parameter('kd').value
        self.speed_max = self.get_parameter('speed_max').value
        self.steer_rate = self.get_parameter('steer_rate').value
        self.i_limit = self.get_parameter('i_limit').value

        self.target_speed = 0.0
        self.target_steer = 0.0
        self.measured_speed = 0.0
        self.cmd_steer = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
        self.last = self.get_clock().now()

        self.create_subscription(AckermannDriveStamped, '/pid/setpoint', self.on_setpoint, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.create_timer(0.1, self.tick)  # 10 Hz

    def on_setpoint(self, msg):
        self.target_speed = msg.drive.speed
        self.target_steer = msg.drive.steering_angle

    def on_odom(self, msg):
        self.measured_speed = msg.twist.twist.linear.x

    def tick(self):
        now = self.get_clock().now()
        dt = (now - self.last).nanoseconds * 1e-9
        self.last = now
        if dt <= 0.0 or dt > 0.5:
            return

        error = self.target_speed - self.measured_speed
        self.integral = max(-self.i_limit, min(self.i_limit, self.integral + error * dt))
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        speed = self.kp * error + self.ki * self.integral + self.kd * derivative
        speed = max(-self.speed_max, min(self.speed_max, speed))

        step = self.steer_rate * dt
        delta = max(-step, min(step, self.target_steer - self.cmd_steer))
        self.cmd_steer += delta

        out = AckermannDriveStamped()
        out.header.stamp = now.to_msg()
        out.header.frame_id = 'base_link'
        out.drive.speed = speed
        out.drive.steering_angle = self.cmd_steer
        self.pub.publish(out)

def main():
    rclpy.init()
    node = PidTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()