#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64

class PidTracker(Node):
    def __init__(self):
        super().__init__('pid_tracker')
        self.declare_parameter('kp', 0.8)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.0)
        self.declare_parameter('steering_angle_to_servo_gain', 1.0)
        self.declare_parameter('steering_angle_to_servo_offset', 0.5)
        self.declare_parameter('speed_to_erpm_gain', 1.0)
        self.declare_parameter('servo_min', 0.0)
        self.declare_parameter('servo_max', 1.0)
        self.declare_parameter('speed_min', -10000.0)
        self.declare_parameter('speed_max', 10000.0)
        self.declare_parameter('i_limit', 0.4)
        self.declare_parameter('steer_rate', 1.0)

        self.kp = self.get_parameter('kp').value
        self.ki = self.get_parameter('ki').value
        self.kd = self.get_parameter('kd').value
        self.i_limit = self.get_parameter('i_limit').value
        self.steer_rate = self.get_parameter('steer_rate').value

        self.steering_angle_to_servo_gain = self.get_parameter('steering_angle_to_servo_gain').value
        self.steering_angle_to_servo_offset = self.get_parameter('steering_angle_to_servo_offset').value
        self.speed_to_erpm_gain = self.get_parameter('speed_to_erpm_gain').value
        self.servo_min = self.get_parameter('servo_min').value
        self.servo_max = self.get_parameter('servo_max').value
        self.min_rpm = self.get_parameter('speed_min').value
        self.max_rpm = self.get_parameter('speed_max').value
        self.speed_max = self.get_parameter('speed_max').value

        self.target_speed = 0.0
        self.target_steer = 0.0
        self.measured_speed = 0.0
        self.cmd_steer = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
        self.last = self.get_clock().now()

        self.create_subscription(AckermannDriveStamped, '/ackermann_cmd', self.on_setpoint, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.pub_servo = self.create_publisher(Float64, '/commands/servo/position', 10)
        self.pub_speed = self.create_publisher(Float64, '/commands/motor/speed', 10)
        self.create_timer(0.01, self.tick)  # 100 Hz

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

        servo = self.steering_angle_to_servo_gain * self.cmd_steer + self.steering_angle_to_servo_offset
        servo = max(self.servo_min, min(self.servo_max, servo))
        servo_msg = Float64()
        servo_msg.data = float(servo)
        self.pub_servo.publish(servo_msg)

        speed_erpm = speed * self.speed_to_erpm_gain
        speed_erpm = max(self.min_rpm, min(self.max_rpm, speed_erpm))
        speed_msg = Float64()
        speed_msg.data = float(speed_erpm)
        self.pub_speed.publish(speed_msg)

def main():
    rclpy.init()
    node = PidTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
