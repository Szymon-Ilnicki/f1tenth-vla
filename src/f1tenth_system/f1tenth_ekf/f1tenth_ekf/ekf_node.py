#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped
from tf2_ros import TransformBroadcaster


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def quat_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class F1TenthEkfNode(Node):
    def __init__(self):
        super().__init__('f1tenth_ekf_node')
        self.get_logger().info("REAL EKF VERSION IS RUNNING")

        # Parameters
        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('fused_odom_topic', '/odometry/filtered')
        self.declare_parameter('gyro_bias_z', 0.0)

        imu_topic = self.get_parameter('imu_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        fused_topic = self.get_parameter('fused_odom_topic').value
        self.gyro_bias_z = float(self.get_parameter('gyro_bias_z').value)

        self.get_logger().info(f"IMU topic: {imu_topic}")
        self.get_logger().info(f"Odom topic: {odom_topic}")
        self.get_logger().info(f"Fused topic: {fused_topic}")

        # Subscribers / Publishers
        self.imu_sub = self.create_subscription(Imu, imu_topic, self.on_imu, 50)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.on_odom, 50)
        self.fused_pub = self.create_publisher(Odometry, fused_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # State: [x, y, yaw, v]
        self.x = np.zeros((4, 1), dtype=float)

        # Covariance
        self.P = np.diag([
            0.2**2,                        # x
            0.2**2,                        # y
            (5.0 * math.pi / 180.0)**2,    # yaw
            0.3**2                         # v
        ])

        # Process noise
        self.Q = np.diag([
            0.02**2,                       # x
            0.02**2,                       # y
            (2.0 * math.pi / 180.0)**2,    # yaw
            0.15**2                        # v
        ])

        # Odom measurement noise: [x, y, yaw, v]
        self.R = np.diag([
            0.05**2,
            0.05**2,
            (3.0 * math.pi / 180.0)**2,
            0.10**2
        ])

        self.last_imu_stamp = None
        self.max_dt = 0.1
        self.omega_z = 0.0
        self.initialized_from_odom = False

        self.imu_count = 0
        self.odom_count = 0
        self.create_timer(1.0, self.report_rates)

    def on_imu(self, msg: Imu):
        self.imu_count += 1

        if msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0:
            t = self.get_clock().now().nanoseconds * 1e-9
        else:
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_imu_stamp is None:
            self.last_imu_stamp = t
            return

        dt = t - self.last_imu_stamp
        self.last_imu_stamp = t

        if dt <= 0.0:
            return
        if dt > self.max_dt:
            dt = self.max_dt

        self.omega_z = float(msg.angular_velocity.z) - self.gyro_bias_z

        if self.initialized_from_odom:
            self.predict(dt, self.omega_z)

    def on_odom(self, msg: Odometry):
        self.odom_count += 1

        odom_x = float(msg.pose.pose.position.x)
        odom_y = float(msg.pose.pose.position.y)
        odom_yaw = quat_to_yaw(msg.pose.pose.orientation)
        odom_v = float(msg.twist.twist.linear.x)

        if not self.initialized_from_odom:
            self.x[0, 0] = odom_x
            self.x[1, 0] = odom_y
            self.x[2, 0] = odom_yaw
            self.x[3, 0] = odom_v
            self.initialized_from_odom = True
            self.publish_fused()
            return

        self.correct(odom_x, odom_y, odom_yaw, odom_v)
        self.publish_fused()

    def predict(self, dt: float, omega_z: float):
        px, py, yaw, v = self.x.flatten()

        yaw_new = wrap_angle(yaw + omega_z * dt)
        px_new = px + v * math.cos(yaw) * dt
        py_new = py + v * math.sin(yaw) * dt
        v_new = v

        self.x = np.array([[px_new], [py_new], [yaw_new], [v_new]])

        F = np.eye(4)
        F[0, 2] = -v * math.sin(yaw) * dt
        F[0, 3] = math.cos(yaw) * dt
        F[1, 2] =  v * math.cos(yaw) * dt
        F[1, 3] = math.sin(yaw) * dt

        self.P = F @ self.P @ F.T + self.Q

    def correct(self, odom_x: float, odom_y: float, odom_yaw: float, odom_v: float):
        z = np.array([[odom_x], [odom_y], [odom_yaw], [odom_v]])
        H = np.eye(4)

        y = z - H @ self.x
        y[2, 0] = wrap_angle(y[2, 0])

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.x[2, 0] = wrap_angle(self.x[2, 0])

        I = np.eye(4)
        self.P = (I - K @ H) @ self.P

    def publish_fused(self):
        msg = Odometry()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"

        px, py, yaw, v = self.x.flatten()

        msg.pose.pose.position.x = float(px)
        msg.pose.pose.position.y = float(py)
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation = yaw_to_quat(float(yaw))

        msg.twist.twist.linear.x = float(v)
        msg.twist.twist.angular.z = float(self.omega_z)

        pose_cov = [0.0] * 36
        pose_cov[0] = float(self.P[0, 0])
        pose_cov[7] = float(self.P[1, 1])
        pose_cov[35] = float(self.P[2, 2])
        pose_cov[14] = 999.0
        pose_cov[21] = 999.0
        pose_cov[28] = 999.0
        msg.pose.covariance = pose_cov

        twist_cov = [0.0] * 36
        twist_cov[0] = float(self.P[3, 3])
        twist_cov[35] = 0.05
        msg.twist.covariance = twist_cov

        self.fused_pub.publish(msg)

        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = float(px)
        t.transform.translation.y = float(py)
        t.transform.translation.z = 0.0
        t.transform.rotation = yaw_to_quat(float(yaw))
        self.tf_broadcaster.sendTransform(t)

    def report_rates(self):
        self.get_logger().info(
            f"IMU: {self.imu_count}/s | "
            f"Odom: {self.odom_count}/s | "
            f"x={self.x[0,0]:.2f}, "
            f"y={self.x[1,0]:.2f}, "
            f"yaw={self.x[2,0]:.2f}, "
            f"v={self.x[3,0]:.2f}"
        )
        self.imu_count = 0
        self.odom_count = 0


def main():
    rclpy.init()
    node = F1TenthEkfNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()