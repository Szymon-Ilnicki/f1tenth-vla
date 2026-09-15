#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.qos import qos_profile_sensor_data

# comment


class GapFollower(Node):
    def __init__(self):
        super().__init__("gap_follower_node")
        # 1. Topics
        lidar_topic = "/scan"
        drive_topic = "/drive"

        # 2. Parameters
        self.max_scan_distance = 12.0
        self.depth_threshold = 1.5
        self.max_speed = 2.5
        self.min_speed = 2.0
        self.prev_steering_angle = 0.0
        self.cb_count = 0

        self.subscription = self.create_subscription(
            LaserScan, lidar_topic, self.lidar_callback, qos_profile_sensor_data
        )
        self.publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.get_logger().info("Gap Follower: Racing Line Stability Mode")

    def preprocess_lidar(self, ranges):
        proc_ranges = np.array(ranges)
        proc_ranges = np.nan_to_num(
            proc_ranges, nan=0.0, posinf=self.max_scan_distance, neginf=0.0
        )
        return np.clip(proc_ranges, 0, self.max_scan_distance)

    def apply_safety_bubble(self, proc_ranges):
        # FIND THE APEX (Closest point)
        closest_idx = np.argmin(proc_ranges)
        min_dist = proc_ranges[closest_idx]

        # INCREASED BUBBLE: This pushes the car away from the inner wall
        if min_dist < 2.0:
            radius = 80  # Large bubble to force a wider line
            start = max(0, closest_idx - radius)
            end = min(len(proc_ranges), closest_idx + radius)
            proc_ranges[start:end] = 0.0
        return proc_ranges

    def find_max_gap(self, proc_ranges):
        mask = proc_ranges > self.depth_threshold
        slices = np.ma.clump_unmasked(np.ma.masked_where(~mask, proc_ranges))
        if slices:
            best_slice = max(slices, key=lambda s: s.stop - s.start)
            return best_slice.start, best_slice.stop - 1
        return None, None

    def lidar_callback(self, data):
        self.get_logger().info("LIDAR callback triggered")
        num_beams = len(data.ranges)

        # 1. FIXED FOV (30% to 70%)
        # This keeps the car focused forward and stops side-wall jitters
        start_idx = int(num_beams * 0.20)
        end_idx = int(num_beams * 0.80)
        front_ranges = data.ranges[start_idx:end_idx]

        # 2. Process
        raw_proc = self.preprocess_lidar(front_ranges)
        proc_ranges = self.apply_safety_bubble(np.copy(raw_proc))
        gap_start, gap_end = self.find_max_gap(proc_ranges)

        if gap_start is not None:
            # 3. Calculate Target Angle
            best_idx = (gap_start + gap_end) // 2
            angle_offset = (start_idx + best_idx) * data.angle_increment
            target_angle = data.angle_min + angle_offset

            # 4. SPEED-SENSITIVE ALPHA (The Anti-Jitter Fix)
            # High speed = 0.08 alpha (heavy damping)
            # Low speed = 0.25 alpha (responsive for hairpins)
            speed_ratio = (self.max_speed - self.min_speed) / self.max_speed
            dynamic_alpha = 0.25 - (speed_ratio * 0.15)

            # 5. DYNAMIC DEADZONE
            if abs(target_angle) < 0.10:
                target_angle = 0.0

            smoothed_angle = (dynamic_alpha * target_angle) + (
                (1.0 - dynamic_alpha) * self.prev_steering_angle
            )
            self.prev_steering_angle = smoothed_angle

            # 6. TRAIL BRAKING
            # Slow down earlier and harder for turns to prevent understeer clipping
            turn_intensity = abs(smoothed_angle) / 0.30
            final_speed = max(self.min_speed, self.max_speed * (1.0 - turn_intensity))

            self.publish_drive(smoothed_angle, final_speed)
        else:
            self.publish_drive(0.0, 1.0)

    def publish_drive(self, steering_angle, speed):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = float(steering_angle)
        drive_msg.drive.speed = float(speed)
        self.publisher.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GapFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
