#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener
from tf2_ros import ConnectivityException, ExtrapolationException, LookupException


class PathFollower(Node):
    def __init__(self):
        super().__init__("path_follower_node")
        self.get_logger().info("Path Follower Initialized")

        # Topics / frames
        self.declare_parameter("path_topic", "/global_centerline")
        self.declare_parameter("ekf_odom_topic", "/odometry/filtered")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("path_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("use_tf_pose", True)

        # Pure pursuit
        self.declare_parameter("lookahead_distance", 0.8)
        self.declare_parameter("wheelbase", 0.25)
        self.declare_parameter("max_steering_angle", 0.5)
        self.declare_parameter("closed_loop_path", True)
        self.declare_parameter("force_closed_path", False)
        self.declare_parameter("reverse_path", False)
        self.declare_parameter("closest_search_back_points", 25)
        self.declare_parameter("closest_search_ahead_points", 500)
        self.declare_parameter("lookahead_min_forward_x", 0.05)
        self.declare_parameter("max_lookahead_target_distance", 3.0)
        self.declare_parameter("enable_path_error_correction", False)
        self.declare_parameter("heading_error_gain", 0.0)
        self.declare_parameter("crosstrack_error_gain", 0.0)
        self.declare_parameter("crosstrack_softening_speed", 0.5)

        # Speed / smoothing
        self.declare_parameter("max_speed", 1.0)
        self.declare_parameter("min_speed", 0.5)
        self.declare_parameter("local_planner_speed", 0.45)
        self.declare_parameter("steering_smoothing", 0.25)
        self.declare_parameter("steering_deadband", 0.03)

        # Local gap fallback and blocker handling
        self.declare_parameter("enable_local_planner", True)
        self.declare_parameter("max_scan_distance", 12.0)
        self.declare_parameter("avoid_distance", 1.2)
        self.declare_parameter("blocker_distance", 0.55)
        self.declare_parameter("stop_on_blocker", True)
        self.declare_parameter("depth_threshold", 1.5)
        self.declare_parameter("front_fov_degrees", 35.0)
        self.declare_parameter("gap_fov_degrees", 140.0)
        self.declare_parameter("bubble_radius_beams", 20)
        self.declare_parameter("local_steering_blend", 1.0)
        self.declare_parameter("wall_avoidance_gain", 0.0)
        self.declare_parameter("enable_boundary_safety", False)
        self.declare_parameter("boundary_safety_distance", 0.4)
        self.declare_parameter("boundary_safety_speed", 0.08)
        self.declare_parameter("boundary_steering_blend", 0.9)
        self.declare_parameter("boundary_escape_steering", 0.5)
        self.declare_parameter("odom_timeout_sec", 0.5)

        # State
        self.path_points = []
        self.path_frame = ""
        self.path_wraps = False
        self.last_closest_idx = None
        self.current_x = None
        self.current_y = None
        self.current_yaw = None
        self.odom_frame = ""
        self.last_odom_time = None
        self.latest_scan = None

        # Parameters
        self.path_topic = self.get_parameter("path_topic").value
        self.ekf_odom_topic = self.get_parameter("ekf_odom_topic").value
        self.scan_topic = self.get_parameter("scan_topic").value
        self.drive_topic = self.get_parameter("drive_topic").value
        self.expected_path_frame = self.get_parameter("path_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.use_tf_pose = bool(self.get_parameter("use_tf_pose").value)

        self.lookahead_distance = float(self.get_parameter("lookahead_distance").value)
        self.wheelbase = float(self.get_parameter("wheelbase").value)
        self.max_steering_angle = float(self.get_parameter("max_steering_angle").value)
        self.closed_loop_path = bool(self.get_parameter("closed_loop_path").value)
        self.force_closed_path = bool(self.get_parameter("force_closed_path").value)
        self.reverse_path = bool(self.get_parameter("reverse_path").value)
        self.closest_search_back_points = int(
            self.get_parameter("closest_search_back_points").value
        )
        self.closest_search_ahead_points = int(
            self.get_parameter("closest_search_ahead_points").value
        )
        self.lookahead_min_forward_x = float(
            self.get_parameter("lookahead_min_forward_x").value
        )
        self.max_lookahead_target_distance = float(
            self.get_parameter("max_lookahead_target_distance").value
        )
        self.enable_path_error_correction = bool(
            self.get_parameter("enable_path_error_correction").value
        )
        self.heading_error_gain = float(self.get_parameter("heading_error_gain").value)
        self.crosstrack_error_gain = float(self.get_parameter("crosstrack_error_gain").value)
        self.crosstrack_softening_speed = float(
            self.get_parameter("crosstrack_softening_speed").value
        )

        self.max_speed = float(self.get_parameter("max_speed").value)
        self.min_speed = float(self.get_parameter("min_speed").value)
        self.local_planner_speed = float(self.get_parameter("local_planner_speed").value)
        self.steering_smoothing = float(self.get_parameter("steering_smoothing").value)
        self.steering_deadband = float(self.get_parameter("steering_deadband").value)

        self.enable_local_planner = bool(self.get_parameter("enable_local_planner").value)
        self.max_scan_distance = float(self.get_parameter("max_scan_distance").value)
        self.avoid_distance = float(self.get_parameter("avoid_distance").value)
        self.blocker_distance = float(self.get_parameter("blocker_distance").value)
        self.stop_on_blocker = bool(self.get_parameter("stop_on_blocker").value)
        self.depth_threshold = float(self.get_parameter("depth_threshold").value)
        self.front_fov = math.radians(float(self.get_parameter("front_fov_degrees").value))
        self.gap_fov = math.radians(float(self.get_parameter("gap_fov_degrees").value))
        self.bubble_radius_beams = int(self.get_parameter("bubble_radius_beams").value)
        self.local_steering_blend = float(self.get_parameter("local_steering_blend").value)
        self.wall_avoidance_gain = float(self.get_parameter("wall_avoidance_gain").value)
        self.enable_boundary_safety = bool(
            self.get_parameter("enable_boundary_safety").value
        )
        self.boundary_safety_distance = float(
            self.get_parameter("boundary_safety_distance").value
        )
        self.boundary_safety_speed = float(
            self.get_parameter("boundary_safety_speed").value
        )
        self.boundary_steering_blend = float(
            self.get_parameter("boundary_steering_blend").value
        )
        self.boundary_escape_steering = float(
            self.get_parameter("boundary_escape_steering").value
        )
        self.odom_timeout_sec = float(self.get_parameter("odom_timeout_sec").value)

        self.prev_steering_angle = 0.0

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        path_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(Path, self.path_topic, self.path_callback, path_qos)
        self.create_subscription(Odometry, self.ekf_odom_topic, self.odom_callback, 10)
        self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            qos_profile_sensor_data
        )

        # Publisher
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)

        # Timer
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info(
            f"Following {self.path_topic} using "
            f"{'TF pose with EKF fallback' if self.use_tf_pose else self.ekf_odom_topic}; "
            f"local fallback={'on' if self.enable_local_planner else 'off'}"
        )

    def path_callback(self, msg):
        had_path = len(self.path_points) > 1
        self.path_points = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if self.reverse_path:
            self.path_points.reverse()
        self.path_frame = msg.header.frame_id
        self.path_wraps = self.force_closed_path or self.path_is_closed()

        if not had_path:
            self.last_closest_idx = None
        elif self.last_closest_idx is not None and self.path_points:
            self.last_closest_idx = min(self.last_closest_idx, len(self.path_points) - 1)

        if self.path_frame and self.path_frame != self.expected_path_frame:
            self.get_logger().warn(
                f"Path frame is '{self.path_frame}', expected '{self.expected_path_frame}'. "
                "Pure pursuit assumes EKF odometry is in the same frame.",
                throttle_duration_sec=5.0
            )

        self.get_logger().info(
            f"Received path with {len(self.path_points)} points in frame "
            f"{msg.header.frame_id}; closed={self.path_wraps}",
            throttle_duration_sec=1.0
        )

    def path_is_closed(self):
        if not self.closed_loop_path or len(self.path_points) < 3:
            return False

        first_x, first_y = self.path_points[0]
        last_x, last_y = self.path_points[-1]
        return math.hypot(last_x - first_x, last_y - first_y) <= self.lookahead_distance * 2.0

    def odom_callback(self, msg):
        self.odom_frame = msg.header.frame_id
        self.current_x = float(msg.pose.pose.position.x)
        self.current_y = float(msg.pose.pose.position.y)
        self.current_yaw = self.quat_to_yaw(msg.pose.pose.orientation)
        self.last_odom_time = self.get_clock().now()

    def scan_callback(self, msg):
        self.latest_scan = msg

    def quat_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def update_pose(self):
        target_frame = self.path_frame or self.expected_path_frame

        if self.use_tf_pose and target_frame:
            if self.update_pose_from_tf(target_frame):
                return True

            self.get_logger().warn(
                f"TF lookup {target_frame}->{self.base_frame} unavailable; "
                f"trying {self.ekf_odom_topic} fallback.",
                throttle_duration_sec=2.0
            )

        return self.odom_is_ready(target_frame)

    def update_pose_from_tf(self, target_frame):
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                self.base_frame,
                rclpy.time.Time()
            )

            t = transform.transform.translation
            q = transform.transform.rotation

            self.current_x = t.x
            self.current_y = t.y
            self.current_yaw = self.quat_to_yaw(q)
            return True

        except (LookupException, ConnectivityException, ExtrapolationException):
            return False

    def odom_is_ready(self, target_frame):
        if self.current_x is None or self.current_y is None or self.current_yaw is None:
            return False

        if self.last_odom_time is None:
            return False

        age = (self.get_clock().now() - self.last_odom_time).nanoseconds * 1e-9
        if age > self.odom_timeout_sec:
            return False

        if target_frame and self.odom_frame and self.odom_frame != target_frame:
            self.get_logger().warn(
                f"Fallback odom frame is '{self.odom_frame}', but path frame is "
                f"'{target_frame}'. Waiting for TF so pose and path match.",
                throttle_duration_sec=2.0
            )
            return False

        return True

    def control_loop(self):
        # Checks for valid global path
        if len(self.path_points) < 2:
            self.get_logger().warn(
                f"Waiting for a usable global path on {self.path_topic}",
                throttle_duration_sec=2.0
            )
            self.publish_drive(0.0, 0.0)
            return
        # Checks for valid pose
        if not self.update_pose():
            self.get_logger().warn(
                f"Waiting for pose in path frame '{self.path_frame or self.expected_path_frame}'",
                throttle_duration_sec=2.0
            )
            self.publish_drive(0.0, 0.0)
            return
        # Finds the lookahead point and computes pure pursuit steering
        target_x, target_y, closest_idx = self.find_lookahead_point()
        steering = self.compute_pure_pursuit_steering(target_x, target_y)
        # Checks if pure pursuit can produce valid steering
        # If not then stop and wait for next cycle check
        if steering is None:
            self.publish_drive(0.0, 0.0)
            return
        # Applies path error correction if enabled
        # Provides smoother steering 
        if self.enable_path_error_correction:
            steering = self.apply_path_error_correction(steering, closest_idx)

        mode = "global"
        boundary_steering = self.compute_boundary_safety_steering()
        blocked, need_avoid = self.path_is_blocked()
        # If boundary safety is triggered - makes avoiding wall top priority
        if boundary_steering is not None:
            blend = max(0.0, min(1.0, self.boundary_steering_blend))
            steering = (1.0 - blend) * steering + blend * boundary_steering
            speed = self.boundary_safety_speed
            mode = "boundary_safety"
        # If path is blocked or too close then use local planner
        elif self.enable_local_planner and need_avoid:
            local_steering = self.compute_gap_steering()
            if blocked and self.stop_on_blocker:
                self.get_logger().warn(
                    "Blocker is too close ahead; stopping until the path clears.",
                    throttle_duration_sec=1.0
                )
                self.publish_drive(0.0, 0.0)
                return

            if local_steering is None:
                self.get_logger().warn(
                    "Obstacle detected ahead but no safe local gap is available; stopping.",
                    throttle_duration_sec=1.0
                )
                self.publish_drive(0.0, 0.0)
                return

            blend = max(0.0, min(1.0, self.local_steering_blend))
            steering = (1.0 - blend) * steering + blend * local_steering
            speed = self.local_planner_speed
            mode = "local_blend"
        # Use pure pursuit - global planner if path is clear
        else:
            speed = self.speed_for_steering(steering)

        steering = self.smooth_steering(steering)
        # drops speed if using local planner for safer steering 
        # only if path is blocked or too close to obstacle
        if mode.startswith("local"):
            speed = min(speed, self.speed_for_steering(steering))

        self.get_logger().info(
            f"mode={mode} idx={closest_idx} car=({self.current_x:.2f},{self.current_y:.2f}) "
            f"target=({target_x:.2f},{target_y:.2f}) "
            f"steering={steering:.3f} speed={speed:.3f}",
            throttle_duration_sec=0.5
        )

        self.publish_drive(steering, speed)
    # Finds lookahead point using the path and current pose
    # Goes along the path from closest point until lookahead distance is reached
    def find_lookahead_point(self):
        closest_idx = self.find_progressive_closest_idx()

        total = 0.0
        target_x, target_y = self.path_points[closest_idx]
        fallback_target = None
        found_forward_target = False
        max_steps = (
            len(self.path_points)
            if self.path_wraps
            else len(self.path_points) - closest_idx - 1
        )

        for j in range(max_steps):
            if self.path_wraps:
                i1 = (closest_idx + j) % len(self.path_points)
                i2 = (closest_idx + j + 1) % len(self.path_points)
            else:
                i1 = closest_idx + j
                i2 = min(closest_idx + j + 1, len(self.path_points) - 1)

            x1, y1 = self.path_points[i1]
            x2, y2 = self.path_points[i2]

            total += math.hypot(x2 - x1, y2 - y1)
            # keep searhing until lookahead distance is reached along the path
            if total < self.lookahead_distance:
                continue
            # If point is too far then skip it - prevents overshooting
            if not self.point_is_near_enough(x2, y2):
                continue

            if fallback_target is None:
                fallback_target = (x2, y2)
            # Prefer a point in front of car if perfect forward point is not available
            if self.point_is_ahead(x2, y2):
                target_x, target_y = x2, y2
                found_forward_target = True
                break

        if not found_forward_target and fallback_target is not None:
            target_x, target_y = fallback_target

        return target_x, target_y, closest_idx

    # Finds closest point on path to the car
    # Uses a progressive search starting from last closest point for efficiency
    # Find where the car is on the path without jumping around
    def find_progressive_closest_idx(self):
        best_dist2 = float("inf")
        # Search the entire path if the closest point is invalid or unknown state
        if self.last_closest_idx is None:
            candidates = range(len(self.path_points))
        # Allow wrap around if a looped track
        elif self.path_wraps:
            start = self.last_closest_idx - self.closest_search_back_points
            stop = self.last_closest_idx + self.closest_search_ahead_points + 1
            candidates = (i % len(self.path_points) for i in range(start, stop))
        # If not a loop then search forwards and backwards within a window
        else:
            start = max(0, self.last_closest_idx - self.closest_search_back_points)
            stop = min(
                len(self.path_points),
                self.last_closest_idx + self.closest_search_ahead_points + 1
            )
            candidates = range(start, stop)

        closest_idx = self.last_closest_idx if self.last_closest_idx is not None else 0

        for i in candidates:
            px, py = self.path_points[i]
            dx = px - self.current_x
            dy = py - self.current_y
            d2 = dx * dx + dy * dy

            if d2 < best_dist2:
                best_dist2 = d2
                closest_idx = i

        self.last_closest_idx = closest_idx
        return closest_idx
    # Transforms point into car local frame
    # Checks if point is ahead of vehicle
    def point_is_ahead(self, point_x, point_y):
        dx = point_x - self.current_x
        dy = point_y - self.current_y
        local_x = math.cos(self.current_yaw) * dx + math.sin(self.current_yaw) * dy
        return local_x > self.lookahead_min_forward_x

    # Checks target is not too far away 
    def point_is_near_enough(self, point_x, point_y):
        return (
            math.hypot(point_x - self.current_x, point_y - self.current_y)
            <= self.max_lookahead_target_distance
        )

    # Computes pure pursuit steering angle to target point
    def compute_pure_pursuit_steering(self, target_x, target_y):
        dx = target_x - self.current_x
        dy = target_y - self.current_y

        local_x = math.cos(self.current_yaw) * dx + math.sin(self.current_yaw) * dy
        local_y = -math.sin(self.current_yaw) * dx + math.cos(self.current_yaw) * dy

        Ld = math.hypot(local_x, local_y)
        if Ld < 1e-6:
            return None

        alpha_pp = math.atan2(local_y, local_x)
        return math.atan2(2.0 * self.wheelbase * math.sin(alpha_pp), Ld)

    # Pulls back to centreline if vehicle starts to drift off path
    def apply_path_error_correction(self, steering, closest_idx):
        if len(self.path_points) < 2:
            return steering

        next_idx = (closest_idx + 1) % len(self.path_points)
        if not self.path_wraps:
            next_idx = min(closest_idx + 1, len(self.path_points) - 1)

        px, py = self.path_points[closest_idx]
        nx, ny = self.path_points[next_idx]
        path_heading = math.atan2(ny - py, nx - px)
        heading_error = self.normalize_angle(path_heading - self.current_yaw)

        dx = self.current_x - px
        dy = self.current_y - py
        crosstrack_error = -math.sin(path_heading) * dx + math.cos(path_heading) * dy
        crosstrack_correction = -math.atan2(
            self.crosstrack_error_gain * crosstrack_error,
            self.crosstrack_softening_speed,
        )

        corrected = (steering + self.heading_error_gain * heading_error + crosstrack_correction)
        return max(-self.max_steering_angle, min(self.max_steering_angle, corrected))

    # Avoid weird angle jumps - keeps between -pi and pi
    def normalize_angle(self, angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    # Checks if there is an obstacle blocking the path in front
    # If blocked then stop but if avoidable then avoid 
    def path_is_blocked(self):
        if self.latest_scan is None:
            return False, False

        ranges, angles = self.scan_ranges_and_angles()
        front_mask = np.abs(angles) <= self.front_fov * 0.5
        front_ranges = ranges[front_mask]

        if front_ranges.size == 0:
            return False, False

        min_front = float(np.min(front_ranges))
        return min_front < self.blocker_distance, min_front < self.avoid_distance

    # Computes steering angle for local planner to find gaps in front of the car
    def compute_gap_steering(self):
        if self.latest_scan is None:
            return None

        ranges, angles = self.scan_ranges_and_angles()
        gap_mask = np.abs(angles) <= self.gap_fov * 0.5
        gap_ranges = ranges[gap_mask]
        gap_angles = angles[gap_mask]

        if gap_ranges.size == 0:
            return None

        wall_steering = self.compute_wall_avoidance_steering(gap_ranges, gap_angles)
        if wall_steering is not None:
            return wall_steering

        closest_idx = int(np.argmin(gap_ranges))
        if gap_ranges[closest_idx] < self.avoid_distance:
            start = max(0, closest_idx - self.bubble_radius_beams)
            end = min(gap_ranges.size, closest_idx + self.bubble_radius_beams + 1)
            gap_ranges[start:end] = 0.0

        free_mask = gap_ranges > self.depth_threshold
        slices = np.ma.clump_unmasked(np.ma.masked_where(~free_mask, gap_ranges))

        if not slices:
            return None

        best_slice = max(slices, key=lambda s: s.stop - s.start)
        best_idx = (best_slice.start + best_slice.stop - 1) // 2
        return float(gap_angles[best_idx])

    # To avoid boundary collision if vehicle gets too close to wall
    def compute_boundary_safety_steering(self):
        if (
            not self.enable_boundary_safety
            or self.latest_scan is None
            or self.wall_avoidance_gain <= 0.0
        ):
            return None

        ranges, angles = self.scan_ranges_and_angles()
        safety_mask = np.abs(angles) <= self.gap_fov * 0.5
        safety_ranges = ranges[safety_mask]
        safety_angles = angles[safety_mask]

        if safety_ranges.size == 0:
            return None

        closest_idx = int(np.argmin(safety_ranges))
        closest_range = float(safety_ranges[closest_idx])

        if closest_range >= self.boundary_safety_distance:
            return None

        closest_angle = float(safety_angles[closest_idx])
        escape_steering = min(self.boundary_escape_steering, self.max_steering_angle)

        if abs(closest_angle) > math.radians(8.0):
            return -math.copysign(escape_steering, closest_angle)

        left_ranges = safety_ranges[safety_angles > 0.0]
        right_ranges = safety_ranges[safety_angles < 0.0]

        if left_ranges.size == 0 or right_ranges.size == 0:
            return escape_steering

        left_clearance = float(np.median(left_ranges))
        right_clearance = float(np.median(right_ranges))
        return escape_steering if left_clearance > right_clearance else -escape_steering

    # Function checks if walls are too close  and computes appropriate steering 
    def compute_wall_avoidance_steering(self, ranges, angles, trigger_distance=None):
        # Do nothing if wall avoidance is off
        if self.wall_avoidance_gain <= 0.0:
            return None

        # Split lidar reading into left and right 
        left_ranges = ranges[angles > 0.0]
        right_ranges = ranges[angles < 0.0]

        if left_ranges.size == 0 or right_ranges.size == 0:
            return None

        left_clearance = float(np.min(left_ranges))
        right_clearance = float(np.min(right_ranges))

        if trigger_distance is None:
            trigger_distance = self.avoid_distance

        if min(left_clearance, right_clearance) >= trigger_distance:
            return None

        # Calculates the steering away from the wall based on closeness
        min_range = max(float(self.latest_scan.range_min), 1e-3)
        left_clearance = max(left_clearance, min_range)
        right_clearance = max(right_clearance, min_range)
        steering = self.wall_avoidance_gain * (
            (1.0 / right_clearance) - (1.0 / left_clearance)
        )
        return max(-self.max_steering_angle, min(self.max_steering_angle, steering))

    # Converts raw lidar into two arrays of distances and matching angles
    def scan_ranges_and_angles(self):
        # Turns LiDAR scan into numpy arrays and handles invalid values
        ranges = np.array(self.latest_scan.ranges, dtype=float)
        ranges = np.nan_to_num(
            ranges,
            nan=0.0,
            posinf=self.max_scan_distance,
            neginf=0.0
        )
        ranges = np.clip(ranges, 0.0, self.max_scan_distance)
        # Calculates angles for every LiDAR distance
        angles = (
            self.latest_scan.angle_min
            + np.arange(ranges.size) * self.latest_scan.angle_increment
        )
        return ranges, angles

    # Steer smoothing 
    def smooth_steering(self, steering):
        if abs(steering) < self.steering_deadband:
            steering = 0.0

        steering = (
            self.steering_smoothing * steering
            + (1.0 - self.steering_smoothing) * self.prev_steering_angle
        )
        steering = max(-self.max_steering_angle, min(self.max_steering_angle, steering))
        self.prev_steering_angle = steering
        return steering

    def speed_for_steering(self, steering):
        turn_scale = max(0.0, 1.0 - abs(steering) / self.max_steering_angle)
        return self.min_speed + (self.max_speed - self.min_speed) * turn_scale

    def publish_drive(self, steering_angle, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.drive.steering_angle = float(steering_angle)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PathFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
