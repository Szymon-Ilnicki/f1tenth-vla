#!/usr/bin/env python3
"""One-file launcher: car stack (+ optional policy) + joystick-controlled bag recorder.

    python3 record.py                                   (config below)
    python3 record.py --out ~/bags/takeovers_r2 --prefix to --task "complete laps clockwise slowly"
    python3 record.py --policy smolvla                  (also start the policy here)

Command-line options override the CONFIG block. Run from anywhere; ROS and the
workspace are sourced automatically if needed.

Replaces rosbags_launch.sh + bag_recorder_node.py. Ctrl+C stops the recording
cleanly and shuts everything down.
"""
# ==================== CONFIG (edit these) ====================
BAG_DIR = "~/bags/takeovers_r2"
BAG_PREFIX = "ep"                     # bags are named <prefix>_<YYYYmmdd_HHMMSS>
TASK = "complete laps anticlockwise slowly"   # published to /task_in at startup

# What to launch (paths relative to the repo root). POLICY_LAUNCH = None for
# manual driving only, or the policy's launch file for takeover sessions.
CAR_LAUNCH = "src/f1tenth_system/f1tenth_stack/launch/car_cam.launch.py"
POLICY_LAUNCH = None
# POLICY_LAUNCH = "src/f1tenth_system/f1tenth_stack/launch/smolvla_node.launch.py"
# POLICY_LAUNCH = "src/f1tenth_system/f1tenth_stack/launch/act_node.launch.py"

TOPICS = []        # empty = record everything (-a); or e.g. ["/camera/camera/color/image_raw", "/teleop", ...]
NOTE = ""          # if set, written to <bag>/info.txt together with the task and policy

START_BUTTON = 1   # joystick buttons that start / stop a bag
STOP_BUTTON = 2
PWM_CHIP = 3       # recording LED
PWM_CHANNEL = 0

ROS_SETUP = "/opt/ros/humble/setup.bash"
WORKSPACE_SETUP = "install/setup.bash"
# =============================================================

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def ensure_ros_env():
    """Re-run this script inside a shell that has sourced ROS + the workspace."""
    if os.environ.get("_RECORD_PY_SOURCED") == "1":
        return
    ws = ROOT / WORKSPACE_SETUP
    src = f"source {ROS_SETUP}" + (f" && source {ws}" if ws.exists() else "")
    args = " ".join(f"'{a}'" for a in sys.argv[1:])
    os.environ["_RECORD_PY_SOURCED"] = "1"
    os.execvp("bash", ["bash", "-c", f"{src} && exec python3 '{Path(__file__).resolve()}' {args}"])


def launch(path):
    """ros2 launch in its own process group, so it can be stopped as a whole."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    print(f"launching {p.name} ...", flush=True)
    return subprocess.Popen(["ros2", "launch", str(p)], start_new_session=True)


def stop_group(proc, name, timeout=10):
    if proc is None or proc.poll() is not None:
        return
    print(f"stopping {name} ...", flush=True)
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def sh(cmd):
    subprocess.run(cmd, shell=True, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


POLICIES = {
    "smolvla": "src/f1tenth_system/f1tenth_stack/launch/smolvla_node.launch.py",
    "act": "src/f1tenth_system/f1tenth_stack/launch/act_node.launch.py",
}


def parse_args():
    global BAG_DIR, BAG_PREFIX, TASK, POLICY_LAUNCH, NOTE
    ap = argparse.ArgumentParser(description="car stack + joystick bag recorder")
    ap.add_argument("--out", help=f"folder for bags (default {BAG_DIR})")
    ap.add_argument("--prefix", help=f"bag name prefix (default {BAG_PREFIX})")
    ap.add_argument("--task", help="task published at startup")
    ap.add_argument("--policy", choices=["none", *POLICIES],
                    help="also launch a policy node (default: CONFIG POLICY_LAUNCH)")
    ap.add_argument("--note", help="written to <bag>/info.txt")
    a = ap.parse_args()
    if a.out:
        BAG_DIR = a.out
    if a.prefix:
        BAG_PREFIX = a.prefix
    if a.task:
        TASK = a.task
    if a.policy:
        POLICY_LAUNCH = None if a.policy == "none" else POLICIES[a.policy]
    if a.note is not None:
        NOTE = a.note


def main():
    parse_args()          # before re-exec too, so --help works without ROS
    ensure_ros_env()
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from sensor_msgs.msg import Joy
    from std_msgs.msg import String

    bag_dir = Path(BAG_DIR).expanduser()
    bag_dir.mkdir(parents=True, exist_ok=True)
    pwm = f"/sys/class/pwm/pwmchip{PWM_CHIP}"

    class Recorder(Node):
        def __init__(self):
            super().__init__("bag_recorder")
            self.proc = None
            self.bag_path = None
            self.prev = (0, 0)
            self.task_sent = False
            self.create_subscription(Joy, "/joy", self.joy_cb, 10)
            self.task_pub = self.create_publisher(String, "/task_in", 10)
            self.create_timer(0.5, self.send_task)
            sh(f"echo {PWM_CHANNEL} | tee {pwm}/export")
            sh(f"echo 1000000 | tee {pwm}/pwm{PWM_CHANNEL}/period")
            sh(f"echo 1 | tee {pwm}/pwm{PWM_CHANNEL}/enable")
            self.set_led(False)

        def set_led(self, on):
            sh(f"echo {1000000 if on else 0} | tee {pwm}/pwm{PWM_CHANNEL}/duty_cycle")

        def send_task(self):
            # wait until task_latch is up, then publish the task once
            if not self.task_sent and self.task_pub.get_subscription_count() > 0:
                self.task_pub.publish(String(data=TASK))
                self.task_sent = True
                self.get_logger().info(f"task: '{TASK}'")

        def start(self):
            if self.proc is not None:
                return
            name = f"{BAG_PREFIX}_{datetime.now():%Y%m%d_%H%M%S}"
            self.bag_path = bag_dir / name
            cmd = ["ros2", "bag", "record", "-o", str(self.bag_path)] + (TOPICS or ["-a"])
            self.proc = subprocess.Popen(cmd, start_new_session=True,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.set_led(True)
            self.get_logger().info(f"recording {name}")

        def stop(self):
            if self.proc is None:
                return
            stop_group(self.proc, "recording")
            self.proc = None
            self.set_led(False)
            if NOTE and self.bag_path is not None and self.bag_path.exists():
                (self.bag_path / "info.txt").write_text(
                    f"{NOTE}\ntask: {TASK}\npolicy: {POLICY_LAUNCH or 'none (manual)'}\n")
            self.get_logger().info(f"saved {self.bag_path.name if self.bag_path else ''}")

        def joy_cb(self, msg):
            b = msg.buttons
            start = b[START_BUTTON] if START_BUTTON < len(b) else 0
            stop = b[STOP_BUTTON] if STOP_BUTTON < len(b) else 0
            if start and not self.prev[0]:
                self.start()
            if stop and not self.prev[1]:
                self.stop()
            self.prev = (start, stop)

    sh("termux-wake-lock")
    car = launch(CAR_LAUNCH)
    policy = None
    rclpy.init()
    node = Recorder()
    try:
        time.sleep(3)
        if POLICY_LAUNCH:
            policy = launch(POLICY_LAUNCH)
        print(f"\nready: button {START_BUTTON} = start bag, button {STOP_BUTTON} = stop bag, "
              f"Ctrl+C = quit\nbags go to {bag_dir}\n", flush=True)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop()
        stop_group(policy, "policy")
        stop_group(car, "car stack")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        sh("termux-wake-unlock")


if __name__ == "__main__":
    main()
