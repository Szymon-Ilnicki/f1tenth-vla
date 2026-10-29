#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
import subprocess
import os
import signal
from datetime import datetime

# ==================== CONFIG (edit these) ====================
BAG_DIR = "/home/hwf1tenth/bags/fwd1"          # folder where bags will be saved
BAG_PREFIX = "ep"                             # bags will be named <BAG_PREFIX>_YYYYMMDD_HHMMSS
TOPICS = [                                    # topics to record (empty list = all topics)
    "/camera/camera/color/image_raw",
    "/teleop",
    "/task",
    # add more if needed
]
START_BUTTON = 0      # button index for start (A / cross on many pads)
STOP_BUTTON  = 1      # button index for stop  (B / circle)
# If you prefer a single toggle button, set STOP_BUTTON = START_BUTTON

PWM_CHIP = 3
PWM_CHANNEL = 0
# =============================================================

class BagRecorder(Node):
    def __init__(self):
        super().__init__('bag_recorder')
        # Listen to the real joystick for start/stop buttons
        self.sub = self.create_subscription(Joy, '/joy', self.joy_cb, 10)
        self.proc = None
        self.recording = False
        self.prev_start = 0
        self.prev_stop = 0

        os.makedirs(BAG_DIR, exist_ok=True)
        self.setup_led()
        self.get_logger().info("Bag recorder ready. Waiting for joystick start/stop.")

    def setup_led(self):
        try:
            subprocess.run(f"echo {PWM_CHANNEL} | sudo tee /sys/class/pwm/pwmchip{PWM_CHIP}/export",
                           shell=True, check=False)
            subprocess.run(f"echo 1000000 | sudo tee /sys/class/pwm/pwmchip{PWM_CHIP}/pwm{PWM_CHANNEL}/period",
                           shell=True, check=False)
            subprocess.run(f"echo 0 | sudo tee /sys/class/pwm/pwmchip{PWM_CHIP}/pwm{PWM_CHANNEL}/duty_cycle",
                           shell=True, check=False)
            subprocess.run(f"echo 1 | sudo tee /sys/class/pwm/pwmchip{PWM_CHIP}/pwm{PWM_CHANNEL}/enable",
                           shell=True, check=False)
        except Exception as e:
            self.get_logger().warn(f"LED setup failed: {e}")

    def set_led(self, on: bool):
        duty = 1000000 if on else 0
        subprocess.run(
            f"echo {duty} | sudo tee /sys/class/pwm/pwmchip{PWM_CHIP}/pwm{PWM_CHANNEL}/duty_cycle",
            shell=True, check=False
        )

    def start_recording(self):
        if self.recording:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bag_path = os.path.join(BAG_DIR, f"{BAG_PREFIX}_{timestamp}")
        cmd = ["ros2", "bag", "record", "-o", bag_path]
        if TOPICS:
            cmd += TOPICS
        else:
            cmd.append("-a")

        self.get_logger().info(f"Starting bag: {bag_path}")
        self.proc = subprocess.Popen(cmd)
        self.recording = True
        self.set_led(True)

    def stop_recording(self):
        if not self.recording or self.proc is None:
            return
        self.get_logger().info("Stopping bag...")
        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None