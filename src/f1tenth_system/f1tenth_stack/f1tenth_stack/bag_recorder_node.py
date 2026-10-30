#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
import subprocess
import os
import signal
from datetime import datetime

# ==================== CONFIG (edit these) ====================
BAG_DIR = "/home/hwf1tenth/bags/fwd_v1"
BAG_PREFIX = "ep"
TOPICS = [
    "/camera/camera/color/image_raw",
    "/teleop",
    "/task",
]
START_BUTTON = 1
STOP_BUTTON  = 2
PWM_CHIP = 3
PWM_CHANNEL = 0
# =============================================================

class BagRecorder(Node):
    def __init__(self):
        super().__init__('bag_recorder')
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
        self.recording = False
        self.set_led(False)

    def joy_cb(self, msg: Joy):
        start_pressed = msg.buttons[START_BUTTON] if START_BUTTON < len(msg.buttons) else 0
        stop_pressed  = msg.buttons[STOP_BUTTON]  if STOP_BUTTON  < len(msg.buttons) else 0

        if start_pressed and not self.prev_start:
            self.start_recording()
        if stop_pressed and not self.prev_stop:
            self.stop_recording()

        self.prev_start = start_pressed
        self.prev_stop = stop_pressed

def main():
    rclpy.init()
    node = BagRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_recording()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
