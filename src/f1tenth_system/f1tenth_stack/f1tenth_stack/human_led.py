#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool
import subprocess

PWM_CHIP = 2
PWM_CHANNEL = 0

class HumanLED(Node):
    def __init__(self):
        super().__init__("human_led")
        self.declare_parameter("deadman_buttons", [5])
        buttons = self.get_parameter("deadman_buttons").value
        self.btn = buttons[0] if buttons else 5
        self.last_joy = None
        self.setup_led()
        self.create_subscription(Joy, "/joy", self.cb, 10)
        self.pub = self.create_publisher(Bool, "/human_control", 10)
        self.create_timer(0.1, self.tick)            # 10 Hz
        self.set_led(True)

    def setup_led(self):
        subprocess.run(f"echo {PWM_CHANNEL} | tee /sys/class/pwm/pwmchip{PWM_CHIP}/export", shell=True, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(f"echo 1000000 | tee /sys/class/pwm/pwmchip{PWM_CHIP}/pwm{PWM_CHANNEL}/period", shell=True, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(f"echo 1 | tee /sys/class/pwm/pwmchip{PWM_CHIP}/pwm{PWM_CHANNEL}/enable", shell=True, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def set_led(self, on: bool):
        duty = 0 if on else 1000000
        subprocess.run(f"echo {duty} | tee /sys/class/pwm/pwmchip{PWM_CHIP}/pwm{PWM_CHANNEL}/duty_cycle", shell=True, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def cb(self, msg: Joy):
        self.last_joy = msg

    def tick(self):
        human = False
        if self.last_joy and len(self.last_joy.buttons) > self.btn:
            human = self.last_joy.buttons[self.btn] == 1
        self.set_led(not human)
        msg = Bool()
        msg.data = human
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = HumanLED()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
