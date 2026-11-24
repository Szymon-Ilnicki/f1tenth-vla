
#!/usr/bin/env python3
"""Human-control flag + LED, plus an "auto-speed" takeover mode.
 
R1 (deadman_buttons): normal manual driving, published by joy_teleop.
R2 (auto_button or auto_axis): this node publishes /teleop itself with a constant
    speed (auto_speed) and steering from the right stick (steer_axis * steer_scale);
    the left stick is ignored. If R1 is also held, R1 wins and this node stays quiet.
 
/human_control is True while R1 or R2 is held (and /joy is fresh), so recorded
takeovers in either mode are picked up by bag_to_lerobot.py --takeover-bags.
 
R2 can be a button (set auto_button) or an analogue trigger axis (set auto_axis,
auto_axis_rest = value when released, auto_axis_pressed = value fully pressed;
it counts as held past halfway). Check with `ros2 topic echo /joy` while pressing R2.
"""
import subprocess
 
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool
 
PWM_CHIP = 2
PWM_CHANNEL = 0
 
 
class HumanLED(Node):
    def __init__(self):
        super().__init__("human_led")
        self.declare_parameter("deadman_buttons", [5])      # R1
        self.declare_parameter("auto_button", -1)            # R2 as a button (-1 = unused)
        self.declare_parameter("auto_axis", -1)              # R2 as a trigger axis (-1 = unused)
        self.declare_parameter("auto_axis_rest", 1.0)
        self.declare_parameter("auto_axis_pressed", -1.0)
        self.declare_parameter("auto_speed", 1.2)            # speed setpoint, ~0.5 m/s
        self.declare_parameter("steer_axis", 2)              # right stick X
        self.declare_parameter("steer_scale", 0.34)
        self.declare_parameter("teleop_topic", "teleop")
        self.declare_parameter("joy_timeout", 0.5)
 
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.btns = list(gp("deadman_buttons")) or [5]
        self.auto_button = int(gp("auto_button"))
        self.auto_axis = int(gp("auto_axis"))
        self.auto_rest = float(gp("auto_axis_rest"))
        self.auto_pressed = float(gp("auto_axis_pressed"))
        self.auto_speed = float(gp("auto_speed"))
        self.steer_axis = int(gp("steer_axis"))
        self.steer_scale = float(gp("steer_scale"))
        self.joy_timeout = float(gp("joy_timeout"))
 
        self.last_joy = None
        self.last_joy_t = None
        self.setup_led()
        self.create_subscription(Joy, "/joy", self.cb, 10)
        self.pub = self.create_publisher(Bool, "/human_control", 10)
        self.teleop_pub = self.create_publisher(AckermannDriveStamped, gp("teleop_topic"), 10)
        self.create_timer(0.1, self.tick)            # 10 Hz
        self.set_led(True)
        self.get_logger().info(
            f"R1 buttons {self.btns}; auto mode: button {self.auto_button}, axis {self.auto_axis} "
            f"(rest {self.auto_rest}, pressed {self.auto_pressed}), speed {self.auto_speed}")
 
    def setup_led(self):
        subprocess.run(f"echo {PWM_CHANNEL} | tee /sys/class/pwm/pwmchip{PWM_CHIP}/export", shell=True, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(f"echo 1000000 | tee /sys/class/pwm/pwmchip{PWM_CHIP}/pwm{PWM_CHANNEL}/period", shell=True, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(f"echo 1 | tee /sys/class/pwm/pwmchip{PWM_CHIP}/pwm{PWM_CHANNEL}/enable", shell=True, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
 
    def set_led(self, on: bool):
        duty = 0 if on else 1000000
        subprocess.run(f"echo {duty} | tee /sys/class/pwm/pwmchip{PWM_CHIP}/pwm{PWM_CHANNEL}/duty_cycle", shell=True, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
 
    # --- button state -------------------------------------------------------
    def r1_held(self, joy):
        return any(0 <= b < len(joy.buttons) and joy.buttons[b] == 1 for b in self.btns)
 
    def r2_held(self, joy):
        if 0 <= self.auto_button < len(joy.buttons) and joy.buttons[self.auto_button] == 1:
            return True
        if 0 <= self.auto_axis < len(joy.axes):
            span = self.auto_pressed - self.auto_rest
            if span != 0 and (joy.axes[self.auto_axis] - self.auto_rest) / span > 0.5:
                return True
        return False
 
    def joy_fresh(self):
        if self.last_joy_t is None:
            return False
        age = (self.get_clock().now() - self.last_joy_t).nanoseconds * 1e-9
        return age < self.joy_timeout
 
    # --- callbacks ----------------------------------------------------------
    def cb(self, msg: Joy):
        self.last_joy = msg
        self.last_joy_t = self.get_clock().now()
        # Auto-speed mode: publish on every /joy message (joy autorepeats ~20 Hz),
        # so the mux keeps the joystick input alive while R2 is held.
        if self.r2_held(msg) and not self.r1_held(msg):
            steer = msg.axes[self.steer_axis] * self.steer_scale if self.steer_axis < len(msg.axes) else 0.0
            out = AckermannDriveStamped()
            out.header.stamp = self.last_joy_t.to_msg()
            out.drive.speed = self.auto_speed
            out.drive.steering_angle = float(steer)
            self.teleop_pub.publish(out)
 
    def tick(self):
        human = False
        if self.last_joy is not None and self.joy_fresh():
            human = self.r1_held(self.last_joy) or self.r2_held(self.last_joy)
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