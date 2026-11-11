#!/usr/bin/env python3
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray, String

import torch
from transformers import AutoTokenizer
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from pathlib import Path
import subprocess

PWM_BASE = "/sys/class/pwm/pwmchip2/pwm0"
INFERENCE_TIME = 1.2

def _pwm_write(path, value):
    try:
        subprocess.run(
            ["sudo", "tee", path],
            input=str(value).encode(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass

def led_on():
    _pwm_write(f"{PWM_BASE}/duty_cycle", 1000000)
    _pwm_write(f"{PWM_BASE}/enable", 1)

def led_off():
    _pwm_write(f"{PWM_BASE}/enable", 0)

def resolve_model_dir(model_path: str) -> Path:
    p = Path(model_path).expanduser()
    if p.is_absolute() and p.is_dir():
        return p
    repo_root = Path(__file__).resolve().parents[4]
    candidate = (repo_root / p).resolve()
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"model not found: {model_path} (tried {candidate})")

def image_msg_to_nchw(msg: Image) -> torch.Tensor:
    h, w = msg.height, msg.width

    if msg.encoding in ("rgb8", "bgr8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = arr.reshape(h, w, 3)
        if msg.encoding == "bgr8":
            img = img[:, :, ::-1]
    elif msg.encoding in ("16UC1", "mono16"):
        arr = np.frombuffer(msg.data, dtype=np.uint16)
        img = arr.reshape(h, w)
        # Normalize to 0–255 and expand to 3 channels
        if img.max() > 0:
            img = (img.astype(np.float32) / img.max() * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        img = np.stack([img, img, img], axis=-1)
    else:
        raise RuntimeError(f"unsupported encoding {msg.encoding}")

    # Resize to 224×224 with nearest-neighbor sampling
    ys = (np.linspace(0, h - 1, 224)).astype(np.int32)
    xs = (np.linspace(0, w - 1, 224)).astype(np.int32)
    img = img[ys][:, xs]

    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)
class SmolVLAOpenLoop(Node):
    def __init__(self):
        super().__init__("smolvla_openloop")

        self.declare_parameter("rgb_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/depth/image_rect_raw")
        self.declare_parameter("task", "drive forward")
        self.declare_parameter("rate_hz", 2.0)
        self.declare_parameter("n_action_steps", 10)
        self.declare_parameter("mode", "rgb")  # rgb | depth | dual
        self.declare_parameter(
            "model_path", "outputs/smoke/checkpoints/last/pretrained_model"
        )

        self.mode = self.get_parameter("mode").value.lower()
        if self.mode not in ("rgb", "depth", "dual"):
            self.get_logger().warn(f"unknown mode '{self.mode}', defaulting to rgb")
            self.mode = "rgb"
        self.get_logger().info(f"mode: {self.mode}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = self.get_parameter("model_path").value
        if model_path and model_path.strip():
            model_dir = resolve_model_dir(model_path)
            self.get_logger().info(f"loading smolvla from {model_dir} on {self.device}")
            self.policy = SmolVLAPolicy.from_pretrained(str(model_dir)).to(self.device).eval()
            self.get_logger().info(f"loaded {model_dir}")
        else:
            self.get_logger().info(f"loading base smolvla on {self.device}")
            self.policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(self.device).eval()
            self.get_logger().info("loaded lerobot/smolvla_base")

        self.n_action_steps = int(self.get_parameter("n_action_steps").value)
        self.n_action_steps = max(1, min(self.n_action_steps, int(self.policy.config.chunk_size)))
        self.policy.config.n_action_steps = self.n_action_steps
        self.policy.reset()

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        actions_to_cover = int(INFERENCE_TIME * self.rate_hz) + 1
        self.trigger_at = max(1, min(actions_to_cover, self.n_action_steps - 1))

        self.get_logger().info(
            f"n_action_steps={self.n_action_steps} "
            f"chunk_size={self.policy.config.chunk_size} "
            f"rate_hz={self.rate_hz:.1f} "
            f"trigger_at={self.trigger_at}"
        )

        self.img_keys = list(self.policy.config.image_features)
        tok_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        self.tok = AutoTokenizer.from_pretrained(tok_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self._set_task(self.get_parameter("task").value)

        self.latest_rgb = None
        self.latest_depth = None
        self.action_queue = []
        self.inference_in_progress = False

        # Always subscribe to both; we just ignore what we don't need
        self.create_subscription(
            Image, self.get_parameter("rgb_topic").value,
            self._on_rgb, qos_profile_sensor_data,
        )
        self.create_subscription(
            Image, self.get_parameter("depth_topic").value,
            self._on_depth, qos_profile_sensor_data,
        )

        self.create_subscription(String, "/task", self._on_task, 10)
        self.pub_act = self.create_publisher(Float32MultiArray, "/smolvla/action", 10)
        self.pub_dt = self.create_publisher(Float32, "/smolvla/infer_dt", 10)

        period = 1.0 / max(self.rate_hz, 0.2)
        self.create_timer(period, self._tick)
        self.get_logger().info(f"image keys {self.img_keys}")

    def _set_task(self, text: str):
        text = (text or "drive forward").strip() + "\n"
        enc = self.tok(text, return_tensors="pt", padding="max_length", max_length=48)
        self.lang_tokens = enc.input_ids.to(self.device)
        self.lang_mask = enc.attention_mask.to(device=self.device, dtype=torch.bool)
        self.get_logger().info(f"task set: {text.strip()}")

    def _on_task(self, msg: String):
        self._set_task(msg.data)

    def _on_rgb(self, msg: Image):
        self.latest_rgb = msg

    def _on_depth(self, msg: Image):
        self.latest_depth = msg

    def _zeros(self):
        return torch.zeros(1, 3, 224, 224, device=self.device)

    def _run_inference(self):
        # Build the two image tensors according to mode
        if self.mode in ("rgb", "dual"):
            if self.latest_rgb is None:
                return None, 0.0
            try:
                cam1 = image_msg_to_nchw(self.latest_rgb).to(self.device)
            except Exception as e:
                self.get_logger().error(f"rgb convert: {e}")
                return None, 0.0
        else:
            cam1 = self._zeros()

        if self.mode in ("depth", "dual"):
            if self.latest_depth is None:
                return None, 0.0
            try:
                cam2 = image_msg_to_nchw(self.latest_depth).to(self.device)
            except Exception as e:
                self.get_logger().error(f"depth convert: {e}")
                return None, 0.0
        else:
            cam2 = self._zeros()

        state = torch.zeros(1, 6, device=self.device)
        batch = {
            "observation.state": state,
            "observation.language.tokens": self.lang_tokens,
            "observation.language.attention_mask": self.lang_mask,
        }
        # Map to the keys the model expects
        for i, k in enumerate(self.img_keys):
            batch[k] = cam1 if i == 0 else cam2

        led_on()
        try:
            t0 = time.perf_counter()
            with torch.no_grad():
                actions = []
                for _ in range(self.n_action_steps):
                    out = self.policy.select_action(batch)
                    if out[0, 0] < 0:
                        out[0, 0] = -out[0, 0]
                    actions.append(out.detach().float().cpu().flatten().tolist())
            if self.device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        finally:
            led_off()
        return actions, dt

    def _tick(self):
        if self.action_queue:
            vec = self.action_queue.pop(0)
            act = Float32MultiArray()
            act.data = vec
            self.pub_act.publish(act)

        if len(self.action_queue) <= self.trigger_at and not self.inference_in_progress:
            self.inference_in_progress = True
            actions, dt = self._run_inference()
            self.inference_in_progress = False
            if actions is not None:
                self.action_queue.extend(actions)
                self.pub_dt.publish(Float32(data=float(dt)))
                self.get_logger().info(
                    f"dt={dt:.3f}s queued {len(actions)} actions "
                    f"(remaining={len(self.action_queue)})"
                )

def main():
    rclpy.init()
    node = SmolVLAOpenLoop()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()