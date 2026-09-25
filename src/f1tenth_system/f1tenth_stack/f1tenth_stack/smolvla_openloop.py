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

def resolve_model_dir(model_path: str) -> Path:
    p = Path(model_path).expanduser()
    if p.is_absolute() and p.is_dir():
        return p

    # smolvla_openloop.py is at
    # <repo>/src/f1tenth_system/f1tenth_stack/f1tenth_stack/smolvla_openloop.py
    repo_root = Path(__file__).resolve().parents[4]
    candidate = (repo_root / p).resolve()
    if candidate.is_dir():
        return candidate

    raise FileNotFoundError(
        f"model not found: {model_path} (tried {candidate})"
    )

def image_msg_to_nchw(msg: Image) -> torch.Tensor:
    h, w = msg.height, msg.width
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = arr.reshape(h, w, 3)
        if msg.encoding == "bgr8":
            img = img[:, :, ::-1]
    else:
        raise RuntimeError(f"unsupported encoding {msg.encoding}")
    # nearest is enough for a probe
    ys = (np.linspace(0, h - 1, 224)).astype(np.int32)
    xs = (np.linspace(0, w - 1, 224)).astype(np.int32)
    img = img[ys][:, xs]
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


class SmolVLAOpenLoop(Node):
    def __init__(self):
        super().__init__("smolvla_openloop")

        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("task", "drive forward")
        self.declare_parameter("rate_hz", 2.0)
        self.declare_parameter("n_action_steps", 10)
        self.declare_parameter(
            "model_path", "outputs/smoke/checkpoints/last/pretrained_model"
        )

        raw = self.get_parameter("model_path").value
        model_dir = resolve_model_dir(self.get_parameter("model_path").value)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"loading smolvla on {self.device}")
        self.policy = (
            SmolVLAPolicy.from_pretrained(str(model_dir))
            .to(self.device)
            .eval()
        )
        self.get_logger().info(f"loaded {model_dir}")
        n = int(self.get_parameter("n_action_steps").value)
        n = max(1, min(n, int(self.policy.config.chunk_size)))
        self.policy.config.n_action_steps = n
        self.policy.reset()
        self.get_logger().info(
            f"n_action_steps={self.policy.config.n_action_steps} "
            f"chunk_size={self.policy.config.chunk_size}"
        )
        self.img_keys = list(self.policy.config.image_features)
        tok_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        self.tok = AutoTokenizer.from_pretrained(tok_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self._set_task(self.get_parameter("task").value)

        self.latest = None
        self.create_subscription(
            Image,
            self.get_parameter("image_topic").value,
            self._on_img,
            qos_profile_sensor_data,
        )
        self.create_subscription(String, "/smolvla/task", self._on_task, 10)
        self.pub_act = self.create_publisher(Float32MultiArray, "/smolvla/action", 10)
        self.pub_dt = self.create_publisher(Float32, "/smolvla/infer_dt", 10)

        period = 1.0 / max(self.get_parameter("rate_hz").value, 0.2)
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

    def _on_img(self, msg: Image):
        self.latest = msg

    def _tick(self):
        if self.latest is None:
            return
        try:
            img = image_msg_to_nchw(self.latest).to(self.device)
        except Exception as e:
            self.get_logger().error(str(e))
            return
        state = torch.zeros(1, 6, device=self.device)
        batch = {
            "observation.state": state,
            "observation.language.tokens": self.lang_tokens,
            "observation.language.attention_mask": self.lang_mask,
        }
        for k in self.img_keys:
            batch[k] = img

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.policy.select_action(batch)
        if self.device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        vec = out.detach().float().cpu().flatten().tolist()
        act = Float32MultiArray()
        act.data = vec
        self.pub_act.publish(act)
        self.pub_dt.publish(Float32(data=float(dt)))
        self.get_logger().info(f"dt={dt:.3f}s action={['%.3f' % x for x in vec]}")


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
