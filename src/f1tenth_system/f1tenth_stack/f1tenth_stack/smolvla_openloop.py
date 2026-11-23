#!/usr/bin/env python3
"""SmolVLA driving node.

Timing: the model was trained on a 10 fps dataset, so each action in a chunk covers
1/rate_hz = 0.1 s. Inference runs continuously in a background thread; the timer
publishes one action every 0.1 s from the most recent chunk, choosing the action
planned for the current moment: index = (now - time the images were taken + lead_s) * rate_hz.
So the actions that fell due while the model was running are skipped, and every
action is executed at the moment it was planned for.
"""
import threading
import time
import cv2
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

DEPTH_MIN = 200
DEPTH_MAX = 2000

PWM_BASE = "/sys/class/pwm/pwmchip2/pwm0"

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

def compat_model_dir(model_dir: Path, logger=None) -> Path:
    """Checkpoints from a newer lerobot can carry config fields this lerobot doesn't know
    (e.g. dtype, pretrained_revision, optimizer_fused), which makes from_pretrained fail.
    If so, build a temporary copy of the folder (symlinks + a filtered config.json) and load
    from that. The model folder itself is never modified."""
    import dataclasses
    import json
    import tempfile
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return model_dir
    cfg = json.loads(cfg_path.read_text())
    known = {f.name for f in dataclasses.fields(SmolVLAConfig)} | {"type"}
    unknown = sorted(k for k in cfg if k not in known)
    if not unknown:
        return model_dir
    tmp = Path(tempfile.mkdtemp(prefix="smolvla_compat_"))
    for f in model_dir.iterdir():
        if f.name != "config.json":
            (tmp / f.name).symlink_to(f.resolve())
    (tmp / "config.json").write_text(json.dumps({k: v for k, v in cfg.items() if k in known}, indent=4))
    if logger is not None:
        logger.warn(f"checkpoint config has fields this lerobot doesn't know, ignoring them: {unknown}")
    return tmp


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
        depth = arr.reshape(h, w)

        invalid = (depth == 0)
        depth_f = depth.astype(np.float32)
        depth_f = np.clip(depth_f, DEPTH_MIN, DEPTH_MAX)
        norm = (depth_f - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)
        norm = (norm * 255).astype(np.uint8)

        color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color[invalid] = [255, 0, 255]   # bright magenta
        img = color
    else:
        raise RuntimeError(f"unsupported encoding {msg.encoding}")

    # Full frame, as in the LeRobot dataset; the policy letterboxes it to 512x512 itself.
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)
class SmolVLAOpenLoop(Node):
    def __init__(self):
        super().__init__("smolvla_openloop")

        self.declare_parameter("rgb_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/depth/image_rect_raw")
        self.declare_parameter("task", "drive forward")
        self.declare_parameter("rate_hz", 10.0)        # must equal the dataset fps
        self.declare_parameter("n_action_steps", 40)   # max actions used from one chunk
        self.declare_parameter("stop_when_empty", True)
        self.declare_parameter("max_image_age_s", 0.5)  # don't plan from images older than this
        self.declare_parameter("lead_s", 0.0)    # publish the action planned this far ahead (actuator lag)
        self.declare_parameter("num_steps", 0)   # denoising steps per inference; 0 = checkpoint default
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
        model_dir = None
        if model_path and model_path.strip():
            model_dir = resolve_model_dir(model_path)
            self.get_logger().info(f"loading smolvla from {model_dir} on {self.device}")
            load_dir = compat_model_dir(model_dir, self.get_logger())
            self.policy = SmolVLAPolicy.from_pretrained(str(load_dir)).to(self.device).eval()
            self.get_logger().info(f"loaded {model_dir}")
        else:
            self.get_logger().info(f"loading base smolvla on {self.device}")
            self.policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(self.device).eval()
            self.get_logger().info("loaded lerobot/smolvla_base")

        self.declare_parameter("stats_path", "")  # dataset meta/stats.json, if the postprocessor won't load
        self.act_mean, self.act_std = self._load_action_stats(
            model_dir, self.get_parameter("stats_path").value)

        self.chunk_size = int(self.policy.config.chunk_size)
        self.n_action_steps = int(self.get_parameter("n_action_steps").value)
        self.n_action_steps = max(1, min(self.n_action_steps, self.chunk_size))
        # whole chunk per inference; we pick the time-aligned part of it ourselves
        self.policy.config.n_action_steps = self.chunk_size
        self.policy.reset()

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.stop_when_empty = bool(self.get_parameter("stop_when_empty").value)
        self.lead_s = float(self.get_parameter("lead_s").value)

        num_steps = int(self.get_parameter("num_steps").value)
        if num_steps > 0:
            self.policy.config.num_steps = num_steps
            inner = getattr(getattr(self.policy, "model", None), "config", None)
            if inner is not None:
                inner.num_steps = num_steps
        self.get_logger().info(f"denoising num_steps={self.policy.config.num_steps}, lead_s={self.lead_s:.2f}")

        self.get_logger().info(
            f"n_action_steps={self.n_action_steps} "
            f"chunk_size={self.chunk_size} "
            f"rate_hz={self.rate_hz:.1f} (action spacing {1000 / self.rate_hz:.0f} ms)"
        )

        self.img_keys = list(self.policy.config.image_features)
        tok_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        self.tok = AutoTokenizer.from_pretrained(tok_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self._set_task(self.get_parameter("task").value)

        self.latest_rgb = None
        self.latest_depth = None
        self.rgb_t = None                # monotonic receive time of latest_rgb / latest_depth
        self.depth_t = None
        self.max_image_age_s = float(self.get_parameter("max_image_age_s").value)
        self.plan = None                 # (chunk [N,2], monotonic time of its images)
        self.plan_lock = threading.Lock()
        self.running = True
        self.warned_empty = False

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

        self.infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.infer_thread.start()

    def _load_action_stats(self, model_dir, stats_path):
        """The policy outputs mean/std-normalised actions; recover speed = raw*std + mean.
        Normally the checkpoint's postprocessor holds these stats."""
        # stats_path: absolute, or relative to the repo root (like model_path). Empty -> use
        # <model folder>/dataset_stats.json if it exists, else the checkpoint's postprocessor.
        if stats_path:
            sp = Path(stats_path).expanduser()
            if not sp.is_absolute():
                sp = Path(__file__).resolve().parents[4] / sp
            stats_path = str(sp)
        elif model_dir is not None and (Path(model_dir) / "dataset_stats.json").exists():
            stats_path = str(Path(model_dir) / "dataset_stats.json")
        if stats_path:
            self.get_logger().info(f"action stats from {stats_path}")
        try:
            if stats_path:
                import json
                s = json.loads(Path(stats_path).read_text())["action"]
                mean = np.asarray(s["mean"], dtype=np.float32)
                std = np.asarray(s["std"], dtype=np.float32)
            else:
                from lerobot.policies.factory import make_pre_post_processors
                _, post = make_pre_post_processors(
                    self.policy.config,
                    pretrained_path=str(model_dir) if model_dir else "lerobot/smolvla_base",
                )
                m = torch.as_tensor(post(torch.zeros(1, 2))).float().cpu().numpy().reshape(-1)
                o = torch.as_tensor(post(torch.ones(1, 2))).float().cpu().numpy().reshape(-1)
                mean, std = m, o - m
            self.get_logger().info(
                f"action un-normalisation: speed = raw*{std[0]:.4f} + {mean[0]:.4f}, "
                f"steering = raw*{std[1]:.4f} + {mean[1]:.4f}")
            return mean[:2], std[:2]
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(
                f"could not load action stats ({e}); publishing RAW normalised actions. "
                "Set the stats_path parameter to <dataset>/meta/stats.json to fix.")
            return np.zeros(2, np.float32), np.ones(2, np.float32)

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
        self.rgb_t = time.monotonic()

    def _on_depth(self, msg: Image):
        self.latest_depth = msg
        self.depth_t = time.monotonic()

    def _zeros(self):
        return torch.zeros(1, 3, 240, 424, device=self.device)

    def _run_inference(self):
        """One model run on the newest images. Returns (chunk [N,2] in setpoint units,
        monotonic time the images were taken, inference seconds), or (None, None, 0)."""
        rgb_msg, depth_msg = self.latest_rgb, self.latest_depth   # snapshot
        # The plan's clock starts when the oldest image it uses arrived, not now.
        used = [t for t, on in ((self.rgb_t, self.mode in ("rgb", "dual")),
                                (self.depth_t, self.mode in ("depth", "dual"))) if on and t is not None]
        obs_time = min(used) if used else time.monotonic()

        # Build the two image tensors according to mode
        if self.mode in ("rgb", "dual"):
            if rgb_msg is None:
                return None, None, 0.0
            try:
                cam1 = image_msg_to_nchw(rgb_msg).to(self.device)
            except Exception as e:
                self.get_logger().error(f"rgb convert: {e}")
                return None, None, 0.0
        else:
            cam1 = self._zeros()

        if self.mode in ("depth", "dual"):
            if depth_msg is None:
                return None, None, 0.0
            try:
                cam2 = image_msg_to_nchw(depth_msg).to(self.device)
            except Exception as e:
                self.get_logger().error(f"depth convert: {e}")
                return None, None, 0.0
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
                if hasattr(self.policy, "predict_action_chunk"):
                    out = self.policy.predict_action_chunk(batch)[0]          # [chunk, action_dim]
                else:                                                         # older lerobot
                    self.policy.reset()
                    out = torch.stack([self.policy.select_action(batch)[0]
                                       for _ in range(self.chunk_size)])
                raw = out.detach().float().cpu().numpy()[:, :2]
            dt = time.perf_counter() - t0
        finally:
            led_off()
        chunk = raw * self.act_std + self.act_mean   # back to setpoint units; clamping is in smolvla_to_drive
        return chunk, obs_time, dt

    def _images_fresh(self):
        now = time.monotonic()
        need = []
        if self.mode in ("rgb", "dual"):
            need.append(self.rgb_t)
        if self.mode in ("depth", "dual"):
            need.append(self.depth_t)
        return all(t is not None and now - t <= self.max_image_age_s for t in need)

    def _inference_loop(self):
        """Background thread: run the model back to back, swap in each new chunk."""
        was_fresh = False
        while self.running and rclpy.ok():
            if not self._images_fresh():
                if was_fresh:                # camera stream stopped or link dropped
                    with self.plan_lock:
                        self.plan = None     # -> _tick publishes stop at once
                    self.get_logger().warn(
                        f"camera images older than {self.max_image_age_s:.1f}s -> plan dropped, stopping")
                was_fresh = False
                time.sleep(0.05)
                continue
            was_fresh = True
            try:
                chunk, obs_time, dt = self._run_inference()
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"inference failed: {e}")
                time.sleep(0.5)
                continue
            if chunk is None:            # no images yet
                time.sleep(0.05)
                continue
            with self.plan_lock:
                self.plan = (chunk[:self.n_action_steps], obs_time)
            age = time.monotonic() - obs_time + self.lead_s
            self.pub_dt.publish(Float32(data=float(dt)))
            self.get_logger().info(
                f"dt={dt:.3f}s new chunk, starting at action {int(round(age * self.rate_hz))}",
                throttle_duration_sec=2.0)
            if 2 * age * self.rate_hz > self.n_action_steps:
                self.get_logger().warn(
                    f"inference ({dt:.2f}s) too slow for n_action_steps={self.n_action_steps} at "
                    f"{self.rate_hz:.0f} Hz; the queue can run dry -- raise n_action_steps",
                    throttle_duration_sec=5.0)

    def _tick(self):
        with self.plan_lock:
            plan = self.plan
        vec = None
        if plan is not None:
            chunk, obs_time = plan
            i = int(round((time.monotonic() - obs_time + self.lead_s) * self.rate_hz))
            if 0 <= i < len(chunk):
                vec = [float(v) for v in chunk[i]]
        if vec is None:
            if not self.stop_when_empty:
                return
            vec = [0.0, 0.0]             # no current plan: stop rather than hold the last command
            if not self.warned_empty:
                self.get_logger().warn("no current plan (startup, or inference stalled) -> publishing stop")
                self.warned_empty = True
        else:
            self.warned_empty = False
        act = Float32MultiArray()
        act.data = vec
        self.pub_act.publish(act)

def main():
    rclpy.init()
    node = SmolVLAOpenLoop()
    try:
        rclpy.spin(node)
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()