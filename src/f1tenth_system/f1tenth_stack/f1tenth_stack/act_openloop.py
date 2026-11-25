#!/usr/bin/env python3
"""ACT driving node -- the ACT counterpart of smolvla_openloop.py.

Publishes on the same topics as the SmolVLA node (/smolvla/action, /smolvla/infer_dt,
/smolvla/chunk_*), so smolvla_to_drive, recording, compare_runs.py and the takeover
pipeline work unchanged.

Differences from the SmolVLA node:
  * Inputs are normalised here: ACT normalises images and state (SmolVLA images are
    not). Stats come from the checkpoint's normaliser files; missing image stats fall
    back to ImageNet mean/std (lerobot's default use_imagenet_stats=True).
  * No text: the task string (parameter or /task) becomes observation.state =
    [direction +1 anticlockwise / -1 clockwise, fast 1/0], exactly as
    bag_to_lerobot.py --state-from-task wrote it.
  * Re-plans on every new camera frame (ACT is fast). Each 1/rate_hz tick publishes
    the time-aligned action, either from the newest plan or, with ensemble=true, a
    weighted average of all plans that cover the current moment (ACT's temporal
    ensembling: weight exp(-ensemble_coeff * k), k = 0 for the oldest plan).
"""
import collections
import dataclasses
import json
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray, String

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

DEPTH_MIN = 200   # must match bag_to_lerobot.py
DEPTH_MAX = 2000
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# --- helpers (same behaviour as smolvla_openloop.py) ----------------------------
def resolve_model_dir(model_path: str) -> Path:
    p = Path(model_path).expanduser()
    if p.is_absolute() and p.is_dir():
        return p
    repo_root = Path(__file__).resolve().parents[4]
    candidate = (repo_root / p).resolve()
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"model not found: {model_path} (tried {candidate})")


def compat_model_dir(model_dir: Path, logger=None) -> Path:
    """Drop config.json fields this lerobot's ACTConfig doesn't know (checkpoint from a
    newer lerobot) via a temporary symlinked copy; the model folder is not modified."""
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return model_dir
    cfg = json.loads(cfg_path.read_text())
    known = {f.name for f in dataclasses.fields(ACTConfig)} | {"type"}
    unknown = sorted(k for k in cfg if k not in known)
    if not unknown:
        return model_dir
    tmp = Path(tempfile.mkdtemp(prefix="act_compat_"))
    for f in model_dir.iterdir():
        if f.name != "config.json":
            (tmp / f.name).symlink_to(f.resolve())
    (tmp / "config.json").write_text(json.dumps({k: v for k, v in cfg.items() if k in known}, indent=4))
    if logger is not None:
        logger.warn(f"checkpoint config has fields this lerobot doesn't know, ignoring them: {unknown}")
    return tmp


def image_msg_to_nchw(msg: Image) -> torch.Tensor:
    """RGB or 16-bit depth (turbo-coloured, invalid = magenta) -> [1,3,H,W] float 0..1."""
    h, w = msg.height, msg.width
    if msg.encoding in ("rgb8", "bgr8"):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
        if msg.encoding == "bgr8":
            img = img[:, :, ::-1]
    elif msg.encoding in ("16UC1", "mono16"):
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
        invalid = depth == 0
        d = np.clip(depth.astype(np.float32), DEPTH_MIN, DEPTH_MAX)
        norm = ((d - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * 255).astype(np.uint8)
        img = cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        img[invalid] = [255, 0, 255]
    else:
        raise RuntimeError(f"unsupported encoding {msg.encoding}")
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


def task_to_state(task: str) -> np.ndarray:
    """Same encoding as bag_to_lerobot.py --state-from-task."""
    t = (task or "").lower()
    d = 1.0 if "anticlockwise" in t else (-1.0 if "clockwise" in t else 0.0)
    return np.array([d, 1.0 if "fast" in t else 0.0], dtype=np.float32)


def load_stats(model_dir: Path, stats_path: str, logger):
    """{feature: {"mean": tensor, "std": tensor}} from the checkpoint's normaliser
    safetensors (exactly what training used), plus an optional stats json on top."""
    stats = collections.defaultdict(dict)
    try:
        from safetensors.torch import load_file
        for f in sorted(model_dir.glob("*normalizer*.safetensors")):
            for k, v in load_file(str(f)).items():
                feat, _, stat = k.rpartition(".")
                if stat in ("mean", "std"):
                    stats[feat][stat] = v.float()
            logger.info(f"normalisation stats from {f.name}")
    except Exception as e:  # noqa: BLE001
        logger.warn(f"could not read normaliser safetensors: {e}")
    sp = None
    if stats_path:
        sp = Path(stats_path).expanduser()
        if not sp.is_absolute():
            sp = Path(__file__).resolve().parents[4] / sp
    elif (model_dir / "dataset_stats.json").exists():
        sp = model_dir / "dataset_stats.json"
    if sp is not None and sp.exists():
        js = json.loads(sp.read_text())
        for feat, s in js.items():
            for stat in ("mean", "std"):
                if stat in s and stat not in stats.get(feat, {}):
                    stats[feat][stat] = torch.tensor(s[stat], dtype=torch.float32)
        logger.info(f"extra stats from {sp}")
    return stats


class ACTOpenLoop(Node):
    def __init__(self):
        super().__init__("act_openloop")
        P = self.declare_parameter
        P("rgb_topic", "/camera/camera/color/image_raw")
        P("depth_topic", "/camera/camera/depth/image_rect_raw")
        P("model_path", "models/track_v5_act")
        P("stats_path", "")
        P("task", "complete laps anticlockwise slowly")
        P("mode", "dual")               # rgb | depth | dual (as trained: dual)
        P("rate_hz", 10.0)              # = dataset fps
        P("lead_s", 0.0)
        P("ensemble", True)             # temporal ensembling over overlapping plans
        P("ensemble_coeff", 0.01)       # ACT default (~plain average); larger = favour OLDER plans (smoother,
                                        # slower to react); negative = favour newer plans
        P("max_image_age_s", 0.5)
        P("stop_when_empty", True)
        P("publish_chunk", True)
        P("action_topic", "/smolvla/action")
        P("infer_dt_topic", "/smolvla/infer_dt")
        gp = lambda n: self.get_parameter(n).value  # noqa: E731

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_dir = resolve_model_dir(gp("model_path"))
        self.get_logger().info(f"loading ACT from {model_dir} on {self.device}")
        self.policy = ACTPolicy.from_pretrained(str(compat_model_dir(model_dir, self.get_logger())))
        self.policy.to(self.device).eval()
        self.chunk_size = int(self.policy.config.chunk_size)
        self.img_keys = list(self.policy.config.image_features)

        # normalisation
        st = load_stats(model_dir, gp("stats_path"), self.get_logger())
        self.img_norm = {}
        for k in self.img_keys:
            if "mean" in st.get(k, {}) and "std" in st[k]:
                m, s = st[k]["mean"], st[k]["std"]
                src = "checkpoint"
            else:
                m, s, src = IMAGENET_MEAN, IMAGENET_STD, "ImageNet default"
            self.img_norm[k] = (m.reshape(1, 3, 1, 1).to(self.device), s.reshape(1, 3, 1, 1).to(self.device))
            self.get_logger().info(f"{k}: image stats from {src}")
        if "observation.state" not in st or "action" not in st:
            raise RuntimeError("state/action stats not found: copy the dataset's meta/stats.json "
                               "to <model>/dataset_stats.json or set stats_path")
        self.state_mean = st["observation.state"]["mean"].reshape(-1).to(self.device)
        self.state_std = st["observation.state"]["std"].reshape(-1).to(self.device)
        self.act_mean = st["action"]["mean"].reshape(-1)[:2].numpy()
        self.act_std = st["action"]["std"].reshape(-1)[:2].numpy()
        self.get_logger().info(
            f"state mean {self.state_mean.tolist()} std {self.state_std.tolist()}; "
            f"action: speed = raw*{self.act_std[0]:.4f} + {self.act_mean[0]:.4f}, "
            f"steering = raw*{self.act_std[1]:.4f} + {self.act_mean[1]:.4f}")

        self.mode = str(gp("mode")).lower()
        self.rate_hz = float(gp("rate_hz"))
        self.lead_s = float(gp("lead_s"))
        self.ensemble = bool(gp("ensemble"))
        self.ens_coeff = float(gp("ensemble_coeff"))
        self.max_image_age_s = float(gp("max_image_age_s"))
        self.stop_when_empty = bool(gp("stop_when_empty"))
        self.publish_chunk = bool(gp("publish_chunk"))
        self._set_task(gp("task"))

        self.latest_rgb = self.latest_depth = None
        self.rgb_t = self.depth_t = None
        self.last_used_rgb_t = None
        self.plans = collections.deque()      # (chunk [N,2], obs_time)
        self.plan_lock = threading.Lock()
        self.running = True
        self.warned_empty = False

        self.create_subscription(Image, gp("rgb_topic"), self._on_rgb, qos_profile_sensor_data)
        self.create_subscription(Image, gp("depth_topic"), self._on_depth, qos_profile_sensor_data)
        self.create_subscription(String, "/task", lambda m: self._set_task(m.data), 10)
        self.pub_act = self.create_publisher(Float32MultiArray, gp("action_topic"), 10)
        self.pub_dt = self.create_publisher(Float32, gp("infer_dt_topic"), 10)
        self.pub_chunk_steer = self.create_publisher(Float32MultiArray, "/smolvla/chunk_steer", 10)
        self.pub_chunk_speed = self.create_publisher(Float32MultiArray, "/smolvla/chunk_speed", 10)
        self.create_timer(1.0 / self.rate_hz, self._tick)
        self.get_logger().info(
            f"chunk_size={self.chunk_size} ({self.chunk_size / self.rate_hz:.1f} s), image keys {self.img_keys}, "
            f"mode={self.mode}, ensemble={self.ensemble} (coeff {self.ens_coeff}), lead_s={self.lead_s}")
        threading.Thread(target=self._inference_loop, daemon=True).start()

    # --- inputs ---------------------------------------------------------------
    def _set_task(self, text):
        self.task = text or ""
        s = torch.from_numpy(task_to_state(self.task)).to(self.device)
        std = self.state_std.clone() if hasattr(self, "state_std") else torch.ones(2, device=self.device)
        mean = self.state_mean if hasattr(self, "state_mean") else torch.zeros(2, device=self.device)
        z = (s - mean) / (std + 1e-8)
        z[std < 1e-6] = 0.0          # a state dim that never varied in training (e.g. no "fast" data)
        self.state = z.unsqueeze(0)
        if s[0] == 0:
            self.get_logger().warn(f"task '{self.task}' has no direction word -> direction 0")
        self.get_logger().info(f"task set: '{self.task}' -> state {s.tolist()}")

    def _on_rgb(self, msg):
        self.latest_rgb, self.rgb_t = msg, time.monotonic()

    def _on_depth(self, msg):
        self.latest_depth, self.depth_t = msg, time.monotonic()

    def _images_fresh(self):
        now = time.monotonic()
        need = ([self.rgb_t] if self.mode in ("rgb", "dual") else []) + \
               ([self.depth_t] if self.mode in ("depth", "dual") else [])
        return all(t is not None and now - t <= self.max_image_age_s for t in need)

    def _cam(self, msg, key):
        x = image_msg_to_nchw(msg).to(self.device)
        m, s = self.img_norm[key]
        return (x - m) / s

    # --- inference ------------------------------------------------------------
    def _run_inference(self):
        rgb, depth = self.latest_rgb, self.latest_depth
        used = [t for t, on in ((self.rgb_t, self.mode in ("rgb", "dual")),
                                (self.depth_t, self.mode in ("depth", "dual"))) if on]
        if any(t is None for t in used) or (self.mode in ("rgb", "dual") and rgb is None) \
                or (self.mode in ("depth", "dual") and depth is None):
            return None, None, 0.0
        obs_time = min(used)
        batch = {"observation.state": self.state}
        for i, k in enumerate(self.img_keys):
            msg = rgb if i == 0 else depth
            if (i == 0 and self.mode == "depth") or (i > 0 and self.mode == "rgb"):
                batch[k] = torch.zeros(1, 3, 240, 424, device=self.device)
            else:
                batch[k] = self._cam(msg, k)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.policy.predict_action_chunk(batch)[0]       # [chunk, action_dim], normalised
        if self.device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        raw = out.float().cpu().numpy()[:, :2]
        return raw * self.act_std + self.act_mean, obs_time, dt

    def _inference_loop(self):
        was_fresh = False
        while self.running and rclpy.ok():
            if not self._images_fresh():
                if was_fresh:
                    with self.plan_lock:
                        self.plans.clear()
                    self.get_logger().warn(f"camera images older than {self.max_image_age_s:.1f}s -> stopping")
                was_fresh = False
                time.sleep(0.05)
                continue
            was_fresh = True
            if self.rgb_t == self.last_used_rgb_t:     # wait for a new frame; no point re-planning
                time.sleep(0.005)
                continue
            self.last_used_rgb_t = self.rgb_t
            try:
                chunk, obs_time, dt = self._run_inference()
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"inference failed: {e}")
                time.sleep(0.5)
                continue
            if chunk is None:
                time.sleep(0.05)
                continue
            with self.plan_lock:
                self.plans.append((chunk, obs_time))
                horizon = self.chunk_size / self.rate_hz + 0.5
                while self.plans and time.monotonic() - self.plans[0][1] > horizon:
                    self.plans.popleft()
            self.pub_dt.publish(Float32(data=float(dt)))
            if self.publish_chunk:
                for pub, col in ((self.pub_chunk_speed, 0), (self.pub_chunk_steer, 1)):
                    pub.publish(Float32MultiArray(data=[float(v) for v in chunk[:, col]]))
            self.get_logger().info(f"dt={dt * 1000:.0f} ms, {len(self.plans)} plans held",
                                   throttle_duration_sec=2.0)

    def _tick(self):
        now = time.monotonic()
        with self.plan_lock:
            plans = list(self.plans)
        vals, weights = [], []
        candidates = plans if self.ensemble else plans[-1:]
        for k, (chunk, obs_time) in enumerate(candidates):   # k = 0 is the oldest plan
            i = max(int(round((now - obs_time + self.lead_s) * self.rate_hz)), 0)
            if i < len(chunk):
                vals.append(chunk[i])
                weights.append(np.exp(-self.ens_coeff * k))
        if vals:
            w = np.asarray(weights)
            vec = [float(v) for v in (np.asarray(vals) * w[:, None]).sum(0) / w.sum()]
            self.warned_empty = False
        else:
            if not self.stop_when_empty:
                return
            vec = [0.0, 0.0]
            if not self.warned_empty:
                self.get_logger().warn("no current plan -> publishing stop")
                self.warned_empty = True
        self.pub_act.publish(Float32MultiArray(data=vec))


def main():
    rclpy.init()
    node = ACTOpenLoop()
    try:
        rclpy.spin(node)
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
