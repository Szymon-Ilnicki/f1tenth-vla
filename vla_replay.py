#!/usr/bin/env python3
"""Offline replay of SmolVLA on a recorded rosbag -- no car on the track needed.

For every sampled camera frame in the bag this:
  1. builds the model input the same way smolvla_openloop.py does on the car,
  2. asks the policy for its full action chunk [speed, steering],
  3. treats the chunk as setpoints for the car's speed PID (pid_controller.py,
     gains from config/vesc.yaml) and simulates PID + motor + rate-limited steering
     + bicycle model (wheelbase 0.25 m) to get the path the car would really drive,
  4. compares it with the path the human actually drove next (measured: /odom speed
     + IMU yaw rate) and with the walls seen by the depth camera.

The motor lag, how well the VESC follows the PID output, and the real steering gain
are fitted per bag from /teleop vs /odom and the IMU, and printed at the start.
--vehicle ideal skips all that and treats commands as exact m/s and radians.

Outputs (in --out):
  replay.mp4   camera view with the planned wheel tracks drawn on the floor,
               next to a top-down view on a 0.5 m grid (= one floor tile).
               --show also displays it live while it runs.
  plans.csv    one row per plan: commands, errors vs human, wall clearance over --check-s
  plans.npz    every raw / deployed / unnormalised chunk and the human reference
  summary.png  steering + speed over time, and a dead-reckoned map with plan fans
  run_info.json  everything needed to reproduce/compare the run: command line, all
               arguments, model details, bag, fitted car model, software versions
               and the result summary. Written at the start and updated as it goes.

Two versions of each plan are drawn:
  deployed      raw policy output -> sign flip on speed -> clamp. This is exactly
                what smolvla_openloop.py + smolvla_to_drive.py send to /drive.
  unnormalised  raw output passed through the checkpoint's postprocessor
                (mean/std un-normalisation) first, then the same clamp.

Reading bags uses rosbag2_py when ROS is sourced, otherwise `pip install rosbags`.

Example:
  python3 vla_replay.py --bag bags/track_v2/ep__20261106_140429 \
      --model models/track_v2_dual-image --out replay_out/ep1
"""
import argparse
import csv
import datetime
import json
import math
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---- constants mirrored from the repo --------------------------------------
DEPTH_MIN, DEPTH_MAX = 200, 2000          # bag_to_lerobot.py / smolvla_openloop.py
WHEELBASE = 0.25                          # config/vesc.yaml
TOKENIZER = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

ACKERMANN_DRIVE_MSG = """float32 steering_angle
float32 steering_angle_velocity
float32 speed
float32 acceleration
float32 jerk
"""
ACKERMANN_STAMPED_MSG = """std_msgs/Header header
AckermannDrive drive
"""

# BGR colours for the video
C_HUMAN = (255, 255, 255)
C_DEPLOYED = (0, 140, 255)
C_UNNORM = (120, 220, 80)
C_WALL = (150, 150, 150)
C_HIT = (60, 60, 255)


# =============================================================================
# Run log
# =============================================================================
RUN_INFO = {}
RUN_INFO_PATH = None


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, float) and v != v:
        return None
    return v


def log_info(section, **values):
    """Record values under RUN_INFO[section] and rewrite run_info.json."""
    RUN_INFO.setdefault(section, {}).update({k: _jsonable(v) for k, v in values.items()})
    if RUN_INFO_PATH is not None:
        RUN_INFO_PATH.write_text(json.dumps(RUN_INFO, indent=2, default=str))


def _git_commit(path):
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("+uncommitted changes" if dirty else "")
    except Exception:  # noqa: BLE001
        return None


def _versions():
    v = {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__}
    for mod in ("torch", "lerobot", "transformers", "rosbags"):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "installed")
        except Exception:  # noqa: BLE001
            v[mod] = None
    try:
        import rosbag2_py  # noqa: F401
        v["bag_reader"] = "rosbag2_py"
    except ImportError:
        v["bag_reader"] = "rosbags"
    return v


# =============================================================================
# Bag reading
# =============================================================================
def read_bag(bag_dir, topics):
    """Yield (topic, t_seconds, msg) for the wanted topics, in bag order."""
    bag_dir = Path(bag_dir)
    try:
        import rosbag2_py  # noqa: F401
        have_ros = True
    except ImportError:
        have_ros = False
    if have_ros:
        yield from _read_rosbag2_py(bag_dir, topics)
    else:
        yield from _read_rosbags(bag_dir, topics)


def _read_rosbag2_py(bag_dir, topics):
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions

    storage = "mcap" if list(bag_dir.glob("*.mcap")) else "sqlite3"
    r = SequentialReader()
    r.open(
        StorageOptions(uri=str(bag_dir), storage_id=storage),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    types = {c.name: get_message(c.type) for c in r.get_all_topics_and_types() if c.name in topics}
    r.set_filter(StorageFilter(topics=list(types)))
    while r.has_next():
        topic, data, t = r.read_next()
        if topic in types:
            yield topic, t * 1e-9, deserialize_message(data, types[topic])


def _read_rosbags(bag_dir, topics):
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    except ImportError:
        sys.exit("No bag reader found: source ROS 2 (rosbag2_py) or `pip install rosbags`.")
    ts = get_typestore(Stores.ROS2_HUMBLE)
    ts.register(get_types_from_msg(ACKERMANN_DRIVE_MSG, "ackermann_msgs/msg/AckermannDrive"))
    ts.register(get_types_from_msg(ACKERMANN_STAMPED_MSG, "ackermann_msgs/msg/AckermannDriveStamped"))
    with AnyReader([bag_dir], default_typestore=ts) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        for conn, t_ns, raw in reader.messages(connections=conns):
            yield conn.topic, t_ns * 1e-9, reader.deserialize(raw, conn.msgtype)


# =============================================================================
# Image conversion (same maths as the repo)
# =============================================================================
def rgb_from_msg(msg):
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    img = arr.reshape(msg.height, msg.width, 3)
    if msg.encoding == "bgr8":
        img = img[:, :, ::-1]
    elif msg.encoding != "rgb8":
        raise RuntimeError(f"unsupported colour encoding {msg.encoding}")
    return np.ascontiguousarray(img)


def depth_from_msg(msg):
    if msg.encoding not in ("16UC1", "mono16"):
        raise RuntimeError(f"unsupported depth encoding {msg.encoding}")
    return np.frombuffer(msg.data, dtype=np.uint8).view(np.uint16).reshape(msg.height, msg.width)


def depth_to_rgb(depth):
    """16-bit depth (mm) -> turbo-coloured RGB, invalid pixels magenta."""
    invalid = depth == 0
    d = np.clip(depth.astype(np.float32), DEPTH_MIN, DEPTH_MAX)
    norm = ((d - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * 255).astype(np.uint8)
    color = cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    color[invalid] = [255, 0, 255]
    return color


def node_resize(img):
    """Nearest-neighbour stretch to 224x224, as image_msg_to_nchw() does on the car."""
    h, w = img.shape[:2]
    ys = np.linspace(0, h - 1, 224).astype(np.int32)
    xs = np.linspace(0, w - 1, 224).astype(np.int32)
    return img[ys][:, xs]


# =============================================================================
# Human commands
# =============================================================================
class Commands:
    """Recorded /teleop commands with zero-order hold; stale commands mean stopped."""

    def __init__(self, t, speed, steer, stale_s=0.5):
        order = np.argsort(t)
        self.t = np.asarray(t, dtype=np.float64)[order]
        self.speed = np.asarray(speed, dtype=np.float64)[order]
        self.steer = np.asarray(steer, dtype=np.float64)[order]
        self.stale_s = stale_s

    def at(self, t):
        """(speed, steer) arrays for an array of times."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        i = np.searchsorted(self.t, t, side="right") - 1
        ok = (i >= 0) & (t - self.t[np.clip(i, 0, None)] <= self.stale_s)
        i = np.clip(i, 0, len(self.t) - 1)
        return np.where(ok, self.speed[i], 0.0), np.where(ok, self.steer[i], 0.0)


def load_signals(bag, a, info_topics):
    """First pass over the bag: teleop setpoints, /odom speed, IMU yaw rate, camera intrinsics."""
    t, sp, st = [], [], []
    odom_t, odom_v, imu_t, imu_w = [], [], [], []
    intr = {}
    for topic, ts, msg in read_bag(bag, {a.cmd_topic, a.odom_topic, a.imu_topic, *info_topics}):
        if topic == a.cmd_topic:
            d = msg.drive if hasattr(msg, "drive") else msg
            t.append(ts)
            sp.append(float(d.speed))
            st.append(float(d.steering_angle))
        elif topic == a.odom_topic:
            odom_t.append(ts)
            odom_v.append(float(msg.twist.twist.linear.x))
        elif topic == a.imu_topic:
            imu_t.append(ts)
            imu_w.append(float(msg.angular_velocity.z))
        elif topic not in intr:
            k = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
            intr[topic] = (k[0, 0], k[1, 1], k[0, 2], k[1, 2])
    if not t:
        sys.exit(f"no messages on {a.cmd_topic} in {bag}")
    odom = (np.array(odom_t), np.array(odom_v)) if odom_t else None
    imu = (np.array(imu_t), np.array(imu_w)) if imu_t else None
    return Commands(t, sp, st), odom, imu, intr


# =============================================================================
# Vehicle: speed PID (pid_controller.py) + motor lag + rate-limited steering
# =============================================================================
H_SIM = 0.01          # pid_controller.py runs at 100 Hz
DEADBAND = 0.05       # |setpoint| below this -> PID outputs 0 and resets the integral


def speed_loop(T, h, v0, kp, ki, ilim, tau, gain, I0=0.0):
    """Simulate the PID + motor for setpoints T (n,). tau/gain may be arrays (grid search).
    Motor: dv/dt = (gain * u - v) / tau, u = PID output in m/s (sent to the VESC as erpm)."""
    tau, gain = np.broadcast_arrays(np.asarray(tau, float), np.asarray(gain, float))
    v = np.full(tau.shape, float(v0))
    I = np.full(tau.shape, float(I0))
    vs, Is = np.empty((len(T),) + tau.shape), np.empty((len(T),) + tau.shape)
    for i, target in enumerate(T):
        if abs(target) < DEADBAND:
            I = np.zeros_like(I)
            u = 0.0
        else:
            e = target - v
            I = np.clip(I + e * h, -ilim, ilim)
            u = kp * e + ki * I
        v = v + (gain * u - v) * h / tau
        vs[i], Is[i] = v, I
    return vs, Is


def steer_track(S, h, rate, s0=0.0):
    """pid_controller.py rate-limits the steering setpoint at `rate` rad/s."""
    out = np.empty(len(S))
    s = s0
    for i, target in enumerate(S):
        s += np.clip(target - s, -rate * h, rate * h)
        out[i] = s
    return out


class Vehicle:
    def __init__(self, a):
        self.a = a
        self.kp, self.ki, self.ilim, self.rate = a.kp, a.ki, a.i_limit, a.steer_rate
        self.tau, self.gain, self.k_steer = 0.2, 1.0, 1.0
        self.yaw_rate = None                  # callable t -> measured yaw rate [rad/s], if IMU usable
        self.odom = None

    # ---- fitting from the human part of the bag ----
    def fit(self, cmds, odom, imu):
        a = self.a
        self.odom = odom
        ot, ov = odom
        tt = np.arange(max(cmds.t[0], ot[0]), min(cmds.t[-1], ot[-1]), H_SIM)
        T, S = cmds.at(tt)
        v_meas = np.interp(tt, ot, ov)

        taus = np.array([0.03, 0.05, 0.08, 0.12, 0.17, 0.25, 0.35, 0.5, 0.7, 1.0, 1.5])
        gains = np.linspace(0.5, 1.8, 27)
        TG, GG = np.meshgrid(taus, gains, indexing="ij")
        vs, _ = speed_loop(T, H_SIM, v_meas[0], self.kp, self.ki, self.ilim, TG, GG)
        rmse = np.sqrt(np.mean((vs - v_meas[:, None, None]) ** 2, axis=0))
        i, j = np.unravel_index(np.argmin(rmse), rmse.shape)
        self.tau, self.gain = float(taus[i]), float(gains[j])
        vs, Is = speed_loop(T, H_SIM, v_meas[0], self.kp, self.ki, self.ilim, self.tau, self.gain)
        self.track_t, self.track_I = tt, Is
        self.track_s = steer_track(S, H_SIM, self.rate)
        print(f"  speed loop fit: motor lag tau={self.tau:.2f}s, VESC gain={self.gain:.2f}, "
              f"speed RMSE {rmse[i, j]:.2f} m/s (setpoint-as-speed RMSE {np.sqrt(np.mean((T - v_meas) ** 2)):.2f})")
        moving = np.abs(T) >= DEADBAND
        if moving.any():
            print(f"  while driving: setpoint mean {T[moving].mean():.2f}, measured speed mean "
                  f"{v_meas[moving].mean():.2f} m/s")
        print("  steady state:  " + ",  ".join(
            f"setpoint {sp:.1f} -> {self.steady(sp):.2f} m/s" for sp in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)))
        log_info("vehicle_fit", motor_tau_s=self.tau, vesc_gain=self.gain, speed_rmse_m_s=float(rmse[i, j]),
                 setpoint_mean_driving=float(T[moving].mean()) if moving.any() else None,
                 measured_speed_mean_driving_m_s=float(v_meas[moving].mean()) if moving.any() else None,
                 steady_state_m_s={str(sp): self.steady(sp) for sp in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)},
                 steering_gain=1.0, imu_used=False)
        if rmse[i, j] > 0.3:
            print("  WARNING: poor speed fit -- check --kp/--ki/--i-limit match the PID used when recording.")

        # ---- steering gain from the IMU ----
        r0 = v_meas * np.tan(self.track_s) / WHEELBASE           # yaw rate if steering is exact
        mask = (np.abs(v_meas) > 0.3) & (np.abs(r0) > 0.05)
        if imu is None or mask.sum() < 100:
            print("  no usable IMU yaw rate -> steering taken as exact (k=1.0), "
                  "human path from /odom speed + commanded steering")
            return
        it, iw = imu
        w = np.interp(tt, it, iw)
        still = (np.abs(v_meas) < 0.03) & (np.abs(T) < DEADBAND)
        bias = float(np.median(w[still])) if still.sum() > 50 else 0.0
        w = w - bias
        b = float(np.sum(w[mask] * r0[mask]) / np.sum(r0[mask] ** 2))
        scale = 1.0
        if abs(b) > 8:                       # VESC firmware reports gyro in deg/s
            scale = math.pi / 180
            b *= scale
        sign = 1.0 if b >= 0 else -1.0
        corr = np.corrcoef(w[mask], r0[mask])[0, 1] * sign
        if corr < 0.5:
            print(f"  IMU yaw rate correlates poorly with steering (r={corr:.2f}) -> not used")
            return
        wz = w * scale * sign
        ks = np.linspace(0.3, 1.8, 151)
        err = [np.mean((wz[mask] - v_meas[mask] * np.tan(k * self.track_s[mask]) / WHEELBASE) ** 2) for k in ks]
        self.k_steer = float(ks[int(np.argmin(err))])
        self.yaw_rate = lambda t: np.interp(t, it, (iw - bias) * scale * sign)
        print(f"  IMU: {'deg/s' if scale != 1 else 'rad/s'}, sign {sign:+.0f}, bias {bias:.3f}, "
              f"r={corr:.2f} -> real steering = {self.k_steer:.2f} x commanded")
        log_info("vehicle_fit", steering_gain=self.k_steer, imu_used=True,
                 imu_units="deg/s" if scale != 1 else "rad/s", imu_sign=sign, imu_bias=bias, imu_corr=float(corr))

    def steady(self, sp):
        v, _ = speed_loop(np.full(int(15 / H_SIM), sp), H_SIM, 0.0, self.kp, self.ki, self.ilim,
                          self.tau, self.gain)
        return float(v[-1])

    # ---- state at a frame ----
    def state(self, t):
        v0 = float(np.interp(t, *self.odom))
        I0 = float(np.interp(t, self.track_t, self.track_I))
        s0 = float(np.interp(t, self.track_t, self.track_s))
        return v0, I0, s0

    # ---- simulate a chunk of setpoints ----
    def simulate(self, actions, dt, sub, v0, I0, s0):
        n_in = max(1, int(round(dt / sub / H_SIM)))
        h = dt / sub / n_in
        x = y = th = 0.0
        v, I, s = v0, I0, s0
        out = [(0.0, 0.0, 0.0)]
        for target, steer in actions:
            for _ in range(sub):
                for _ in range(n_in):
                    if abs(target) < DEADBAND:
                        I, u = 0.0, 0.0
                    else:
                        e = target - v
                        I = min(max(I + e * h, -self.ilim), self.ilim)
                        u = self.kp * e + self.ki * I
                    v += (self.gain * u - v) * h / self.tau
                    s += min(max(steer - s, -self.rate * h), self.rate * h)
                    w = v * math.tan(self.k_steer * s) / WHEELBASE
                    x += v * math.cos(th) * h
                    y += v * math.sin(th) * h
                    th += w * h
                out.append((x, y, th))
        return np.array(out)

    # ---- what the human really did ----
    def measured(self, t0, n_steps, step):
        tt = t0 + np.arange(n_steps + 1) * step
        v = np.interp(tt, *self.odom)
        if self.yaw_rate is not None:
            w = self.yaw_rate(tt)
        else:
            s = np.interp(tt, self.track_t, self.track_s)
            w = v * np.tan(self.k_steer * s) / WHEELBASE
        out = np.zeros((n_steps + 1, 3))
        for i in range(n_steps):
            vm, wm = 0.5 * (v[i] + v[i + 1]), 0.5 * (w[i] + w[i + 1])
            th_mid = out[i, 2] + 0.5 * wm * step
            out[i + 1] = out[i] + (vm * math.cos(th_mid) * step, vm * math.sin(th_mid) * step, wm * step)
        return out


def iter_frames(bag, rgb_topic, depth_topic, hz):
    """Second pass: (t, rgb, raw_depth) at <= hz, pairing each colour frame with the last depth."""
    last_depth, last_keep = None, -1e18
    for topic, ts, msg in read_bag(bag, {rgb_topic, depth_topic}):
        if topic == depth_topic:
            last_depth = msg
        elif last_depth is not None and ts - last_keep >= 1.0 / hz - 1e-3:
            last_keep = ts
            yield ts, rgb_from_msg(msg), depth_from_msg(last_depth)


# =============================================================================
# Policies
# =============================================================================
class SmolVLARunner:
    def __init__(self, a):
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from transformers import AutoTokenizer

        self.torch = torch
        self.a = a
        self.device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = Path(a.model).expanduser()
        if not model_dir.is_dir():
            sys.exit(f"model dir not found: {model_dir}")
        print(f"loading {model_dir} on {self.device}")
        self.policy = SmolVLAPolicy.from_pretrained(str(model_dir)).to(self.device).eval()
        cfg = self.policy.config
        self.chunk = int(cfg.chunk_size)
        cfg.n_action_steps = self.chunk
        if a.num_steps > 0:                 # same as the node's num_steps parameter
            cfg.num_steps = a.num_steps
            inner = getattr(getattr(self.policy, "model", None), "config", None)
            if inner is not None:
                inner.num_steps = a.num_steps
        print(f"denoising steps={cfg.num_steps}")
        self.autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(a.precision)
        if self.autocast_dtype is not None and not self.device.startswith("cuda"):
            sys.exit("--precision fp16/bf16 needs a CUDA device")
        print(f"precision: {a.precision}" + (" (autocast)" if self.autocast_dtype else ""))
        self.policy.reset()
        self.img_keys = list(cfg.image_features)
        print(f"chunk_size={self.chunk} image keys={self.img_keys}")

        tok = AutoTokenizer.from_pretrained(getattr(cfg, "vlm_model_name", TOKENIZER))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        enc = tok(a.task.strip() + "\n", return_tensors="pt", padding="max_length", max_length=48)
        self.lang_tokens = enc.input_ids.to(self.device)
        self.lang_mask = enc.attention_mask.to(device=self.device, dtype=torch.bool)

        self.mean, self.std = self._load_action_stats(model_dir)
        self.infer_times = []
        train_cfg = {}
        tc = model_dir / "train_config.json"
        if tc.exists():
            try:
                t = json.loads(tc.read_text())
                train_cfg = {"dataset": (t.get("dataset") or {}).get("repo_id"), "steps": t.get("steps"),
                             "batch_size": t.get("batch_size"), "job_name": t.get("job_name")}
            except Exception:  # noqa: BLE001
                pass
        log_info("model", path=str(model_dir.resolve()), chunk_size=self.chunk,
                 num_steps=int(cfg.num_steps), precision=a.precision, image_keys=self.img_keys, device=self.device,
                 gpu=torch.cuda.get_device_name(0) if self.device.startswith("cuda") else None,
                 normalization={str(k): str(v) for k, v in getattr(cfg, "normalization_mapping", {}).items()},
                 action_mean=self.mean, action_std=self.std, training=train_cfg)

    def _load_action_stats(self, model_dir):
        """Recover the action un-normalisation as raw * std + mean."""
        torch = self.torch
        if self.a.stats:
            import json
            s = json.loads(Path(self.a.stats).read_text())["action"]
            mean, std = np.array(s["mean"], dtype=np.float64), np.array(s["std"], dtype=np.float64)
        else:
            try:
                from lerobot.policies.factory import make_pre_post_processors
                _, post = make_pre_post_processors(self.policy.config, pretrained_path=str(model_dir))
                m = torch.as_tensor(post(torch.zeros(1, 2))).float().cpu().numpy().reshape(-1)
                o = torch.as_tensor(post(torch.ones(1, 2))).float().cpu().numpy().reshape(-1)
                mean, std = m.astype(np.float64), (o - m).astype(np.float64)
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: could not load the checkpoint's postprocessor ({e}).\n"
                      "         Only the 'deployed' plan will be shown; pass --stats "
                      "<dataset>/meta/stats.json to get the unnormalised one.")
                return None, None
        print(f"action un-normalisation: speed = raw*{std[0]:.4f} + {mean[0]:.4f}, "
              f"steering = raw*{std[1]:.4f} + {mean[1]:.4f}")
        return mean, std

    def _img(self, img):
        if self.a.preproc == "node":
            img = node_resize(img)          # 224x224 stretch, as on the car
        t = self.torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(self.device)

    def plan(self, rgb, depth_rgb, human_future):
        torch = self.torch
        zeros = lambda ref: torch.zeros_like(ref)  # noqa: E731
        cam1 = self._img(rgb)
        cam2 = self._img(depth_rgb)
        if self.a.mode == "rgb":
            cam2 = zeros(cam2)
        elif self.a.mode == "depth":
            cam1 = zeros(cam1)
        batch = {
            "observation.state": torch.zeros(1, 6, device=self.device),
            "observation.language.tokens": self.lang_tokens,
            "observation.language.attention_mask": self.lang_mask,
        }
        for i, k in enumerate(self.img_keys):
            batch[k] = cam1 if i == 0 else cam2
        if self.a.seed >= 0:
            torch.manual_seed(self.a.seed)   # same denoising noise for every frame and every run
        t_inf = time.perf_counter()
        import contextlib
        ac = (torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
              if self.autocast_dtype is not None else contextlib.nullcontext())
        with torch.no_grad(), ac:
            if hasattr(self.policy, "predict_action_chunk"):
                out = self.policy.predict_action_chunk(batch)
            else:
                self.policy.reset()
                out = torch.stack([self.policy.select_action(batch) for _ in range(self.chunk)], dim=1)
        raw = out[0].float().cpu().numpy().astype(np.float64)[:, :2]
        self.infer_times.append(time.perf_counter() - t_inf)
        unnorm = raw * self.std + self.mean if self.mean is not None else None
        return raw, unnorm


class DummyRunner:
    """No model: echoes the human's future commands with noise. For checking the pipeline."""

    def __init__(self, a):
        self.rng = np.random.default_rng(0)
        self.mean, self.std = np.array([0.6, 0.0]), np.array([0.25, 0.2])

    def plan(self, rgb, depth_rgb, human_future):
        unnorm = human_future + self.rng.normal(0, [0.05, 0.04], human_future.shape)
        return (unnorm - self.mean) / self.std, unnorm


# =============================================================================
# Kinematics and geometry
# =============================================================================
def deploy_transform(actions, a):
    """smolvla_openloop.py speed sign flip + smolvla_to_drive.py clamp."""
    speed = np.clip(np.abs(actions[:, 0]), a.speed_min, a.speed_max)
    steer = np.clip(actions[:, 1], -a.steer_max, a.steer_max)
    return np.stack([speed, steer], axis=1)


def rollout(actions, dt, sub=5):
    """Kinematic bicycle from the rear axle at (0,0,0). x forward, y left, +steer = left.
    Returns (N*sub+1, 3) poses [x, y, yaw]."""
    x = y = th = 0.0
    out = [(0.0, 0.0, 0.0)]
    h = dt / sub
    for v, d in actions:
        w = v * math.tan(d) / WHEELBASE
        for _ in range(sub):
            x += v * math.cos(th) * h
            y += v * math.sin(th) * h
            th += w * h
            out.append((x, y, th))
    return np.array(out)


def wall_points(depth, intr, a):
    """Depth image -> top-down obstacle points (x forward, y left) in the car frame."""
    fx, fy, cx, cy = intr
    h, w = depth.shape
    step = 2
    v, u = np.mgrid[0:h:step, 0:w:step]
    z = depth[::step, ::step].astype(np.float32) / 1000.0
    ok = (z > 0.15) & (z < a.wall_range)
    z, u, v = z[ok], u[ok], v[ok]
    xc = (u - cx) * z / fx                 # right
    yc = (v - cy) * z / fy                 # down
    p = math.radians(a.cam_pitch)
    fwd = z * math.cos(p) - yc * math.sin(p)
    down = z * math.sin(p) + yc * math.cos(p)
    height = a.cam_height - down
    keep = (height > a.wall_min_h) & (height < 0.6)
    pts = np.stack([fwd[keep] + a.cam_x, -xc[keep]], axis=1)
    if len(pts) == 0:
        return pts
    # 5 cm occupancy cells, drop cells with fewer than 3 hits (depth speckle)
    cells = np.floor(pts / 0.05).astype(np.int64)
    uniq, counts = np.unique(cells, axis=0, return_counts=True)
    return (uniq[counts >= 3] + 0.5) * 0.05


def clearance(path, walls, dt, sub, a):
    """Smallest wall distance along the path, and time of first contact (nan if none).
    The car is two discs of radius half_width at the rear and front axles."""
    if len(walls) == 0:
        return float("nan"), float("nan")
    front = path[:, :2] + WHEELBASE * np.stack([np.cos(path[:, 2]), np.sin(path[:, 2])], axis=1)
    best = np.full(len(path), np.inf)
    for pts in (path[:, :2], front):
        d = np.linalg.norm(pts[:, None, :] - walls[None, :, :], axis=2).min(axis=1)
        best = np.minimum(best, d)
    hits = np.nonzero(best < a.half_width)[0]
    hit_t = float(hits[0]) * dt / sub if len(hits) else float("nan")
    return float(best.min()), hit_t


def project_floor(xy, intr, a):
    """Floor points (car frame) -> colour image pixels. Returns (N,2) with NaN behind the camera."""
    fx, fy, cx, cy = intr
    p = math.radians(a.cam_pitch)
    f = xy[:, 0] - a.cam_x
    down = a.cam_height
    zc = f * math.cos(p) + down * math.sin(p)
    yc = -f * math.sin(p) + down * math.cos(p)
    xc = -xy[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([fx * xc / zc + cx, fy * yc / zc + cy], axis=1)
    uv[zc < 0.08] = np.nan
    return uv


def wheel_tracks(path, half_width):
    n = np.stack([-np.sin(path[:, 2]), np.cos(path[:, 2])], axis=1)
    return path[:, :2] + half_width * n, path[:, :2] - half_width * n


# =============================================================================
# Drawing
# =============================================================================
TD = 480           # top-down panel size, px
PPM = 100          # px per metre -> 0.5 m tile = 50 px
TD_ORIGIN = (TD // 2, TD - 60)


def td_px(xy):
    return np.stack([TD_ORIGIN[0] - xy[:, 1] * PPM, TD_ORIGIN[1] - xy[:, 0] * PPM], axis=1)


def polyline(img, pts, color, thick):
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) >= 2:
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, color, thick, cv2.LINE_AA)


def draw_frame(t_rel, rgb, walls, plans, n_exec_pts, intr_color, a, info_lines):
    """plans: list of (label, colour, path, hit_t)."""
    scale = 2
    cam = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_LINEAR)
    cam = cv2.resize(cam, (int(cam.shape[1] * TD / cam.shape[0]), TD))
    k = cam.shape[0] / rgb.shape[0]
    for _, color, path, _ in plans:
        for side in wheel_tracks(path, a.half_width):
            polyline(cam, project_floor(side, intr_color, a) * k, color, 2)

    td = np.full((TD, TD, 3), 30, np.uint8)
    for i in range(-20, 21):                                   # 0.5 m tile grid
        g = int(TD_ORIGIN[0] + i * 0.5 * PPM)
        cv2.line(td, (g, 0), (g, TD), (55, 55, 55), 1)
        g = int(TD_ORIGIN[1] - i * 0.5 * PPM)
        cv2.line(td, (0, g), (TD, g), (55, 55, 55), 1)
    for x, y in td_px(walls) if len(walls) else []:
        if 0 <= x < TD and 0 <= y < TD:
            cv2.circle(td, (int(x), int(y)), 2, C_WALL, -1)
    car = np.array([[-0.10, 0.15], [0.40, 0.15], [0.40, -0.15], [-0.10, -0.15]])
    cv2.polylines(td, [np.round(td_px(car)).astype(np.int32)], True, (200, 200, 200), 1, cv2.LINE_AA)
    for _, color, path, hit_t in plans:
        px = td_px(path[:, :2])
        polyline(td, px, color, 1)
        polyline(td, px[:n_exec_pts + 1], color, 3)            # the part the car would execute
        if hit_t == hit_t:                                     # not NaN
            i = min(int(round(hit_t / a.dt * a.sub)), len(px) - 1)
            cv2.drawMarker(td, tuple(np.round(px[i]).astype(int)), C_HIT, cv2.MARKER_TILTED_CROSS, 14, 2)
    yy = 18
    for label, color, _, _ in plans:
        cv2.putText(td, label, (8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        yy += 17
    cv2.putText(td, "grid 0.5 m", (TD - 92, TD - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1, cv2.LINE_AA)

    bar = np.full((22 * len(info_lines) + 8, cam.shape[1] + TD, 3), 0, np.uint8)
    for i, line in enumerate(info_lines):
        cv2.putText(bar, line, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([np.hstack([cam, td]), bar])


class Viewer:
    """Live window: OpenCV if it has GUI support, else matplotlib, else nothing."""

    def __init__(self, period):
        self.period, self.kind, self.paused, self.quit = period, None, False, False

    def _init(self, frame):
        import os
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print("  --show: no display (DISPLAY/WAYLAND_DISPLAY unset); continuing without live view.")
            self.kind = "none"
            return
        try:
            cv2.namedWindow("vla_replay", cv2.WINDOW_NORMAL)
            self.kind = "cv2"
            return
        except cv2.error:
            pass
        try:
            import matplotlib
            for backend in ("TkAgg", "QtAgg", "GTK3Agg"):
                try:
                    matplotlib.use(backend, force=True)
                    import matplotlib.pyplot as plt
                    plt.figure("vla_replay")
                    break
                except Exception:  # noqa: BLE001
                    continue
            else:
                raise RuntimeError("no interactive matplotlib backend")
            self.plt = plt
            self.fig = plt.gcf()
            self.fig.set_size_inches(frame.shape[1] / 100, frame.shape[0] / 100)
            self.ax = self.fig.add_axes([0, 0, 1, 1])
            self.ax.axis("off")
            self.im = self.ax.imshow(frame[:, :, ::-1])
            self.fig.canvas.mpl_connect("key_press_event", self._key)
            self.fig.canvas.mpl_connect("close_event", lambda e: setattr(self, "quit", True))
            plt.show(block=False)
            self.kind = "mpl"
        except Exception as e:  # noqa: BLE001
            print(f"  --show: no GUI available ({e}); continuing without live view.\n"
                  "         sudo apt install python3-tk, or pip install "
                  "opencv-python in place of opencv-python-headless, to enable it.")
            self.kind = "none"

    def _key(self, e):
        if e.key == " ":
            self.paused = not self.paused
        elif e.key == "q":
            self.quit = True

    def show(self, frame):
        """Returns False when the user asked to quit."""
        if self.kind is None:
            self._init(frame)
        if self.kind == "cv2":
            cv2.imshow("vla_replay", frame)
            key = cv2.waitKey(max(1, int(1000 * self.period))) & 0xFF
            if key == ord(" "):
                key = cv2.waitKey(0) & 0xFF
            return key != ord("q")
        if self.kind == "mpl":
            self.im.set_data(frame[:, :, ::-1])
            self.plt.pause(self.period)
            while self.paused and not self.quit:
                self.plt.pause(0.1)
            return not self.quit
        return True

    def close(self):
        if self.kind == "cv2":
            cv2.destroyAllWindows()
        elif self.kind == "mpl":
            self.plt.close(self.fig)
            self.plt.switch_backend("Agg")


def open_writer(path, fps, shape):
    """H.264 if this OpenCV build can encode it (plays everywhere), else MPEG-4 Part 2."""
    h, w = shape[:2]
    for fourcc in ("avc1", "mp4v"):
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if wr.isOpened():
            if fourcc == "mp4v":
                print("  video: mp4v codec -- if your player shows nothing, open it in VLC or run\n"
                      f"         ffmpeg -i {path} -c:v libx264 -pix_fmt yuv420p replay_h264.mp4")
            return wr
    sys.exit("OpenCV could not open a video writer; rerun with --no-video")


def write_summary(rows, human_xyth, fans, a, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([r["t"] for r in rows])
    has_unn = rows[0]["unn_steer0"] == rows[0]["unn_steer0"]
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1])
    for row, key, ylabel in ((0, "steer", "steering [rad]"), (1, "speed", "speed setpoint [m/s]")):
        ax = fig.add_subplot(gs[row, 0])
        ax.plot(t, [r[f"human_{key}"] for r in rows], color="#222222", lw=1.6, label="human (/teleop)")
        if key == "speed" and rows[0]["odom_speed"] == rows[0]["odom_speed"]:
            ax.plot(t, [r["odom_speed"] for r in rows], color="#222222", lw=1.0, ls="--",
                    label="measured speed (/odom)")
            ax.legend(loc="upper right", fontsize=8)
        ax.plot(t, [r[f"dep_{key}0"] for r in rows], color="#e8710a", lw=1.2, label="VLA as deployed")
        if has_unn:
            ax.plot(t, [r[f"unn_{key}0"] for r in rows], color="#2e9e4f", lw=1.2, label="VLA unnormalised")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        if row == 0:
            ax.set_title("First action of each plan vs what the human commanded at that frame")
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.set_xlabel("time in bag [s]")
    ax = fig.add_subplot(gs[:, 1])
    ax.plot(-human_xyth[:, 1], human_xyth[:, 0], color="#222222", lw=1.8, label="human, dead-reckoned (odom+IMU)")
    for i, (pose, dep, unn) in enumerate(fans):
        for path, color, lab in ((dep, "#e8710a", "VLA as deployed"), (unn, "#2e9e4f", "VLA unnormalised")):
            if path is None:
                continue
            c, s = math.cos(pose[2]), math.sin(pose[2])
            gx = pose[0] + c * path[:, 0] - s * path[:, 1]
            gy = pose[1] + s * path[:, 0] + c * path[:, 1]
            ax.plot(-gy, gx, color=color, lw=0.9, alpha=0.8, label=lab if i == 0 else None)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25)
    ax.set_title(f"Plans ({a.map_horizon:.1f} s each) branching off the driven path, 0.5 m grid\n"
                 "(path is integrated, so it drifts over a lap)", fontsize=9)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=130)
    plt.close(fig)


# =============================================================================
# Main loop
# =============================================================================
def run(a, cmds, frames, intr_depth, intr_color, runner, veh=None):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    nan = float("nan")
    rows, fans, store = [], [], {"t": [], "raw": [], "deployed": [], "unnorm": [], "human": []}
    writer = None
    viewer = Viewer(1.0 / (a.hz * a.playback)) if a.show and not a.no_video else None
    t0 = None
    last_fan = -1e18
    tic = time.time()

    for n, (ts, rgb, depth) in enumerate(frames):
        if t0 is None:
            t0 = ts
        if a.start is not None and ts - t0 < a.start:
            continue
        if a.end is not None and ts - t0 > a.end:
            break

        H = a.horizon
        hs, hd = cmds.at(ts + np.arange(H) * a.dt)
        human = np.stack([hs, hd], axis=1)
        raw, unn = runner.plan(rgb, depth_to_rgb(depth), human)
        raw = raw[:H]
        dep = deploy_transform(raw, a)
        unn = deploy_transform(unn[:H], a) if unn is not None else None

        walls = wall_points(depth, intr_depth, a) if not a.no_walls else np.zeros((0, 2))
        variants = [("human, %s (next %.1f s)" % ("measured" if veh else "commands", H * a.dt),
                     C_HUMAN, human, "human")]
        variants.append(("VLA as deployed", C_DEPLOYED, dep, "dep"))
        if unn is not None:
            variants.append(("VLA unnormalised", C_UNNORM, unn, "unn"))

        row = {"t": ts - t0, "human_speed": hs[0], "human_steer": hd[0],
               "raw_speed0": raw[0, 0], "raw_steer0": raw[0, 1],
               "dep_speed0": dep[0, 0], "dep_steer0": dep[0, 1],
               "unn_speed0": unn[0, 0] if unn is not None else nan,
               "unn_steer0": unn[0, 1] if unn is not None else nan}
        plans, paths = [], {}
        k1 = min(int(round(1.0 / a.dt)), H) * a.sub              # path index at 1 s
        if veh is not None:
            v0, I0, s0 = veh.state(ts)
            row["odom_speed"] = v0
        else:
            row["odom_speed"] = nan
        for label, color, acts, key in variants:
            if veh is None:
                path = rollout(acts, a.dt, a.sub)
            elif key == "human":
                path = veh.measured(ts, H * a.sub, a.dt / a.sub)
            else:
                path = veh.simulate(acts, a.dt, a.sub, v0, I0, s0)
            paths[key] = path
            kc = int(round(a.check_s / a.dt)) * a.sub + 1        # model error grows with time
            clr, hit_t = clearance(path[:kc], walls, a.dt, a.sub, a)
            row[f"{key}_clearance_m"] = clr
            row[f"{key}_hit_t"] = hit_t
            plans.append((label, color, path, hit_t))
        for key in ("dep", "unn"):
            if key in paths:
                row[f"{key}_dev_1s_m"] = float(np.linalg.norm(paths[key][k1, :2] - paths["human"][k1, :2]))
                row[f"{key}_steer_mae_exec"] = float(np.mean(np.abs(
                    (dep if key == "dep" else unn)[:a.n_exec, 1] - human[:a.n_exec, 1])))
            else:
                row[f"{key}_dev_1s_m"] = row[f"{key}_steer_mae_exec"] = nan
                row[f"{key}_clearance_m"] = row[f"{key}_hit_t"] = nan
        rows.append(row)

        store["t"].append(ts - t0)
        store["raw"].append(raw)
        store["deployed"].append(dep)
        store["human"].append(human)
        store["unnorm"].append(unn if unn is not None else np.full_like(dep, nan))

        if ts - last_fan >= a.map_every:
            last_fan = ts
            km = min(int(round(a.map_horizon / a.dt)), H) * a.sub + 1
            fans.append((ts, paths["dep"][:km], paths["unn"][:km] if "unn" in paths else None))

        if not a.no_video:
            fmt = lambda v: "%+.2f" % v  # noqa: E731
            lines = [
                f"t={ts - t0:6.2f}s   human: speed {fmt(hs[0])} steer {fmt(hd[0])}   "
                f"raw model out: {fmt(raw[0, 0])} {fmt(raw[0, 1])}",
                f"deployed: speed {fmt(dep[0, 0])} steer {fmt(dep[0, 1])}"
                + (f"   unnormalised: speed {fmt(unn[0, 0])} steer {fmt(unn[0, 1])}" if unn is not None else "")
                + f"   thick = first {a.n_exec} actions, X = wall contact",
            ]
            if veh is not None:
                lines.append(f"measured speed {row['odom_speed']:.2f} m/s   (speeds above are PID setpoints; "
                             f"paths are simulated through the PID)")
            frame = draw_frame(ts - t0, rgb, walls, plans, a.n_exec * a.sub, intr_color, a, lines)
            if writer is None:
                writer = open_writer(out / "replay.mp4", a.hz * a.playback, frame.shape)
            writer.write(frame)
            if viewer is not None and not viewer.show(frame):
                break
        if len(rows) % 25 == 0:
            print(f"  {len(rows)} plans, t={ts - t0:.1f}s, {len(rows) / (time.time() - tic):.1f} plans/s")

    if writer is not None:
        writer.release()
    if viewer is not None:
        viewer.close()
    if not rows:
        sys.exit("no frames processed -- check --rgb-topic / --depth-topic / --start / --end")

    with open(out / "plans.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    np.savez_compressed(out / "plans.npz", dt=a.dt, **{k: np.array(v) for k, v in store.items()})

    # dead-reckoned human path for the map, and the pose at each fan's start time
    t_start, t_end = t0 + rows[0]["t"], t0 + rows[-1]["t"]
    if veh is not None:
        human_xyth = veh.measured(t_start, int((t_end - t_start) / 0.02) + 1, 0.02)
    else:
        tt = np.arange(t_start, t_end + 0.02, 0.02)
        sp, st = cmds.at(tt)
        human_xyth = rollout(np.stack([sp, st], axis=1), 0.02, 1)
    fans = [(human_xyth[min(int(round((ts - t_start) / 0.02)), len(human_xyth) - 1)], d, u) for ts, d, u in fans]
    write_summary(rows, human_xyth, fans, a, out)

    # ---- console summary ----
    col = lambda k: np.array([r[k] for r in rows], dtype=np.float64)  # noqa: E731
    print(f"\n{len(rows)} plans over {rows[-1]['t'] - rows[0]['t']:.1f} s -> {out}/")
    print(f"raw model output: speed mean {col('raw_speed0').mean():+.2f} "
          f"(negative in {100 * np.mean(col('raw_speed0') < 0):.0f}% of plans), "
          f"steering mean {col('raw_steer0').mean():+.2f} std {col('raw_steer0').std():.2f}")
    print(f"human commands:   speed setpoint mean {col('human_speed').mean():+.2f}"
          + (f" (measured {np.nanmean(col('odom_speed')):.2f} m/s)" if veh is not None else "") + ", "
          f"steering mean {col('human_steer').mean():+.2f} std {col('human_steer').std():.2f}")
    print(f"{'':18s}{'steer MAE':>10s}{'dev @1s':>10s}{'wall contact':>14s}")
    hh = col("human_hit_t")
    print(f"{'human':18s}{'-':>10s}{'-':>10s}{100 * np.mean(~np.isnan(hh)):>13.0f}%")
    for key, name in (("dep", "VLA as deployed"), ("unn", "VLA unnormalised")):
        if np.all(np.isnan(col(f"{key}_dev_1s_m"))):
            continue
        print(f"{name:18s}{np.nanmean(col(f'{key}_steer_mae_exec')):>10.3f}"
              f"{np.nanmean(col(f'{key}_dev_1s_m')):>9.2f}m"
              f"{100 * np.mean(~np.isnan(col(f'{key}_hit_t'))):>13.0f}%")
    res = {"plans": len(rows), "duration_s": rows[-1]["t"] - rows[0]["t"],
           "raw_speed_mean": float(col("raw_speed0").mean()), "raw_steer_mean": float(col("raw_steer0").mean()),
           "raw_steer_std": float(col("raw_steer0").std()),
           "human_speed_setpoint_mean": float(col("human_speed").mean()),
           "human_steer_mean": float(col("human_steer").mean()), "human_steer_std": float(col("human_steer").std()),
           "human_wall_contact_pct": float(100 * np.mean(~np.isnan(hh)))}
    if veh is not None:
        res["measured_speed_mean_m_s"] = float(np.nanmean(col("odom_speed")))
    for key, name in (("dep", "deployed"), ("unn", "unnormalised")):
        if not np.all(np.isnan(col(f"{key}_dev_1s_m"))):
            res[name] = {"steer_mae_rad": float(np.nanmean(col(f"{key}_steer_mae_exec"))),
                         "dev_at_1s_m": float(np.nanmean(col(f"{key}_dev_1s_m"))),
                         "wall_contact_pct": float(100 * np.mean(~np.isnan(col(f"{key}_hit_t"))))}
    log_info("results", **res)
    if not a.no_walls and np.mean(~np.isnan(hh)) > 0.1:
        print("NOTE: the human path 'hits' walls in >10% of frames, so the wall map is off. "
              "Tune --cam-height / --cam-pitch / --cam-x / --wall-min-h before trusting contacts.")
    return rows


def default_intrinsics(w, h, hfov_deg):
    f = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    return (f, f, w / 2, h / 2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bag", required=True, help="rosbag2 directory (one episode)")
    p.add_argument("--out", default=None, help="output dir (default: replay_out/<bag name>)")
    g = p.add_argument_group("policy (defaults match config/smolvla.yaml)")
    g.add_argument("--policy", choices=["smolvla", "dummy"], default="smolvla",
                   help="dummy = no model, echoes the human commands with noise to test the pipeline")
    g.add_argument("--model", default="models/track_v2_dual-image")
    g.add_argument("--task", default="complete laps around the track anticlockwise")
    g.add_argument("--mode", choices=["rgb", "depth", "dual"], default="dual")
    g.add_argument("--preproc", choices=["node", "train"], default="train",
                   help="node: old 224x224 stretch (smolvla_openloop.py before the fix); "
                        "train: full 424x240 frame like the LeRobot dataset")
    g.add_argument("--stats", default=None, help="dataset meta/stats.json, if the postprocessor can't be loaded")
    g.add_argument("--device", default=None)
    g.add_argument("--seed", type=int, default=0, help="fixed denoising noise; -1 = random like on the car")
    g.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32",
                   help="node's precision parameter: fp16/bf16 run the model under autocast")
    g.add_argument("--num-steps", type=int, default=0,
                   help="denoising steps per plan (node's num_steps); 0 = checkpoint default (10)")
    g.add_argument("--speed-min", type=float, default=0.0)
    g.add_argument("--speed-max", type=float, default=1.0)
    g.add_argument("--steer-max", type=float, default=0.4)
    g = p.add_argument_group("timing")
    g.add_argument("--hz", type=float, default=5.0, help="plans per second of bag")
    g.add_argument("--dt", type=float, default=0.1,
                   help="seconds per action. 0.1 = dataset FPS 10 (what the model learned); "
                        "0.5 = how the car plays them back at rate_hz 2")
    g.add_argument("--horizon", type=int, default=30, help="actions of each chunk to roll out")
    g.add_argument("--n-exec", type=int, default=3, help="n_action_steps on the car (drawn thick)")
    g.add_argument("--start", type=float, default=None, help="skip to this many seconds into the bag")
    g.add_argument("--end", type=float, default=None)
    g = p.add_argument_group("topics")
    g.add_argument("--rgb-topic", default="/camera/camera/color/image_raw")
    g.add_argument("--depth-topic", default="/camera/camera/depth/image_rect_raw")
    g.add_argument("--cmd-topic", default="/teleop")
    g.add_argument("--odom-topic", default="/odom")
    g.add_argument("--imu-topic", default="/sensors/imu/raw")
    g = p.add_argument_group("vehicle model (defaults from config/vesc.yaml pid_controller)")
    g.add_argument("--vehicle", choices=["pid", "ideal"], default="pid",
                   help="pid: simulate pid_controller.py, fitted to /odom; ideal: commands are exact m/s")
    g.add_argument("--kp", type=float, default=0.4)
    g.add_argument("--ki", type=float, default=0.4)
    g.add_argument("--i-limit", type=float, default=0.8)
    g.add_argument("--steer-rate", type=float, default=2.0)
    g = p.add_argument_group("camera mounting (measure these on the car)")
    g.add_argument("--cam-height", type=float, default=0.15, help="lens height above the floor [m]")
    g.add_argument("--cam-x", type=float, default=0.33, help="lens distance ahead of the rear axle [m]")
    g.add_argument("--cam-pitch", type=float, default=0.0, help="degrees, positive = looking down")
    g.add_argument("--half-width", type=float, default=0.15, help="half the car width [m]")
    g.add_argument("--wall-min-h", type=float, default=0.05, help="ignore depth points lower than this [m]")
    g.add_argument("--wall-range", type=float, default=4.0)
    g.add_argument("--check-s", type=float, default=1.5, help="seconds of each path checked against walls")
    g.add_argument("--no-walls", action="store_true")
    g = p.add_argument_group("output")
    g.add_argument("--no-video", action="store_true")
    g.add_argument("--show", action="store_true", help="live window while running (space pause, q quit)")
    g.add_argument("--playback", type=float, default=1.0, help="video speed, 1 = real time")
    g.add_argument("--map-every", type=float, default=1.0, help="seconds between plans drawn on the map")
    g.add_argument("--map-horizon", type=float, default=1.5, help="seconds of each plan drawn on the map")
    a = p.parse_args(argv)
    a.sub = 5
    if a.out is None:
        a.out = str(Path("replay_out") / Path(a.bag).name)
    return a


def main(argv=None):
    global RUN_INFO_PATH
    a = parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    RUN_INFO_PATH = out / "run_info.json"
    here = Path(__file__).resolve().parent
    log_info("run", started=datetime.datetime.now().isoformat(timespec="seconds"), status="running",
             command=" ".join([Path(sys.argv[0]).name] + sys.argv[1:]) if argv is None else str(argv),
             host=socket.gethostname(), script=str(Path(__file__).resolve()), script_git=_git_commit(here))
    log_info("args", **{k: v for k, v in vars(a).items()})
    log_info("versions", **_versions())
    info_d = a.depth_topic.rsplit("/", 1)[0] + "/camera_info"
    info_c = a.rgb_topic.rsplit("/", 1)[0] + "/camera_info"
    print(f"reading commands from {a.bag}")
    cmds, odom, imu, intr = load_signals(a.bag, a, {info_d, info_c})
    print(f"  {len(cmds.t)} commands over {cmds.t[-1] - cmds.t[0]:.1f} s, "
          f"{0 if odom is None else len(odom[0])} odom, {0 if imu is None else len(imu[0])} imu messages")
    log_info("bag", path=str(Path(a.bag).resolve()), name=Path(a.bag).name,
             duration_s=float(cmds.t[-1] - cmds.t[0]), commands=len(cmds.t),
             odom_msgs=0 if odom is None else len(odom[0]), imu_msgs=0 if imu is None else len(imu[0]))
    veh = None
    if a.vehicle == "pid":
        if odom is None:
            print(f"  no {a.odom_topic} in bag -> falling back to --vehicle ideal")
        else:
            veh = Vehicle(a)
            veh.fit(cmds, odom, imu)
    frames = iter_frames(a.bag, a.rgb_topic, a.depth_topic, a.hz)
    first = next(frames, None)
    if first is None:
        sys.exit("no colour+depth frames found -- check --rgb-topic / --depth-topic")
    h, w = first[1].shape[:2]
    dh, dw = first[2].shape
    for name, topic in (("depth", info_d), ("colour", info_c)):
        if topic not in intr:
            print(f"  no {topic} in bag, using nominal RealSense D435 {name} intrinsics")
    intr_depth = intr.get(info_d, default_intrinsics(dw, dh, 87.0))
    intr_color = intr.get(info_c, default_intrinsics(w, h, 69.0))
    log_info("bag", rgb_size=[w, h], depth_size=[dw, dh],
             intrinsics_source="bag" if info_c in intr else "nominal D435")

    def chained():
        yield first
        yield from frames

    runner = DummyRunner(a) if a.policy == "dummy" else SmolVLARunner(a)
    run(a, cmds, chained(), intr_depth, intr_color, runner, veh)
    if getattr(runner, "infer_times", None):
        t = np.array(runner.infer_times[3:] or runner.infer_times)   # skip warm-up
        print(f"model run time on this machine: median {np.median(t) * 1000:.0f} ms "
              f"(compare runs with different --num-steps; the Jetson is slower in absolute terms)")
        log_info("results", model_run_time_median_ms=float(np.median(t) * 1000))
    log_info("run", finished=datetime.datetime.now().isoformat(timespec="seconds"), status="complete")
    print(f"settings and results saved to {RUN_INFO_PATH}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:  # record interrupted/failed runs too, then re-raise
        if RUN_INFO_PATH is not None and not isinstance(e, SystemExit):
            log_info("run", status=f"failed: {type(e).__name__}: {e}"[:300],
                     finished=datetime.datetime.now().isoformat(timespec="seconds"))
        raise
