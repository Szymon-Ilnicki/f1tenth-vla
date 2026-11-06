#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from sensor_msgs.msg import Imu

BAGS_DIR = Path(__file__).resolve().parent / "bags/track_v1"
OUT = Path(__file__).resolve().parent / "lerobot_output/track_v1"
TASK = "complete laps around the track"
IMG_TOPIC = "/camera/camera/color/image_raw"
CMD_TOPIC = "/teleop"
ODOM_TOPIC = "/odom"
IMU_TOPIC = "/sensors/imu/raw"
FPS = 10
REPO_ID = "local/track_v0"

def open_bag(path):
    r = SequentialReader()
    r.open(
        StorageOptions(uri=str(path), storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    return r


def img_np(msg: Image):
    h, w = msg.height, msg.width
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = arr.reshape(h, w, 3)
        if msg.encoding == "bgr8":
            img = img[:, :, ::-1].copy()
        return img
    raise RuntimeError(msg.encoding)


def cmd_vec(msg):
    d = msg.drive if hasattr(msg, "drive") else msg
    speed = float(d.speed)
    return np.array([speed, d.steering_angle], dtype=np.float32)


def iter_synced(bag_dir):
    reader = open_bag(bag_dir)
    wanted = {IMG_TOPIC, CMD_TOPIC, ODOM_TOPIC, IMU_TOPIC}
    types = {
        c.name: get_message(c.type)
        for c in reader.get_all_topics_and_types()
        if c.name in wanted
    }
    last_cmd = None
    last_linear_x = 0.0
    last_angular_z = 0.0
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in types:          # skip unwanted topics
            continue
        msg = deserialize_message(data, types[topic])
        if topic == CMD_TOPIC:
            last_cmd = cmd_vec(msg)
        elif topic == ODOM_TOPIC:
            last_linear_x = float(msg.twist.twist.linear.x)
        elif topic == IMU_TOPIC:
            last_angular_z = float(msg.angular_velocity.z)
        elif topic == IMG_TOPIC and last_cmd is not None:
            state = np.array([last_linear_x, last_angular_z], dtype=np.float32)
            yield t * 1e-9, img_np(msg), last_cmd, state

def verify(dataset_root: Path):
    import pandas as pd
    import json
    from pathlib import Path

    root = Path(dataset_root)
    print("\n=== dataset verification ===")

    # task
    tasks_path = root / "meta" / "tasks.parquet"
    if tasks_path.exists():
        tasks = pd.read_parquet(tasks_path)
        print("task:", tasks.iloc[0, 0] if len(tasks) else "(empty)")
    else:
        print("task: tasks.parquet missing")

    # main parquet
    pq_files = list((root / "data").rglob("*.parquet"))
    if not pq_files:
        print("no parquet files found")
        return
    df = pd.concat([pd.read_parquet(p) for p in pq_files], ignore_index=True)

    # duration from timestamps or frame count
    if "timestamp" in df.columns:
        duration = df["timestamp"].max() - df["timestamp"].min()
    else:
        duration = len(df) / FPS
    print(f"frames: {len(df)}")
    print(f"duration: {duration:.1f} s  ({duration/60:.2f} min)")

    # action speed
    if "action" in df.columns:
        speeds = np.stack(df["action"].values)[:, 0]
        print(f"speed  min/max/mean: {speeds.min():.3f} / {speeds.max():.3f} / {speeds.mean():.3f}")
        neg = (speeds < 0).sum()
        if neg:
            print(f"  WARNING: {neg} negative speed frames")

    # observation.state
    if "observation.state" in df.columns:
        state = np.stack(df["observation.state"].values)
        print(f"state shape: {state.shape}")
        print(f"linear_x  min/max/mean: {state[:,0].min():.3f} / {state[:,0].max():.3f} / {state[:,0].mean():.3f}")
        if state.shape[1] > 1:
            print(f"angular_z min/max/mean: {state[:,1].min():.3f} / {state[:,1].max():.3f} / {state[:,1].mean():.3f}")
    else:
        print("observation.state: missing")

    # video duration (rough)
    mp4s = list((root / "videos").rglob("*.mp4"))
    if mp4s:
        print(f"mp4 files: {len(mp4s)}")
        # optional: use ffprobe if available
        try:
            import subprocess
            out = subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(mp4s[0])],
                text=True)
            print(f"first mp4 duration: {float(out):.1f} s")
        except Exception:
            print("(ffprobe not available, skipped mp4 duration)")
    else:
        print("no mp4 files found")

    print("=== end verification ===\n")


def main():
    bags = sorted(BAGS_DIR.glob("ep*"))
    if not bags:
        raise SystemExit(f"no ep* bags found in {BAGS_DIR}")

    print(f"found {len(bags)} bags in {BAGS_DIR}")

    ds = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        robot_type="f1tenth",
        features={
            "observation.images.camera1": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["linear_x", "angular_z"],
            },
            "action": {"dtype": "float32", "shape": (2,), "names": ["speed", "steering"]},
        },
        root=str(OUT),
        use_videos=True,
    )

    for bag in bags:
        print("episode", bag.name)
        n = 0
        t0 = None
        last_keep = None
        for t, img, act, odom in iter_synced(bag):
            if t0 is None:
                t0 = t
                last_keep = t - 1
            if t - last_keep < 1.0 / FPS - 1e-3:
                continue
            last_keep = t
            ds.add_frame({
                "observation.images.camera1": img,
                "observation.state": odom,
                "action": act,
                "task": TASK,
            })
            n += 1
        if n == 0:
            print("  skipped (no synced frames — check topics)")
            continue
        ds.save_episode()
        print("  frames", n)
    ds.finalize()
    print("wrote", OUT)
    verify(OUT)


if __name__ == "__main__":
    main()