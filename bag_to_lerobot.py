#!/usr/bin/env python3
"""Convert rosbag2 episodes into a LeRobot dataset for SmolVLA.

Each bag gets a task string for its lap direction, detected from the sign of its
mean /teleop steering (+ = left = anticlockwise). With --mirror, every bag is
added a second time flipped left-right with steering negated and the opposite
task, which doubles the data and balances the two directions.

observation.state is written as zeros (SmolVLA needs a state input); the replay
script and smolvla_openloop.py feed zeros too.

Example:
  python3 bag_to_lerobot.py --bags bags/track_v2 --out lerobot_output/track_v3 \
      --repo-id local/track_v3 --mirror
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image

HERE = Path(__file__).resolve().parent
IMG_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"
CMD_TOPIC = "/teleop"
TASKS = {+1: "complete laps around the track anticlockwise",
         -1: "complete laps around the track clockwise"}

# depth normalization (mm) -- must match smolvla_openloop.py
DEPTH_MIN = 200
DEPTH_MAX = 2000


def open_bag(path, topics=None):
    r = SequentialReader()
    storage = "mcap" if list(Path(path).glob("*.mcap")) else "sqlite3"
    r.open(
        StorageOptions(uri=str(path), storage_id=storage),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    if topics:
        r.set_filter(StorageFilter(topics=list(topics)))
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


def depth_to_rgb(msg: Image):
    """16-bit depth -> turbo-coloured RGB; invalid (0) pixels bright magenta."""
    h, w = msg.height, msg.width
    depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
    invalid = depth == 0
    depth_f = np.clip(depth.astype(np.float32), DEPTH_MIN, DEPTH_MAX)
    norm = ((depth_f - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * 255).astype(np.uint8)
    color = cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    color[invalid] = [255, 0, 255]
    return color


def cmd_vec(msg):
    d = msg.drive if hasattr(msg, "drive") else msg
    return np.array([float(d.speed), float(d.steering_angle)], dtype=np.float32)


def bag_types(reader, wanted):
    return {c.name: get_message(c.type) for c in reader.get_all_topics_and_types() if c.name in wanted}


def lap_direction(bag_dir, min_speed):
    """+1 anticlockwise, -1 clockwise, from mean steering while driving."""
    reader = open_bag(bag_dir, [CMD_TOPIC])
    types = bag_types(reader, {CMD_TOPIC})
    steers = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in types:
            speed, steer = cmd_vec(deserialize_message(data, types[topic]))
            if speed > min_speed:
                steers.append(steer)
    if not steers:
        return 0, 0.0
    m = float(np.mean(steers))
    return (1 if m > 0 else -1), m


def iter_synced(bag_dir):
    wanted = {IMG_TOPIC, DEPTH_TOPIC, CMD_TOPIC}
    reader = open_bag(bag_dir, wanted)
    types = bag_types(reader, wanted)
    last_cmd = None
    last_depth = None
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in types:
            continue
        msg = deserialize_message(data, types[topic])
        if topic == CMD_TOPIC:
            last_cmd = cmd_vec(msg)
        elif topic == DEPTH_TOPIC:
            last_depth = depth_to_rgb(msg)
        elif topic == IMG_TOPIC and last_cmd is not None and last_depth is not None:
            yield t * 1e-9, img_np(msg), last_depth, last_cmd


def add_episode(ds, bag, task, fps, mirror):
    n = 0
    last_keep = None
    for t, img, depth_img, act in iter_synced(bag):
        if last_keep is None:
            last_keep = t - 1
        if t - last_keep < 1.0 / fps - 1e-3:
            continue
        last_keep = t
        act = act.copy()
        if mirror:
            img = np.ascontiguousarray(img[:, ::-1])
            depth_img = np.ascontiguousarray(depth_img[:, ::-1])
            act[1] = -act[1]
        ds.add_frame({
            "observation.images.camera1": img,
            "observation.images.camera2": depth_img,
            "observation.state": np.zeros(2, dtype=np.float32),
            "action": act,
            "task": task,
        })
        n += 1
    if n:
        ds.save_episode()
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bags", default=str(HERE / "bags/track_v2"), help="folder with ep* bag directories")
    p.add_argument("--out", default=str(HERE / "lerobot_output/track_v3"))
    p.add_argument("--repo-id", default="local/track_v3")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--mirror", action="store_true", help="also add every bag flipped left-right")
    p.add_argument("--min-speed", type=float, default=0.3, help="ignore stationary commands for direction")
    a = p.parse_args()

    bags = sorted(Path(a.bags).expanduser().glob("ep*"))
    if not bags:
        raise SystemExit(f"no ep* bags found in {a.bags}")
    print(f"found {len(bags)} bags in {a.bags}")

    ds = LeRobotDataset.create(
        repo_id=a.repo_id,
        fps=a.fps,
        robot_type="f1tenth",
        features={
            "observation.images.camera1": {
                "dtype": "video", "shape": (240, 424, 3), "names": ["height", "width", "channels"]},
            "observation.images.camera2": {
                "dtype": "video", "shape": (240, 424, 3), "names": ["height", "width", "channels"]},
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["dummy1", "dummy2"]},
            "action": {"dtype": "float32", "shape": (2,), "names": ["speed", "steering"]},
        },
        root=str(Path(a.out).expanduser()),
        use_videos=True,
    )

    counts = {}
    for bag in bags:
        d, mean_steer = lap_direction(bag, a.min_speed)
        if d == 0:
            print(f"{bag.name}: no driving commands, skipped")
            continue
        passes = [(False, d)] + ([(True, -d)] if a.mirror else [])
        for mirror, direction in passes:
            task = TASKS[direction]
            n = add_episode(ds, bag, task, a.fps, mirror)
            tag = "mirrored" if mirror else "original"
            print(f"{bag.name} ({tag}): mean steer {mean_steer:+.3f} -> '{task}', {n} frames")
            counts[task] = counts.get(task, 0) + n
    ds.finalize()
    print("frames per task:", counts)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
