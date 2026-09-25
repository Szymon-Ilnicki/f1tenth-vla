#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from lerobot.datasets.lerobot_dataset import LeRobotDataset
BAGS = [
    Path(__file__).resolve().parent / "ep000",
    Path(__file__).resolve().parent / "ep001",
    Path(__file__).resolve().parent / "ep002",
]
OUT = Path(__file__).resolve().parent / "outputs/fwd_v0"
TASK = "drive forward"
IMG_TOPIC = "/camera/camera/color/image_raw"
CMD_TOPIC = "/ackermann_cmd"
FPS = 10

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
    return np.array([d.speed, d.steering_angle], dtype=np.float32)


def iter_synced(bag_dir):
    reader = open_bag(bag_dir)
    types = {c.name: get_message(c.type) for c in reader.get_all_topics_and_types()}
    last_cmd = None
    while reader.has_next():
        topic, data, t = reader.read_next()
        msg = deserialize_message(data, types[topic])
        if topic == CMD_TOPIC:
            last_cmd = cmd_vec(msg)
        elif topic == IMG_TOPIC and last_cmd is not None:
            yield t * 1e-9, img_np(msg), last_cmd


def main():
    bags = BAGS
    if not bags:
        raise SystemExit(f"no bags in {BAGS}")

    ds = LeRobotDataset.create(
        repo_id="local/fwd_v1",
        fps=FPS,
        robot_type="f1tenth",
        features={
            "observation.images.camera1": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
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
        for t, img, act in iter_synced(bag):
            if t0 is None:
                t0 = t
                last_keep = t - 1
            if t - last_keep < 1.0 / FPS - 1e-3:
                continue
            last_keep = t
            ds.add_frame(
                {
                    "observation.images.camera1": img,
                    "action": act,
                    "task": TASK,
                }
            )
            n += 1
        if n == 0:
            print("  skipped (no synced frames — check CMD_TOPIC)")
            continue
        ds.save_episode()
        print("  frames", n)

    print("wrote", OUT)


if __name__ == "__main__":
    main()
