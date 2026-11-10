#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import cv2

BAGS_DIR = Path(__file__).resolve().parent / "bags/track_v1"
OUT = Path(__file__).resolve().parent / "lerobot_output/track_v1_dual-image"
TASK = "complete laps around the track"
IMG_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"
CMD_TOPIC = "/teleop"
FPS = 10
REPO_ID = "local/track_v2_dual-image"

# depth normalization (mm)
DEPTH_MIN = 200
DEPTH_MAX = 2000

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

def depth_to_rgb(msg: Image):
    """Convert 16-bit depth to a colorized RGB image.
    Invalid (0) pixels become bright magenta so they are distinct.
    Valid depths are normalized 200–2000 mm and mapped with turbo colormap.
    """
    h, w = msg.height, msg.width
    depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)

    # invalid mask
    invalid = (depth == 0)

    # clip and normalize valid depths to 0–1
    depth_f = depth.astype(np.float32)
    depth_f = np.clip(depth_f, DEPTH_MIN, DEPTH_MAX)
    norm = (depth_f - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)
    norm = (norm * 255).astype(np.uint8)

    # apply turbo colormap
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    # mark invalid pixels as bright magenta
    color[invalid] = [255, 0, 255]

    return color

def cmd_vec(msg):
    d = msg.drive if hasattr(msg, "drive") else msg
    speed = float(d.speed)
    return np.array([speed, d.steering_angle], dtype=np.float32)

def iter_synced(bag_dir):
    reader = open_bag(bag_dir)
    wanted = {IMG_TOPIC, DEPTH_TOPIC, CMD_TOPIC}
    types = {
        c.name: get_message(c.type)
        for c in reader.get_all_topics_and_types()
        if c.name in wanted
    }
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
            "observation.images.camera2": {
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
        for t, img, depth_img, act in iter_synced(bag):
            if t0 is None:
                t0 = t
                last_keep = t - 1
            if t - last_keep < 1.0 / FPS - 1e-3:
                continue
            last_keep = t
            ds.add_frame({
                "observation.images.camera1": img,
                "observation.images.camera2": depth_img,
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

if __name__ == "__main__":
    main()