#!/usr/bin/env python3
"""Convert rosbag2 episodes into a LeRobot dataset for SmolVLA.
 
Each frame's task is the /task string recorded in the bag at that moment (published
by task_latch while recording). With --mirror, every bag is added a second time
flipped left-right, with steering negated and "clockwise" <-> "anticlockwise"
swapped in the task, which doubles the data and balances the two directions.
 
As a check, each bag's lap direction is also detected from the sign of its mean
/teleop steering (+ = left = anticlockwise) and compared with its task; a mismatch
is reported, and with --strict the bag is skipped. Bags without /task fall back to
the detected direction and --default-task.
 
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
TASK_TOPIC = "/task"
 
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
 
 
def mirror_task(task: str) -> str:
    """Swap the direction word; "anticlockwise" contains "clockwise", so go via a placeholder."""
    return (task.replace("anticlockwise", "\0")
                .replace("clockwise", "anticlockwise")
                .replace("\0", "clockwise"))
 
 
def task_direction(task: str) -> int:
    if "anticlockwise" in task:
        return 1
    if "clockwise" in task:
        return -1
    return 0
 
 
def scan_bag(bag_dir, min_speed):
    """Direction from mean steering while driving (+1 anticlockwise, -1 clockwise),
    mean steering, and the /task strings recorded in the bag (in order, deduplicated)."""
    reader = open_bag(bag_dir, [CMD_TOPIC, TASK_TOPIC])
    types = bag_types(reader, {CMD_TOPIC, TASK_TOPIC})
    steers, tasks = [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in types:
            continue
        msg = deserialize_message(data, types[topic])
        if topic == CMD_TOPIC:
            speed, steer = cmd_vec(msg)
            if speed > min_speed:
                steers.append(steer)
        elif msg.data and (not tasks or tasks[-1] != msg.data):
            tasks.append(msg.data)
    if not steers:
        return 0, 0.0, tasks
    m = float(np.mean(steers))
    return (1 if m > 0 else -1), m, tasks
 
 
def iter_synced(bag_dir, first_task, use_recorded=True):
    """(t, rgb, depth, action, task) per colour frame; task = latest /task so far
    (first_task until the first /task message of the bag arrives)."""
    wanted = {IMG_TOPIC, DEPTH_TOPIC, CMD_TOPIC, TASK_TOPIC}
    reader = open_bag(bag_dir, wanted)
    types = bag_types(reader, wanted)
    last_cmd = None
    last_depth = None
    task = first_task
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in types:
            continue
        msg = deserialize_message(data, types[topic])
        if topic == CMD_TOPIC:
            last_cmd = cmd_vec(msg)
        elif topic == TASK_TOPIC:
            if use_recorded and msg.data:
                task = msg.data
        elif topic == DEPTH_TOPIC:
            last_depth = depth_to_rgb(msg)
        elif topic == IMG_TOPIC and last_cmd is not None and last_depth is not None:
            yield t * 1e-9, img_np(msg), last_depth, last_cmd, task
 
 
def add_episode(ds, bag, first_task, fps, mirror, counts, use_recorded=True):
    n = 0
    last_keep = None
    for t, img, depth_img, act, task in iter_synced(bag, first_task, use_recorded):
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
            task = mirror_task(task)
        counts[task] = counts.get(task, 0) + 1
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
    p.add_argument("--default-task", default="complete laps {dir} around the track",
                   help="for bags with no /task; {dir} becomes anticlockwise/clockwise from steering")
    p.add_argument("--strict", action="store_true",
                   help="skip bags whose recorded task direction disagrees with their steering")
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
 
    counts, problems = {}, []
    for bag in bags:
        d, mean_steer, tasks = scan_bag(bag, a.min_speed)
        if d == 0:
            print(f"{bag.name}: no driving commands, skipped")
            continue
        use_recorded = bool(tasks) and all(task_direction(t) != 0 for t in tasks)
        if tasks and not use_recorded:
            problems.append(f"{bag.name}: recorded task has no direction word {tasks}; "
                            f"using --default-task from steering instead")
        if use_recorded:
            first_task = tasks[0]
            task_dirs = {task_direction(t) for t in tasks}
            if task_dirs - {0} and task_dirs - {0} != {d}:
                msg = (f"{bag.name}: recorded task {tasks} but steering says "
                       f"{'anticlockwise' if d > 0 else 'clockwise'} (mean steer {mean_steer:+.3f})")
                problems.append(msg)
                if a.strict:
                    print("SKIPPED", msg)
                    continue
            if len(tasks) > 1:
                print(f"{bag.name}: task changed during recording: {tasks}")
        else:
            first_task = a.default_task.format(dir="anticlockwise" if d > 0 else "clockwise")
            if not tasks:
                problems.append(f"{bag.name}: no /task recorded, using '{first_task}' from steering")
        for mirror in ([False, True] if a.mirror else [False]):
            n = add_episode(ds, bag, first_task, a.fps, mirror, counts, use_recorded)
            shown = mirror_task(first_task) if mirror else first_task
            print(f"{bag.name} ({'mirrored' if mirror else 'original'}): mean steer "
                  f"{mean_steer:+.3f}, task '{shown}', {n} frames")
    ds.finalize()
    print("\nframes per task:")
    for task, n in sorted(counts.items()):
        print(f"  {n:6d}  {task}")
    if problems:
        print("\nCHECK THESE:")
        for m in problems:
            print("  " + m)
    print("wrote", a.out)
 
 
if __name__ == "__main__":
    main()
 
