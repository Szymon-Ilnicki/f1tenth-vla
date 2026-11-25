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

Takeover mode (--takeover-bags): bags recorded while the model drove and a human
took over when needed (HG-DAgger). Only frames where /human_control is True, /teleop
is fresh (the joystick only publishes while the deadman is held) and the commanded
speed is >= --min-speed are kept; reversing and standing still are dropped. Each
contiguous takeover becomes its own episode, so action chunks never mix in the
model's own actions (LeRobot pads past the episode end and masks the padding).
Takeovers shorter than --takeover-min-len are dropped, and each one is added
--takeover-repeat times (oversampling) and mirrored with --mirror. The task is the
recorded /task at that moment; frames whose task has no direction word are dropped.

--state-from-task: observation.state = [direction, fast] taken from each frame's
task (+1 anticlockwise / -1 clockwise; 1 if the task says "fast", else 0), so
policies that cannot read text (e.g. ACT) still get the command. Mirrored frames
get the mirrored task, so the direction sign flips with the image.

Task override: a file task.txt inside a bag's folder replaces the recorded /task
for every frame of that bag (e.g. when the wrong task was set while recording).

Examples:
  python3 bag_to_lerobot.py --bags bags/track_v2 --out lerobot_output/track_v3 \
      --repo-id local/track_v3 --mirror
  python3 bag_to_lerobot.py --bags ~/bags/track_v4_all \
      --takeover-bags ~/bags/experiment ~/bags/takeovers_r1 --takeover-repeat 3 \
      --out lerobot_output/track_v5 --repo-id local/track_v5 --mirror \
      --default-task "complete laps {dir} fast"
"""
import argparse
import re
import sys
import time
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
HUMAN_TOPIC = "/human_control"
 
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
    """Mirror a task: swap clockwise <-> anticlockwise and left <-> right (whole words).
    "anticlockwise" contains "clockwise", so go via placeholders."""
    t = (task.replace("anticlockwise", "\0")
             .replace("clockwise", "anticlockwise")
             .replace("\0", "clockwise"))
    t = re.sub(r"\bleft\b", "\1", t)
    t = re.sub(r"\bright\b", "left", t)
    return t.replace("\1", "right")


# Tasks that are valid without a direction word (e.g. "stop"); set with --nodir-tasks.
NODIR_TASKS = {"stop"}


def task_valid(task: str) -> bool:
    return task_direction(task) != 0 or task.strip().lower() in NODIR_TASKS
 
 
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
 
 
def iter_synced(bag_dir, first_task, use_recorded=True, max_cmd_age=None, stats=None):
    """(t, rgb, depth, action, task) per colour frame; task = latest /task so far
    (first_task until the first /task message of the bag arrives). With max_cmd_age,
    frames whose latest /teleop is older than that are skipped (the joystick only
    publishes while a deadman button is held, so an old command is not what the
    car was doing); they are counted in stats["stale"]."""
    wanted = {IMG_TOPIC, DEPTH_TOPIC, CMD_TOPIC, TASK_TOPIC}
    reader = open_bag(bag_dir, wanted)
    types = bag_types(reader, wanted)
    last_cmd = None
    last_cmd_t = None
    last_depth = None
    task = first_task
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in types:
            continue
        if topic == IMG_TOPIC and max_cmd_age is not None and last_cmd_t is not None \
                and (t - last_cmd_t) * 1e-9 > max_cmd_age:
            if stats is not None:
                stats["stale"] = stats.get("stale", 0) + 1
            continue
        msg = deserialize_message(data, types[topic])
        if topic == CMD_TOPIC:
            last_cmd = cmd_vec(msg)
            last_cmd_t = t
        elif topic == TASK_TOPIC:
            if use_recorded and msg.data:
                task = msg.data
        elif topic == DEPTH_TOPIC:
            last_depth = depth_to_rgb(msg)
        elif topic == IMG_TOPIC and last_cmd is not None and last_depth is not None:
            yield t * 1e-9, img_np(msg), last_depth, last_cmd, task
 
 
STATE_FROM_TASK = False


def state_vec(task):
    if not STATE_FROM_TASK:
        return np.zeros(2, dtype=np.float32)
    return np.array([float(task_direction(task)), 1.0 if "fast" in task else 0.0], dtype=np.float32)


class Progress:
    """'[i/N] bag' prefix plus an in-place frame counter on the current line."""

    def __init__(self, total):
        self.total, self.i, self.t0 = total, 0, time.time()
        self.tty = sys.stdout.isatty()

    def next_bag(self, name):
        self.i += 1
        self.name = name
        self.last = 0.0
        self.tick(0)

    def tick(self, n, label="frames"):
        if not self.tty or time.time() - self.last < 0.5:
            return
        self.last = time.time()
        el = time.time() - self.t0
        print(f"\r\033[K[{self.i}/{self.total}] {self.name}: {n} {label}  ({el / 60:.1f} min elapsed)",
              end="", flush=True)

    def clear(self):
        if self.tty:
            print("\r\033[K", end="", flush=True)

    @property
    def tag(self):
        return f"[{self.i}/{self.total}]"


PROGRESS = None


class FrameGrid:
    """Pick camera frames on a fixed 1/fps time grid. The camera runs at ~15 Hz, so
    "at least 1/fps since the last kept frame" would keep every 2nd frame (7.5 fps)
    while the dataset claims 10 fps; stepping a grid keeps the true average rate."""

    def __init__(self, fps):
        self.dt = 1.0 / fps
        self.next = None

    def keep(self, t):
        if self.next is None:
            self.next = t
        if t < self.next - 1e-3:
            return False
        self.next += self.dt
        if t - self.next > self.dt:  # after a gap in the bag, restart the grid
            self.next = t + self.dt
        return True


def add_episode(ds, bag, first_task, fps, mirror, counts, use_recorded=True, max_cmd_age=None):
    n = 0
    grid = FrameGrid(fps)
    stats = {}
    for t, img, depth_img, act, task in iter_synced(bag, first_task, use_recorded, max_cmd_age, stats):
        if not grid.keep(t):
            continue
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
            "observation.state": state_vec(task),
            "action": act,
            "task": task,
        })
        n += 1
        if PROGRESS:
            PROGRESS.tick(n, "mirrored frames" if mirror else "frames")
    if PROGRESS:
        PROGRESS.clear()
    if stats.get("stale") and not mirror:
        print(f"    {bag.name}: skipped {stats['stale']} camera frames with stale /teleop "
              f"(no joystick command for > {max_cmd_age} s)")
    if n:
        ds.save_episode()
    return n
 
 
def task_override(bag_dir):
    """Contents of <bag>/task.txt, or None."""
    f = Path(bag_dir) / "task.txt"
    if f.is_file():
        t = f.read_text().strip()
        return t or None
    return None


def iter_takeover(bag_dir, teleop_max_age, override=None):
    """Per colour frame: (t, human_ok, rgb_msg, depth_msg, action, task).
    human_ok = /human_control True and /teleop newer than teleop_max_age.
    Images are returned undecoded so unused frames cost nothing."""
    wanted = {IMG_TOPIC, DEPTH_TOPIC, CMD_TOPIC, TASK_TOPIC, HUMAN_TOPIC}
    reader = open_bag(bag_dir, wanted)
    types = bag_types(reader, wanted)
    if HUMAN_TOPIC not in types:
        raise ValueError(f"no {HUMAN_TOPIC} topic")
    last_cmd, last_cmd_t = None, -1e18
    last_depth, task, human = None, override or "", False
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic not in types:
            continue
        t = t_ns * 1e-9
        if topic == IMG_TOPIC:
            if last_depth is None:
                continue
            ok = human and last_cmd is not None and (t - last_cmd_t) <= teleop_max_age
            yield t, ok, data, last_depth, last_cmd, task
            continue
        if topic == DEPTH_TOPIC:
            last_depth = data
            continue
        msg = deserialize_message(data, types[topic])
        if topic == CMD_TOPIC:
            last_cmd, last_cmd_t = cmd_vec(msg), t
        elif topic == TASK_TOPIC:
            if msg.data and not override:
                task = msg.data
        elif topic == HUMAN_TOPIC:
            human = bool(msg.data)


def takeover_segments(bag, a):
    """List of takeovers, each a list of (t, rgb, depth_rgb, action, task); plus stats."""
    segs, cur = [], []
    grid = FrameGrid(a.fps)
    stats = {"short": 0, "no_dir": 0}
    for t, ok, img_data, depth_data, act, task in iter_takeover(bag, a.teleop_max_age,
                                                                task_override(bag)):
        if not grid.keep(t):
            continue
        if ok and act[0] >= a.min_speed and not task_valid(task):
            stats["no_dir"] += 1
            ok = False
        if PROGRESS:
            PROGRESS.tick(sum(map(len, segs)) + len(cur), "takeover frames found")
        if ok and act[0] >= a.min_speed:
            cur.append((t, img_np(deserialize_message(img_data, Image)),
                        depth_to_rgb(deserialize_message(depth_data, Image)), act.copy(), task))
        elif cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    if PROGRESS:
        PROGRESS.clear()
    out = []
    for s in segs:
        s = [f for f in s if f[0] - s[0][0] >= a.takeover_start_trim]
        if not s or (s[-1][0] - s[0][0] + 1.0 / a.fps) < a.takeover_min_len:
            stats["short"] += 1
            continue
        out.append(s)
    return out, stats


def add_frames(ds, frames, mirror, counts):
    for _, img, depth_img, act, task in frames:
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
            "observation.state": state_vec(task),
            "action": act,
            "task": task,
        })
    ds.save_episode()


def find_bags(folders, pattern="*"):
    """All bag directories (containing metadata.yaml) at any depth under each folder,
    whose name matches pattern, or the folder itself if it is a bag. Sorted by path,
    duplicates removed."""
    found = []
    for f in folders:
        f = Path(f).expanduser()
        if (f / "metadata.yaml").is_file():
            cands = [f]
        else:
            cands = [m.parent for m in f.rglob("metadata.yaml")]
        found += [b for b in cands if b.match(pattern)]
    return sorted(set(found))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bags", nargs="*", default=[],
                   help="folder(s) searched recursively for bags of human-driven laps")
    p.add_argument("--takeover-bags", nargs="*", default=[],
                   help="folder(s) searched recursively for model-driven bags; "
                        "only human takeovers are used")
    p.add_argument("--bag-glob", default="*",
                   help="only use bag folders whose name matches this (e.g. 'ep*')")
    p.add_argument("--takeover-repeat", type=int, default=3,
                   help="add each takeover episode this many times (oversampling)")
    p.add_argument("--takeover-min-len", type=float, default=0.5,
                   help="drop takeovers shorter than this [s]")
    p.add_argument("--takeover-start-trim", type=float, default=0.0,
                   help="drop this many seconds from the start of each takeover")
    p.add_argument("--teleop-max-age", type=float, default=0.2,
                   help="a takeover frame needs a /teleop message at most this old [s]")
    p.add_argument("--out", default=str(HERE / "lerobot_output/track_v3"))
    p.add_argument("--repo-id", default="local/track_v3")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--mirror", action="store_true", help="also add every bag flipped left-right")
    p.add_argument("--min-speed", type=float, default=0.3,
                   help="ignore stationary commands for direction; takeover frames below it are dropped")
    p.add_argument("--default-task", default="complete laps {dir} around the track",
                   help="for bags with no /task; {dir} becomes anticlockwise/clockwise from steering")
    p.add_argument("--lap-teleop-max-age", type=float, default=0.2,
                   help="lap bags: skip frames whose /teleop is older than this [s]; 0 = keep all (old behaviour)")
    p.add_argument("--nodir-tasks", nargs="*", default=["stop"],
                   help="tasks that are valid without a direction word (default: stop)")
    p.add_argument("--state-from-task", action="store_true",
                   help="write observation.state = [direction +-1, fast 0/1] from the task (for ACT)")
    p.add_argument("--strict", action="store_true",
                   help="skip bags whose recorded task direction disagrees with their steering")
    a = p.parse_args()
    global STATE_FROM_TASK, NODIR_TASKS
    STATE_FROM_TASK = a.state_from_task
    NODIR_TASKS = {t.strip().lower() for t in a.nodir_tasks}
    if a.lap_teleop_max_age <= 0:
        a.lap_teleop_max_age = None
 
    bags = find_bags(a.bags, a.bag_glob)
    tbags = find_bags(a.takeover_bags, a.bag_glob)
    if a.bags and not bags:
        raise SystemExit(f"no bags matching '{a.bag_glob}' found in {a.bags}")
    if a.takeover_bags and not tbags:
        raise SystemExit(f"no bags found in {a.takeover_bags}")
    if not bags and not tbags:
        raise SystemExit("give --bags and/or --takeover-bags")
    print(f"found {len(bags)} lap bags, {len(tbags)} takeover bags")
    for label, group, roots in (("lap", bags, a.bags), ("takeover", tbags, a.takeover_bags)):
        per = {}
        for b in group:
            rel = next((b.parent.relative_to(Path(r).expanduser()) for r in roots
                        if b.is_relative_to(Path(r).expanduser())), b.parent)
            per[str(rel)] = per.get(str(rel), 0) + 1
        for folder, n in sorted(per.items()):
            print(f"  {label:8s} {n:3d}  {folder}")
    global PROGRESS
    PROGRESS = Progress(len(bags) + len(tbags))
 
    ds = LeRobotDataset.create(
        repo_id=a.repo_id,
        fps=a.fps,
        robot_type="f1tenth",
        features={
            "observation.images.camera1": {
                "dtype": "video", "shape": (240, 424, 3), "names": ["height", "width", "channels"]},
            "observation.images.camera2": {
                "dtype": "video", "shape": (240, 424, 3), "names": ["height", "width", "channels"]},
            "observation.state": {"dtype": "float32", "shape": (2,),
                                  "names": ["direction", "fast"] if a.state_from_task else ["dummy1", "dummy2"]},
            "action": {"dtype": "float32", "shape": (2,), "names": ["speed", "steering"]},
        },
        root=str(Path(a.out).expanduser()),
        use_videos=True,
    )
 
    counts, problems = {}, []
    for bag in bags:
        PROGRESS.next_bag(bag.name)
        d, mean_steer, tasks = scan_bag(bag, a.min_speed)
        if d == 0:
            print(f"{bag.name}: no driving commands, skipped")
            continue
        ov = task_override(bag)
        if ov:
            print(f"{bag.name}: task.txt overrides recorded {tasks} -> '{ov}'")
            if task_direction(ov) == 0:
                problems.append(f"{bag.name}: task.txt '{ov}' has no direction word, skipped")
                continue
            if task_direction(ov) != d:
                msg = (f"{bag.name}: task.txt '{ov}' but steering says "
                       f"{'anticlockwise' if d > 0 else 'clockwise'} (mean steer {mean_steer:+.3f})")
                problems.append(msg)
                if a.strict:
                    print("SKIPPED", msg)
                    continue
            for mirror in ([False, True] if a.mirror else [False]):
                n = add_episode(ds, bag, ov, a.fps, mirror, counts, use_recorded=False,
                                max_cmd_age=a.lap_teleop_max_age)
                print(f"{PROGRESS.tag} {bag.name} ({'mirrored' if mirror else 'original'}): mean steer "
                      f"{mean_steer:+.3f}, task '{mirror_task(ov) if mirror else ov}', {n} frames")
            continue
        use_recorded = bool(tasks) and all(task_valid(t) for t in tasks) \
            and any(task_direction(t) != 0 for t in tasks)
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
            n = add_episode(ds, bag, first_task, a.fps, mirror, counts, use_recorded,
                        max_cmd_age=a.lap_teleop_max_age)
            shown = mirror_task(first_task) if mirror else first_task
            print(f"{PROGRESS.tag} {bag.name} ({'mirrored' if mirror else 'original'}): mean steer "
                  f"{mean_steer:+.3f}, task '{shown}', {n} frames")
    lap_frames = sum(counts.values())
    takeover_frames = 0
    for bag in tbags:
        PROGRESS.next_bag(bag.name)
        try:
            segs, st = takeover_segments(bag, a)
        except ValueError as e:
            problems.append(f"{bag.name}: {e}, skipped")
            continue
        if task_override(bag):
            print(f"{bag.name}: task.txt override '{task_override(bag)}'")
        if st["no_dir"]:
            problems.append(f"{bag.name}: {st['no_dir']} takeover frames dropped, /task has no direction word")
        secs = sum(s[-1][0] - s[0][0] + 1.0 / a.fps for s in segs)
        print(f"{PROGRESS.tag} {bag.name} (takeovers): {len(segs)} kept, {st['short']} too short, "
              f"{secs:.1f} s, x{a.takeover_repeat}{' x2 mirrored' if a.mirror else ''}")
        for seg in segs:
            for _ in range(a.takeover_repeat):
                for mirror in ([False, True] if a.mirror else [False]):
                    add_frames(ds, seg, mirror, counts)
                    takeover_frames += len(seg)
    print(f"finalizing dataset ... ({(time.time() - PROGRESS.t0) / 60:.1f} min so far)", flush=True)
    ds.finalize()
    if tbags:
        print(f"\nlap frames {lap_frames}, takeover frames {takeover_frames} (incl. repeats)")
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
 
