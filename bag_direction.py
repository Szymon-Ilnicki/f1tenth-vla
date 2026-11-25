#!/usr/bin/env python3
"""Quick per-bag summary: mean steering, inferred lap direction, recorded /task.

Usage (ROS 2 sourced):
    python bag_direction.py ~/bags/*/            # one or more bag folders
    python bag_direction.py ~/bags/track_v4_all/* --topic /teleop --min-speed 0.3

Positive steering = left, so mean > 0 => anticlockwise laps.
"""
import argparse
import glob
import os
from collections import Counter

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


def storage_id(path):
    return "mcap" if glob.glob(os.path.join(path, "*.mcap")) else "sqlite3"


def scan(path, topic, min_speed):
    r = SequentialReader()
    r.open(StorageOptions(uri=path, storage_id=storage_id(path)),
           ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    if topic not in types:
        return None, f"no {topic} topic"
    msg_t = get_message(types[topic])
    task_t = get_message(types["/task"]) if "/task" in types else None

    steer, speeds, stamps, tasks = [], [], [], Counter()
    while r.has_next():
        name, data, t_ns = r.read_next()
        if name == topic:
            m = deserialize_message(data, msg_t)
            d = getattr(m, "drive", m)  # AckermannDriveStamped or AckermannDrive
            if d.speed >= min_speed:
                steer.append(d.steering_angle)
                speeds.append(d.speed)
            stamps.append(t_ns)
        elif task_t is not None and name == "/task":
            tasks[deserialize_message(data, task_t).data] += 1

    if not steer:
        return None, "no moving frames"
    s = np.asarray(steer)
    dur = (stamps[-1] - stamps[0]) / 1e9 if len(stamps) > 1 else 0.0
    return dict(
        mean=s.mean(),
        speed=float(np.mean(speeds)),
        left=(s > 0.05).mean(),
        right=(s < -0.05).mean(),
        dur=dur,
        dir="anticlockwise" if s.mean() > 0 else "clockwise",
        tasks=tasks,
    ), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--topic", default="/teleop")
    ap.add_argument("--min-speed", type=float, default=0.3)
    a = ap.parse_args()

    print(f"{'bag':40s} {'dur':>6s} {'speed':>6s} {'mean':>7s} {'%L':>5s} {'%R':>5s}  direction      task(s)")
    for b in sorted(a.bags):
        b = b.rstrip("/")
        if not os.path.isfile(os.path.join(b, "metadata.yaml")):
            continue
        try:
            res, err = scan(b, a.topic, a.min_speed)
        except Exception as e:  # noqa: BLE001
            res, err = None, str(e)
        name = os.path.basename(b)[:40]
        if res is None:
            print(f"{name:40s}  -- {err}")
            continue
        tasks = "; ".join(f"{k} (x{v})" for k, v in res["tasks"].items()) or "-"
        flag = ""
        if res["tasks"] and not any(res["dir"] in k and
                                    ("anti" in k) == (res["dir"] == "anticlockwise")
                                    for k in res["tasks"]):
            flag = "  <-- task/steering mismatch?"
        print(f"{name:40s} {res['dur']:6.0f} {res['speed']:6.2f} {res['mean']:+7.3f} "
              f"{100*res['left']:5.0f} {100*res['right']:5.0f}  "
              f"{res['dir']:13s}  {tasks}{flag}")


if __name__ == "__main__":
    main()
