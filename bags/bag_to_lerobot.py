#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

BAGS = [
    Path.home() / "bags" / "ep000",
    Path.home() / "bags" / "ep002",
]
OUT = Path.home() / "datasets" / "f1tenth_fwd_smoke"
TASK = "drive forward"
IMG_TOPIC = "/camera/camera/color/image_raw"
CMD_TOPIC = "/ackermann_cmd"
FPS = 10

def decode_image(msg):
    h, w = msg.height, msg.width
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8"):
        img = arr.reshape(h, w, 3)
        if enc == "bgr8":
            img = img[:, :, ::-1]
    elif enc in ("rgba8", "bgra8"):
        img = arr.reshape(h, w, 4)[:, :, :3]
        if enc == "bgra8":
            img = img[:, :, ::-1]
    else:
        raise RuntimeError(f"unsupported encoding {msg.encoding}")
    return np.ascontiguousarray(img)

def cmd_to_vec(msg):
    d = msg.drive if hasattr(msg, "drive") else msg
    return np.array([float(d.speed), float(d.steering_angle)], dtype=np.float32)

def read_bag(path: Path):
    imgs, cmds = [], []
    with AnyReader([path]) as reader:
        for conn, ts, raw in reader.messages():
            if conn.topic == IMG_TOPIC:
                imgs.append((ts, decode_image(reader.deserialize(raw, conn.msgtype))))
            elif conn.topic == CMD_TOPIC:
                cmds.append((ts, cmd_to_vec(reader.deserialize(raw, conn.msgtype))))
    return imgs, cmds

def nearest_cmd(cmds, t):
    if not cmds:
        return np.zeros(2, dtype=np.float32)
    i = min(range(len(cmds)), key=lambda k: abs(cmds[k][0] - t))
    return cmds[i][1]

def subsample(imgs, fps):
    if len(imgs) < 2:
        return imgs
    dt = 1e9 / fps
    out, next_t = [], imgs[0][0]
    for t, im in imgs:
        if t + 1e6 >= next_t:
            out.append((t, im))
            next_t = t + dt
    return out

def main():
    first_imgs, _ = read_bag(BAGS[0])
    h, w = first_imgs[0][1].shape[:2]
    OUT.parent.mkdir(parents=True, exist_ok=True)

    ds = LeRobotDataset.create(
        repo_id="local/f1tenth_fwd_smoke",
        root=OUT,
        fps=FPS,
        robot_type="f1tenth",
        use_videos=True,
        features={
            "observation.images.camera1": {
                "dtype": "video",
                "shape": (h, w, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["speed", "steering"],
            },
            "action": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["speed", "steering"],
            },
        },
    )

    for bag in BAGS:
        print("reading", bag)
        imgs, cmds = read_bag(bag)
        frames = subsample(imgs, FPS)
        print(f"  images={len(imgs)} cmds={len(cmds)} kept={len(frames)}")
        if len(frames) < 5:
            print("  skip, too short")
            continue
        for t, im in frames:
            act = nearest_cmd(cmds, t)
            ds.add_frame(
                {
                    "observation.images.camera1": im,
                    "observation.state": act,
                    "action": act,
                    "task": TASK,
                }
            )
        ds.save_episode()

    if hasattr(ds, "finalize"):
        ds.finalize()
    print("wrote", OUT)

if __name__ == "__main__":
    main()