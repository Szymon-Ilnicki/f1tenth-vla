#!/usr/bin/env python3
"""Compare autonomous SmolVLA runs across bags (e.g. a good day vs a bad day).

Only the autonomous parts are analysed: /human_control says the VLA is driving,
the car is moving, and a fresh /smolvla/action exists. Idle time, pauses and
teleop segments are dropped automatically.

Usage (ROS 2 sourced, on the Jetson or anywhere with the bags):
    python compare_runs.py ~/bags/experiment/run1 ~/bags/experiment/today_v4nc ... \
        --out compare_out

Prints a side-by-side table and writes compare_out/compare.png,
compare_out/frames.png and compare_out/summary.json.
"""
import argparse
import glob
import json
import os

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from rosidl_runtime_py.utilities import get_message

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RGB = "/camera/camera/color/image_raw"
DEPTH = "/camera/camera/depth/image_rect_raw"
TOPICS = ["/human_control", "/drive", "/ackermann_cmd", "/smolvla/action",
          "/smolvla/infer_dt", "/odom", "/sensors/imu/raw", "/task",
          "/commands/servo/position", RGB, DEPTH]


def storage_id(path):
    return "mcap" if glob.glob(os.path.join(path, "*.mcap")) else "sqlite3"


def read_bag(path, img_every):
    r = SequentialReader()
    r.open(StorageOptions(uri=path, storage_id=storage_id(path)), ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    present = [t for t in TOPICS if t in types]
    r.set_filter(StorageFilter(topics=present))
    cls = {t: get_message(types[t]) for t in present}

    d = {k: [] for k in ["hc", "drive", "cmd", "act", "idt", "odom", "imu", "task",
                         "servo", "rgb", "depth"]}
    frames = []
    n_rgb = n_depth = 0
    t0 = None
    while r.has_next():
        name, data, t_ns = r.read_next()
        t = t_ns * 1e-9
        if t0 is None:
            t0 = t
        t -= t0
        if name == RGB:
            n_rgb += 1
            if n_rgb % img_every:
                continue
        if name == DEPTH:
            n_depth += 1
            if n_depth % img_every:
                continue
        m = deserialize_message(data, cls[name])
        if name == "/human_control":
            d["hc"].append((t, float(m.data)))
        elif name == "/drive":
            d["drive"].append((t, m.drive.steering_angle, m.drive.speed))
        elif name == "/ackermann_cmd":
            d["cmd"].append((t, m.drive.steering_angle, m.drive.speed))
        elif name == "/smolvla/action":
            v = list(m.data) + [np.nan, np.nan]
            d["act"].append((t, v[0], v[1]))  # [speed, steer]
        elif name == "/smolvla/infer_dt":
            d["idt"].append((t, m.data))
        elif name == "/odom":
            d["odom"].append((t, m.twist.twist.linear.x))
        elif name == "/sensors/imu/raw":
            d["imu"].append((t, m.angular_velocity.z))
        elif name == "/task":
            d["task"].append((t, m.data))
        elif name == "/commands/servo/position":
            d["servo"].append((t, m.data))
        elif name == RGB:
            img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, : m.width * 3]
            img = img.reshape(m.height, m.width, 3)
            if m.encoding.lower().startswith("bgr"):
                img = img[..., ::-1]
            d["rgb"].append((t, float(img.mean())))
            frames.append((t, img.copy()))
        elif name == DEPTH:
            dep = np.frombuffer(m.data, np.uint16)
            d["depth"].append((t, float((dep == 0).mean())))
    out = {}
    for k, v in d.items():
        if k == "task":
            out[k] = v
        else:
            out[k] = np.array(v, dtype=float) if v else np.zeros((0, 2))
    return out, frames, present


def hold(arr, grid, col=1):
    """Zero-order hold of arr[:, col] at grid times; also returns sample age."""
    if len(arr) == 0:
        return np.full(len(grid), np.nan), np.full(len(grid), np.inf)
    ts = arr[:, 0]
    idx = np.searchsorted(ts, grid, side="right") - 1
    val = np.full(len(grid), np.nan)
    age = np.full(len(grid), np.inf)
    ok = idx >= 0
    val[ok] = arr[idx[ok], col]
    age[ok] = grid[ok] - ts[idx[ok]]
    return val, age


def analyse(path, a):
    d, frames, present = read_bag(path, a.img_every)
    tend = max(x[-1, 0] for k, x in d.items() if k != "task" and len(x))
    g = np.arange(0, tend, 1.0 / a.grid_hz)

    hc, _ = hold(d["hc"], g)
    human = hc > 0.5 if not a.invert_human else hc < 0.5
    human[np.isnan(hc)] = True
    _, act_age = hold(d["act"], g)
    v, _ = hold(d["odom"], g)
    wz, _ = hold(d["imu"], g)
    steer_src = d["cmd"] if len(d["cmd"]) else d["drive"]

    auto = (~human) & (act_age < 0.5) & (v > a.min_speed)
    still = np.abs(v) < 0.02

    # task per grid point
    task_t = np.array([t for t, _ in d["task"]]) if d["task"] else np.zeros(0)
    tidx = np.searchsorted(task_t, g, side="right") - 1
    tasks = np.array([d["task"][i][1] if i >= 0 else "" for i in tidx]) if len(task_t) else np.array([""] * len(g))

    gyro_bias = float(np.nanmean(wz[still])) if still.any() else 0.0
    wz_c = wz - gyro_bias

    # curvature fit with servo lag search: kappa = gain * tan(delta)/L + offset
    best = None
    for lag in np.arange(0, 0.51, 0.05):
        dl, _ = hold(steer_src, g - lag)
        x = np.tan(dl) / a.wheelbase
        k = wz_c / np.maximum(v, 1e-3)
        m = auto & np.isfinite(x) & np.isfinite(k) & (np.abs(k) < 5)
        if m.sum() < 50:
            continue
        c = np.corrcoef(x[m], k[m])[0, 1]
        if best is None or c > best[0]:
            best = (c, lag, x[m], k[m])
    fit = {}
    if best:
        c, lag, x, k = best
        gain, off = np.polyfit(x, k, 1)
        fit = dict(corr=float(c), lag_s=float(lag), gain=float(gain), offset_curv=float(off),
                   steer_offset_rad=float(np.arctan(-off * a.wheelbase / gain)) if gain else np.nan,
                   x=x, k=k)

    dl_now, _ = hold(steer_src, g)
    act_speed, _ = hold(d["act"], g, col=1)
    servo, _ = hold(d["servo"], g)

    def stats(mask):
        if mask.sum() == 0:
            return None
        s = dl_now[mask]
        return dict(seconds=float(mask.sum() / a.grid_hz),
                    speed_mean=float(np.nanmean(v[mask])),
                    act_speed_mean=float(np.nanmean(act_speed[mask])),
                    steer_mean=float(np.nanmean(s)),
                    steer_abs_mean=float(np.nanmean(np.abs(s))),
                    steer_sat_frac=float(np.nanmean(np.abs(s) >= a.steer_sat)),
                    servo_mean=float(np.nanmean(servo[mask])))

    per_dir = {}
    for name, key in [("anticlockwise", "anticlockwise"), ("clockwise", None)]:
        if key:
            m = auto & (np.char.find(tasks.astype(str), "anticlockwise") >= 0)
        else:
            m = auto & (np.char.find(tasks.astype(str), "clockwise") >= 0) & \
                (np.char.find(tasks.astype(str), "anticlockwise") < 0)
        per_dir[name] = stats(m)

    # straight-line check: commanded ~0, what did the car do?
    straight = auto & (np.abs(dl_now) < 0.02)
    straight_curv = float(np.nanmean(wz_c[straight] / v[straight])) if straight.sum() > 10 else np.nan

    idt = d["idt"][:, 1] if len(d["idt"]) else np.zeros(0)
    def auto_vals(arr):
        if len(arr) == 0 or auto.sum() == 0:
            return np.zeros(0)
        m, _ = hold(np.c_[g, auto.astype(float)], arr[:, 0])
        return arr[m > 0.5, 1]

    return dict(
        name=os.path.basename(path.rstrip("/")),
        topics=present,
        auto_seconds=float(auto.sum() / a.grid_hz),
        tasks=sorted(set(t for _, t in d["task"])),
        infer_dt_mean=float(idt.mean()) if len(idt) else np.nan,
        infer_dt_p90=float(np.percentile(idt, 90)) if len(idt) else np.nan,
        gyro_bias=gyro_bias,
        straight_curv=straight_curv,
        fit=fit,
        per_dir=per_dir,
        rgb_mean=float(np.mean(auto_vals(d["rgb"]))) if len(auto_vals(d["rgb"])) else np.nan,
        depth_invalid=float(np.mean(auto_vals(d["depth"]))) if len(auto_vals(d["depth"])) else np.nan,
        _speed=v[auto], _steer=dl_now[auto], _idt=idt,
        _rgb=auto_vals(d["rgb"]), _depth=auto_vals(d["depth"]),
        _frames=[(t, f) for t, f in frames if auto[min(int(t * a.grid_hz), len(auto) - 1)]],
    )


def fmt(x, f="{:+.3f}"):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f.format(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--out", default="compare_out")
    ap.add_argument("--wheelbase", type=float, default=0.33)
    ap.add_argument("--min-speed", type=float, default=0.3)
    ap.add_argument("--steer-sat", type=float, default=0.39)
    ap.add_argument("--grid-hz", type=float, default=20.0)
    ap.add_argument("--img-every", type=int, default=15, help="analyse every Nth camera frame")
    ap.add_argument("--invert-human", action="store_true",
                    help="use if /human_control is False while the human drives")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    res = []
    for b in a.bags:
        print(f"reading {b} ...", flush=True)
        res.append(analyse(b, a))

    rows = [
        ("autonomous time [s]", lambda r: fmt(r["auto_seconds"], "{:.0f}")),
        ("infer_dt mean / p90 [s]", lambda r: f"{fmt(r['infer_dt_mean'], '{:.2f}')} / {fmt(r['infer_dt_p90'], '{:.2f}')}"),
        ("steer gain (1 = ideal)", lambda r: fmt(r["fit"].get("gain"), "{:.2f}")),
        ("steer offset [rad]", lambda r: fmt(r["fit"].get("steer_offset_rad"))),
        ("servo lag [s]", lambda r: fmt(r["fit"].get("lag_s"), "{:.2f}")),
        ("fit corr", lambda r: fmt(r["fit"].get("corr"), "{:.2f}")),
        ("curvature on 'straights' [1/m]", lambda r: fmt(r["straight_curv"])),
        ("gyro bias [rad/s]", lambda r: fmt(r["gyro_bias"], "{:+.4f}")),
        ("RGB mean brightness", lambda r: fmt(r["rgb_mean"], "{:.0f}")),
        ("depth invalid fraction", lambda r: fmt(r["depth_invalid"], "{:.3f}")),
    ]
    for dname in ["anticlockwise", "clockwise"]:
        for key, lab, f in [("seconds", "time [s]", "{:.0f}"), ("speed_mean", "real speed [m/s]", "{:.2f}"),
                            ("act_speed_mean", "VLA speed setpoint", "{:.2f}"),
                            ("steer_mean", "steer mean [rad]", "{:+.3f}"),
                            ("steer_abs_mean", "|steer| mean [rad]", "{:.3f}"),
                            ("steer_sat_frac", "steer saturated frac", "{:.2f}"),
                            ("servo_mean", "servo position mean", "{:.3f}")]:
            rows.append((f"{dname[:4]}: {lab}",
                         lambda r, dn=dname, k=key, f=f: fmt((r["per_dir"][dn] or {}).get(k), f)))

    w = max(len(r["name"]) for r in res) + 2
    print("\n" + " " * 34 + "".join(f"{r['name']:>{max(w, 14)}}" for r in res))
    for lab, fn in rows:
        print(f"{lab:34s}" + "".join(f"{fn(r):>{max(w, 14)}}" for r in res))
    for r in res:
        print(f"\n{r['name']} tasks: {r['tasks']}")
        missing = [t for t in TOPICS if t not in r["topics"]]
        if missing:
            print(f"  missing topics: {missing}")

    # plots
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for r in res:
        f = r["fit"]
        if f:
            idx = np.random.default_rng(0).choice(len(f["x"]), min(1500, len(f["x"])), replace=False)
            ax[0, 0].scatter(f["x"][idx], f["k"][idx], s=3, alpha=0.3)
            xs = np.linspace(f["x"].min(), f["x"].max(), 10)
            ax[0, 0].plot(xs, f["gain"] * xs + f["offset_curv"],
                          label=f"{r['name']}: gain {f['gain']:.2f}, off {f['steer_offset_rad']:+.3f} rad")
        ax[0, 1].hist(r["_idt"], bins=30, alpha=0.5, label=r["name"])
        ax[0, 2].hist(r["_speed"], bins=40, alpha=0.5, label=r["name"])
        ax[1, 0].hist(r["_steer"][np.isfinite(r["_steer"])], bins=41, alpha=0.5, label=r["name"])
        ax[1, 1].hist(r["_rgb"], bins=30, alpha=0.5, label=r["name"])
        ax[1, 2].hist(r["_depth"], bins=30, alpha=0.5, label=r["name"])
    lim = ax[0, 0].get_xlim()
    ax[0, 0].plot(lim, lim, "k--", lw=0.8, label="ideal")
    titles = ["measured curvature vs commanded tan(δ)/L", "infer_dt [s]", "real speed (autonomous) [m/s]",
              "commanded steering (autonomous) [rad]", "RGB mean brightness", "depth invalid fraction"]
    for axx, t in zip(ax.flat, titles):
        axx.set_title(t)
        axx.legend(fontsize=7)
        axx.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "compare.png"), dpi=110)

    nf = 4
    fig, ax = plt.subplots(len(res), nf, figsize=(4 * nf, 2.6 * len(res)), squeeze=False)
    for i, r in enumerate(res):
        fr = r["_frames"]
        pick = [fr[j] for j in np.linspace(0, len(fr) - 1, nf).astype(int)] if fr else []
        for j in range(nf):
            ax[i, j].axis("off")
            if j < len(pick):
                ax[i, j].imshow(pick[j][1])
                ax[i, j].set_title(f"{r['name']}  t={pick[j][0]:.0f}s", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "frames.png"), dpi=100)

    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in res]
    for c in clean:
        c["fit"] = {k: v for k, v in c["fit"].items() if k not in ("x", "k")}
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(clean, fh, indent=2, default=float)
    print(f"\nwrote {a.out}/compare.png, frames.png, summary.json")


if __name__ == "__main__":
    main()
