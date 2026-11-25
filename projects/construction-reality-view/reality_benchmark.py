import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def synthetic():
    """Known translations with changes in brightness, blur and occlusion."""
    rng = np.random.default_rng(17)
    marker = rng.integers(15, 245, (17, 17), dtype=np.uint8)
    frames, points, hidden = [], [], []
    for index in range(72):
        x = 70 + round(24 * np.sin(index * .22))
        y = 48 + round(12 * np.cos(index * .17))
        obscured = 43 <= index < 50
        frame = np.full((96, 144), 100, dtype=np.uint8)
        frame[y-8:y+9, x-8:x+9] = marker
        if 16 <= index < 28:
            frame = np.clip(frame.astype(float) * .65 + 55, 0, 255).astype(np.uint8)
        if 28 <= index < 43:
            frame = np.asarray(Image.fromarray(frame).filter(ImageFilter.GaussianBlur(1.4)))
        if obscured:
            frame[y-8:y+9, x-8:x+9] = 100
        frames.append(np.repeat(frame[..., None], 3, axis=2))
        points.append((x, y))
        hidden.append(obscured)
    return np.stack(frames), np.array(points)[None], np.array(hidden)[None]


def davis(path, clip=None):
    # Load only the publisher's trusted archive. Pickle can execute code.
    with path.open("rb") as source:
        videos = pickle.load(source)
    name = clip or sorted(videos)[0]
    data = videos[name]
    video = data["video"]
    height, width = video.shape[1:3]
    points = np.asarray(data["points"]) * [width, height]
    return name, video, points, np.asarray(data["occluded"], dtype=bool)


def baseline(video, queries):
    """Reproduce the HTML's fixed template, 2px sampling and ±12px search."""
    frames = video.astype(np.uint16)
    gray = (frames[..., 0]*77 + frames[..., 1]*150 + frames[..., 2]*29) >> 8
    height, width = gray.shape[1:]
    points = np.full((len(queries), len(video), 2), np.nan)
    visible = np.zeros(points.shape[:2], dtype=bool)
    for point, (start, raw_x, raw_y) in enumerate(queries):
        x, y = int(raw_x), int(raw_y)
        if not (8 <= x < width-8 and 8 <= y < height-8):
            continue
        pattern = gray[start, y-8:y+9:2, x-8:x+9:2].astype(float)
        if np.ptp(pattern) < 45:
            continue
        for index in range(start, len(video)):
            ylo, yhi = max(8, y-12), min(height-9, y+12)
            xlo, xhi = max(8, x-12), min(width-9, x+12)
            region = gray[index, ylo-8:yhi+9, xlo-8:xhi+9]
            patches = np.lib.stride_tricks.sliding_window_view(region, (17, 17))[..., ::2, ::2]
            errors = ((patches - pattern)**2).mean(axis=(-1, -2))
            dy, dx = np.unravel_index(errors.argmin(), errors.shape)
            if errors[dy, dx] <= 1600:
                x, y = xlo+dx, ylo+dy
                points[point, index] = x, y
                visible[point, index] = True
    return points, visible


def score(pred, visible, truth, hidden, queries):
    evaluated = np.arange(truth.shape[1])[None, :] > queries[:, 0, None]
    valid = evaluated & ~hidden
    seen = valid & visible
    errors = np.linalg.norm(pred - truth, axis=-1)
    obscured = evaluated & hidden
    recoveries = []
    for mask, predictions in zip(obscured, seen & (errors <= 4)):
        for end in np.flatnonzero(mask[:-1] & ~mask[1:]) + 1:
            next_occlusion = np.flatnonzero(mask[end:])
            stop = end + next_occlusion[0] if len(next_occlusion) else len(mask)
            hits = np.flatnonzero(predictions[end:stop])
            recoveries.append(int(hits[0]) if len(hits) else None)
    return {
        "visible_truth_frames": int(valid.sum()),
        "missed_visible_frames": int((valid & ~visible).sum()),
        "false_visible_frames": int((obscured & visible).sum()),
        "median_reported_error_px": float(np.median(errors[seen])) if seen.any() else None,
        "within_4px_rate": float((seen & (errors <= 4)).sum() / valid.sum()) if valid.any() else None,
        "recovery_frames_after_occlusion": recoveries,
    }


def track(args):
    name = "controlled-simulation"
    if args.davis:
        name, frames, truth, hidden = davis(args.davis, args.clip)
    else:
        frames, truth, hidden = synthetic()
    eligible = np.flatnonzero((~hidden).any(axis=1))[:args.points]
    if not len(eligible):
        raise ValueError("No visible query points")
    truth, hidden = truth[eligible], hidden[eligible]
    starts = (~hidden).argmax(axis=1)
    queries = np.column_stack((starts, truth[np.arange(len(starts)), starts])).astype(int)
    start = time.perf_counter()
    pred, visible = baseline(frames, queries)
    result = {"clip": name, "resolution": list(frames.shape[1:3][::-1]),
              "baseline": score(pred, visible, truth, hidden, queries),
              "baseline_processing_sec": time.perf_counter() - start}
    if args.export:
        np.savez_compressed(args.export, frames=frames, queries=queries)
    if args.predictions:
        with np.load(args.predictions) as data:
            pred, visible = data["points"], data["visible"].astype(bool)
        if pred.shape != truth.shape or visible.shape != hidden.shape:
            raise ValueError("Predictions must match the exported query order and frame count")
        if not np.isfinite(pred[visible]).all():
            raise ValueError("Visible predictions must contain finite coordinates")
        result["ai"] = score(pred, visible, truth, hidden, queries)
    print(json.dumps(result, indent=2))


def geometry(args):
    depth = np.asarray(Image.open(args.depth))
    if depth.ndim != 2 or args.factor <= 0 or args.step < 1 or min(args.fx, args.fy) <= 0:
        raise ValueError("Expected single-channel depth and positive focal lengths, scale and step")
    y, x = np.mgrid[:depth.shape[0]:args.step, :depth.shape[1]:args.step]
    z = depth[y, x] / args.factor
    points = np.stack(((x-args.cx)*z/args.fx, (y-args.cy)*z/args.fy, z), axis=-1)
    valid = z > 0
    if args.rgb:
        rgb = np.asarray(Image.open(args.rgb).convert("RGB"))
        if rgb.shape[:2] != depth.shape:
            raise ValueError("RGB and depth must be aligned and have the same resolution")
        values = np.column_stack((points[valid], rgb[y, x][valid]))
        props = ["property uchar red", "property uchar green", "property uchar blue"]
        fmt = ["%.6f"]*3 + ["%d"]*3
    else:
        values, props, fmt = points[valid], [], "%.6f"
    header = ["ply", "format ascii 1.0", f"element vertex {len(values)}",
              "property float x", "property float y", "property float z", *props, "end_header"]
    with args.output.open("w", encoding="ascii") as file:
        file.write("\n".join(header) + "\n")
        np.savetxt(file, values, fmt=fmt)
    print(f"{len(values)} points in camera coordinates (meters): {args.output}")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(required=True)
    tracking = commands.add_parser("track")
    tracking.add_argument("--davis", type=Path)
    tracking.add_argument("--clip")
    tracking.add_argument("--points", type=int, default=8)
    tracking.add_argument("--export", type=Path)
    tracking.add_argument("--predictions", type=Path)
    tracking.set_defaults(action=track)
    geo = commands.add_parser("geometry")
    geo.add_argument("--depth", required=True, type=Path)
    geo.add_argument("--rgb", type=Path)
    geo.add_argument("--output", required=True, type=Path)
    for key in ("fx", "fy", "cx", "cy"):
        geo.add_argument("--"+key, type=float, required=True)
    geo.add_argument("--factor", type=float, default=5000)
    geo.add_argument("--step", type=int, default=2)
    geo.set_defaults(action=geometry)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
