#!/usr/bin/env python3
"""LeRobot episode web visualizer.

Usage:
    python server.py <dataset_path> [--port PORT]
"""
import argparse
import io
import json
from pathlib import Path

import jinja2
import numpy as np
from aiohttp import web
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _build_episode_bounds(ds, max_episodes=None):
    """Compute [start, end) global index for each episode.

    Leverages the fact that episodes are stored contiguously — breaks early
    once we have found `max_episodes` episode boundaries.
    """
    ep_indices = ds.hf_dataset["episode_index"]
    bounds = {}
    for i, ep_idx in enumerate(ep_indices):
        e = ep_idx.item()
        if e not in bounds:
            if max_episodes is not None and e >= max_episodes:
                break  # all following episodes are past the limit
            bounds[e] = [i, i + 1]
        else:
            bounds[e][1] = i + 1
    return [bounds[i] for i in sorted(bounds)]


def _tensor_to_jpeg(tensor):
    """Convert [C,H,W] float32 [0,1] tensor to JPEG bytes."""
    arr = (tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _load_raw_info(dataset_path):
    """Load names from meta/info.json (not exposed by LeRobotDataset.meta)."""
    info_path = Path(dataset_path) / "meta" / "info.json"
    raw = json.loads(info_path.read_text())
    feat = raw.get("features", {})
    info = {}
    for key in ("observation.state", "action"):
        if key in feat:
            info[key] = {
                "names": feat[key].get("names", []),
                "shape": feat[key].get("shape", []),
            }
    return info


def build_app(dataset_path, max_episodes=None):
    ds = LeRobotDataset("local", root=str(dataset_path), video_backend="pyav")
    ep_bounds = _build_episode_bounds(ds, max_episodes)
    total_episodes = ds.num_episodes
    raw_info = _load_raw_info(dataset_path)

    template_loader = jinja2.FileSystemLoader(
        str(Path(__file__).parent / "templates")
    )
    env = jinja2.Environment(loader=template_loader)

    app = web.Application()

    # ── Static info payload ──
    info = {
        "robot_type": ds.meta.robot_type,
        "total_episodes": total_episodes,
        "loaded_episodes": len(ep_bounds),
        "total_frames": ds.num_frames,
        "fps": ds.meta.fps,
        "camera_keys": list(ds.meta.camera_keys),
        "state_names": raw_info.get("observation.state", {}).get("names", []),
        "action_names": raw_info.get("action", {}).get("names", []),
        "dataset_name": Path(dataset_path).name,
    }

    # ── Helper: resolve (ep, idx) to global index ──
    def resolve_ep_idx(ep, idx):
        if ep < 0 or ep >= len(ep_bounds):
            return None
        start, end = ep_bounds[ep]
        if idx < 0 or idx >= (end - start):
            return None
        return start + idx

    # ── Routes ──

    async def index(_request):
        html = env.get_template("index.html").render(info=info)
        return web.Response(text=html, content_type="text/html")

    async def api_info(_request):
        return web.json_response(info)

    async def api_episodes(_request):
        episodes = []
        for ep_idx, (start, end) in enumerate(ep_bounds):
            task_idx = ds.hf_dataset[start]["task_index"].item()
            try:
                task = ds.meta.tasks.iloc[task_idx].name
            except Exception:
                task = f"task_{task_idx}"
            episodes.append(
                {
                    "episode": ep_idx,
                    "num_frames": end - start,
                    "task": task if isinstance(task, str) else str(task),
                }
            )
        return web.json_response(episodes)

    async def api_episode(request):
        ep = int(request.match_info["ep"])
        if ep < 0 or ep >= len(ep_bounds):
            return web.json_response({"error": "episode not found"}, status=404)
        start, end = ep_bounds[ep]

        raw_states = ds.hf_dataset[start:end]["observation.state"]
        raw_actions = ds.hf_dataset[start:end]["action"]

        return web.json_response(
            {
                "episode": ep,
                "num_frames": end - start,
                "states": [s.tolist() for s in raw_states],
                "actions": [a.tolist() for a in raw_actions],
                "state_names": info["state_names"],
                "action_names": info["action_names"],
            }
        )

    async def api_frame(request):
        ep = int(request.match_info["ep"])
        idx = int(request.match_info["idx"])
        camera = request.query.get("camera", "")

        global_idx = resolve_ep_idx(ep, idx)
        if global_idx is None:
            return web.json_response({"error": "frame not found"}, status=404)

        frame = ds[global_idx]
        if camera:
            if camera not in frame:
                return web.json_response({"error": f"camera {camera} not found"}, status=404)
            img_tensor = frame[camera]
        else:
            img_tensor = frame[ds.meta.camera_keys[0]]

        jpeg = _tensor_to_jpeg(img_tensor)
        return web.Response(body=jpeg, content_type="image/jpeg")

    app.router.add_get("/", index)
    app.router.add_get("/api/info", api_info)
    app.router.add_get("/api/episodes", api_episodes)
    app.router.add_get("/api/episode/{ep}", api_episode)
    app.router.add_get("/api/frame/{ep}/{idx}", api_frame)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LeRobot episode visualizer")
    parser.add_argument("dataset_path", help="Path to LeRobot dataset")
    parser.add_argument("--port", type=int, default=8866, help="Port to serve on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=100,
        help="Load only first N episodes for faster startup (default: 100)",
    )
    args = parser.parse_args()

    path = Path(args.dataset_path).resolve()
    print(f"Loading dataset: {path} (max {args.max_episodes} episodes)")
    app = build_app(str(path), max_episodes=args.max_episodes)
    print(
        f"Serving at http://localhost:{args.port} "
        f"({args.max_episodes} episodes loaded, full dataset has more)"
    )
    web.run_app(app, host=args.host, port=args.port)
