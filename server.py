#!/usr/bin/env python3
"""LeRobot episode web visualizer — multi-dataset support.

Usage:
    # Single dataset (backward compat)
    python server.py <dataset_path> [--port PORT]

    # Directory containing multiple datasets (up to 2 levels deep)
    python server.py <directory_path> [--port PORT]
"""
import argparse
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import jinja2
import numpy as np
from aiohttp import web
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

@dataclass
class DatasetItem:
    group: str       # top-level group name (e.g. "乐聚_k4pro")
    name: str        # display name (e.g. "乐聚_k4pro/00345526")
    path: str        # absolute filesystem path
    idx: int = 0     # index in the list


def discover_datasets(base_dir: str) -> list[DatasetItem]:
    """Scan up to 2 levels deep for directories containing ``meta/``."""
    base = Path(base_dir)
    items: list[DatasetItem] = []

    for level1 in sorted(base.iterdir()):
        if not level1.is_dir():
            continue
        # Level-1 dataset:  dir / meta /
        if (level1 / "meta").is_dir():
            items.append(DatasetItem(
                group=level1.name,
                name=level1.name,
                path=str(level1.resolve()),
            ))
            continue
        # Level-2 datasets:  dir / uuid / meta /
        for level2 in sorted(level1.iterdir()):
            if level2.is_dir() and (level2 / "meta").is_dir():
                short = level2.name[:8]
                items.append(DatasetItem(
                    group=level1.name,
                    name=f"{level1.name}/{short}",
                    path=str(level2.resolve()),
                ))

    for i, item in enumerate(items):
        item.idx = i
    return items


# ---------------------------------------------------------------------------
# Dataset manager — lazy-load / cache
# ---------------------------------------------------------------------------

class DatasetManager:
    """Lazy-load LeRobot datasets and cache them for the session."""

    def __init__(self, items: list[DatasetItem], max_episodes: int | None = None):
        self.items = items
        self.max_episodes = max_episodes
        self._cache: dict[int, dict] = {}

    def _load(self, idx: int) -> dict:
        item = self.items[idx]
        ds = LeRobotDataset("local", root=item.path, video_backend="pyav")
        ep_bounds = _build_episode_bounds(ds, self.max_episodes)
        raw_info = _load_raw_info(item.path)
        info = {
            "robot_type": ds.meta.robot_type,
            "total_episodes": ds.num_episodes,
            "loaded_episodes": len(ep_bounds),
            "total_frames": ds.num_frames,
            "fps": ds.meta.fps,
            "camera_keys": list(ds.meta.camera_keys),
            "state_names": raw_info.get("observation.state", {}).get("names", []),
            "action_names": raw_info.get("action", {}).get("names", []),
            "dataset_name": item.name,
        }
        self._cache[idx] = {"ds": ds, "ep_bounds": ep_bounds, "info": info}
        return self._cache[idx]

    def get_info(self, idx: int) -> dict:
        return self._load(idx)["info"]

    def get_ds(self, idx: int) -> LeRobotDataset:
        return self._load(idx)["ds"]

    def get_ep_bounds(self, idx: int) -> list:
        return self._load(idx)["ep_bounds"]

    def resolve_ep_idx(self, ds_idx: int, ep: int, frame: int) -> int | None:
        bounds = self.get_ep_bounds(ds_idx)
        if ep < 0 or ep >= len(bounds):
            return None
        start, end = bounds[ep]
        if frame < 0 or frame >= (end - start):
            return None
        return start + frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_episode_bounds(ds, max_episodes=None):
    """Compute [start, end) global index for each episode."""
    ep_indices = ds.hf_dataset["episode_index"]
    bounds = {}
    for i, ep_idx in enumerate(ep_indices):
        e = ep_idx.item()
        if e not in bounds:
            if max_episodes is not None and e >= max_episodes:
                break
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


_ROBOT_MAPPING_OVERRIDE = {"a2_zy": "a2_zy_sl"}


def _resolve_dim_names(raw, key):
    """When info.json has placeholder names=["dim"], look up the robot-specific mapping."""
    names = raw.get("features", {}).get(key, {}).get("names", [])
    if names != ["dim"]:
        return names
    robot_type = raw.get("robot_type", "")
    if not robot_type:
        return names
    mapping_key = _ROBOT_MAPPING_OVERRIDE.get(robot_type, robot_type)
    mapping_file = (Path(__file__).resolve().parent.parent.parent
                    / "data" / "Dataset--May-2026" / "readme" / "mapping"
                    / f"{mapping_key}_mapping.json")
    if not mapping_file.exists():
        return names
    try:
        mapping = json.loads(mapping_file.read_text())
        entries = mapping.get("features", {}).get(key, [])
        return [e["name"] for e in sorted(entries, key=lambda e: e.get("lerobot_index", 0))]
    except Exception:
        return names


def _load_raw_info(dataset_path):
    """Load names from meta/info.json, fall back to robot mapping for placeholder 'dim'."""
    info_path = Path(dataset_path) / "meta" / "info.json"
    raw = json.loads(info_path.read_text())
    feat = raw.get("features", {})
    info = {}
    for key in ("observation.state", "action"):
        if key in feat:
            info[key] = {
                "names": _resolve_dim_names(raw, key),
                "shape": feat[key].get("shape", []),
            }
    return info


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def build_app(manager: DatasetManager) -> web.Application:
    template_loader = jinja2.FileSystemLoader(
        str(Path(__file__).parent / "templates")
    )
    env = jinja2.Environment(loader=template_loader)

    app = web.Application()

    # ── Index ──

    async def index(_request):
        datasets_info = [
            {"name": d.name, "group": d.group, "idx": d.idx}
            for d in manager.items
        ]
        html = env.get_template("index.html").render(datasets=datasets_info)
        return web.Response(text=html, content_type="text/html")

    # ── API: list datasets ──

    async def api_datasets(_request):
        return web.json_response([
            {"name": d.name, "group": d.group, "idx": d.idx}
            for d in manager.items
        ])

    # ── API: dataset info ──

    async def api_info(request):
        ds_idx = int(request.query.get("ds", "0"))
        try:
            info = manager.get_info(ds_idx)
            return web.json_response(info)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # ── API: episodes list ──

    async def api_episodes(request):
        ds_idx = int(request.query.get("ds", "0"))
        try:
            ds = manager.get_ds(ds_idx)
            bounds = manager.get_ep_bounds(ds_idx)
            episodes = []
            for ep_idx, (start, end) in enumerate(bounds):
                task_idx = ds.hf_dataset[start]["task_index"].item()
                try:
                    task = ds.meta.tasks.iloc[task_idx].name
                except Exception:
                    task = f"task_{task_idx}"
                episodes.append({
                    "episode": ep_idx,
                    "num_frames": end - start,
                    "task": task if isinstance(task, str) else str(task),
                })
            return web.json_response(episodes)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # ── API: single episode data ──

    async def api_episode(request):
        ds_idx = int(request.query.get("ds", "0"))
        ep = int(request.match_info["ep"])
        try:
            ds = manager.get_ds(ds_idx)
            bounds = manager.get_ep_bounds(ds_idx)
            if ep < 0 or ep >= len(bounds):
                return web.json_response({"error": "episode not found"}, status=404)
            start, end = bounds[ep]

            raw_states = ds.hf_dataset[start:end]["observation.state"]
            raw_actions = ds.hf_dataset[start:end]["action"]
            info = manager.get_info(ds_idx)

            return web.json_response({
                "episode": ep,
                "num_frames": end - start,
                "states": [s.tolist() for s in raw_states],
                "actions": [a.tolist() for a in raw_actions],
                "state_names": info["state_names"],
                "action_names": info["action_names"],
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # ── API: frame image ──

    async def api_frame(request):
        ds_idx = int(request.query.get("ds", "0"))
        ep = int(request.match_info["ep"])
        idx = int(request.match_info["idx"])
        camera = request.query.get("camera", "")

        try:
            ds = manager.get_ds(ds_idx)
            global_idx = manager.resolve_ep_idx(ds_idx, ep, idx)
            if global_idx is None:
                return web.json_response({"error": "frame not found"}, status=404)

            frame = ds[global_idx]
            info = manager.get_info(ds_idx)
            if camera:
                if camera not in frame:
                    return web.json_response({"error": f"camera {camera} not found"}, status=404)
                img_tensor = frame[camera]
            else:
                img_tensor = frame[info["camera_keys"][0]]

            jpeg = _tensor_to_jpeg(img_tensor)
            return web.Response(body=jpeg, content_type="image/jpeg")
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    app.router.add_get("/", index)
    app.router.add_get("/api/datasets", api_datasets)
    app.router.add_get("/api/info", api_info)
    app.router.add_get("/api/episodes", api_episodes)
    app.router.add_get("/api/episode/{ep}", api_episode)
    app.router.add_get("/api/frame/{ep}/{idx}", api_frame)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _resolve_datasets(path: Path) -> list[DatasetItem]:
    """If path is a single dataset return it as one-item list, otherwise discover."""
    if (path / "meta").is_dir():
        return [DatasetItem(group=path.name, name=path.name, path=str(path.resolve()))]
    return discover_datasets(str(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LeRobot episode visualizer")
    parser.add_argument("dataset_path", help="Path to LeRobot dataset or directory containing multiple datasets")
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
    datasets = _resolve_datasets(path)
    for d in datasets:
        d.idx = list(datasets).index(d)

    print(f"Found {len(datasets)} dataset(s) in {path}")
    for d in datasets:
        print(f"  [{d.idx}] {d.name}")

    manager = DatasetManager(datasets, max_episodes=args.max_episodes)
    app = build_app(manager)

    if len(datasets) == 1:
        info = manager.get_info(0)
        print(f"Serving at http://localhost:{args.port} "
              f"({info['loaded_episodes']} episodes loaded, full dataset has more)")
    else:
        print(f"Serving at http://localhost:{args.port} (datasets: {len(datasets)})")

    web.run_app(app, host=args.host, port=args.port)
