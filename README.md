# Episode Web Visualizer

LeRobot 数据集 Episode 可视化工具。加载一个数据集，在浏览器里逐帧浏览每段 episode 的相机画面和 state/action 信号。

## 启动

```bash
conda activate embodied-data-evaluator

# 指定数据集路径
python validation-archive/web-viz/server.py data/svla_so101_pickplace_up

# 指定端口（默认 8866）
python validation-archive/web-viz/server.py data/svla_so101_pickplace_up --port 8000
```

打开 `http://localhost:8866`。

## 依赖

项目已有（无需额外安装）：
- `aiohttp` — HTTP 服务
- `jinja2` — 模板渲染
- `Pillow` — 图片编码
- `lerobot` — 数据集加载与视频解码

## 交互

| 操作 | 方式 |
|------|------|
| 选 Episode | 左栏点击 |
| 逐帧 | 拖动 Slider / ← → 键 |
| 播放/暂停 | ▶ 按钮 / Space 键 |
| 调速 | FPS 下拉框 |
| 选 State 维度 | 页面底部勾选框 |
| 跳转帧 | 点击曲线图上的数据点 |

## 支持的 API

| 路由 | 说明 |
|------|------|
| `GET /` | 主页面 |
| `GET /api/info` | 数据集元信息 |
| `GET /api/episodes` | Episode 列表 |
| `GET /api/episode/{ep}` | 指定 episode 的 state/action 数据 |
| `GET /api/frame/{ep}/{idx}?camera=xxx` | 单帧图片（JPEG） |
