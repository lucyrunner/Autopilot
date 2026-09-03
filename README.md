# 🚗 Autopilot Vision System

A production-style perception and control pipeline for autonomous driving — combining classical computer vision for lane detection with a YOLOv8/ONNX model for traffic sign detection, wired into a rule-based decision engine and served over a FastAPI backend.

---

## ✨ Features

- 🛣️ **Lane Detection** — Traditional CV pipeline (Canny edges → Hough transform → polynomial fit → temporal smoothing). Fast, deterministic, and interpretable — no GPU required.
- 🚦 **Traffic Sign Detection** — YOLOv8 running via ONNX Runtime for real-time, multi-class sign recognition.
- 🧠 **Rule-Based Decision Engine** — Translates perception outputs into control events (`KEEP_LANE`, `SLOW_DOWN`, `FULL_STOP`, `LANE_DEPARTURE_WARNING`, `TURN_LEFT`/`TURN_RIGHT`) with clear, auditable priority logic.
- ⚡ **Two Inference Modes**
  - **Simple pipeline** — direct, synchronous inference for development and edge devices.
  - **Async worker pipeline** — Redis-queued inference for horizontal scaling across multiple workers/machines.
- 🌐 **FastAPI Server** — Sync (`/infer`) and async (`/async/submit`, `/async/result`, `/async/status`) endpoints, health checks, CORS, and structured logging.
- 📊 **Observability** — Per-stage timing breakdowns, metrics collection, and Prometheus/Sentry hooks for production monitoring.
- 🐳 **Dockerized** — Ready-to-run `docker-compose` stack (API + Redis).
- 🧪 **Tested & Linted** — Pytest suite, `ruff` linting, `mypy` type checking, and coverage reporting via `make check`.

---

## 🏗️ Architecture

```
Camera Frame
     │
     ▼
┌─────────────────────┐
│   Preprocessing      │
└──────────┬───────────┘
           ▼
   ┌───────────────┐        ┌────────────────────┐
   │ Lane Detector │        │  Sign Detector       │
   │ (OpenCV / CV) │        │  (YOLOv8 / ONNX)     │
   └───────┬───────┘        └──────────┬──────────┘
           └───────────┬───────────────┘
                        ▼
              ┌───────────────────┐
              │  Decision Engine   │  🧠 Rule-based
              └─────────┬──────────┘
                        ▼
                 Control Event
      (KEEP_LANE / SLOW_DOWN / FULL_STOP / …)
```

---

## 📁 Project Structure

```
.
├── config/              ⚙️  Centralized Pydantic settings (models, API, Redis, logging)
├── src/
│   ├── perception/       👁️  Lane detection & sign detection
│   ├── control/          🧠  Rule-based decision engine
│   ├── inference/        🔄  Simple pipeline + async Redis worker
│   ├── api/               🌐  FastAPI app, routes, schemas
│   ├── ui/                 🖥️  Visualizer
│   └── utils/               🛠️  Logging, metrics, timing, reproducibility
├── scripts/              📜  Benchmarking & inference runner scripts
├── examples/             🧪  Minimal end-to-end usage example
├── tests/                 ✅  Pytest suite
├── docker/                🐳  Dockerfile + docker-compose stack
├── requirements/          📦  base / dev / prod dependency sets
└── Makefile                🔧  Common dev commands
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
make install          # base dependencies
make install-dev       # + dev tooling (pytest, mypy, black, isort)
```

### 2. Run the API server

```bash
make run
# → http://localhost:8000
```

### 3. Try the minimal example

```bash
python examples/minimal_inference.py
```

### 4. Run with Docker 🐳

```bash
make docker-build
make docker-run
```

---

## 🧪 Testing & Quality

```bash
make test              # 🧪 run tests
make test-coverage      # 📊 run tests with coverage report
make lint                # 🔍 ruff linting
make typecheck            # 🔍 mypy type checking
make check                 # ✅ lint + typecheck + test
```

---

## 📈 Benchmarking

```bash
python scripts/benchmark.py
```

Reports mean/P95/P99 latency, FPS, and memory usage for both lane and sign detection.

---

## ⚙️ Configuration

All settings are managed centrally via `config/settings.py` (Pydantic Settings) and can be overridden with environment variables:

| Prefix | Controls |
|---|---|
| `MODEL_` | Lane/sign detection thresholds, temporal smoothing |
| `API_` | Host, port, workers, CORS, rate limiting |
| `REDIS_` | Queue connection & TTL settings |
| `LOG_` | Log level, format, output |

Example: `MODEL_SIGN_CONFIDENCE_THRESHOLD=0.7`

---

## 🗺️ Roadmap Ideas

- 🎯 Sensor fusion (camera + radar/lidar)
- 🧭 Trajectory planning & state machine for complex maneuvers
- 🔁 Model retraining pipeline
