# Project Upgrade Plan: Real-Time Distributed Intrusion Detection

Current state: offline ML benchmarking (sklearn notebooks, UNSW-NB15 CSV data).
Target state: production-grade, real-time intrusion detection system with C++ hot paths, Kafka-backed streaming, ONNX inference, and Kubernetes deployment.

---

## Target Architecture

```
Live traffic (pcap / UNSW-NB15 replay)
        ↓
[Phase 1] C++ packet parser (libpcap → flow features)
        ↓
[Phase 2] Kafka topic: raw-flows
        ↓
[Phase 2] C++ feature extraction worker
        ↓
[Phase 2] Kafka topic: model-ready-features
        ↓
[Phase 3] ONNX Runtime inference server (C++, XGBoost → ONNX)
        ↓
[Phase 3] Redis (alert dedup + result caching)
        ↓
[Phase 4] gRPC alert stream
        ↓
[Phase 5] Kubernetes (HPA on inference pods) + Prometheus/Grafana
```

---

## Phase 1 — C++ Feature Extraction (Foundation)

**Goal:** Replace offline CSV preprocessing with a live C++ pipeline that parses packets and computes UNSW-NB15-style flow features in real time.

**Why start here:** This is the most authentic C++ use case, most technically differentiating, and everything downstream depends on it.

### Tasks

- [ ] **1.1** Set up a `cpp/` directory with a CMake build system
  - Dependencies: `libpcap`, `librdkafka` (for later), `nlohmann/json`
  - Use `vcpkg` or `conan` for dependency management

- [ ] **1.2** Write a pcap replay tool (`pcap_replay.cpp`)
  - Reads UNSW-NB15 raw pcap files (if available) OR replays the CSV as simulated packets
  - Emits packet structs: `{src_ip, dst_ip, src_port, dst_port, proto, timestamp, payload_len, flags}`

- [ ] **1.3** Write a flow aggregator (`flow_aggregator.cpp`)
  - Groups packets into bidirectional flows (5-tuple key: src/dst IP, src/dst port, proto)
  - Computes the 49 UNSW-NB15 features per flow:
    - Duration, byte counts (`sbytes`, `dbytes`), packet counts (`spkts`, `dpkts`)
    - Service detection (HTTP, FTP, DNS, etc.)
    - TCP flags (SYN, FIN, RST counts)
    - Load, loss, jitter features
  - Exports completed flows as JSON or protobuf

- [ ] **1.4** Validate output against `data/processed/training_full.csv`
  - Run the C++ extractor on the raw UNSW-NB15 CSVs and diff features against the preprocessed output
  - Target: feature values within 1% of the Python preprocessing pipeline

- [ ] **1.5** Benchmark throughput
  - Measure flows/sec on a single core
  - Target: >10k flows/sec (real network sensors run at 100k+; show you know the gap)

---

## Phase 2 — Kafka Streaming Pipeline

**Goal:** Decouple packet ingestion from inference with a Kafka message bus. Enables replay, backpressure, and horizontal scaling.

### Tasks

- [ ] **2.1** Spin up Kafka locally via Docker Compose
  - Services: Zookeeper, Kafka broker, Kafka UI (for visibility)
  - File: `infra/docker-compose.yml`

- [ ] **2.2** Create two Kafka topics
  - `raw-flows`: output from C++ flow aggregator (JSON/protobuf per flow)
  - `scored-flows`: output from inference server (flow + prediction + confidence)

- [ ] **2.3** Add Kafka producer to `flow_aggregator.cpp`
  - Use `librdkafka` C++ client
  - Publish completed flows to `raw-flows` topic
  - Tune: batch size, linger time, compression (snappy) for throughput

- [ ] **2.4** Write a Kafka consumer in C++ (`feature_consumer.cpp`)
  - Consumes from `raw-flows`
  - Applies the same log-transform and encoding as `src/preprocessing/preprocessing.py`
  - Publishes model-ready feature vectors to an inference request queue

- [ ] **2.5** Load test the pipeline
  - Replay the 163K UNSW-NB15 records as fast as possible
  - Measure end-to-end latency from packet ingestion to Kafka publish
  - Target: saturate a single Kafka partition at >50k msg/sec

---

## Phase 3 — ONNX Inference Server

**Goal:** Export the trained XGBoost model to ONNX and serve it from a C++ inference server using ONNX Runtime. Replaces ad-hoc Python `predict()` calls.

### Tasks

- [ ] **3.1** Export XGBoost to ONNX (Python)
  - Use `sklearn-onnx` / `onnxmltools` to convert the trained model from `notebooks/xgboost.ipynb`
  - Validate: run ONNX predictions against original sklearn predictions on the test set
  - Target: identical predictions (ONNX vs. sklearn) on all 55,945 test records
  - Save to `models/xgboost_intrusion.onnx`

- [ ] **3.2** Write a C++ inference server (`inference_server.cpp`)
  - Loads the ONNX model at startup via ONNX Runtime C++ API
  - Consumes feature vectors from the Kafka consumer (Phase 2)
  - Runs batched inference (batch size = 32–128 for throughput)
  - Publishes `{flow_id, label, confidence, latency_us}` to `scored-flows` Kafka topic

- [ ] **3.3** Add Redis alert deduplication
  - For each predicted attack, check Redis for a recent alert on the same src_ip/dst_ip/proto key
  - TTL: 60 seconds (suppress duplicate alerts within a time window)
  - Use `hiredis` C++ client

- [ ] **3.4** Benchmark inference throughput and latency
  - Target: p99 inference latency <1ms per flow at batch size 64
  - Target: >50k scored flows/sec on a single inference server instance

- [ ] **3.5** Expose a gRPC alert stream
  - Define a `.proto` schema: `AlertStream` service, `FlowAlert` message
  - Stream predicted attacks to subscribers in real time
  - Python test client that prints alerts to stdout

---

## Phase 4 — Observability

**Goal:** Instrument the pipeline so you can measure and demonstrate performance — the thing that turns architecture into quantifiable resume bullets.

### Tasks

- [ ] **4.1** Instrument the C++ pipeline with Prometheus metrics
  - Use `prometheus-cpp` library
  - Expose: `flows_ingested_total`, `inference_latency_seconds` (histogram), `alerts_fired_total`, `kafka_lag_messages`

- [ ] **4.2** Add a Grafana dashboard
  - Panels: flows/sec, p50/p99 inference latency, alert rate, Kafka consumer lag
  - File: `infra/grafana/dashboard.json`

- [ ] **4.3** Add structured logging
  - JSON logs with: `flow_id`, `timestamp`, `prediction`, `confidence`, `latency_us`
  - Pipe to a log file or stdout for easy grepping

---

## Phase 5 — Kubernetes Deployment

**Goal:** Deploy the full pipeline on Kubernetes with autoscaling inference pods. This is the infra story that rounds out the resume.

### Tasks

- [ ] **5.1** Containerize all components
  - `Dockerfile.flow-extractor` — C++ flow aggregator binary
  - `Dockerfile.inference-server` — C++ ONNX inference server
  - `Dockerfile.feature-consumer` — C++ Kafka consumer/preprocessor
  - Use multi-stage builds: build in `gcc` image, copy binary to `debian:slim`

- [ ] **5.2** Write Helm charts
  - Chart per component: `flow-extractor`, `feature-consumer`, `inference-server`
  - Values: replica count, Kafka broker URL, Redis URL, ONNX model path

- [ ] **5.3** Configure Horizontal Pod Autoscaler (HPA) on inference pods
  - Scale on Kafka consumer lag (via KEDA — Kubernetes Event-Driven Autoscaler)
  - Min replicas: 1, Max replicas: 10
  - Target: lag < 1000 messages per partition

- [ ] **5.4** Deploy locally with `minikube` or `kind`
  - Full stack: Kafka, Zookeeper, Redis, flow-extractor, feature-consumer, inference-server
  - Load test: replay 163K records and watch HPA scale inference pods

- [ ] **5.5** Set up GitHub Actions CI/CD
  - On push to `main`: build Docker images, run ONNX validation test, push to GHCR
  - On tag: deploy to the k8s cluster

---

## Suggested Build Order

| Week | Focus |
|------|-------|
| 1–2 | Phase 1: C++ flow extractor + CMake setup |
| 3   | Phase 2: Kafka Docker Compose + C++ producer |
| 4   | Phase 3: ONNX export + C++ inference server |
| 5   | Phase 3 cont.: Redis dedup + gRPC stream |
| 6   | Phase 4: Prometheus + Grafana |
| 7–8 | Phase 5: Dockerfiles + Helm + k8s + HPA |

---

## Key Libraries & Tools

| Component | Library |
|-----------|---------|
| Packet parsing | `libpcap` |
| Kafka client (C++) | `librdkafka` |
| ONNX inference | `onnxruntime` C++ API |
| Redis client (C++) | `hiredis` |
| gRPC | `grpc` + `protobuf` |
| Prometheus metrics | `prometheus-cpp` |
| C++ build | `CMake` + `vcpkg` |
| Container orchestration | `Kubernetes` + `Helm` + `KEDA` |
| CI/CD | GitHub Actions |
| Local k8s | `minikube` or `kind` |
