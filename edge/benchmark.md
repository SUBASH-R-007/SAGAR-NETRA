# SAGAR-NETRA edge benchmark

- Date: 2026-08-29T22:12:35+00:00
- Machine: LOKESH-ROG — Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
- OS: Windows-10-10.0.26200-SP0
- Python: 3.11.9, weights: `detector.pt`, imgsz 640

| Stage | Metric | Value |
|---|---|---|
| L1 preprocess | pings/s | 2821.6 |
| L1 preprocess | full 800-ping survey | 0.28 s -> 12 tiles |
| Detector (PyTorch CPU) | ms/tile | 64.7 |
| Detector (PyTorch CPU) | tiles/s | 15.45 |
| Detector (ONNX RT CPU) | ms/tile | 29.9 |
| Detector (ONNX RT CPU) | tiles/s | 33.44 |
| Detector (ONNX INT8 CPU) | ms/tile | 292.3 |
| Detector (ONNX INT8 CPU) | tiles/s | 3.42 |
| Anomaly AE (CPU) | ms/tile | 35.3 |
| Anomaly AE (CPU) | tiles/s | 28.35 |

TensorRT INT8 numbers require a Jetson-class device; see `edge/trt_int8.md`.
