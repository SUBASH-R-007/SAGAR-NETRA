# SAGAR-NETRA edge benchmark

- Date: 2026-08-27T12:58:03+00:00
- Machine: LOKESH-ROG — Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
- OS: Windows-10-10.0.26200-SP0
- Python: 3.11.9, weights: `detector.pt`, imgsz 512

| Stage | Metric | Value |
|---|---|---|
| L1 preprocess | pings/s | 2717.3 |
| L1 preprocess | full 800-ping survey | 0.29 s -> 12 tiles |
| Detector (PyTorch CPU) | ms/tile | 55.7 |
| Detector (PyTorch CPU) | tiles/s | 17.96 |
| Detector (ONNX RT CPU) | ms/tile | 26.2 |
| Detector (ONNX RT CPU) | tiles/s | 38.14 |
| Detector (ONNX INT8 CPU) | ms/tile | 223.6 |
| Detector (ONNX INT8 CPU) | tiles/s | 4.47 |
| Anomaly AE (CPU) | ms/tile | 40.7 |
| Anomaly AE (CPU) | tiles/s | 24.57 |

TensorRT INT8 numbers require a Jetson-class device; see `edge/trt_int8.md`.
