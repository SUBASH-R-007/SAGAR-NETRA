# SAGAR-NETRA edge benchmark

- Date: 2026-08-26T13:56:53+00:00
- Machine: LOKESH-ROG — Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
- OS: Windows-10-10.0.26200-SP0
- Python: 3.11.9, weights: `detector.pt`, imgsz 512

| Stage | Metric | Value |
|---|---|---|
| L1 preprocess | pings/s | 2233.4 |
| L1 preprocess | full 800-ping survey | 0.36 s -> 12 tiles |
| Detector (PyTorch CPU) | ms/tile | 56.6 |
| Detector (PyTorch CPU) | tiles/s | 17.67 |
| Detector (ONNX RT CPU) | ms/tile | 23.2 |
| Detector (ONNX RT CPU) | tiles/s | 43.16 |
| Anomaly AE (CPU) | ms/tile | 47.7 |
| Anomaly AE (CPU) | tiles/s | 20.98 |

TensorRT INT8 numbers require a Jetson-class device; see `edge/trt_int8.md`.
