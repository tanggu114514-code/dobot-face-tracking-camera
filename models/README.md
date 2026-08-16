# Runtime Models

This directory is intentionally empty in source control.

On first use, `src/face_engine.py` downloads:

- `face_detection_yunet_2023mar.onnx`
- `face_recognition_sface_2021dec.onnx`

The URLs point directly to the official OpenCV Zoo repository. Model binaries
are ignored by Git to keep the repository small and preserve clear provenance.
