"""Download and export ONNX models for the AdGaze pipeline.

Usage:
    python -m adgaze.scripts.download_models

Downloads YOLOv8n (person) and YOLOv8n-face models, then exports to ONNX.
"""

from pathlib import Path


def download_yolov8n_person():
    """Download YOLOv8n and export to ONNX for person detection."""
    from ultralytics import YOLO

    out_dir = Path("models/person_detector")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "yolov8n.onnx"

    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return

    print("Downloading YOLOv8n...")
    model = YOLO("yolov8n.pt")
    print("Exporting to ONNX...")
    model.export(format="onnx", imgsz=640, opset=17, simplify=True)

    # Ultralytics exports to same dir as .pt — move it
    exported = Path("yolov8n.onnx")
    if exported.exists():
        exported.rename(out_path)
    print(f"[done] Saved to {out_path}")


def download_yolov8n_face():
    """Download YOLOv8n-face and export to ONNX for face detection."""
    from ultralytics import YOLO

    out_dir = Path("models/face_detector")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "yolov8n-face.onnx"

    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return

    print("Downloading YOLOv8n-face...")
    # yolov8n-face is available from the ultralytics-face community
    # If not available directly, you can use a face-specific model:
    try:
        model = YOLO("yolov8n-face.pt")
    except Exception:
        print("YOLOv8n-face not found in ultralytics hub.")
        print("Alternative: download manually from https://github.com/akanametov/yolov8-face")
        print("Or use the standard YOLOv8n as a fallback (less accurate for faces).")
        print("Falling back to standard YOLOv8n for face detection...")
        model = YOLO("yolov8n.pt")

    print("Exporting to ONNX...")
    model.export(format="onnx", imgsz=640, opset=17, simplify=True)

    exported = Path("yolov8n-face.onnx")
    if not exported.exists():
        exported = Path("yolov8n.onnx")
    if exported.exists():
        exported.rename(out_path)
    print(f"[done] Saved to {out_path}")


def main():
    print("=" * 50)
    print("AdGaze Model Download & Export")
    print("=" * 50)
    download_yolov8n_person()
    print()
    download_yolov8n_face()
    print()
    print("All models ready.")


if __name__ == "__main__":
    main()
