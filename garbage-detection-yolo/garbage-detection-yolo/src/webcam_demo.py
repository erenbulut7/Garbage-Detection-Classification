"""
webcam_demo.py

Runs the trained model on a live webcam feed for real-time detection, and
optionally measures average FPS over a fixed number of frames.

Tested on:
  - Google Colab + NVIDIA A100 (via saved video/frames, no direct webcam access)
  - Local machine with Apple Silicon (MPS backend) and a physical webcam

Usage:
    python webcam_demo.py --weights runs/train/baseline/weights/best.pt --device mps
    python webcam_demo.py --weights runs/train/baseline/weights/best.pt --device cpu --fps_test
"""

import argparse
import time
import cv2
from ultralytics import YOLO


def run_live_demo(model, device, conf):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions.")

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, device=device, verbose=False, conf=conf)
        annotated_frame = results[0].plot()

        cv2.imshow("Garbage Detection - Real Time", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_fps_test(model, device, conf, n_frames=100):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions.")

    frame_count = 0
    start = time.time()

    while frame_count < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        _ = model.predict(frame, device=device, verbose=False, conf=conf)
        frame_count += 1

    end = time.time()
    cap.release()

    fps = frame_count / (end - start)
    print(f"Average FPS over {frame_count} frames on device='{device}': {fps:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Real-time webcam demo for the garbage detection model.")
    parser.add_argument("--weights", required=True, help="Path to trained model weights (best.pt)")
    parser.add_argument("--device", default="cpu", help="Inference device: 'cpu', 'mps' (Apple Silicon), or '0' (CUDA GPU index)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--fps_test", action="store_true", help="Run an FPS benchmark instead of the live demo")
    parser.add_argument("--n_frames", type=int, default=100, help="Number of frames for FPS test")
    args = parser.parse_args()

    model = YOLO(args.weights)

    if args.fps_test:
        run_fps_test(model, args.device, args.conf, args.n_frames)
    else:
        run_live_demo(model, args.device, args.conf)


if __name__ == "__main__":
    main()
