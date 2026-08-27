"""
evaluate.py

Evaluates a trained model on the test set and reports per-class Precision,
Recall, mAP50, mAP50-95, and derived F1-scores.

Usage:
    python evaluate.py --weights runs/train/baseline/weights/best.pt --data data/data.yaml
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLOv8 model on the test set.")
    parser.add_argument("--weights", required=True, help="Path to trained model weights (best.pt)")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split, plots=True)

    class_names = list(model.names.values())
    print(f"\n{'Class':<15} {'Precision':<10} {'Recall':<10} {'mAP50':<10} {'F1-Score':<10}")

    f1_scores = []
    p, r = metrics.box.p, metrics.box.r
    map50 = metrics.box.maps  # per-class mAP50-95, not mAP50; use metrics.box.ap50 for per-class mAP50 if available
    ap50 = metrics.box.ap50 if hasattr(metrics.box, "ap50") else None

    for i, name in enumerate(class_names):
        pi = p[i] if i < len(p) else 0
        ri = r[i] if i < len(r) else 0
        m50 = ap50[i] if ap50 is not None and i < len(ap50) else float("nan")
        f1 = 2 * pi * ri / (pi + ri) if (pi + ri) > 0 else 0
        f1_scores.append(f1)
        print(f"{name:<15} {pi:<10.3f} {ri:<10.3f} {m50:<10.3f} {f1:<10.3f}")

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    print(f"\nMacro F1-Score: {macro_f1:.3f}")
    print(f"Overall mAP50: {metrics.box.map50:.3f}")
    print(f"Overall mAP50-95: {metrics.box.map:.3f}")
    print(f"\nResults (confusion matrix, PR curve, etc.) saved to: {metrics.save_dir}")


if __name__ == "__main__":
    main()
