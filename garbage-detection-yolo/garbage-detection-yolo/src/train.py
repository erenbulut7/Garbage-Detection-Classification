"""
train.py

Trains a YOLOv8 model on the garbage classification dataset. Supports all
experiment configurations used in this project (baseline, dropout, SGD,
cosine scheduler) through command-line arguments, so every experiment is
reproducible with a single command.

Usage examples:

    # Baseline (AdamW-auto, no dropout, 60 epochs)
    python train.py --data data/data.yaml --model yolov8n.pt \
        --epochs 60 --batch 32 --seed 42 --patience 15 --name baseline

    # Dropout experiment (custom architecture)
    python train.py --data data/data.yaml --model models/yolov8n_dropout.yaml \
        --pretrained yolov8n.pt --epochs 60 --batch 32 --seed 42 --patience 15 \
        --name dropout02

    # SGD experiment
    python train.py --data data/data.yaml --model yolov8n.pt \
        --epochs 40 --batch 32 --seed 42 --patience 10 \
        --optimizer SGD --lr0 0.01 --momentum 0.937 --name optimizer_sgd

    # Cosine LR scheduler experiment
    python train.py --data data/data.yaml --model yolov8n.pt \
        --epochs 40 --batch 32 --seed 42 --patience 10 \
        --optimizer AdamW --lr0 0.001 --cos_lr --name scheduler_cosine
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model for garbage classification.")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt",
                         help="Model checkpoint or architecture YAML (e.g. yolov8n.pt or models/yolov8n_dropout.yaml)")
    parser.add_argument("--pretrained", default=None,
                         help="If --model is an architecture YAML, path/name of pretrained weights to load into it")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--optimizer", default="auto", help="auto, SGD, AdamW, etc.")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate (omit to use optimizer default)")
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--cos_lr", action="store_true", help="Enable cosine LR scheduler")
    parser.add_argument("--project", default="runs/train", help="Directory to save results")
    parser.add_argument("--name", required=True, help="Experiment name (subfolder under --project)")
    args = parser.parse_args()

    if args.pretrained:
        model = YOLO(args.model).load(args.pretrained)
    else:
        model = YOLO(args.model)

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        patience=args.patience,
        optimizer=args.optimizer,
        cos_lr=args.cos_lr,
        project=args.project,
        name=args.name,
    )
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.momentum is not None:
        train_kwargs["momentum"] = args.momentum

    results = model.train(**train_kwargs)
    print(f"\nTraining complete. Results saved to: {args.project}/{args.name}")


if __name__ == "__main__":
    main()
