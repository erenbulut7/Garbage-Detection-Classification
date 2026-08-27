# Real-Time Garbage Detection-Classification Using YOLO

CENG 476 – Introduction to Deep Learning, Deep Learning Project (Summer Term 2026)

A YOLOv8-based real-time object detection system for classifying waste into 6 categories: **BIODEGRADABLE, CARDBOARD, GLASS, METAL, PAPER, PLASTIC**.

## Project Overview

This project fine-tunes a YOLOv8n model (transfer learning from COCO pretrained weights) on a garbage classification dataset, with a custom architecture modification (Dropout layers added to the detection head) and a series of ablation experiments comparing optimizers, learning rate schedulers, and regularization settings. See `report/` for the full project report with detailed methodology and results.

**Key result:** Best model (baseline configuration) achieves **test mAP50 = 0.641**, **mAP50-95 = 0.448**, **macro F1 = 0.633**.

**Notable finding:** The original dataset split (as provided) had a severe class-imbalance issue (the GLASS class had zero examples in the original test set). This was identified and corrected by re-splitting the pooled dataset with a class-stratified strategy, which alone improved test mAP50 from 0.433 to 0.641. See Section 5.3 of the report for details.

## Repository Structure

```
garbage-detection-yolo/
├── README.md                      <- this file
├── requirements.txt                <- Python dependencies
├── data/
│   └── data.yaml                   <- dataset config (update 'path' before use)
├── models/
│   └── yolov8n_dropout.yaml        <- custom architecture with Dropout layers
├── src/
│   ├── prepare_data.py             <- pools + re-splits the dataset (stratified 70/15/15)
│   ├── train.py                    <- trains a model (parameterized for all experiments)
│   ├── evaluate.py                 <- evaluates a trained model on the test set (incl. F1)
│   └── webcam_demo.py              <- real-time webcam inference + FPS benchmark
└── notebooks/
    └── main_experiments.ipynb      <- consolidated notebook demonstrating all experiments
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the dataset

The dataset is publicly available on Kaggle:
https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/data?select=standardized_384&classId=0f315e5f-a84b-4641-b878-7e2d09ef70e8&assignmentId=2dc9c6d9-85c5-4ab0-984e-79d9ddd56392&submissionId=97578ce3-d9b4-8c7a-0d52-0af16348f145

Download it in YOLOv8 format and place it anywhere on disk (e.g. `./raw_dataset/`), keeping the default `train/valid/test` folder structure with `images/` and `labels/` subfolders in each.

### 3. Re-split the dataset (important — see "Notable finding" above)

```bash
python src/prepare_data.py \
    --input_dir ./raw_dataset \
    --output_dir ./garbage_dataset_resplit \
    --train_ratio 0.7 --val_ratio 0.15 --seed 42
```

### 4. Update `data/data.yaml`

Set `path:` to the absolute path of the resplit dataset created in step 3.

## Reproducing the Experiments

All experiments use `seed=42` for reproducibility. Each command below corresponds to one experiment reported in the project report.

**Baseline** (AdamW-auto, no dropout, 60 epochs):
```bash
python src/train.py --data data/data.yaml --model yolov8n.pt \
    --epochs 60 --batch 32 --seed 42 --patience 15 --name baseline
```

**Dropout experiment** (custom architecture, p=0.2, 60 epochs):
```bash
python src/train.py --data data/data.yaml --model models/yolov8n_dropout.yaml \
    --pretrained yolov8n.pt --epochs 60 --batch 32 --seed 42 --patience 15 \
    --name dropout02
```

**SGD optimizer experiment** (40 epochs):
```bash
python src/train.py --data data/data.yaml --model yolov8n.pt \
    --epochs 40 --batch 32 --seed 42 --patience 10 \
    --optimizer SGD --lr0 0.01 --momentum 0.937 --name optimizer_sgd
```

**Cosine LR scheduler experiment** (40 epochs):
```bash
python src/train.py --data data/data.yaml --model yolov8n.pt \
    --epochs 40 --batch 32 --seed 42 --patience 10 \
    --optimizer AdamW --lr0 0.001 --cos_lr --name scheduler_cosine
```

## Evaluation

```bash
python src/evaluate.py --weights runs/train/baseline/weights/best.pt --data data/data.yaml
```

This prints per-class Precision, Recall, mAP50, and F1-score, and saves the confusion matrix and PR curve to the results directory.

## Real-Time Webcam Demo

Live detection (press `q` to quit):
```bash
python src/webcam_demo.py --weights runs/train/baseline/weights/best.pt --device cpu
```

On Apple Silicon Macs, use `--device mps` for GPU acceleration:
```bash
python src/webcam_demo.py --weights runs/train/baseline/weights/best.pt --device mps
```

FPS benchmark (no display window):
```bash
python src/webcam_demo.py --weights runs/train/baseline/weights/best.pt --device mps --fps_test --n_frames 100
```

## Notebook

`notebooks/main_experiments.ipynb` contains a consolidated, step-by-step walkthrough of the full pipeline (data preparation, all four training experiments, evaluation, and result visualization) intended for use in Google Colab with a GPU runtime.

## Hardware Used

All training was performed on Google Colab with an NVIDIA A100 GPU (80GB VRAM). The webcam demo was additionally tested locally on a MacBook Air (Apple M-series, MPS backend).

## Results Summary

| Experiment       | Epochs | Precision | Recall | mAP50 | mAP50-95 |
|-------------------|--------|-----------|--------|-------|----------|
| Baseline          | 60     | 0.732     | 0.560  | 0.641 | 0.448    |
| Dropout (p=0.2)   | 60     | 0.722     | 0.540  | 0.616 | 0.424    |
| SGD               | 40     | 0.717     | 0.550  | 0.625 | 0.433    |
| Cosine Scheduler  | 40     | 0.725     | 0.557  | 0.636 | 0.440    |

See the full project report for detailed analysis, per-class results, and discussion.

## License / Attribution

- Model: [Ultralytics YOLOv8](https://docs.ultralytics.com)
- Dataset: [Garbage Classification Dataset](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/data?select=standardized_384&classId=0f315e5f-a84b-4641-b878-7e2d09ef70e8&assignmentId=2dc9c6d9-85c5-4ab0-984e-79d9ddd56392&submissionId=97578ce3-d9b4-8c7a-0d52-0af16348f145) 
