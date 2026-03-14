# AI-Based Vehicle Detection using YOLOv8

## Overview
This project implements vehicle detection (car, bus, truck) using Ultralytics YOLOv8 on a custom Roboflow dataset. Optimized for CPU training.

**Classes:**
- 0: car
- 1: bus  
- 2: truck

Dataset: ~200+ train images in `dataset/train/`, with `valid/` and `test/` splits.

Previous trainings: `runs/detect/train[1-9]/` (use `train9/weights/best.pt`).

## Setup
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. (Optional) Download pre-trained `yolov8n.pt` if missing.

## Training
Run CPU-optimized training (30 epochs, batch=8):
```
python train.py
```
- Outputs to `runs/detect/train10/`
- Monitor with `tensorboard --logdir runs`
- Plots, metrics, best.pt saved automatically.

## Detection/Inference
Enhanced CLI support for images, folders, videos, webcam.

### Examples:
```
# Single image
python detect.py --source images/test.jpg

# Folder (e.g., valid set)
python detect.py --source dataset/valid/images --save-txt

# Video
python detect.py --source video.mp4

# Webcam
python detect.py --source 0 --show

# Custom model/confidence
python detect.py --model runs/detect/train9/weights/best.pt --source img.jpg --conf 0.25 --iou 0.45
```

**Options:**
- `--model`: Weights path (default: train9/best.pt)
- `--source`: Input (default: sample image)
- `--conf`: Confidence (0.5)
- `--iou`: NMS IoU (0.5)
- `--save-txt`: Save labels
- `--show`: Display results

Results: `runs/detect/predict_latest/` (images, labels, conf/visuals with class names).

## Expected Performance
From prior runs:
- Check `runs/detect/train9/results.png` for mAP@50, F1, etc.
- confusion_matrix.png for class-wise accuracy.

## Troubleshooting (CPU)
- Low VRAM: Reduce `batch`/`imgsz` in train.py.
- Slow training: Normal on CPU (~minutes/epoch).
- No GPU: `device='cpu'` enforced.

## Next Steps
- Train `train10`.
- Test on new videos/images.
- Export to ONNX/TensorRT for deployment.
- Fine-tune hypers with `hyp.yaml`.

Enjoy vehicle detection! 🚗🚌🚚

