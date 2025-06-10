# ⚡ Lightning Object Detection

This project explores the use of object detection models on video footage of lightning storms. It includes hand-labeled datasets, preprocessing and augmentation scripts, training workflows, and a real-time inference pipeline.

## 📁 Project Structure

- **/data**
  - `images/train` and `labels/train`: Training data
  - `images/val` and `labels/val`: Validation data
- **/code**
  - Scripts for processing, augmenting, and up-sampling data
  - Automatically updates annotations accordingly
- **main.py**: Script for training object detection models (currently set up for YOLOv5)
- **real_time_pred.py**: Script for running real-time video inference using the trained model

## 📝 Dataset & Labeling

Frames were hand-labeled using [CVAT](https://www.cvat.ai/). All videos used for training, validation, and testing are included in the repository **except for `video4`**, which exceeds GitHub’s file size limits.

## 🧪 Model Training

- The training script is in `main.py`
- The current setup uses YOLOv5, but you can swap in other object detection models by updating:
  - `main.py`
  - `real_time_pred.py`
- A custom YOLOv5 hyperparameter file is included to control which augmentations are applied during training

**Note:** YOLO models typically handle data augmentation internally via hyperparameters. You likely won’t need additional augmentation scripts unless using a different model.

## 🎥 Real-Time Prediction

- Use `real_time_pred.py` to run real-time predictions on videos
- This script expects a reference to the **most recent training run**
- After training, copy the folder containing the latest weights into the root directory and update `real_time_pred.py` to point to it

## ⚠️ Important Notes

- Video `video4` is **not included** in the repository due to size limitations
- Make sure to update paths in the scripts based on your model and training runs

---

Feel free to open an issue or contribute if you'd like to experiment with different models or datasets.
