# Multi-Label Image Classification

This repository contains the code and results for the second deep learning project, developed as part of a classification competition. The goal was to train a multi-label image classifier capable of assigning multiple tags to individual images. Over the course of seven days, several architectures and training strategies were explored to improve macro F1 score performance.

## 🔍 Problem Statement

Multi-label classification differs from standard classification tasks in that each image can belong to multiple categories simultaneously. This project focuses on optimizing the macro F1 score—a metric that equally weights performance across all classes, which is particularly important for imbalanced datasets.

## 📁 Project Structure

```
├── train_example_dayX.py   # Training scripts for each experimental iteration
├── results_Themisto_dayX.xlsx # Output result files with predictions and F1 scores
├── model_Themisto_dayX.pt  # Saved PyTorch model weights
├── test_Themisto.py        # Unified test script for evaluating trained models
├── report_Themisto.docx    # Final narrative report aligned with rubric
├── README.md               # This file
```

## 🏗️ Model Development Timeline

| Day | Architecture    | Image Size | Augmentation          | Class Weighting | Macro F1 |
|-----|-----------------|------------|------------------------|------------------|----------|
| 1   | Custom CNN      | 100x100    | Horizontal Flip        | No               | 0.32     |
| 2   | ResNet18        | 100x100    | Horizontal Flip        | No               | 0.53     |
| 3   | ResNet50        | 100x100    | Rotation, Jitter       | No               | 0.35     |
| 4   | ResNet18        | 100x100    | Rotation, Jitter       | Yes              | 0.58     |
| 7   | EfficientNet-B4 | 380x380    | Full Augmentation      | Yes              | 0.6295   |

## ⚙️ Training Details

- Optimizer: Adam
- Loss: `BCEWithLogitsLoss` with optional `pos_weight`
- Learning Rate: 0.001
- Scheduler: ReduceLROnPlateau
- Early Stopping: Day 7 onward
- Evaluation Metric: Macro F1 Score

## 🧪 How to Run

1. Place your dataset in a `Data/` folder and metadata in `excel/`.
2. Choose a script (e.g., `train_example_day4.py`) and run:
   ```bash
   python3 train_example_day4.py
   ```
3. Evaluate your trained model using:
   ```bash
   python3 test_Themisto.py --path . --split test
   ```

## 📈 Results

Final macro F1 score (Day 7, EfficientNet-B4): **0.6295**  
Class-wise F1 scores are available in the result `.xlsx` files.

## 📄 Report

The full development process, experiments, and findings are detailed in `report_Themisto.docx`.

## 🧑‍💻 Author

Haya Monawwar  
April 2025
