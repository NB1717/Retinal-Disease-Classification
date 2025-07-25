# Retinal Disease Classification using OCT Images

This project classifies retinal diseases from OCT (Optical Coherence Tomography) images using both traditional machine learning and deep learning techniques.

## 📌 Project Overview

The classification process was performed in two main stages:

1. **First milestone**: Traditional ML models (e.g., SVM, Random Forest) on extracted features.
2. **Second stage**: Deep learning with transfer learning using CNNs (ResNet-18, DenseNet-101) on raw OCT images.

## 🛠 Technologies Used

- Python
- PyTorch
- Scikit-learn
- CNN Architectures: ResNet-18, DenseNet-101
- Transfer Learning
- Data Augmentation
- F1-micro Score Evaluation

## 📁 Files

- `first_milestone.py`: Implements traditional ML models using extracted features.
- `new_approach.ipynb`: Final deep learning solution with transfer learning and augmentation.
- `metadata.yml`: Project metadata and configuration.
- `README.md`: Project documentation.

## 📊 Results

- Achieved **F1-micro Score: 83.89%** using DenseNet-101.
- ResNet-18 also showed strong performance on augmented data.

## 🚀 How to Run

```bash
git clone https://github.com/NB1717/Retinal-Disease-Classification.git
cd Retinal-Disease-Classification
