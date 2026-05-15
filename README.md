# Crop Disease Detection using CNN

A Deep Learning based image classification project for detecting crop diseases using Convolutional Neural Networks (CNN).

---

## Project Overview

This project uses CNN architectures to classify crop leaf diseases from images.  
The model was trained using TensorFlow/Keras and deployed using Streamlit for real-time predictions.

---

## Features

- Image classification using CNN
- Training and evaluation pipeline
- Accuracy and loss visualization
- Confusion matrix generation
- Streamlit web application
- Multiple optimizer comparison:
  - Adam
  - SGD with Momentum

---

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Streamlit

---

## Dataset

Dataset contains categorized crop leaf disease images used for training and testing.

Example classes:

- Healthy
- Leaf Blight
- Rust
- Powdery Mildew

---

## Model Architecture

The CNN model contains:

- Convolution Layers
- MaxPooling Layers
- Dropout
- Dense Layers
- Softmax Output

---

## Training Details

| Parameter | Value |
|---|---|
| Epochs | 20 / 50 |
| Batch Size | 32 |
| Optimizer | Adam / SGD |
| Loss Function | Categorical Crossentropy |

---

## Results

### Classification Report

<img width="812" height="357" alt="image" src="https://github.com/user-attachments/assets/c2f5b926-85ef-423e-ac87-3162cefae8b1" />


### Confusion Matrix

<img width="842" height="872" alt="image" src="https://github.com/user-attachments/assets/98367a18-2511-4569-9480-3619238082e4" />


---

## Optimizer Comparison

### Adam
- Faster convergence
- Better validation accuracy
- Adaptive learning rate

### SGD with Momentum
- More stable learning
- Better generalization in some cases
- Slower convergence

---

## Streamlit Application

The Streamlit app allows users to:

- Upload crop images
- Predict disease
- Display confidence score
- Compares different CNN models

Run locally:

```bash
streamlit run app.py
```

---

## Installation

Clone repository:

```bash
git clone https://github.com/SHEisSONALI/Crop-disease-model
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Future Improvements

- Add Transfer Learning
- Deploy on Cloud
- Mobile App Integration
- Real-time Camera Detection

---

## Author

Sonali Shekhawat
