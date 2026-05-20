# Crop Disease Detection using CNN

A Deep Learning based image classification project for detecting crop diseases using Convolutional Neural Networks (CNN).

---

## Project Overview

This project uses CNN architectures to classify crop leaf diseases from images. 
Multiple CNN architectures were trained on the basic model.
1. MobileNet : it is Lightweight, uses depthwise separable convolutions.
2. ResNet : it Introduces *skip connections* to solve vanishing gradient. (was developed using both keras/.pth)
3. leNet : is a ancient version of all.

The model was trained using TensorFlow/Keras and deployed using Streamlit for real-time predictions.

adding Graphs for all models
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
Graph for ADAM
<img width="556" height="435" alt="image" src="https://github.com/user-attachments/assets/659451a2-b957-4430-a51b-132731b45bab" />
<img width="556" height="435" alt="image" src="https://github.com/user-attachments/assets/c538f7fc-3ae5-4ecf-8346-dfa4f99db8c6" />


### SGD with Momentum
- More stable learning
- Better generalization in some cases
- Slower convergence
Graph for SGD
<img width="556" height="435" alt="image" src="https://github.com/user-attachments/assets/151f4391-4a7b-4bd0-9a39-548f1ac277f7" />
<img width="556" height="435" alt="image" src="https://github.com/user-attachments/assets/5d43faf3-e2fe-47da-8423-6265752b1af6" />

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
