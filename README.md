
![dataset-card](https://github.com/user-attachments/assets/f302e055-5e75-47f2-a54c-5f85ffca9dd2)


# Brain Tumor Classification Using CNN

This project implements a brain tumor classification model using Convolutional Neural Networks (CNNs) to classify brain tumor images as either 'no tumor' or 'tumor'. The model is built using TensorFlow and Keras.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)

## Project Overview

The goal of this project is to build a deep learning model that can automatically classify brain tumor images into two categories: "no tumor" and "tumor". The model uses a Convolutional Neural Network (CNN) architecture, which is effective in image classification tasks.

The project covers:
- Data preprocessing (image loading, splitting, and augmentation)
- Model development (CNN architecture)
- Model training and evaluation
- Image prediction for new data

## Dataset

The dataset used in this project consists of brain tumor images categorized into two classes:
- **'no'**: No tumor present
- **'yes'**: Tumor present

The dataset is split into three subsets:
- **Training**: 70% of the images
- **Validation**: 20% of the images
- **Testing**: 10% of the images

Data augmentation techniques such as rotation, zooming, shearing, and shifting are applied to the training set to increase the model's ability to generalize.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/brain-tumor-classification.git
   cd brain-tumor-classification
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Preparation
Ensure that you have the brain tumor dataset ready and placed in the `data/brain_tumor_dataset` directory. The dataset should contain images organized into two folders: `no` and `yes`.

### 2. Model Training

Run the following command to train the model:

```bash
python train_model.py
```

The model will train on the dataset, and the best-performing model will be saved as `best_model.keras`.

### 3. Model Evaluation

After training, the model will be evaluated on the test set. The accuracy and loss will be displayed.

To visualize the training and validation accuracy/loss over epochs, you can plot the graphs using the code provided.

### 4. Image Prediction

To make predictions on new images, place the images in the `data/brain_tumor_dataset/check` directory and run:

```bash
python predict_images.py
```

This will display the predictions (tumor or no tumor) for each image.

## Model Architecture

The CNN model consists of the following layers:
1. **Input Layer**: Input image size of (256, 256, 3) (RGB image of size 256x256)
2. **Convolutional Layers**: 
   - 3 Conv2D layers with increasing number of filters (16, 32, 64) and ReLU activation
   - MaxPooling2D layers after each Conv2D layer to reduce the spatial dimensions
3. **Fully Connected Layers**:
   - Flatten the output of the convolutional layers
   - A dense layer with 256 neurons and ReLU activation
   - Output layer with 1 neuron and Sigmoid activation (binary classification)

## Training

The model is trained using the following parameters:
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Metrics**: Accuracy
- **Epochs**: 40 (with early stopping)

## Evaluation

After training, the model's performance is evaluated on the test set, and the test accuracy is printed. The accuracy achieved was **76.92%**.

## Results

The following metrics were observed during the training:

- **Best Test Accuracy**: 76.92%
- **Loss during Training**: Decreased steadily with a slight fluctuation
- **Validation Accuracy**: Stabilized around 80%

### Training and Validation Accuracy:

![Training and Validation Accuracy](./images/training_accuracy.png)

### Training and Validation Loss:

![Training and Validation Loss](./images/training_loss.png)

