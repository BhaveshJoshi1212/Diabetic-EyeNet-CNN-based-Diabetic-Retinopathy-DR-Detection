# Diabetic-EyeNet-CNN-based-Diabetic-Retinopathy-DR-Detection
CNN based DR detection

# Diabetic Retinopathy Detection

## Problem Statement

Diabetic Retinopathy (DR) is a serious eye condition that can lead to blindness if not detected and treated early. It is a common complication of diabetes, where damage to the blood vessels in the retina occurs. Early detection and diagnosis are crucial for preventing vision loss.

The challenge in this project is to develop a machine learning model to accurately classify images of the retina into different stages of Diabetic Retinopathy. The goal is to automate the diagnostic process, assisting healthcare professionals in the early detection of DR by leveraging deep learning techniques on retinal images.

## Objectives

1. **Data Collection**: Utilize a dataset of retinal images with labeled categories corresponding to the severity of Diabetic Retinopathy.
2. **Data Preprocessing**: Perform necessary preprocessing steps to prepare the images for model training, including resizing, normalization, and splitting into training, validation, and test sets.
3. **Model Development**: Build a Convolutional Neural Network (CNN) model to classify the retinal images into different stages of DR.
4. **Model Evaluation**: Assess the performance of the model using appropriate metrics and validate its effectiveness on unseen test data.
5. **Prediction**: Implement a function to predict the DR stage of new retinal images.

## Data

The dataset used for this project consists of retinal images with associated labels indicating the severity of Diabetic Retinopathy. The images have been processed to a consistent size and format for model training.

## Techniques Used

- **Data Preprocessing**: Image resizing, normalization, data augmentation
- **Machine Learning Model**: Convolutional Neural Network (CNN) using TensorFlow and Keras.
- **Evaluation Metrics**: Accuracy, loss.

## Results

The CNN model was trained and evaluated on a dataset of retinal images, achieving an accuracy of approximately 93% on the test set. The model can classify images into two categories: 'No DR' and 'DR'.

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/diabetic-retinopathy-detection.git

