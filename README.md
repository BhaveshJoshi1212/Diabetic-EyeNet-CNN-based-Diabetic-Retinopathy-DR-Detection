# Diabetic-EyeNet-CNN-and-Transfer-Learning-based-Diabetic-Retinopathy-DR-Detection


## Problem Statement

**Diabetic retinopathy (DR)** is a leading cause of blindness among individuals with diabetes, characterized by progressive damage to the retina due to prolonged high blood sugar levels. Early diagnosis and intervention are crucial to prevent severe vision loss and manage the disease effectively. Traditional diagnostic methods rely heavily on manual examination of retinal images by ophthalmologists, which can be time-consuming and prone to human error.

This project aims to develop an automated system for detecting and classifying diabetic retinopathy using deep learning techniques. By employing convolutional neural networks (CNNs), the goal is to create a model that can accurately categorize retinal images into different stages of diabetic retinopathy.

![image](https://github.com/user-attachments/assets/312fd2f5-8172-40ca-b64b-0b2da85afe83)



## Data

The dataset used for this project consists of retinal images with associated labels indicating the severity of Diabetic Retinopathy. The images have been processed to a consistent size and format for model training.

## Objectives:
* **Data Preprocessing**: Implement a pipeline to handle and prepare retinal images for model training, including image resizing and normalization.
* **Model Development**: Design and train a convolutional neural network (CNN) to classify retinal images based on the severity of diabetic retinopathy.
* **Evaluation**: Assess the performance of the trained model on a separate test dataset to ensure its accuracy and reliability in predicting diabetic retinopathy.
  
**Scope:**
The project includes data handling and preprocessing, model architecture design, training, and evaluation.
The model will be tested on a dataset consisting of retinal images with various levels of diabetic retinopathy severity.
Results will be analyzed to determine the effectiveness of the model in assisting healthcare professionals with DR diagnosis.
By automating the DR detection process, this project aims to support early intervention and improve diagnostic accuracy, ultimately contributing to better patient outcomes and reduced healthcare costs.


## Major Techniques Used

### Data Preprocessing
1. **Image Resizing and Normalization**:
   - **Resizing**: The images are resized to a consistent dimension of 224x224 pixels to ensure uniform input size for the model.
   - **Normalization**: Pixel values are scaled to the range [0, 1] by dividing by 255.0. This helps in improving model convergence during training.

2. **Data transformation**:
   - **ImageDataGenerator**: Utilized for data augmentation and normalization. It helps in creating additional variations of the training images by applying random transformations such as rotations, shifts, and flips. This improves the model’s robustness and generalization.

### Model Development
1. **Convolutional Neural Network (CNN)**:
   - **Architecture**: The model consists of several convolutional layers followed by max-pooling layers. These layers extract hierarchical features from the images.
   - **Activation Functions**: ReLU (Rectified Linear Unit) is used for introducing non-linearity and enhancing the model's ability to capture complex patterns.
   - **Batch Normalization**: Applied after convolutional layers to stabilize and accelerate the training process by normalizing the activations.
   - **Dense Layers**: Fully connected layers at the end of the network, culminating in a softmax activation function to output class probabilities.
2. **VGG16 AND ResNet50**
3. **Training and Evaluation**:
   - **CategoricalCrossentropy**: Used as the loss function for Multiclass classification (no dr, moderate dr, severe dr).
   - **Adam Optimizer**: An adaptive learning rate optimizer used to minimize the loss function during training.

### OpenCV for Prediction
1. **Image Processing**:
   - **Loading and Conversion**: The OpenCV library (`cv2`) is used to load and convert images from BGR (OpenCV default) to RGB (for model compatibility).
   - **Resizing**: Images are resized to match the input size expected by the trained model.
   - **Prediction**: The processed image is fed into the trained CNN model to predict the presence and severity of diabetic retinopathy.

## How the Project Works

1. **Data Preparation**:
   - **Loading Data**: The dataset containing retinal images and their corresponding labels is loaded from CSV files.
   - **Data Splitting**: The dataset is split into training, validation, and test sets using stratified sampling to ensure balanced class distribution.

2. **Image Preprocessing**:
   - Images are resized and normalized. Data augmentation techniques are applied to enhance the training dataset and prevent overfitting.

3. **Model Training**:
   - A convolutional neural network (CNN) is designed and trained on the preprocessed training images. The model is optimized using the Adam optimizer and binary crossentropy loss function.

4. **Model Evaluation**:
   - The trained model is evaluated on the validation and test sets to measure its performance in terms of accuracy and loss.

5. **Prediction**:
   - For new retinal images, OpenCV is used to preprocess the image (loading, resizing, and converting to RGB). The preprocessed image is then fed into the trained model to predict the diabetic retinopathy classification.

6. **Results**:
   - The model’s predictions are outputted, indicating whether the image shows signs of diabetic retinopathy and, if so, the severity of the condition.

## Tags

```tags
Machine Learning, Deep Learning, Computer Vision, Diabetic Retinopathy, CNN, TensorFlow, OpenCV, Image Classification

