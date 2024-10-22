# Face Mask Detection - Deep Learning (CNN)
This project involves building a Convolutional Neural Network (CNN) to classify images as either "with mask" or "without mask". The goal is to create a model capable of detecting whether individuals in images are wearing face masks, a crucial application in ensuring public safety during health crises.

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Deep Learning](#deep-learning)
- [Results and Impact](#results-and-impact)
- [Future Improvements](#future-improvements)

## Project Overview
The Face Mask Detection project uses a Convolutional Neural Network (CNN) to classify images as either with mask or without mask. The dataset contains 7,553 images, and the images were resized, labeled, and split into training and testing sets. The model, built using TensorFlow and Keras, includes convolutional and dense layers with dropout for regularization. After training, the model achieved high accuracy in predicting mask usage and can be used for monitoring public health compliance.

## Problem Statement
In the context of public health crises, such as the COVID-19 pandemic, enforcing mask mandates has become a crucial measure to limit the spread of infectious diseases. However, manually monitoring mask compliance in crowded public spaces is both time-consuming and prone to human error.

The challenge is to develop an automated, reliable, and scalable system that can detect whether individuals are wearing face masks from images or live video feeds. This system needs to accurately classify images into "with mask" or "without mask" categories, ensuring rapid and effective enforcement of mask policies, especially in high-traffic areas like airports, malls, and transportation hubs.

This project aims to solve the problem by building a machine learning model capable of real-time mask detection, thereby supporting public health efforts and enhancing safety in various settings.

## Dataset
The [dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) used in this project consists of 7,553 images categorized into two classes:
- **With Mask:** 3,725 images
- **Without Mask:** 3,828 images

**Key Features:**
- **Image Resolution:** All images were resized to 128x128 pixels for uniformity.
- **Image Format:** Each image was converted to RGB format.
- **Class Labels:**
  - 1 for images where the person is wearing a mask.
  - 0 for images where the person is not wearing a mask.

The dataset contains a diverse set of images with varying backgrounds, lighting conditions, and angles, making it a robust source for training a model to detect mask usage in real-world scenarios.

## Deep Learning
### Install and Configure Kaggle API
- Install the Kaggle API and set up the configuration to access the dataset.
- Download the dataset using the Kaggle API and extract the contents.

### Import Required Libraries
- Import essential libraries like `os`, `numpy`, `matplotlib`, `cv2`, `PIL`, and TensorFlow/Keras for image processing, visualization, and model building.

### Load Dataset
- Use `os.listdir()` to list the images in the with_mask and without_mask directories.
- Print the filenames and check the number of images in each category.

### Label Creation
- Create labels for the dataset:
  - Assign `1` for images in the with_mask category.
  - Assign `0` for images in the without_mask category.
- Combine both labels into a single list.

### Display Sample Images
- Use `matplotlib` to display a sample image from both the with_mask and without_mask categories.

### Preprocess Images
- Load and preprocess the images:
  - Resize all images to 128x128 pixels.
  - Convert each image to RGB format.
  - Store the images as numpy arrays in a list.

### Combine Data and Labels
- Convert the list of images and labels into numpy arrays (`X` for images, `Y` for labels).
- Print the shape of `X` and `Y` to verify.

### Train-Test Split
- Split the dataset into training and testing sets using `train_test_split()` with an 80-20 ratio.

### Scale the Image Data
- Normalize the pixel values by dividing the training and testing data by 255 to scale them between 0 and 1.

### Build the CNN Model
- Define a **Sequential** model with the following layers:
  - **Convolutional** layers with ReLU activation for feature extraction.
  - **MaxPooling** layers to reduce spatial dimensions.
  - **Flatten** the output to feed into fully connected layers.
  - **Dense** layers with ReLU activation.
  - **Dropout** layers for regularization.
  - An output layer with **sigmoid activation** to classify images into two categories.

### Compile the Model
- Compile the model using Adam optimizer and sparse categorical cross-entropy loss.
- Set accuracy as the evaluation metric.

### Train the Model
- Train the model on the scaled training data for 5 epochs, using a 10% validation split.

### Evaluate the Model
- Test the model on the scaled testing data and print the test accuracy.

### Plot Loss and Accuracy
- Visualize the training and validation loss and accuracy using matplotlib.

### Make Predictions on New Images
- Accept a user input for an image path.
- Preprocess the image (resize and normalize), and use the trained model to predict whether the person is wearing a mask.
- Display the input image and prediction result.

## Results and Impact
**Results:**
- The Convolutional Neural Network (CNN) achieved high accuracy in detecting whether a person is wearing a mask. After training the model for 5 epochs, the test accuracy demonstrated reliable performance in classifying images into the two categories: with mask and without mask.
- Loss and accuracy plots indicated a stable training process with both training and validation metrics improving over time.

**Impact:**
- **Automation** of Mask Detection: This system can be implemented in public areas, surveillance systems, or healthcare environments to automatically detect whether individuals are following mask-wearing protocols.
- **Public Health Support:** By enabling real-time detection of mask usage, this model can help in ensuring compliance with health regulations during pandemics or other health crises, reducing the need for manual monitoring.
- **Scalable Solution:** The model can be integrated into various platforms, such as CCTV systems or mobile devices, to enhance safety in high-traffic areas, improving overall public safety and health measures.

## Future Improvements
- **Model Enhancement:**
  - Implement deeper CNN architectures and utilize transfer learning with pretrained models to boost accuracy, particularly in challenging scenarios.
- **Real-Time and Edge Detection:**
  - Optimize the model for real-time detection and integrate it with live video feeds while reducing computational complexity for deployment on edge devices.
- **Expanded Functionality:**
  - Enhance detection capabilities to identify improper mask usage and incorporate additional data sources, such as temperature readings and emotion analysis, for comprehensive health monitoring.
