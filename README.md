# Cat and Dog Image Classification Challenge

This project implements an image classification algorithm to distinguish between images of cats and dogs using TensorFlow 2.0 and Keras. The objective is to achieve at least **63% accuracy** on the test set, with extra credit for achieving **70% accuracy or more**.

---

## Project Overview

The challenge is to create a Convolutional Neural Network (CNN) that classifies images into either "cat" or "dog" categories. The dataset is structured into training, validation, and test sets. Users are provided with partial code and guided through completing it.

### Features

- Uses **ImageDataGenerator** to preprocess and augment image data.
- Implements a sequential CNN architecture with layers like `Conv2D` and `MaxPooling2D`.
- Trains and evaluates the model using training and validation datasets.
- Visualizes model performance (accuracy and loss) over time.
- Outputs predictions for the test dataset, with confidence scores for each image.

---

## Dataset Structure

The dataset is organized into three main directories:

```plaintext
cats_and_dogs
|__ train:
    |______ cats: [cat.0.jpg, cat.1.jpg ...]
    |______ dogs: [dog.0.jpg, dog.1.jpg ...]
|__ validation:
    |______ cats: [cat.2000.jpg, cat.2001.jpg ...]
    |______ dogs: [dog.2000.jpg, dog.2001.jpg ...]
|__ test: [1.jpg, 2.jpg ...]
