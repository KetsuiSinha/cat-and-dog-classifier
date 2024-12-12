# Image Classification: Cats vs. Dogs

This project implements an image classification model using TensorFlow 2.0 and Keras to distinguish between images of cats and dogs. The challenge requires achieving an accuracy of at least 63%, with extra credit for reaching 70% accuracy.

## Dataset Structure
The dataset is organized into three directories:
- **train**: Contains subdirectories `cats` and `dogs` with labeled images.
- **validation**: Contains subdirectories `cats` and `dogs` for model validation.
- **test**: Contains unlabeled images.

```
cats_and_dogs
|__ train:
    |______ cats: [cat.0.jpg, cat.1.jpg ...]
    |______ dogs: [dog.0.jpg, dog.1.jpg ...]
|__ validation:
    |______ cats: [cat.2000.jpg, cat.2001.jpg ...]
    |______ dogs: [dog.2000.jpg, dog.2001.jpg ...]
|__ test: [1.jpg, 2.jpg ...]
```

## Objective
- Create a convolutional neural network (CNN) to classify images.
- Achieve a minimum accuracy of 63% on the test set.

## Implementation Steps

### 1. Data Preparation
- **Image Generators**: Use `ImageDataGenerator` to preprocess the images by rescaling pixel values to the range [0, 1].
- **Data Loading**: Use `flow_from_directory` to load images from the `train`, `validation`, and `test` directories.
- Ensure `shuffle=False` for `test_data_gen` to maintain the test image order.

### 2. Data Augmentation
- Use random transformations in `ImageDataGenerator` to increase the diversity of the training set and reduce overfitting. Include transformations such as:
  - Rotation
  - Horizontal flip
  - Zoom
  - Brightness adjustment
  - Shearing

### 3. Model Architecture
- Build a CNN using the Keras `Sequential` API with:
  - Multiple `Conv2D` and `MaxPooling2D` layers.
  - Flatten layer followed by fully connected (`Dense`) layers.
  - ReLU activation for intermediate layers and sigmoid activation for the output layer.

### 4. Compilation
- Compile the model with:
  - Optimizer: Adam or SGD
  - Loss: Binary cross-entropy
  - Metrics: Accuracy

### 5. Training
- Train the model using `fit` with:
  - Training data
  - Validation data
  - Parameters such as `epochs`, `batch_size`, and `steps_per_epoch`.

### 6. Evaluation
- Evaluate the model on the validation set.
- Plot accuracy and loss curves to analyze performance.

### 7. Testing and Visualization
- Predict probabilities for test images.
- Use the `plotImages` function to visualize predictions with confidence levels.

## Example Usage

```python
# Create image generators
train_image_generator = ImageDataGenerator(rescale=1.0/255, rotation_range=40, width_shift_range=0.2,
                                           height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                           horizontal_flip=True)

train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
)

# Model creation
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=validation_steps
)

# Test predictions
probabilities = model.predict(test_data_gen)
plotImages(test_images, probabilities)
```
