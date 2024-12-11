# cat-and-dog-classifier
For this challenge, the task is to complete the code to classify images of dogs and cats using TensorFlow 2.0 and Keras. The goal is to create a convolutional neural network that classifies images of cats and dogs with at least 63% accuracy (with extra credit if it reaches 70% accuracy!).

Some parts of the code are already provided, while other sections require the user to complete the code. Instructions for each section are given in text cells, so the user knows what they need to do.

The structure of the dataset files looks as follows:
cats_and_dogs
|__ train:
    |______ cats: [cat.0.jpg, cat.1.jpg ...]
    |______ dogs: [dog.0.jpg, dog.1.jpg ...]
|__ validation:
    |______ cats: [cat.2000.jpg, cat.2001.jpg ...]
    |______ dogs: [dog.2000.jpg, dog.2001.jpg ...]
|__ test: [1.jpg, 2.jpg ...]
The following instructions correspond to specific code cells, with comments indicating which step each corresponds to (e.g., # 3).

Cell 3
In this cell, the user is required to set the variables correctly (they should no longer equal None).

The task is to create image generators for each of the three image datasets (train, validation, test). The user should use ImageDataGenerator to read, decode the images, and convert them into floating-point tensors. They should use the rescale argument to scale the pixel values from the range 0-255 to 0-1.

For the train, validation, and test data generators, the user should use the flow_from_directory method. Parameters like batch size, directory, target size ((IMG_HEIGHT, IMG_WIDTH)), class mode, and any other required arguments must be passed in. For the test data generator, it’s important to pass shuffle=False to maintain the order of images for final predictions. The user should ensure the test dataset’s directory structure is observed.

After running the code, the output should look something like:

vbnet
Copy code
Found 2000 images belonging to 2 classes.
Found 1000 images belonging to 2 classes.
Found 50 images belonging to 1 class.
Cell 4
The plotImages function, which is given, is used to plot images. It takes an array of images and a list of probabilities (though the probabilities list is optional). If the user has set up the train_data_gen correctly, running this cell will plot five random images from the training dataset.

Cell 5
In this cell, the user should recreate the train_image_generator using ImageDataGenerator.

To address overfitting, the user should augment the training data by adding 4-6 random transformations as arguments to the ImageDataGenerator, along with rescaling as before.

Cell 6
This cell is similar to Cell 4 but now includes augmented training images. The user should check that the train_data_gen variable is created using the new train_image_generator and then plot an image five different times with different variations of the transformations.

Cell 7
The user should now create a neural network model using the Keras Sequential model. The model should output class probabilities, and it will likely involve a combination of Conv2D and MaxPooling2D layers, followed by a fully connected layer with a ReLU activation function.

The model should then be compiled with the appropriate optimizer and loss functions, and the accuracy metric should be included to track the training and validation accuracy during the epochs.

Cell 8
In this cell, the user should use the fit method to train the model. They need to specify arguments such as x, steps_per_epoch, epochs, validation_data, and validation_steps.

Cell 9
After training the model, this cell can be used to visualize the model’s accuracy and loss over time.

Cell 10
Now that the model is trained, the user should use it to predict whether a new image is of a cat or a dog.

For each test image (from test_data_gen), the user needs to calculate the probability that the image is a dog or a cat. These probabilities should be stored in a list. The plotImages function should then be called with the test images and the corresponding probabilities to display them. After running this cell, all 50 test images should be shown, with labels indicating the model’s confidence in whether each image is a cat or a dog.

Cell 11
In this final cell, the user can check if they have passed the challenge or if further adjustments to the model are needed to improve accuracy.

This approach will guide the user step-by-step through the image classification process, with clear directions and expected results at each stage.
