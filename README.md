# Handwritten Digit Recognition with Convolutional Neural Networks

This project demonstrates how to build and train a Convolutional Neural Network (CNN) to recognize handwritten digits using the MNIST dataset. The MNIST dataset contains 60,000 training images and 10,000 testing images of handwritten digits from 0 to 9.

## Libraries and Techniques Used

- **TensorFlow**: For building and training the neural network.
- **Keras**: High-level API of TensorFlow for building neural network models.
- **Matplotlib**: For visualizing the data.
- **NumPy**: For numerical operations and data manipulation.

## Code Explanation

1. **Importing Libraries**:
   The necessary libraries are imported, including `tensorflow`, `matplotlib`, and `numpy`.

2. **Loading the Data**:
   The MNIST dataset is loaded using `tensorflow.keras.datasets.mnist.load_data()`. This dataset is divided into training and testing sets.

3. **Visualizing the Data**:
   A sample image from the training set is displayed using `matplotlib.pyplot.imshow()`.

4. **Normalizing the Data**:
   The pixel values of the images are normalized to a range of 0 to 1 to improve the performance of the neural network.

5. **Reshaping the Data**:
   The images are reshaped to include a single channel, converting them from shape `(28, 28)` to `(28, 28, 1)`.

6. **Building the Model**:
   A Sequential model is built using `tensorflow.keras.Sequential()`. The model consists of:
   - Three convolutional layers (`Conv2D`) with ReLU activation and max pooling (`MaxPooling2D`).
   - A flattening layer (`Flatten`) to convert the 2D outputs of the convolutional layers to 1D.
   - Two dense layers (`Dense`) with ReLU activation.
   - An output layer with 10 neurons (one for each digit) and softmax activation.

7. **Compiling the Model**:
   The model is compiled using `sparse_categorical_crossentropy` loss, `adam` optimizer, and accuracy as the evaluation metric.

8. **Training the Model**:
   The model is trained on the training data for 5 epochs with a validation split of 30%.

9. **Evaluating the Model**:
   The model's performance is evaluated on the testing data, and the loss and accuracy are printed.

10. **Making Predictions**:
   Predictions are made on the testing data, and the predicted digit for a sample image is printed and visualized.
