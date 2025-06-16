# MNIST-NN
This project is a complete implementation of a Neural Network built from scratch using only the NumPy library. The goal of this network is to recognize handwritten digits (from 0 to 9) based on the well-known MNIST dataset. All the code for data handling, model training, and evaluation is contained within the NN.ipynb Jupyter Notebook.

Expected Output
When you run the notebook, you can expect the following outputs:
Live Training Progress: During training, the notebook will print the model's accuracy on both the training and development (validation) sets every 20 iterations.
Accuracy Plot: A graph will be generated showing how the training and validation accuracy improve over the training iterations.
Final Accuracy: The final accuracy of the trained model on the unseen development set will be printed (approximately 97.9%).
Prediction Visualization: A visual output of 5 sample images from the development set will be displayed, showing the model's prediction alongside the actual correct label.
Key Concepts Used
The neural network implementation demonstrates several fundamental concepts in deep learning:
Network Architecture: A multi-layer feedforward network (784-256-128-10).
Activation Functions:
ReLU (Rectified Linear Unit) for the hidden layers.
Softmax for the output layer to get class probabilities.
Optimization:
Adam Optimizer for efficient gradient-based learning.
Mini-Batch Gradient Descent for stable and fast training.
Learning Rate Decay to help the model converge better.
Regularization (to prevent overfitting):
L2 Regularization to penalize large weights.
Dropout to randomly ignore neurons during training.
Parameter Initialization: He Initialization to ensure gradients flow effectively through the network.
Core Mechanics: Forward Propagation, Backpropagation, and Gradient Descent.
Workflow Steps
The project follows a clear, step-by-step workflow:
Data Loading and Preprocessing:
The train.csv (MNIST dataset) is uploaded and loaded using pandas.
The data is separated into features (pixel values) and labels (the actual digits).
Pixel values are normalized to a range of [0, 1].
The dataset is shuffled and split into a training set and a development (validation) set.
Model Definition:
All the necessary functions for the neural network are defined from scratch using NumPy.
This includes functions for parameter initialization, forward propagation (making predictions), backpropagation (calculating errors), and the Adam optimizer rule (updating weights).
Training the Model:
The main training loop (gradien_descent) is executed.
It iterates over the training data in mini-batches, and for each batch, it performs:
A forward pass to get predictions.
A backward pass to calculate gradients (errors).
An update of the model's weights and biases using the Adam optimizer.
Evaluation and Visualization:
After training is complete, the model's final performance is measured on the unseen development set.
The accuracy curve is plotted to show the learning progress.
A few sample predictions are visualized to give a qualitative sense of the model's performance.
