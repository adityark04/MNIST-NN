# MNIST-NN
This project is a complete implementation of a Neural Network built from scratch using only the NumPy library. The goal of this network is to recognize handwritten digits (from 0 to 9) based on the well-known MNIST dataset. All the code for data handling, model training, and evaluation is contained within the NN.ipynb Jupyter Notebook.

Expected Output
Live Training Progress: The notebook prints the model's accuracy on both the training and development sets at regular intervals.
Final Accuracy Report: After training, the final accuracy on the unseen development set is displayed (~97.9%).
Accuracy Plot: A graph is generated to visualize how the model's accuracy improves over the training iterations.
Prediction Visualization: A sample of 5 test images is shown with their correct labels and the model's predicted labels for a quick qualitative check.
Key Concepts
Network Architecture: A multi-layer feedforward network.
Input Layer: 784 nodes (28x28 pixels).
Hidden Layers: Two hidden layers with 256 and 128 nodes respectively.
Output Layer: 10 nodes (for digits 0-9).
Activation Functions:
ReLU: Used in the hidden layers for non-linearity.
Softmax: Used in the output layer to produce probabilities.
Optimization Techniques:
Adam Optimizer: For adaptive learning rate adjustments.
Mini-Batch Gradient Descent: To train the model efficiently.
Learning Rate Decay: To fine-tune convergence.
Regularization (to prevent overfitting):
L2 Regularization: Penalizes large weights.
Dropout: Randomly deactivates neurons during training.
Parameter Initialization:
He Initialization: To maintain healthy gradient flow.
