import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=300)
np.set_printoptions(precision=4)

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# One-hot encoding for the output labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizing the data
X_train = X_train / np.max(X_train)
X_test = X_test / np.max(X_test)

input_size = X_train.shape[1]
epochs = 5000
learning_rate = 0.002
hidden_size = 10
output_size = y_train.shape[1]

class FNN:
    """
    Feedforward Neural Network (FNN) for multi-class classification.
    It consists of one hidden layer with a ReLU activation function and an output layer with a softmax activation function.
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        """
        Initializes the neural network with random weights and biases.
        :param input_size: Number of features in the input data.
        :param hidden_size: Number of neurons in the hidden layer.
        :param output_size: Number of classes (neurons) in the output layer.
        :param learning_rate: The step size used for gradient descent optimization.
        """
        np.random.seed(168)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def softmax(self, x):
        """Softmax activation function."""
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def relu_derivative(self, x):
        """Derivative of the ReLU function."""
        return (x > 0) * 1

    def forward(self, X):
        """
        Performs forward propagation through the network.
        :param X: Input feature matrix.
        :return: Output probabilities.
        """
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def backward(self, X, y, y_pred):
        """
        Performs backward propagation to compute gradients for weights and biases.
        :param X: Input feature matrix.
        :param y: True labels (one-hot encoded).
        :param output: Output from the forward pass.
        """
        error_output = y_pred - y
        d_W2 = np.dot(self.A1.T, error_output)
        d_b2 = np.sum(error_output, axis=0, keepdims=True)

        error_hidden = np.dot(error_output, self.W2.T) * self.relu_derivative(self.A1)
        d_W1 = np.dot(X.T, error_hidden)
        d_b1 = np.sum(error_hidden, axis=0, keepdims=True)

        self.optimizer(d_W1, d_b1, d_W2, d_b2)

    def optimizer(self, d_W1, d_b1, d_W2, d_b2):
        """
        Update the weights and biases using gradient descent.
        :param d_W1, d_b1: Gradients for weights and biases of the first layer.
        :param d_W2, d_b2: Gradients for weights and biases of the second layer.
        """
        self.W1 -= self.learning_rate * d_W1
        self.b1 -= self.learning_rate * d_b1
        self.W2 -= self.learning_rate * d_W2
        self.b2 -= self.learning_rate * d_b2

    def loss(self, y_true, y_pred):
        """
        Calculates the cross-entropy loss.
        :param y_true: True labels (one-hot encoded).
        :param y_pred: Predicted probabilities from the forward pass.
        :return: Computed cross-entropy loss.
        """
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    def train(self, X, y, epochs):
        """
        Trains the neural network.
        :param X: Training feature matrix.
        :param y: Training labels (one-hot encoded).
        :param epochs: Number of iterations to train the network.
        """
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss_value = self.loss(y, y_pred)
            self.backward(X, y, y_pred)

            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss: {loss_value}")

    def predict(self, X):
        """
        Makes predictions on new data.
        :param X: Input feature matrix.
        :return: Predicted class labels.
        """
        output = self.forward(X)
        return np.argmax(output, axis=1)

# Create an instance of the neural network
fnn = FNN(input_size, hidden_size, output_size, learning_rate)
fnn.train(X_train, y_train, epochs=5000)

# Test the model
predictions = fnn.predict(X_test)
accuracy = np.mean(np.argmax(y_test, axis=1) == predictions)
print("Accuracy on the test set:", accuracy)