import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=300)
np.set_printoptions(precision=3)

# Load data and preprocess
iris = load_iris()
X, y = iris.data, iris.target
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = X_train / np.max(X_train)
X_test = X_test / np.max(X_test)

def initialize_network(input_size, hidden_size, output_size):
    # Setting a random seed ensures that the random number generation is reproducible, 
    # which means that you'll get the same set of 'random' numbers (and therefore the same 
    # initial weights and biases) every time you run your code. 
    np.random.seed(168)
    
    weights = {
        "W1": np.random.randn(input_size, hidden_size) * 0.01,
        "W2": np.random.randn(hidden_size, output_size) * 0.01
    }
    biases = {
        "b1": np.zeros((1, hidden_size)),
        "b2": np.zeros((1, output_size))
    }
    return weights, biases

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0) * 1

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward(X, weights, biases):
    Z1 = np.dot(X, weights["W1"]) + biases["b1"]
    A1 = relu(Z1)
    Z2 = np.dot(A1, weights["W2"]) + biases["b2"]
    A2 = softmax(Z2)
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

def backward(X, y, cache, weights, biases):
    error_output = cache["A2"] - y
    d_W2 = np.dot(cache["A1"].T, error_output)
    d_b2 = np.sum(error_output, axis=0, keepdims=True)

    error_hidden = np.dot(error_output, weights["W2"].T) * relu_derivative(cache["Z1"])
    d_W1 = np.dot(X.T, error_hidden)
    d_b1 = np.sum(error_hidden, axis=0, keepdims=True)

    grads = {"d_W1": d_W1, "d_b1": d_b1, "d_W2": d_W2, "d_b2": d_b2}
    return grads

def update_parameters(weights, biases, grads, learning_rate):
    weights["W1"] -= learning_rate * grads["d_W1"]
    biases["b1"] -= learning_rate * grads["d_b1"]
    weights["W2"] -= learning_rate * grads["d_W2"]
    biases["b2"] -= learning_rate * grads["d_b2"]
    return weights, biases

def compute_loss(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def predict(X, weights, biases):
    y_pred, _ = forward(X, weights, biases)
    return np.argmax(y_pred, axis=1)

def train(X_train, y_train, epochs, learning_rate, input_size, hidden_size, output_size):
    weights, biases = initialize_network(input_size, hidden_size, output_size)
    for epoch in range(epochs):
        output, cache = forward(X_train, weights, biases)
        loss = compute_loss(y_train, output)
        grads = backward(X_train, y_train, cache, weights, biases)
        weights, biases = update_parameters(weights, biases, grads, learning_rate)

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
    
    return weights, biases  # Return the trained weights and biases

input_size = X_train.shape[1]
epochs = 5000
learning_rate = 0.002
hidden_size = 10
output_size = y_train.shape[1]

# Train the model and get the trained weights and biases
weights, biases = train(X_train, y_train, epochs, learning_rate, input_size, hidden_size, output_size)

# Test the model
predictions = predict(X_test, weights, biases)
accuracy = np.mean(np.argmax(y_test, axis=1) == predictions)
print("Accuracy on the test set:", accuracy)
