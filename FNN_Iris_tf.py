import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

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

# Parameters
input_size = X_train.shape[1]
hidden_size = 10
output_size = y_train.shape[1]
learning_rate = 0.002

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(hidden_size, input_dim=input_size, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 5000
model.fit(X_train, y_train, epochs=epochs, verbose=0)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy on the test set:", accuracy)
