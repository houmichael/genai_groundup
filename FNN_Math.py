import numpy as np

def binary_cross_entropy_loss(y_true, y_pred):
    """
    Calculate binary cross-entropy loss.
    
    The formula for binary cross-entropy loss is:
    -sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)) / N
    
    :param y_true: Array of true labels (0 or 1).
    :param y_pred: Array of predicted probabilities.
    :return: Computed binary cross-entropy loss.
    """

    # Small epsilon to avoid log(0) error.
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate loss
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example usage
y_true = np.array([0, 1, 1, 0])
y_pred = np.array([0.1, 0.7, 0.8, 0.2])
loss = binary_cross_entropy_loss(y_true, y_pred)
print("Binary Cross-Entropy Loss:", loss)
