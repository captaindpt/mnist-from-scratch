import numpy as np

class NeuralNetwork:
    def __init__(self):
        # Initialize weights and biases
        self.weights = {
            'hidden': np.random.randn(784, 128) / np.sqrt(784),
            'output': np.random.randn(128, 10) / np.sqrt(128)
        }
        self.biases = {
            'hidden': np.zeros((1, 128)),
            'output': np.zeros((1, 10))
        }
    
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU."""
        return np.where(x <= 0, 0, 1)

    def softmax(self, x):
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
    
    def forward_pass(self, x):
        """Compute the forward pass."""
        self.z_hidden = np.dot(x, self.weights['hidden']) + self.biases['hidden']
        self.a_hidden = self.relu(self.z_hidden)
        
        self.z_output = np.dot(self.a_hidden, self.weights['output']) + self.biases['output']
        self.a_output = self.softmax(self.z_output)
        
        return self.a_output
    
    def compute_loss(self, y_true, y_pred):
        """Compute the loss using cross-entropy."""
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss
    
    def backward_pass(self, x, y_true):
        """Compute backward pass."""
        m = y_true.shape[0]

        # Ensure x is a 2D array with shape (1, 784)
        x = x.reshape(1, -1)  # Reshapes x to (1, 784) if it's not already

        # Output layer error
        delta_output = self.a_output - y_true
        
        # Hidden layer error
        delta_hidden = np.dot(delta_output, self.weights['output'].T) * self.relu_derivative(self.z_hidden)
        
        # Gradients for weights and biases
        dw_output = np.dot(self.a_hidden.T, delta_output) / m
        db_output = np.sum(delta_output, axis=0, keepdims=True) / m
        dw_hidden = np.dot(x.T, delta_hidden) / m
        db_hidden = np.sum(delta_hidden, axis=0, keepdims=True) / m
        
        # Update weights and biases
        learning_rate = 0.01  # Learning rate can be adjusted
        self.weights['hidden'] -= learning_rate * dw_hidden
        self.weights['output'] -= learning_rate * dw_output
        self.biases['hidden'] -= learning_rate * db_hidden
        self.biases['output'] -= learning_rate * db_output
    
    def train(self, x_train, y_train, epochs=10):
        """Training loop."""
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):  # Assuming x_train and y_train are pre-batched
                y_pred = self.forward_pass(x)
                loss = self.compute_loss(y, y_pred)
                self.backward_pass(x, y)
            
            print(f"Epoch {epoch+1}, Loss: {loss}")


# to train
