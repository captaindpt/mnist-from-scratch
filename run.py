import numpy as np
import json
from load_data import MnistDataloader
from neural_network import NeuralNetwork
from os.path import join
import random
import matplotlib.pyplot as plt

# Assuming the NeuralNetwork class and MnistDataloader are defined as before

def load_model(model_path='./model/model_parameters.json'):
    with open(model_path, 'r') as file:
        model_parameters = json.load(file)
    nn = NeuralNetwork()
    nn.weights = {k: np.array(v) for k, v in model_parameters['weights'].items()}
    nn.biases = {k: np.array(v) for k, v in model_parameters['biases'].items()}
    return nn

def display_prediction(x, predicted_label, true_label):
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {predicted_label}, True: {true_label}')
    plt.show()

def run_random_predictions(x_test, y_test, nn, num_samples=5):
    for _ in range(num_samples):
        # Randomly select an image from the test set
        idx = random.randint(0, len(x_test) - 1)
        x = x_test[idx]
        y_true = np.argmax(y_test[idx])
        
        # Make a prediction
        y_pred = nn.forward_pass(x)
        predicted_label = np.argmax(y_pred)
        
        # Display the result
        display_prediction(x, predicted_label, y_true)

# Load the MNIST dataset
input_path = './archive'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_and_prepare_data()

# Load the model
nn = load_model()

# Run random predictions
run_random_predictions(x_test, y_test, nn, num_samples=20)
