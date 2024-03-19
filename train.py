from load_data import MnistDataloader
from neural_network import NeuralNetwork
from os.path import join, exists
from os import makedirs
import numpy as np
import json
from tqdm import tqdm

# Initialize the data loader and load the prepared data
input_path = './archive'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

# Load MNIST dataset
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_and_prepare_data()

# Initialize the neural network
nn = NeuralNetwork()

def train_and_save_model(nn, x_train, y_train, epochs=10, model_path='./model'):
    if not exists(model_path):
        makedirs(model_path)
        
    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in tqdm(zip(x_train, y_train), total=len(x_train), desc=f'Epoch {epoch+1}/{epochs}'):
            nn.forward_pass(x)
            loss = nn.compute_loss(y, nn.a_output)
            epoch_loss += loss
            nn.backward_pass(x, y)
        avg_loss = epoch_loss / len(x_train)
        print(f'Epoch {epoch+1}: Average Loss: {avg_loss:.4f}')
        
    # Save model parameters
    model_parameters = {
        'weights': {k: v.tolist() for k, v in nn.weights.items()},
        'biases': {k: v.tolist() for k, v in nn.biases.items()}
    }
    with open(join(model_path, 'model_parameters.json'), 'w') as file:
        json.dump(model_parameters, file)
    print('Model saved successfully.')

# Train the neural network and save the model
train_and_save_model(nn, x_train, y_train, epochs=10)
