
# MNIST Handwritten Digit Classification From Scratch

This project implements a basic feedforward neural network (FNN) in Python to classify handwritten digits from the MNIST dataset, entirely from scratch. The purpose of this project is educational, aiming to demonstrate the fundamentals of neural network architecture, data preprocessing, training, and evaluation without relying on high-level machine learning frameworks.

## Project Structure

- `load_data.py`: Contains the `MnistDataloader` class for loading and preparing the MNIST dataset.
- `neural_network.py`: Implements the `NeuralNetwork` class, defining the architecture and functionalities of the neural network.
- `train.py`: Script for training the neural network using the MNIST dataset.
- `run.py`: Script for testing the trained model on random samples from the test set and displaying the predictions.
- `model/`: Directory where the trained model parameters are saved.
- `archive/`: Directory containing the MNIST dataset files.

## Setup

### Prerequisites

Ensure you have Python 3.x installed on your system. You will also need the following packages:
- NumPy
- Matplotlib
- tqdm (optional for progress bars)

### Installing Dependencies

Install the required Python packages by running the following command:

```bash
pip install -r requirements.txt
```

### Preparing the Dataset

Download the MNIST dataset from [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/) or another source. Ensure the dataset files are placed in the `./archive` directory with the following structure:

- `./archive/train-images-idx3-ubyte/train-images-idx3-ubyte`
- `./archive/train-labels-idx1-ubyte/train-labels-idx1-ubyte`
- `./archive/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte`
- `./archive/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte`

## Usage

### Training the Model

To train the model, simply run:

```bash
python train.py
```

This script will train the neural network on the MNIST dataset and save the model parameters in the `./model` directory.

### Running Predictions

After training, you can test the model on random samples from the test set by running:

```bash
python run.py
```

This will display the model's predictions along with the actual labels for several random images from the test set.

## Note

This project is intended for learning purposes, showcasing how to implement a basic neural network from scratch for image classification. It is not optimized for production use.
