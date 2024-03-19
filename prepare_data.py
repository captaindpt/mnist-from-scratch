import numpy as np
from os.path import join

class DataPreparation:
    def __init__(self, dataloader, batch_size=32):
        self.dataloader = dataloader
        self.batch_size = batch_size
    
    def normalize(self, images):
        # Normalize images from 0-255 to 0-1
        images = np.array(images, dtype=np.float32) / 255.0
        return images
    
    def create_batches(self, data, batch_size):
        # Shuffle the data
        np.random.shuffle(data)
        # Create batches
        return [data[k:k+batch_size] for k in range(0, len(data), batch_size)]
    
    def prepare_data(self):
        # Load data using MnistDataloader
        (x_train, y_train), (x_test, y_test) = self.dataloader.load_data()
        
        # Normalize the training and test images
        x_train = self.normalize(x_train)
        x_test = self.normalize(x_test)
        
        # Flatten the images for our simple neural network
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
        
        # Combine images and labels for shuffling and batching
        train_data = list(zip(x_train, y_train))
        test_data = list(zip(x_test, y_test))
        
        # Create batches
        train_batches = self.create_batches(train_data, self.batch_size)
        test_batches = self.create_batches(test_data, self.batch_size)
        
        return train_batches, test_batches

# Example usage:
# Set file paths for your dataset
input_path = '../input'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

# Initialize the MnistDataloader
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

# Initialize DataPreparation with the dataloader
data_preparation = DataPreparation(mnist_dataloader, batch_size=32)

# Prepare the data
train_batches, test_batches = data_preparation.prepare_data()
