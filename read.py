import random
import numpy as np
import struct
import gzip

from cnn import Layer, Network3, Neuron

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows * cols)  # flatten to 784
        images = images.astype(np.float32) / 255.0       # normalize
        return images


def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels  # integers 0–9


def load_mnist_dataset(image_file, label_file):
    images = load_mnist_images(image_file)
    labels = load_mnist_labels(label_file)
    return images, labels

x_train, y_train = load_mnist_dataset(
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz"
)

x_test, y_test = load_mnist_dataset(
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
)

print(x_train.shape)  # (60000, 784)
print(y_train.shape)  # (60000,)
print(x_test.shape)   # (10000, 784)
print(y_test.shape)   # (10000,)

input_layer = Layer([Neuron() for _ in range(28*28)])
hidden_layer = Layer([Neuron() for _ in range(128)])
output_layer = Layer([Neuron() for _ in range(10)])

network = Network3(input_layer, hidden_layer, output_layer)
network.connect()

indices = list(range(len(x_train)))

for epoch in range(10):
    random.shuffle(indices)
    for i in (indices):
        
        # Set input activations using your function
        network.inputLayer.changeNeuronActivations(x_train[i])
        
        # Forward pass
        network.forwardPass()
        network.softmax()
        
        # Backpropagation
        network.backpropagate(y_train[i])
 
    print(f"Epoch {epoch+1} complete")

score = 0
num_tests = len(x_test)

for j in range(num_tests):
    i = random.randint(1,1000)

    # Set input activations
    network.inputLayer.changeNeuronActivations(x_test[i])

    # Forward pass only (no backprop)
    network.forwardPass()
    network.softmax()

    predicted = np.argmax(network.softMaxOutput)
    actual = y_test[i]

    if predicted == actual:
        score += 1

print("\nFinal Score:", score, "/", num_tests)
print("Accuracy:", score / num_tests)