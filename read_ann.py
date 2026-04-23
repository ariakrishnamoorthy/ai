import random
import numpy as np
import struct
import gzip
import os
import pickle

from ann import  Layer, Network3, Neuron

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
print("running")

SAVE_PATH = 'network.pkl'

# comment out later!


if os.path.exists(SAVE_PATH):
    # Skip training entirely — load saved weights
    print("Loading saved network...")
    with open(SAVE_PATH, 'rb') as f:
        network = pickle.load(f)

else:
    print("yay")

    input_layer = Layer([Neuron() for _ in range(28*28)])
    hidden_layer = Layer([Neuron() for _ in range(128)])
    output_layer = Layer([Neuron() for _ in range(10)])



    network = Network3(input_layer, hidden_layer, output_layer)

    network.connect()

    print("hi")




    indices = list(range(len(x_train)))

    for epoch in range(10):
        random.shuffle(indices)
        for i in (indices):
            img_2d = x_train[i].reshape(28, 28)
            conv_map = network.convolve(img_2d)
            flat = conv_map.reshape(-1).tolist()
            network.inputLayer.changeNeuronActivations(flat)
            network.forwardPass()
            network.softmax()
            network.backpropagate(img_2d, y_train[i])
        
        print(f"Epoch {epoch+1} complete")
        with open(SAVE_PATH, 'wb') as f:
            pickle.dump(network, f)
        print("Network saved.")


score = 0
num_tests = len(x_test)

for j in range(num_tests):

    # Set input activations
    img_2d = x_test[j].reshape(28, 28)
    conv_map = network.convolve(img_2d)
    flat = conv_map.reshape(-1).tolist()
    network.inputLayer.changeNeuronActivations(flat)
    network.forwardPass()
    network.softmax()

    predicted = np.argmax(network.softMaxOutput)
    actual = y_test[j]

    if predicted == actual:
        score += 1

print("\nFinal Score:", score, "/", num_tests)
print("Accuracy:", score / num_tests)