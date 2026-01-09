import gzip
import random
import struct
from matplotlib import pyplot as plt
import matplotlib
import numpy as np


from cnn import Layer, Network3, Neuron, Weight, Kernel 
from os.path import join

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        # Read header info
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read all image data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape to (num_images, 28*28)
        data = data.reshape(num, rows * cols)
        # Normalize to [0,1]
        return data / 255.0

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def one_hot(labels, num_classes=10):
    one_hot_labels = np.zeros((labels.size, num_classes))
    one_hot_labels[np.arange(labels.size), labels] = 1
    return one_hot_labels

def load_mnist_dataset(image_file, label_file):
    images = load_mnist_images(image_file)
    labels = load_mnist_labels(label_file)
    labels_one_hot = one_hot(labels)
    return images, labels_one_hot


# Example usage
x_train, y_train = load_mnist_dataset("train-images-idx3-ubyte.gz",
                                      "train-labels-idx1-ubyte.gz")
x_test, y_test   = load_mnist_dataset("t10k-images-idx3-ubyte.gz",
                                      "t10k-labels-idx1-ubyte.gz")

print(x_train.shape)  # (60000, 784)
print(y_train.shape)  # (60000, 10)

print(x_test.shape)
print(y_test.shape)

input_neurons = []

for i in range(784):
    input_neurons.append(Neuron([],0, 0))
inputLayer = Layer(input_neurons)

hidden_neurons = []
for i in range(32):
    neuron_weights = []
    for inputneuron in input_neurons:
        neuron_weights.append(Weight(0, inputneuron))

    neuron = Neuron(neuron_weights, 0, 0)

    hidden_neurons.append(neuron)

hiddenLayer = Layer(hidden_neurons)

output_neurons = []
for i in range(10):
    neuron_weights = []
    for hiddenneuron in hidden_neurons:
        neuron_weights.append(Weight(0, hiddenneuron))
    neuron = Neuron(neuron_weights, 0, 0)
    output_neurons.append(neuron)

outputLayer = Layer(output_neurons)
# make network 3
kernel1 = Kernel()
network = Network3(inputLayer, hiddenLayer, outputLayer)
network.connect()

for i in range(0, len(x_train)):
    data = x_train[i]
    data_2d = np.array(data).reshape(28, 28)
    network.convolve(data_2d)
    network.relu()
    input_activations = network.return_input_layer()
    input_activations_list = input_activations.tolist()
    inputLayer.changeNeuronActivations(input_activations_list)
    network.forwardPass()
    network.backprop(data_2d, y_train[i])

print("finished training")

correct = 0
for j in range(100):
    i = random.randint(1,1000)
    inputLayer.changeNeuronActivations(x_test[i])
    network.forwardPass()
    print(network.results())
    print(y_test[i])
    a = network.number(network.results())
    b = network.number(y_test[i])
    if a == b:
        correct += 1

    print("The network thought it was a " + str(a))
    print("It is actually a " + str(b))

    index = i  # change this number to view different images
    img = x_test[index].reshape(28, 28)  # reshape 784 -> 28x28

    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.title(f"Label: {np.argmax(y_test[index])}")
    plt.axis("off")
    plt.show(block=False)
    plt.pause(2)
    plt.close()

print("the network got " + str(correct) + " out of 100 correct")







  
