import math
import random
import numpy as np



class SpikingNeuron():
    def __init__(self, activation = 0): 
        self.weights = []
        self.bias = random.uniform(-0.3, 0.3)
        self.z = 0
        self.activation = activation
        self.threshold = random.uniform(0.5, 1.5)
        self.membrane = 0
        self.decay = 0.9

    def activate(self):
        z = 0
        for i in self.weights:
            z += i.startNeuron.activation * i.strength
        z += self.bias
        self.z = z

        self.membrane = self.membrane * self.decay + self.z
        if (self.membrane >= self.threshold):
            self.activation = 1
            self.membrane -= self.threshold
        else:
            self.activation = 0

    def reset(self):
        self.activation = 0
        self.membrane = 0
        

class Layer():
    # i think self.neurons should be a list of neurons
    def __init__(self, neurons):
        self.neurons = neurons  

    def connect(self, prevLayer):
        std = math.sqrt(2.0 / len(prevLayer.neurons))  # He init for ReLU
        for j in self.neurons:
            weights = []
            for k in prevLayer.neurons:
                weights.append(Weight(random.gauss(0, std), k))
            j.weights = weights
    
    def changeNeuronActivations(self, newneuronactivations):
        for i in range(0, len(self.neurons)):
            self.neurons[i].activation = newneuronactivations[i]

    def reset_state(self):
        for neuron in self.neurons:
            if hasattr(neuron, "reset_state"):
                neuron.reset_state()



class Weight():
    def __init__(self, strength, start):
        self.strength = strength
        self.startNeuron = start


# should take image in the function instelf, not in the network cuz network does not depend on image
class SpikingNetwork():
    def __init__(self, inputLayer, hiddenLayer, outputLayer):
        self.inputLayer = inputLayer
        self.hiddenLayer = hiddenLayer
        self.outputLayer = outputLayer


    def connect(self):
        self.hiddenLayer.connect(self.inputLayer)
        self.outputLayer.connect(self.hiddenLayer)

    def reset_state(self):
        self.hiddenLayer.reset_state()
        self.outputLayer.reset_state()

        
    def encode_input(self, x, num_steps):
        spike_trains = []
        for t in range(num_steps):
            spikes = []
            for pixel in x:
                spikes.append(1 if random.random() < pixel else 0)
            spike_trains.append(spikes)
        return spike_trains
    
    def forwardPass(self, x, num_steps):
        self.reset_state()

        output_counts = [0] * len(self.outputLayer.neurons)
        input_spikes = self.encode_input(x, num_steps)
        for t in range(num_steps):
            self.inputLayer.changeNeuronActivations(input_spikes[t])

            for neuron in self.hiddenLayer.neurons:
                neuron.activate()

            for neuron in self.outputLayer.neurons:
                neuron.activate()

            for i, neuron in enumerate(self.outputLayer.neurons):
                output_counts[i] += neuron.activation

        return output_counts

        


        




        




