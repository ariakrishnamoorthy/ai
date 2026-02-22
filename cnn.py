import math
import random
import numpy as np


class Neuron():
    # from the previous layer to this neuron
    def __init__(self, activation = 0):
        self.activation = activation
        self.weights = []
        self.bias = random.uniform(-0.3, 0.3)
        self.z = 0

    def activate(self, useRelu):
        z = 0
        for i in self.weights:
            z += i.startNeuron.activation * i.strength
        z += self.bias
        self.z = z
        if(useRelu):
             self.activation = max(0, self.z)
        else:
            self.activation = self.z
       
    

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



class Weight():
    def __init__(self, strength, start):
        self.strength = strength
        self.startNeuron = start


# should take image in the function instelf, not in the network cuz network does not depend on image
class Network3():
    def __init__(self, inputLayer, hiddenLayer, outputLayer):
        self.inputLayer = inputLayer
        self.hiddenLayer = hiddenLayer
        self.outputLayer = outputLayer
        self.softMaxOutput = []
      
        #self.kernel2 = kernel2
    
    def connect(self):
        self.hiddenLayer.connect(self.inputLayer)
        self.outputLayer.connect(self.hiddenLayer)


    def forwardPass(self):
        for neuron in self.hiddenLayer.neurons:
            neuron.activate(True)
        for neuron in self.outputLayer.neurons:
            neuron.activate(False)

  
    def weightbetween(self, start, end):
        for weight in end.weights:
            if weight.startNeuron == start:
                return weight.strength
            
    def get_last_layer(self):
        last_layer = []
        for neuron in self.outputLayer.neurons:
            last_layer.append(neuron.activation)
        return last_layer
            
    def softmax(self):
        last_layer = self.get_last_layer()

        max = last_layer[0]
        for a in last_layer:
            if a > max:
                max = a

        sum = 0
        exp_values = []
        for a in last_layer :
            e = math.exp(a - max)
            exp_values.append(e)
            sum += e
        probabilities = []
        for e in exp_values:
            probabilities.append(e/sum)
        self.softMaxOutput = probabilities

    def cross_entropy_loss(self, label):
        epsilon = 0.0000000001
        actual = self.softMaxOutput[label]
        return -math.log(actual + epsilon)
    
    def backpropagate(self, label):
        target = [0]*10
        target[label] = 1
        dz = []
        
        for i in range(len(self.outputLayer.neurons)):
            dz.append(self.softMaxOutput[i] - target[i])

        learning_rate = 0.001
        for i, neuron in enumerate(self.outputLayer.neurons):
            dz_i = dz[i]

            # update weights
            for weight in neuron.weights:
                weight.strength -= learning_rate * dz_i * weight.startNeuron.activation

            # update bias
            neuron.bias -= learning_rate * dz_i

        
        for h, hiddenneuron in enumerate(self.hiddenLayer.neurons):
            dz_h = 0
            for i, outputneuron in enumerate(self.outputLayer.neurons):
                dz_h += dz[i] * self.weightbetween(hiddenneuron, outputneuron)

            if hiddenneuron.z <= 0:
                dz_h = 0

            for weight in hiddenneuron.weights:
                weight.strength -= learning_rate * dz_h * weight.startNeuron.activation
            
            hiddenneuron.bias -= learning_rate * dz_h

        




