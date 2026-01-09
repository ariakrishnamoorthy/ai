import math
import random
import struct
import numpy as np

class Neuron():
    # from the previous layer to this neuron
    def __init__(self, weights, bias, activation = 0):
        self.activation = activation
        self.weights = weights
        self.bias = bias
        self.z = 0

    def activate(self):
        z = 0
        for i in self.weights:
            z += i.startNeuron.activation * i.strength
        z += self.bias
        self.z = z
        self.activation = 1/(1+(math.e)**(-z))
    

class Layer():
    # i think self.neurons should be a list of neurons
    def __init__(self, neurons):
        self.neurons = neurons  

    def connect(self, prevLayer):
        for j in self.neurons:
            weights = []
            for k in prevLayer.neurons:
                # make the strength be a random number
                weights.append(Weight(random.uniform(-0.3,0.3), k))
            j.weights = weights
    
    def changeNeuronActivations(self, newneuronactivations):
        for i in range(0, len(self.neurons)):
            self.neurons[i].activation = newneuronactivations[i]



class Weight():
    def __init__(self, strength, start):
        self.strength = strength
        self.startNeuron = start



class Network():
    def __init__(self, inputLayer, hiddenLayer, outputLayer):
        self.inputLayer = inputLayer
        self.hiddenLayer = hiddenLayer
        self.outputLayer = outputLayer
    
    def connect(self):
        self.hiddenLayer.connect(self.inputLayer)
        self.outputLayer.connect(self.hiddenLayer)


    def forwardPass(self):
        for neuron in self.hiddenLayer.neurons:
            neuron.activate()
        for neuron in self.outputLayer.neurons:
            neuron.activate()

  
    def weightbetween(self, start, end):
        for weight in end.weights:
            if weight.startNeuron == start:
                return weight.strength

    def cost(self, goal):
        cost = 0
        for neuron in self.outputLayer.neurons:
            cost += (neuron.activation - goal[self.outputLayer.neurons.index(neuron)])**2
        return cost
    
    def results(self):
        output = []
        for neuron in self.outputLayer.neurons:
            output.append(neuron.activation)
        return output
    
    def number(self, numbers):
        maxindex = 0
        max = 0
        for i in range(len(numbers)):
            if numbers[i] > max:
                max = numbers[i]
                maxindex = i 
        return maxindex


    # goal will be like [0, 0 0, 0, 1 0 0]
    def backprop(self, goal):
        hiddenLayerGoal = []
        hL_affects_loss = []
        derivativeActivation = 0

        for neuron in self.outputLayer.neurons:
            derivativeBias = 2*(neuron.activation - goal[self.outputLayer.neurons.index(neuron)]) * neuron.activation * (1-neuron.activation)
            neuron.bias -= 0.01 * derivativeBias
    
            for weight in neuron.weights:
                # not goal[ neuron, change later
                sigmoid = neuron.activation * (1-neuron.activation)
                derivativeWeight = 2*(neuron.activation - goal[self.outputLayer.neurons.index(neuron)]) * sigmoid * weight.startNeuron.activation
                weight.strength -= 0.01 * derivativeWeight
        

        # code for how much a hiddenlayer neuron affects the output 
        
        for neuron in self.hiddenLayer.neurons:
            for neuron2 in self.outputLayer.neurons:
                wjk = self.weightbetween(neuron, neuron2)
                sigmoid = neuron2.activation * (1-neuron2.activation)
                derivativeActivation += 2*(neuron2.activation - goal[self.outputLayer.neurons.index(neuron2)]) * sigmoid * wjk
            derivativeActivation *= (neuron.activation * (1-neuron.activation))
            hiddenLayerGoal.append(derivativeActivation)
            derivativeActivation = 0
        
        for neuron in self.hiddenLayer.neurons:
            neuron.bias -= 0.01 * hiddenLayerGoal[self.hiddenLayer.neurons.index(neuron)]
            for weight in neuron.weights:
                weight.strength -= 0.01 * hiddenLayerGoal[self.hiddenLayer.neurons.index(neuron)] * weight.startNeuron.activation

        
        for neuron in self.hiddenLayer.neurons:
            for neuron2 in self.outputLayer.neurons:
                wjk = self.weightbetween(neuron, neuron2)
                sigmoid = neuron2.activation * (1-neuron2.activation)
                derivativeActivation += 2*(neuron2.activation - goal[self.outputLayer.neurons.index(neuron2)]) * sigmoid * wjk
            derivativeActivation *= (neuron.activation * (1-neuron.activation))
            hL_affects_loss.append(derivativeActivation)
            derivativeActivation = 0

        indx = 0
        newthing = 0
        input_affects_loss = []

        # need to figure out this part, not completely there yet

        for input_neuron in self.inputLayer.neurons:
            for hidden_neuron in self.hiddenLayer.neurons:
                wjk = self.weightbetween(input_neuron, hidden_neuron)
                newthing += (hL_affects_loss[indx] * wjk)
                indx += 1
            input_affects_loss.append(newthing)
            newthing = 0
            indx = 0
        

            





