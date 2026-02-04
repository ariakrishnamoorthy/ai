import math
import random
import numpy as np


class Neuron():
    # from the previous layer to this neuron
    def __init__(self, weights, bias, activation = 0):
        self.activation = activation
        self.weights = weights
        self.bias = bias
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


# should take image in the function instelf, not in the network cuz network does not depend on image
class Network3():
    def __init__(self, inputLayer, hiddenLayer, outputLayer, kernel1):
        self.inputLayer = inputLayer
        self.hiddenLayer = hiddenLayer
        self.outputLayer = outputLayer
        self.softMaxOutput = []
      
        #self.kernel2 = kernel2
    
    def connect(self):
        self.hiddenLayer.connect(self.inputLayer, True)
        self.outputLayer.connect(self.hiddenLayer, False)


    def forwardPass(self):
        for neuron in self.hiddenLayer.neurons:
            neuron.activate()
        for neuron in self.outputLayer.neurons:
            neuron.activate()

  
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
        max = numbers[0]
        for i in range(len(numbers)):
            if numbers[i] > max:
                max = numbers[i]
                maxindex = i 
        return maxindex

    def convolve(self, image):
        self.Z = np.zeros()
        length, width = self.Z.shape
        kSize = self.kernel1.size
        

        for i in range(length):
            for j in range(width):
                patch = image[i:i+kSize, j:j+kSize]
                self.Z[i,j] = np.sum(patch * self.kernel1.kernel) + self.kernel1.bias

    def return_input_layer(self):
        return self.A
        
    def ReLU(self):
        self.A = self.Z.copy()
        self.A[self.A < 0] = 0
    # change order becuase cant update weights before updating kernel cuz updating kernel relies on weights

    # goal will be like [0, 0 0, 0, 1 0 0]
    def backprop(self, image, goal):


        hL_affects_loss = []
        derivativeActivation = 0

                # how much each hidden neuron affects loss, represented as a vector
        for neuron in self.hiddenLayer.neurons:
            for neuron2 in self.outputLayer.neurons:
                wjk = self.weightbetween(neuron, neuron2)
                sigmoid = neuron2.activation * (1-neuron2.activation)
                derivativeActivation += 2*(neuron2.activation - goal[self.outputLayer.neurons.index(neuron2)]) * sigmoid * wjk
            derivativeActivation *= (neuron.activation * (1-neuron.activation))
            hL_affects_loss.append(derivativeActivation)
            derivativeActivation = 0

        indx = 0
        length, width = self.Z.shape
        kSize = self.kernel1.size
        newthing = 0
        input_affects_loss = []


        # how much each input neuron affects the hidden layer and then multiply by how much the hidden layer affects loss
        for input_neuron in self.inputLayer.neurons:
            for hidden_neuron in self.hiddenLayer.neurons:
                wjk = self.weightbetween(input_neuron, hidden_neuron)
                newthing += (hL_affects_loss[indx] * wjk)
                indx += 1
            input_affects_loss.append(newthing)
            newthing = 0
            indx = 0

                # how much the input la

        for x in range(length):
            for y in range(width):
                if self.Z[x, y] <= 0:
                    input_affects_loss[x*width + y] = 0

        kernel_affects_loss = []
        
        indx2 = 0
        newthing2 = 0
        
        for i in range(kSize):
            for j in range(kSize):
                for a in range(length):
                    for b in range(width):
                        indx2 = a*width + b
                        newthing2 += image[i+a][j+b] * input_affects_loss[indx2]
                kernel_affects_loss.append(newthing2)
                newthing2 = 0
        
        for i in range(kSize):
            for j in range(kSize):
                self.kernel1.kernel[i,j] -= 0.01 * kernel_affects_loss[i * kSize + j]

        self.kernel1.bias -= 0.01 * sum(input_affects_loss)

    
        # changing the weights and biases between the hidden layer and the output layer
        for neuron in self.outputLayer.neurons:
            derivativeBias = 2*(neuron.activation - goal[self.outputLayer.neurons.index(neuron)]) * neuron.activation * (1-neuron.activation)
            neuron.bias -= 0.01 * derivativeBias
    
            for weight in neuron.weights:
                # not goal[ neuron, change later
                sigmoid = neuron.activation * (1-neuron.activation)
                derivativeWeight = 2*(neuron.activation - goal[self.outputLayer.neurons.index(neuron)]) * sigmoid * weight.startNeuron.activation
                weight.strength -= 0.01 * derivativeWeight
        
         
        # chainging the weights and biases between the input layer and hidden layer
        for neuron in self.hiddenLayer.neurons:
            neuron.bias -= 0.01 * hL_affects_loss[self.hiddenLayer.neurons.index(neuron)]
            for weight in neuron.weights:
                weight.strength -= 0.01 * hL_affects_loss[self.hiddenLayer.neurons.index(neuron)] * weight.startNeuron.activation


            


      


class Kernel():
    def __init__(self, bias):
        self.size = 3
        self.kernel = np.random.uniform(-1, 1, (3, 3))
        self.bias = random.uniform(-1, 1)




