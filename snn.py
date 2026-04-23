import math
import random
import numpy as np



class SpikingNeuron():
    def __intit__(self, activation = 0): 
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
            self.membrane -= self.activation
        else:
            self.activation = 0

    def reset(self):
        self.activation = 0
        self.membrane = 0

        
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

class Kernel():
    def __init__(self, size):
        self.kernel = np.random.rand(3,3);
        self.size = size


# should take image in the function instelf, not in the network cuz network does not depend on image
class Network3():
    def __init__(self, inputLayer, hiddenLayer, outputLayer, kernel1):
        self.inputLayer = inputLayer
        self.hiddenLayer = hiddenLayer
        self.outputLayer = outputLayer
        self.softMaxOutput = []
        self.convolved = None
        self.conv_pre_relu = None

        self.kernel = kernel1
    
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
    
    def convolve(self, image):
        H = self.kernel.size
        W = self.kernel.size
        H_img, W_img = image.shape
        out_h = H_img - H + 1
        out_w = W_img - W + 1
        pre = [[0.0 for _ in range(out_w)] for _ in range(out_h)]
        layer = [[0.0 for _ in range(out_w)] for _ in range(out_h)]


        for i in range(out_h):
            for j in range(out_w):
                z = self.multiply_kernel(image, i, j)
                pre[i][j] = z
                layer[i][j] = max(0.0, z)
        
        self.conv_pre_relu = pre
        np_layer = np.array(layer)
        
        return np_layer
        

    def multiply_kernel(self, image, index1, index2):
        sum = 0
        for i in range(index1, index1+ self.kernel.size):
            for j in range(index2, index2 + self.kernel.size):
                sum += self.kernel.kernel[i-index1][j-index2] * image[i][j]

        return sum

    
    def backpropagate(self, image,label):
        target = [0]*10
        target[label] = 1
        dz = []
        dh = []
        delta_flat = []
        
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
            dh.append(dz_h)
            if hiddenneuron.z <= 0:
                dz_h = 0

            for weight in hiddenneuron.weights:
                weight.strength -= learning_rate * dz_h * weight.startNeuron.activation
            
            hiddenneuron.bias -= learning_rate * dz_h

    

        for i, inputneuron in enumerate(self.inputLayer.neurons):
            dz_i = 0
            for h, hiddenneuron in enumerate(self.hiddenLayer.neurons):
                dz_i += dh[h] * self.weightbetween(inputneuron, hiddenneuron)
            delta_flat.append(dz_i)   
        self.backprop_kernel(image, delta_flat)
        

        

    def backprop_kernel(self, image, delta_flat):
        learning_rate = 0.001
        H = self.kernel.size
        W = self.kernel.size
        H_img, W_img = image.shape
        out_h = H_img - H + 1
        out_w = W_img - W + 1

        delta = np.array(delta_flat).reshape(out_h, out_w)

        delta_pre = np.zeros_like(delta)
        dK = [[0.0 for _ in range(W)] for _ in range(H)]
        for i in range(out_h):
            for j in range(out_w):
                if (self.conv_pre_relu[i][j] > 0):
                    delta_pre[i][j] = delta[i][j]
                else:
                    delta_pre[i][j] = 0
               

        kernel_grad = 0
        for u in range (H):
            for v in range (W):
                kernel_grad = 0;
                for i in range(out_h):
                    for j in range(out_w):
                        kernel_grad += delta_pre[i][j] * image[i + u][j + v]

            dK[u][v] = kernel_grad
        for u in range(H):
            for v in range(W):
                self.kernel.kernel[u][v] -= learning_rate * dK[u][v]

        




        




