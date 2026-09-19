import math
import random
import numpy as np

class Neuron():
    # from the previous layer to this neuron
    def __init__(self, activation = 0):
        self.activation = activation
        self.weights = []
        self.bias = 0
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
        self.embed_dim = 8
        self.E = [[random.gauss(0, 0.1) for _ in range(8)] for _ in range(8)]
        self._last_i = 0
        self._last_j = 0 
        self._t = 0
        self._b1 = 0.9
        self._b2 = 0.999
        self._eps = 1e-8
        self._wd = 0.5
        self._m = {}
        self._v = {}
        self._grads = {}   
    
    def connect(self):
        self.hiddenLayer.connect(self.inputLayer)
        self.outputLayer.connect(self.hiddenLayer)

    def embed_and_forward(self, i, j):
        self._last_i = i
        self._last_j = j
        #a = min(i,j)
        #b = max(i,j)

        vec = [self.E[i][k] + self.E[j][k] for k in range(8)]
        self.inputLayer.changeNeuronActivations(vec)
        self.forwardPass()

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
  
        num_outputs = len(self.outputLayer.neurons)
        target = [0] * num_outputs
        target[label] = 1
        dz = [self.softMaxOutput[i] - target[i] for i in range(num_outputs)]

        # Hidden layer gradients
        dh = []
        for h, hn in enumerate(self.hiddenLayer.neurons):
            g = sum(dz[i] * self.outputLayer.neurons[i].weights[h].strength
                    for i in range(num_outputs))
            dh.append(0.0 if hn.z <= 0 else g)

        # Embedding gradients
        num_hidden = len(self.hiddenLayer.neurons)
        dInput = [sum(self.hiddenLayer.neurons[h].weights[k].strength * dh[h]
                    for h in range(num_hidden)) for k in range(8)]

        # Store all gradients for adamw_step to consume
        self._grads = {
            'dz': dz,
            'dh': dh,
            'dE_i': dInput,
            'dE_j': dInput,
    }


    def adamw_step(self, learning_rate):
        """Applies AdamW update using gradients computed by backpropagate()."""
        self._t += 1
        t = self._t
        dz  = self._grads['dz']
        dh  = self._grads['dh']
        dEi = self._grads['dE_i']
        dEj = self._grads['dE_j']

        def step(key, param, grad, apply_wd=True):
            if key not in self._m:
                self._m[key] = 0.0
                self._v[key] = 0.0
            self._m[key] = self._b1 * self._m[key] + (1 - self._b1) * grad
            self._v[key] = self._b2 * self._v[key] + (1 - self._b2) * grad * grad
            mhat = self._m[key] / (1 - self._b1 ** t)
            vhat = self._v[key] / (1 - self._b2 ** t)
            update = learning_rate * mhat / (math.sqrt(vhat) + self._eps)
            if apply_wd:
                update += learning_rate * self._wd * param
            return param - update

        # Output layer
        for i, neuron in enumerate(self.outputLayer.neurons):
            for w in neuron.weights:
                w.strength = step(id(w), w.strength, dz[i] * w.startNeuron.activation)
            neuron.bias = step(('b', id(neuron)), neuron.bias, dz[i], apply_wd=False)

        # Hidden layer
        for h, hn in enumerate(self.hiddenLayer.neurons):
            for w in hn.weights:
                w.strength = step(id(w), w.strength, dh[h] * w.startNeuron.activation)
            hn.bias = step(('b', id(hn)), hn.bias, dh[h], apply_wd=False)

        # Embedding matrix
        for k in range(8):

            if self._last_i == self._last_j:
                self.E[self._last_i][k] = step(('E', self._last_i, k),
                self.E[self._last_i][k], dEi[k] + dEj[k])
            else:
                self.E[self._last_i][k] = step(('E', self._last_i, k), self.E[self._last_i][k], dEi[k])
                self.E[self._last_j][k] = step(('E', self._last_j, k), self.E[self._last_j][k], dEj[k])
            #self.E[self._last_i][k] = step(('E', self._last_i, k), self.E[self._last_i][k], dEi[k])
            #self.E[self._last_j][k] = step(('E', self._last_j, k), self.E[self._last_j][k], dEj[k])

    def backpropagate2(self, label, learning_rate):
        num_outputs = len(self.outputLayer.neurons)
        target = [0] * num_outputs
        target[label] = 1
        dz = []
        dh = []
        
        for i in range(num_outputs):
            dz.append(self.softMaxOutput[i] - target[i])

        # 1. Calculate hidden layer gradients using the current operational weights
        for h, hiddenneuron in enumerate(self.hiddenLayer.neurons):
            dz_h = 0
            for i, outputneuron in enumerate(self.outputLayer.neurons):
                dz_h += dz[i] * outputneuron.weights[h].strength
            
            # Apply ReLU gradient thresholding
            if hiddenneuron.z <= 0:
                dz_h = 0
            dh.append(dz_h)

        weight_decay = 0.0005
        
        for i, neuron in enumerate(self.outputLayer.neurons):
            dz_i = dz[i]
            for weight in neuron.weights:
                weight.strength -= learning_rate * (dz_i * weight.startNeuron.activation + weight_decay * weight.strength)
            neuron.bias -= learning_rate * (dz_i + weight_decay * neuron.bias)


        for h, hiddenneuron in enumerate(self.hiddenLayer.neurons):
            dz_h = dh[h]
            for weight in hiddenneuron.weights:
                weight.strength -= learning_rate * (dz_h * weight.startNeuron.activation + weight_decay * weight.strength)
            hiddenneuron.bias -= learning_rate * (dz_h + weight_decay * hiddenneuron.bias)

        num_hidden = len(self.hiddenLayer.neurons)
        dInput = []
        for k in range(16):
            g = sum(self.hiddenLayer.neurons[h].weights[k].strength * dh[h]
                for h in range(num_hidden))
            dInput.append(g)

        for k in range(8):
            self.E[self._last_i][k] -= learning_rate * (dInput[k] + weight_decay * self.E[self._last_i][k])
            self.E[self._last_j][k] -= learning_rate * (dInput[k + 8] + weight_decay * self.E[self._last_j][k])
            

        
