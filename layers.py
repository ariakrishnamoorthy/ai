class Neuron():
    def __init__(self,  activation, weights, bias):
        self.activation = activation
        self.weights = weights
        self.bias = bias
        

class inputLayer():
    # i think self.neurons should be a list of neurons
    def __init__(self, neurons):
        self.neurons = neurons  

class Layer():
    def __init__(self, numNeurons, bias):
        self.numNeurons = numNeurons
        self.bias = bias
        self.neurons = []
        for i in range(0, numNeurons):
            self.neurons.append(Neuron(0))

    def connections(layer):
        for i in range(0, layer.neurons.length):
            for j in range(0, self.numNeurons):


class Weight():
    def __init__(self, weight, start, end):
        self.weight = weight
        self.start = start
        self.end = end


inputLayer = inputLayer([Neuron(0.4), Neuron(0.3)])

layer1 = Layer(3)

