# MAIN.PY
import math

import random
import numpy as np
import os
import pickle


from finitefields import reduce_field, FIELD_ELEMENTS
from finitefieldann import Neuron, Layer, Network3
from collections import Counter

SEED = 8
FRAC = 0.9
for path in ['field_network.pkl', 'weight_history.pkl']:
    if os.path.exists(path):
        os.remove(path)
        print(f"Cleaned up old checkpoint: {path}")

def make_one_hot(index, size=8):
    """Generates an 8-dimensional vector with a 1 at the specified index."""
    vec = [0] * size
    vec[index] = 1
    return vec

def generate_dataset():
    inputs, labels = [], []
    for i in range(8):
        for j in range(8):
            res = reduce_field(mod=3, maxdeg=1, equal=[2],
                               elem1=FIELD_ELEMENTS[i],
                               elem2=FIELD_ELEMENTS[j])
            labels.append(FIELD_ELEMENTS.index(res))
            inputs.append([i, j])   # just two integers now
    return np.array(inputs), np.array(labels)

def extract_w1_matrix(network):
    matrix = []
    for neuron in network.hiddenLayer.neurons:
        neuron_weights = [w.strength for w in neuron.weights]
        matrix.append(neuron_weights)
    return np.array(matrix)



x_data, y_data = generate_dataset()

indices = list(range(len(x_data)))
random.seed(SEED)
random.shuffle(indices)


cut = int(len(x_data) * FRAC)
train_idx, test_idx = indices[:cut], indices[cut:]

x_train, y_train = x_data[train_idx], y_data[train_idx]
x_test, y_test = x_data[test_idx], y_data[test_idx]

counts = Counter()
for (i, j) in x_train:
    counts[int(i)] += 1
    counts[int(j)] += 1
for e in range(8):
    print(f"element {e}: appears in {counts[e]} training pairs")



SAVE_PATH = 'field_network.pkl'
HISTORY_PATH = 'weight_history.pkl'

# Check for existing checkpoints to save training time
if os.path.exists(SAVE_PATH) and os.path.exists(HISTORY_PATH):
    print("Loading saved finite field network and weight history...")
    with open(SAVE_PATH, 'rb') as f:
        network = pickle.load(f)
    with open(HISTORY_PATH, 'rb') as f:
        weight_history = pickle.load(f)
else:
    print("Initializing from-scratch structural network...")
    random.seed(SEED + 1000) 
    input_layer = Layer([Neuron() for _ in range(8)])
    hidden_layer = Layer([Neuron() for _ in range(32)])
    output_layer = Layer([Neuron() for _ in range(8)])

    network = Network3(input_layer, hidden_layer, output_layer)
    network.connect()

        # PROBE 1: are any test labels never seen as a training label?
    train_labels = set(int(y) for y in y_train)
    test_labels  = set(int(y) for y in y_test)
    print("labels missing from training:", test_labels - train_labels)

    # PROBE 2: on 5 test pairs, print predicted vs true
    for k in range(5):
        i, j = int(x_test[k][0]), int(x_test[k][1])
        network.embed_and_forward(i, j); network.softmax()
        pred = int(np.argmax(network.softMaxOutput))
        print(f"({i},{j}) -> pred {pred}, true {int(y_test[k])}")

    # PROBE 3: is the network's prediction EVER outside the set of training labels?
    preds = []
    for k in range(len(x_test)):
        i, j = int(x_test[k][0]), int(x_test[k][1])
        network.embed_and_forward(i, j); network.softmax()
        preds.append(int(np.argmax(network.softMaxOutput)))
    print("distinct predictions on test:", sorted(set(preds)))
    print("distinct train labels:", sorted(train_labels))

    print("Beginning Training Pipeline...")
    train_indices = list(range(len(x_train)))
    
    weight_history = {} 
    TOTAL_EPOCHS = 20000

    for epoch in range(TOTAL_EPOCHS):
        current_lr = 0.001
        random.shuffle(train_indices)
        loss_total = 0
        for idx in train_indices:
            i, j = int(x_train[idx][0]), int(x_train[idx][1])
            network.embed_and_forward(i, j)
            network.softmax()
            loss_total += network.cross_entropy_loss(y_train[idx])
            
            # Pass the calibrated learning rate and label cleanly
            network.backpropagate(int(y_train[idx]))
            network.adamw_step(current_lr)
        
        # Log the raw weight matrix every 500 epochs
        if epoch % 500 == 0 or epoch == TOTAL_EPOCHS - 1:
            weight_history[epoch] = {
                'W1': extract_w1_matrix(network),
                'E':  np.array(network.E),
                'norm': sum(w.strength**2
                            for L in [network.hiddenLayer, network.outputLayer]
                            for n in L.neurons for w in n.weights),
            }

        
        # Every 500 epochs, run an audit on both train and test metrics
        if (epoch + 1) % 500 == 0:
            # 1. Calculate Live Training Accuracy
            train_score = 0
            for t in range(len(x_train)):
                i, j = int(x_train[t][0]), int(x_train[t][1])
                network.embed_and_forward(i, j)
                network.softmax()
                if np.argmax(network.softMaxOutput) == y_train[t]:
                    train_score += 1
            
            # 2. Calculate Live Test Accuracy
            test_score = 0
            for k in range(len(x_test)):
                i, j = int(x_test[k][0]), int(x_test[k][1])
                network.embed_and_forward(i, j)
                network.softmax()
                if np.argmax(network.softMaxOutput) == y_test[k]:
                    test_score += 1
            
            avg_train_loss = loss_total / len(x_train)
            train_accuracy = (train_score / len(x_train)) * 100
            test_accuracy = (test_score / len(x_test)) * 100
            
            # Print metrics side-by-side
            print(f"Epoch {epoch+1:5d}/{TOTAL_EPOCHS} | Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:6.2f}% | Test Acc: {test_accuracy:6.2f}%")
            
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(network, f)
    with open(HISTORY_PATH, 'wb') as f:
        pickle.dump(weight_history, f)
    print("Network weights and complete training history compiled and saved.")

