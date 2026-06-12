# MAIN.PY
import math
import random
import numpy as np
import os
import pickle

# Import custom structural modules from the same directory
from finitefields import reduce_field, FIELD_ELEMENTS
from finitefieldann import Neuron, Layer, Network3

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
        for j in range(i,8):
            res = reduce_field(mod=3, maxdeg=1, equal=[2],
                               elem1=FIELD_ELEMENTS[i],
                               elem2=FIELD_ELEMENTS[j])
            labels.append(FIELD_ELEMENTS.index(res))
            inputs.append([i, j])   # just two integers now
    return np.array(inputs), np.array(labels)

def extract_w1_matrix(network):
    """
    Extracts the weights connecting the 16 inputs to the 64 hidden neurons.
    Returns a 2D NumPy array of shape (64, 16).
    """
    matrix = []
    for neuron in network.hiddenLayer.neurons:
        neuron_weights = [w.strength for w in neuron.weights]
        matrix.append(neuron_weights)
    return np.array(matrix)

# ==========================================
# DATA PIPELINE SETUP
# ==========================================

x_data, y_data = generate_dataset()

# Split dataset into train and test sets cleanly (80/20 split)
indices = list(range(len(x_data)))
random.seed(42)  # For reproducible splits
random.shuffle(indices)

train_idx = indices[:32]
test_idx = indices[32:]

x_train, y_train = x_data[train_idx], y_data[train_idx]
x_test, y_test = x_data[test_idx], y_data[test_idx]

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
    # 16 Inputs (Concatenated One-Hots) -> 64 Hidden -> 8 Outputs (Field Elements)
    input_layer = Layer([Neuron() for _ in range(16)])
    hidden_layer = Layer([Neuron() for _ in range(32)])
    output_layer = Layer([Neuron() for _ in range(8)])

    network = Network3(input_layer, hidden_layer, output_layer)
    network.connect()

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
        if epoch % 500 == 0 or epoch == (TOTAL_EPOCHS - 1):
            weight_history[epoch] = extract_w1_matrix(network)
        
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

    # After training, print the full predicted multiplication table
print("\nPredicted vs True multiplication table:")
print("Format: predicted (true)")
header = "     " + "  ".join(f"e{j}" for j in range(8))
print(header)

correct = 0
total = 0
for i in range(8):
    row = f"e{i}: "
    for j in range(8):
        network.embed_and_forward(i, j)
        network.softmax()
        pred = int(np.argmax(network.softMaxOutput))
        res = reduce_field(mod=3, maxdeg=1, equal=[2],
                           elem1=FIELD_ELEMENTS[i],
                           elem2=FIELD_ELEMENTS[j])
        true = FIELD_ELEMENTS.index(res)
        match = "✓" if pred == true else "✗"
        row += f"{pred}({true}){match} "
        if pred == true:
            correct += 1
        total += 1
    print(row)

print(f"\nFull table accuracy: {correct}/{total} = {correct/total*100:.1f}%")

# ==========================================
#