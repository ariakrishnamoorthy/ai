import math
import random
import numpy as np
import os
import pickle

# Import custom structural modules from the same directory
from finitefields import reduce_field, FIELD_ELEMENTS
from finitefieldann import Neuron, Layer, Network3

with open('field_network.pkl', 'rb') as f:
    network = pickle.load(f)

print("--- failures ---")
for i in range(8):
    for j in range(8):
        network.embed_and_forward(i, j); network.softmax()
        pred = int(np.argmax(network.softMaxOutput))
        true = FIELD_ELEMENTS.index(reduce_field(mod=3, maxdeg=1, equal=[2],
                    elem1=FIELD_ELEMENTS[i], elem2=FIELD_ELEMENTS[j]))
        if pred != true:
            print(f"({i},{j}) pred {pred} true {true}")

# --- Block 2: do different-product pairs collide to the same embedding sum? ---
print("--- collisions in E[i]+E[j] ---")
E = np.array(network.E)
sums = {}
for i in range(8):
    for j in range(8):
        s = tuple(np.round(E[i] + E[j], 1))
        true = FIELD_ELEMENTS.index(reduce_field(mod=3, maxdeg=1, equal=[2],
                    elem1=FIELD_ELEMENTS[i], elem2=FIELD_ELEMENTS[j]))
        sums.setdefault(s, []).append(((i, j), true))
for s, entries in sums.items():
    if len(set(t for _, t in entries)) > 1:
        print("COLLISION:", entries)

# Sanity: is this the network we think it is?
print("weight decay _wd:", network._wd)
print("adam step count _t:", network._t)

# The decisive check: recompute FULL accuracy right here on the loaded net
import numpy as np
from finitefields import reduce_field, FIELD_ELEMENTS
correct = 0
for i in range(8):
    for j in range(8):
        network.embed_and_forward(i, j); network.softmax()
        pred = int(np.argmax(network.softMaxOutput))
        true = FIELD_ELEMENTS.index(reduce_field(mod=3, maxdeg=1, equal=[2],
                    elem1=FIELD_ELEMENTS[i], elem2=FIELD_ELEMENTS[j]))
        correct += (pred == true)
print("full-table accuracy of loaded network:", correct/64)

print("--- 1 * e_j (identity) ---")
for j in range(8):
    network.embed_and_forward(0, j); network.softmax()
    pred = int(np.argmax(network.softMaxOutput))
    print(f"1 * e{j}: pred {pred}, true {j}  {'OK' if pred==j else 'X'}")

mism = 0
for i in range(8):
    for j in range(8):
        network.embed_and_forward(i, j); network.softmax()
        a = int(np.argmax(network.softMaxOutput))
        network.embed_and_forward(j, i); network.softmax()
        b = int(np.argmax(network.softMaxOutput))
        if a != b: mism += 1
print(f"commutativity: {mism}/64 pairs disagree with transpose")




def findorder(i):
    elem = FIELD_ELEMENTS[i]
    ans = elem
    ord = 1
    while (ans != [1,0]):
        ans = reduce_field(mod=3, maxdeg=1, equal=[2], elem1 = ans, elem2 = elem)
        ord += 1
    return ord

def findgenerator():
    prim = 0
    ord_list = []
    for i in range(8):
        elem_ord = findorder(i)
        if (elem_ord == 8):
            ord_list.append(i)
    return ord_list
        

def reorderrows(i):
    roworder = []
    gen = findgenerator()[i]
    elem = FIELD_ELEMENTS[gen]
    ans = [1,0]
    for i in range(8):
        ind = FIELD_ELEMENTS.index(ans)
        roworder.append(ind)
        ans = reduce_field(mod=3, maxdeg=1, equal=[2], elem1 = ans, elem2 = elem)
    
    return roworder

def reorderembedding(i):
    embed = network.E
    roworder = reorderrows(i)
    newE = [[0 for _ in range(8)] for _ in range(8)]
    for i in range(8):
        j = roworder[i]
        newE[i] = embed[j]
    return newE

def findsimilarity(i, j, matrix):
    dot = np.dot(matrix[i], matrix[j])
    magnitudei = math.sqrt(sum(x**2 for x in matrix[i]))
    magnitudej = math.sqrt(sum(x**2 for x in matrix[j]))
    return (dot/(magnitudei * magnitudej))


def cosinesimilarity(i):
    matrix = reorderembedding(i)
    c = [[0 for _ in range(8)] for _ in range(8)]
    for i in range(8):
        for j in range(8):
            x = findsimilarity(i,j, matrix)
            x = float(x)
            c[i][j] = x
    return c

def full_accuracy(network):
    correct = 0
    for i in range(8):
        for j in range(8):
            network.embed_and_forward(i, j)
            network.softmax()
            pred = int(np.argmax(network.softMaxOutput))
            true = FIELD_ELEMENTS.index(reduce_field(mod=3, maxdeg=1, equal=[2],
                        elem1=FIELD_ELEMENTS[i], elem2=FIELD_ELEMENTS[j]))
            correct += (pred == true)
    return correct / 64

print(full_accuracy(network))
        
def fourier_power(gen_index):
    # 1. clock-ordered embedding as a numpy array (8 elements x 8 coordinates)
    E_clock = np.array(reorderembedding(gen_index))

    # 2. accumulate power at each of the 8 frequencies
    total_power = np.zeros(8)

    # 3. loop over the 8 coordinates (columns)
    for col in range(8):
        sequence = E_clock[:, col]          # the 8 values of this coordinate, around the clock
        coeffs   = np.fft.fft(sequence)     # Fourier transform -> 8 complex numbers
        power    = np.abs(coeffs) ** 2      # squared magnitude = power at each frequency
        total_power += power                # add this coordinate's power to the running total

    return total_power




print(findorder(0))
print(findgenerator())


for i in range(1):
    newE = reorderembedding(i)
    print(len(newE), len(newE[0]))   # should be 8 8
    for row in newE:
        print([round(x, 2) for x in row])

    print("This is now cosine similarity")

    c = cosinesimilarity(i)
    for row in c:
        print([round(x, 2) for x in row])

    by_dist = {d: [] for d in range(5)}
    for a in range(8):
        for b in range(8):
            d = min((a-b) % 8, (b-a) % 8)   # circular distance 0..4
            by_dist[d].append(c[a][b])
    for d in range(5):
        print(f"distance {d}: avg cosine {np.mean(by_dist[d]):+.3f}")

print("---------------------")
# run it for your best generator (index 0 = generator 4)
p = fourier_power(0)
for freq in range(8):
    print(f"frequency {freq}: power {p[freq]:.2f}")
