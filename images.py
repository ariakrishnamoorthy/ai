import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num, rows, cols)
        return images

# Load images (make sure the file is in your working directory)
images = load_mnist_images("train-images-idx3-ubyte.gz")

# Pick an image to display
index = 0
img = images[index]

# Show using matplotlib
plt.imshow(img, cmap='gray', interpolation='nearest')
plt.title(f"MNIST Image #{index}")
plt.axis("off")
plt.show()

