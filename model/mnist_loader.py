"""
MNIST loading, reshaped into the format Network expects.

We use torchvision purely as a download/decode convenience for the raw
MNIST files — no torch tensors or torch training machinery are used
anywhere in the network itself.
"""

import numpy as np
from torchvision import datasets


def vectorized_result(j):
    """
    Turn a digit label (0-9) into a (10, 1) one-hot column vector, e.g.
    3 -> [0,0,0,1,0,0,0,0,0,0]^T.

    Training needs this shape because the quadratic cost (eq. 6) and
    backprop's output-layer error (eq. BP1) compare the target directly
    against the network's (10, 1) output activation — a plain integer
    label isn't something you can subtract from a vector.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def _to_vectors(dataset):
    """
    Convert a torchvision MNIST dataset into a list of (784, 1) input
    vectors, each pixel scaled to [0, 1].

    We flatten each 28x28 image into a single column vector because the
    network's input layer (eq. 22, first application) treats the image as
    784 independent input activations, not a 2D grid — the network has no
    notion of pixel adjacency.
    """
    images = dataset.data.numpy().astype(np.float64) / 255.0
    images = images.reshape(len(dataset), 784, 1)
    labels = dataset.targets.numpy()
    return images, labels


def load_data(data_dir="data"):
    """
    Download (if needed) and return MNIST as (training_data, test_data),
    matching the shapes Network.SGD / Network.evaluate expect:

    - training_data: list of (x, y) with x a (784, 1) input vector and y
      a (10, 1) one-hot vector (needed to compute a cost gradient).
    - test_data: list of (x, y) with x a (784, 1) input vector and y a
      plain int label (evaluate() only needs argmax comparison).
    """
    train_raw = datasets.MNIST(data_dir, train=True, download=True)
    test_raw = datasets.MNIST(data_dir, train=False, download=True)

    train_images, train_labels = _to_vectors(train_raw)
    test_images, test_labels = _to_vectors(test_raw)

    training_data = [
        (x, vectorized_result(y)) for x, y in zip(train_images, train_labels)
    ]
    test_data = list(zip(test_images, test_labels))

    return training_data, test_data
