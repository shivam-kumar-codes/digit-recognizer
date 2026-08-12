"""
Feedforward neural network with mini-batch stochastic gradient descent,
implemented from scratch in numpy.

Follows Michael Nielsen's "Neural Networks and Deep Learning", chapters 1-2:
http://neuralnetworksanddeeplearning.com/chap1.html
(equation numbers referenced below are from that book)
"""

import random

import numpy as np


def sigmoid(z):
    # The sigmoid squashes any real-valued input into (0, 1), which is what
    # lets us interpret a neuron's output as an "activation" — how strongly
    # it's firing. eq. 3.
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    # Derivative of the sigmoid, sigma'(z) = sigma(z)(1 - sigma(z)). We need
    # this during backprop because the chain rule requires how sensitive a
    # neuron's *output* is to its *weighted input* z — that sensitivity is
    # exactly this derivative. Used in eq. BP1.
    s = sigmoid(z)
    return s * (1.0 - s)


class Network:
    """
    A fully-connected feedforward network.

    `sizes` gives the number of neurons in each layer, e.g. [784, 30, 10]
    for MNIST: 784 input pixels, one hidden layer of 30 neurons, and 10
    output neurons (one per digit 0-9).

    Weights and biases are initialized randomly (Gaussian, mean 0, std 1).
    This isn't the most principled init (Nielsen improves on it later in
    the book), but it's what chapters 1-2 use and is enough to train a
    reasonable MNIST classifier.
    """

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes

        # No biases for the input layer — it has no incoming weighted sums,
        # it just holds the input activations. Each bias vector has shape
        # (n, 1) — a column vector, one bias per neuron in that layer.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # weights[l] connects layer l to layer l+1, and has shape
        # (n_{l+1}, n_l): rows = neurons in the next layer, cols = neurons
        # in the current layer. That shape is exactly what makes
        # `np.dot(w, a) + b` (eq. 22) work as ordinary matrix multiplication.
        self.weights = [
            np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])
        ]

    def feedforward(self, a):
        """
        Return the network's output for input activation `a` (shape (n, 1)).

        Implements eq. 22: a' = sigma(w a + b), applied layer by layer. Each
        layer's output activation becomes the next layer's input.
        """
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(
        self,
        training_data,
        epochs,
        mini_batch_size,
        eta,
        test_data=None,
    ):
        """
        Train using mini-batch stochastic gradient descent.

        `training_data` is a list of (x, y) tuples: x is a (784, 1) input
        vector, y is a (10, 1) one-hot vector (the eq. 6 cost needs the
        target in the same shape as the network's output).

        We use *mini-batches* rather than the full dataset (batch GD) or a
        single example (pure online GD) as a middle ground: mini-batches
        approximate the true gradient (over all training examples) cheaply,
        while still updating weights often enough to converge quickly. This
        is the core idea behind eq. 20-21.

        If `test_data` is given, accuracy is evaluated after every epoch —
        useful for tracking progress, but it slows training down.
        """
        training_data = list(training_data)
        n = len(training_data)

        if test_data is not None:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            # Shuffling each epoch means each mini-batch is a fresh random
            # sample of the training set, so the gradient estimate it
            # produces isn't biased by a fixed ordering.
            random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data is not None:
                print(
                    f"Epoch {j}: {self.evaluate(test_data)} / {n_test}"
                )
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta):
        """
        Update weights and biases via a single step of gradient descent,
        using backprop to compute the gradient for this one mini-batch.

        Implements eq. 20-21:
            w -> w - (eta/m) * sum_x(dC_x/dw)
            b -> b - (eta/m) * sum_x(dC_x/db)
        where m = len(mini_batch). We approximate the true gradient (which
        would require summing over *all* training examples) by averaging
        over just this mini-batch — cheap enough to compute often, close
        enough to the true gradient to make steady progress.
        """
        # Accumulators for the summed gradient across the mini-batch, same
        # shapes as the params themselves so we can add into them directly.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        m = len(mini_batch)
        self.weights = [
            w - (eta / m) * nw for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / m) * nb for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        """
        Return (nabla_b, nabla_w), the gradient of the quadratic cost (eq. 6)
        for a single training example (x, y), as lists of numpy arrays
        matching self.biases / self.weights shapes.

        Backprop works backwards from the output error because the chain
        rule naturally composes that way: the error at layer l depends on
        the error at layer l+1 (eq. BP2), so we can't compute any layer's
        error until we've computed the one after it.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # --- forward pass ---
        # We stash every activation and weighted input (z) along the way
        # because backprop's equations (BP1-BP4) need both: activations to
        # compute weight gradients (eq. BP4), z's to compute how each
        # layer's error propagates backwards (eq. BP2).
        activation = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # --- backward pass ---
        # eq. BP1: error at the output layer =
        #   dC/da * sigma'(z), where dC/da = (a - y) for quadratic cost (eq. 6).
        # This measures how much the cost would change if we nudged the
        # output layer's weighted input z — the natural starting point for
        # backprop since it's the one layer whose error we can compute
        # directly from the cost function.
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        # eq. BP4: dC/dw = delta * a_in. The gradient w.r.t. a weight
        # scales with how active the neuron feeding into it was — an
        # inactive input neuron can't be "blamed" for the error no matter
        # how wrong the weight is.
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Walk backwards through the remaining layers. l=2 means "2nd to
        # last layer", l=3 "3rd to last", etc. — Python's negative indexing
        # makes this natural without renumbering layers.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            # eq. BP2: propagate the error backwards one layer. w^T maps
            # "error contribution per next-layer neuron" back onto "error
            # contribution per this-layer neuron" — the transpose flips the
            # forward-pass direction of the weight matrix.
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def cost_derivative(self, output_activations, y):
        """
        dC/da for the quadratic cost C = (1/2)||y - a||^2 (eq. 6).
        The derivative w.r.t. a single output activation a is (a - y) —
        the network's error on that output neuron, sign and all.
        """
        return output_activations - y

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the network's highest-
        activation output neuron matches the true label.

        Unlike training data, test labels here are plain integers (0-9)
        rather than one-hot vectors, since we only need argmax comparison,
        not a cost gradient.
        """
        test_results = [
            (np.argmax(self.feedforward(x)), y) for (x, y) in test_data
        ]
        return sum(int(x == y) for (x, y) in test_results)

    def save(self, path):
        """Persist weights and biases so predict.py can reload them without retraining."""
        np.savez(
            path,
            sizes=np.array(self.sizes),
            **{f"weight_{i}": w for i, w in enumerate(self.weights)},
            **{f"bias_{i}": b for i, b in enumerate(self.biases)},
        )

    @classmethod
    def load(cls, path):
        """Reconstruct a Network from weights saved via `save`."""
        data = np.load(path)
        sizes = data["sizes"].tolist()
        net = cls(sizes)
        net.weights = [data[f"weight_{i}"] for i in range(net.num_layers - 1)]
        net.biases = [data[f"bias_{i}"] for i in range(net.num_layers - 1)]
        return net
