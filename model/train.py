"""
CLI script: train the numpy Network on MNIST and save the resulting weights.

Usage:
    uv run model/train.py [--epochs 30] [--mini-batch-size 10] [--eta 3.0]
"""

import argparse
from pathlib import Path

from mnist_loader import load_data
from network import Network

DEFAULT_WEIGHTS_PATH = Path(__file__).parent / "weights" / "mnist_weights.npz"


def main():
    parser = argparse.ArgumentParser(description="Train the numpy MNIST network.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--mini-batch-size", type=int, default=10)
    # eta=3.0 is the learning rate Nielsen uses for this [784, 30, 10]
    # architecture with quadratic cost — large enough to converge in a
    # reasonable number of epochs without diverging.
    parser.add_argument("--eta", type=float, default=3.0)
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=30,
        help="Size of the single hidden layer (architecture: [784, hidden, 10]).",
    )
    parser.add_argument("--weights-out", type=Path, default=DEFAULT_WEIGHTS_PATH)
    args = parser.parse_args()

    print("Loading MNIST...")
    training_data, test_data = load_data()
    print(f"{len(training_data)} training examples, {len(test_data)} test examples")

    net = Network([784, args.hidden_size, 10])
    net.SGD(
        training_data,
        epochs=args.epochs,
        mini_batch_size=args.mini_batch_size,
        eta=args.eta,
        test_data=test_data,
    )

    final_correct = net.evaluate(test_data)
    accuracy = final_correct / len(test_data)
    print(f"Final test accuracy: {final_correct} / {len(test_data)} ({accuracy:.2%})")

    args.weights_out.parent.mkdir(parents=True, exist_ok=True)
    net.save(args.weights_out)
    print(f"Saved weights to {args.weights_out}")


if __name__ == "__main__":
    main()
