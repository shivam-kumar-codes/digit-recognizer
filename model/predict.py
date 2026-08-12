"""
CLI script: load trained weights and predict a single digit.

Two input modes:
  1. An image file:      uv run model/predict.py path/to/digit.png
  2. An MNIST test index: uv run model/predict.py --test-index 42
     (useful for verifying the trained model works before any frontend
     exists to draw real images with)

Per PROJECT_SPEC.md's build order, this script is how Phase 1 gets
verified end-to-end before the FastAPI backend is wrapped around it.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from network import Network

DEFAULT_WEIGHTS_PATH = Path(__file__).parent / "weights" / "mnist_weights.npz"


def load_image_as_input(path, invert):
    """
    Convert an arbitrary image file into the (784, 1), [0, 1]-normalized
    column vector the network's input layer expects (same preprocessing
    as mnist_loader._to_vectors, applied to a single external image).
    """
    img = Image.open(path).convert("L").resize((28, 28))
    pixels = np.asarray(img, dtype=np.float64) / 255.0
    if invert:
        # MNIST digits are white strokes on a black background. A photo or
        # a canvas export drawn with the opposite convention (dark strokes
        # on light paper) needs flipping to match what the network was
        # trained on.
        pixels = 1.0 - pixels
    return pixels.reshape(784, 1)


def main():
    parser = argparse.ArgumentParser(description="Predict a digit with the trained network.")
    parser.add_argument("image", type=Path, nargs="?", help="Path to a digit image file.")
    parser.add_argument(
        "--test-index",
        type=int,
        help="Instead of an image file, predict MNIST test example at this index.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert pixel values (use for dark-on-light images; MNIST is light-on-dark).",
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    args = parser.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"No weights found at {args.weights}. Run train.py first.")
    net = Network.load(args.weights)

    if args.test_index is not None:
        from mnist_loader import load_data

        _, test_data = load_data()
        x, true_label = test_data[args.test_index]
        print(f"True label: {true_label}")
    elif args.image is not None:
        x = load_image_as_input(args.image, args.invert)
    else:
        parser.error("Provide either an image path or --test-index.")

    output = net.feedforward(x)
    prediction = int(np.argmax(output))
    confidence = float(output[prediction, 0])
    print(f"Predicted digit: {prediction} (confidence {confidence:.2%})")


if __name__ == "__main__":
    main()
