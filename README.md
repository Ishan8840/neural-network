# Simple Neural Network from Scratch in Python

This repository contains a **fully-connected neural network implemented from scratch in Python** using only NumPy, designed to classify spiral data. The network demonstrates **forward propagation, backward propagation, activation functions, loss calculation, and training** without relying on high-level frameworks like TensorFlow or PyTorch.

---

## Features

* **Dense (Fully-Connected) Layers**
  Implements weight initialization with He initialization and bias handling.

* **Activation Functions**

  * ReLU (Rectified Linear Unit)
  * Softmax

* **Loss Functions**

  * Categorical Crossentropy
  * Combined Softmax + Categorical Crossentropy for efficient backward pass

* **Training Loop**

  * Gradient descent optimization with dynamic learning rate decay
  * Calculates both **loss** and **accuracy** during training

* **Data**
  Uses `nnfs.datasets.spiral_data` for a multi-class classification task.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/simple-neural-network.git
cd simple-neural-network
```

2. Install dependencies:

```bash
pip install nnfs numpy
```

> `nnfs` is used for dataset generation and consistent initialization.

---

## Usage

Run the main Python script:

```bash
python neural_network.py
```

You will see the training progress printed every 300 epochs:

```
Epoch 0: Loss=1.0986, Accuracy=0.333
Epoch 300: Loss=0.9254, Accuracy=0.573
...
```

The network learns to classify points in a spiral dataset into 3 classes.

---

## Project Structure

```
.
├── neural_network.py      # Main neural network implementation and training loop
└── README.md
```

---

## Example Output

* Initial loss and accuracy: ~1.0986 and 0.33
* Final loss after training: decreases significantly
* Accuracy improves as training progresses

---

## Notes

* Learning rate decreases over time using a simple decay formula:

```python
learning_rate = 1.0 / (1 + 0.0001 * epoch)
```

* Training is performed for 10,001 epochs by default.

---

## Contributing

Contributions are welcome! You can:

* Add new activation functions (e.g., Sigmoid, LeakyReLU)
* Add support for more datasets
* Implement more optimization algorithms (e.g., Adam, RMSProp)

---

## License

This project is licensed under the MIT License.

---
