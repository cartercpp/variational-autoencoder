# Variational Autoencoder (VAE) from Scratch

A variational autoencoder implemented in **C++26**, without external machine-learning libraries. The demo trains on **MNIST** handwritten digits and shows the original image alongside its reconstruction directly in the terminal.

## How it works

The encoder predicts a mean (\(\mu\)) and log variance (\(\log \sigma^2\)) for a 16-dimensional latent space. A latent vector is sampled with the reparameterization trick:

\[
z = \mu + \exp(0.5\log\sigma^2) \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0,I)
\]

The decoder maps that latent vector back to 784 pixel intensities using a sigmoid output layer. Training minimizes **binary cross-entropy reconstruction loss + KL divergence** to a standard normal prior.

- **Encoder:** 784 → 128 → 64 → 32 (16 means + 16 log variances)
- **Decoder:** 16 → 64 → 128 → 784
- **Activations:** ReLU hidden layers, linear encoder output, sigmoid decoder output
- **Training:** per-image stochastic gradient descent, learning rate 0.001 for both networks
- **Backpropagation:** implemented manually, including gradients through reparameterization

The model uses custom vector and matrix types and the C++ standard library, including `<simd>`.

## Build

Requires **GCC 16** with C++26 `<simd>` support (as used in this project), or a compatible compiler and standard library.

```bash
g++-16 -std=c++26 -O3 -march=native -pthread \
  '-Dpre(...)=' '-Dpost(...)=' \
  main.cpp neural_network.cpp variational_autoencoder.cpp -o vae
```

The `-Dpre(...)=` and `-Dpost(...)=` flags disable the experimental contract annotations for this build.

## Run

The repository includes the uncompressed MNIST training-image file `train-images-idx3-ubyte`. Run:

```bash
./vae train-images-idx3-ubyte 100000
```

The arguments are **MNIST image file** and **number of training steps**. Without arguments, the program uses `train-images-idx3-ubyte` and **180,000 steps**.

If you need to download the MNIST images separately:

```bash
curl -LO https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
gzip -dk train-images-idx3-ubyte.gz
```

The program reads the uncompressed IDX image file; labels are not needed.

## Terminal visualization

The demo displays **Original** and **Reconstruction** side by side in a true-color terminal layout of at most **80 columns**. Each displayed original image is the actual input passed to `regenerate()` for the displayed reconstruction.

- During training, the view refreshes every **500 steps** and shows **BCE per pixel** for the displayed pair.
- After training, it cycles through MNIST examples and their reconstructions every **1.5 seconds**.
- Press **Ctrl+C** to exit.

The displayed BCE is a reconstruction metric, **not** the full VAE objective (which also includes the KL term). Reconstructions are stochastic because `encode()` samples a latent vector.

**Note:** This demo does not save or load trained weights, so each run trains a fresh model.

## Code layout

- `variational_autoencoder.hpp/.cpp` — encoding, decoding, reparameterization and training
- `neural_network.hpp/.cpp` — feedforward network and manual backpropagation
- `math_vector.hpp`, `matrix.hpp` — vector and matrix operations
- `activation_functions.hpp` — activation function options
- `main.cpp` — MNIST loader, training loop and terminal visualization
