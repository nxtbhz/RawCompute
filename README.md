# RawCompute - Neural Networks from Scratch in C

A lightweight, educational framework for building neural networks in pure C, designed to understand the fundamentals of deep learning without external dependencies.

## Project Overview

RawCompute demonstrates how neural networks are built from first principles, providing low-level implementations of:
- **Matrix operations** (multiplication, transposition, indexing)
- **Vector operations** (dot product, vector-matrix multiplication, bias addition)
- **Dense layers** (fully connected layers with weights and biases)

This project is perfect for learning how neural network computations work at the C level, before moving to high-level frameworks like TensorFlow or PyTorch.

## Project Status

🚧 **Work in Progress** - This project is actively being developed as I work through [Neural Networks from Scratch in Python](https://nnfs.io/). Each chapter introduces new concepts that are being translated and implemented in C to deepen understanding of neural network fundamentals.

**Current Progress:**
- ✅ Basic tensor operations (matrices and vectors)
- ✅ Dense layers with forward propagation
- ✅ Activation functions
- ✅ Loss functions
- ✅ Optimization
- ⏳ Backpropagation
- ⏳ Additional layer types

## Core Components

### 1. Matrix Operations (`r_matrix.h/c`)

Implements a 2D matrix structure and essential operations:

- **`RMatrix *r_create_matrix(size_t rows, size_t cols)`** - Allocate a matrix
- **`void r_free_matrix(RMatrix *matrix)`** - Free matrix memory
- **`RMatrix *r_mat_mul(RMatrix *mat1, RMatrix *mat2)`** - Standard matrix multiplication
- **`RMatrix *r_mat_transpose(RMatrix *matrix)`** - Matrix transposition
- **`void r_print_matrix(RMatrix *m, char *name)`** - Pretty-print a matrix

### 2. Vector Operations (`r_vector.h/c`)

Implements a 1D vector structure and operations:

- **`RVector *r_create_vector(size_t size)`** - Allocate a vector
- **`void r_free_vector(RVector *vector)`** - Free vector memory
- **`RVector *r_mat_vec_mul(RMatrix *matrix, RVector *vector)`** - Matrix-vector multiplication
- **`float r_vec_dot(RVector *v1, RVector *v2)`** - Dot product of two vectors
- **`void r_add_bias(RVector *vector, float bias)`** - Add bias to a vector
- **`void r_print_vector(RVector *vector, char *name)`** - Pretty-print a vector

### 3. Dense Layer (`r_layer_dense.h/c`)

Implements a fully connected neural network layer:

```c
typedef struct r_layer_dense_t {
    RMatrix *weights;   // Layer weights (neurons × inputs)
    RVector *biases;    // Layer biases (neurons)
} RLayerDense;
```

- **`RLayerDense *r_create_layer(size_t n_inputs, size_t n_neurons)`** - Create a dense layer
- **`void r_free_layer(RLayerDense *layer)`** - Free layer memory
- **`RMatrix *r_layer_forward(RLayerDense *layer, RMatrix *inputs)`** - Forward pass computation

The forward pass computes: `output = input × weights^T + bias`

## Building and Running

### Build
```bash
gcc gcc tests/test_activation.c src/r_matrix.c src/r_activation.c src/r_vector.c src/r_layer_dense.c -I./include/ -o test_neuron -lm
```

### Run
```bash
./test_neuron
```

### Example Output
The test program demonstrates three scenarios:
1. **Single Neuron** - Basic neuron computation
2. **Layer with 3 Neurons** - Multiple neurons processing the same input
3. **Batch Processing** - Processing multiple samples through a dense layer

## Usage Example

```c
#include <rc/r_matrix.h>
#include <rc/r_vector.h>
#include <rc/r_layer_dense.h>

// Create a dense layer with 4 inputs and 3 neurons
RLayerDense *layer = r_create_layer(4, 3);

// Create batch of 2 samples with 4 features each
RMatrix *input = r_create_matrix(2, 4);
// ... populate input data ...

// Forward pass
RMatrix *output = r_layer_forward(layer, input);

// Print results
r_print_matrix(output, "Layer Output");

// Cleanup
r_free_matrix(input);
r_free_matrix(output);
r_free_layer(layer);
```

## Key Features

- **Pure C implementation** - No external dependencies, standard C library only
- **Educational focus** - Clear, readable code to understand neural network mechanics
- **Memory management** - Explicit allocation and deallocation for learning purposes
- **Basic tensor operations** - Essential linear algebra for neural networks
- **Batch processing support** - Compute multiple samples simultaneously

## Technical Details

### Memory Layout

Matrices and vectors use flat C arrays with row-major ordering:
- Matrix element `[i, j]` is stored at index `i * cols + j`
- Uses the macro: `RMatrixIDX(i, j, cols)`

### Data Type

All computations use `float` (32-bit floating-point).

## Limitations and Future Improvements

Currently planned features being implemented:
- Activation functions (ReLU, Sigmoid, Tanh, Softmax)
- Loss functions (MSE, Cross-Entropy)
- Optimization algorithms (SGD, Adam, RMSprop)
- Backpropagation implementation
- Support for multiple layer types (Convolutional, Recurrent, etc.)

**Inspiration:** This project follows the curriculum and concepts from [Neural Networks from Scratch in Python](https://nnfs.io/) but implements them in C for educational purposes.

## License

This project is provided as-is for educational purposes.
