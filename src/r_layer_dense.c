#include <rc/r_layer_dense.h>

RLayerDense *r_create_layer(size_t n_inputs, size_t n_neurons)
{
    RLayerDense *layer = malloc(sizeof(RLayerDense));

    layer->biases = r_create_vector(n_neurons);
    layer->weights = r_create_matrix(n_neurons, n_inputs);

    return layer;
}

void r_free_layer(RNONNULL RLayerDense *layer)
{
    r_free_matrix(layer->weights);
    r_free_vector(layer->biases);
    free(layer);
}

RMatrix *r_layer_forward(const RNONNULL RLayerDense *layer, const RNONNULL RMatrix *inputs)
{
    RMatrix *transposed_weights = r_mat_transpose(layer->weights);
    RMatrix *result = r_mat_mul(inputs, transposed_weights);

    r_free_matrix(transposed_weights);

    for (size_t i = 0; i < result->rows; i++)
    {
        // seg fault?
        // r_add_bias(result->data, layer->biases->data[i]);
        for (size_t j = 0; j < result->cols; j++)
        {
            result->data[RMatrixIDX(i, j, result->cols)] += layer->biases->data[j];
        }
    }
    return result;
}
