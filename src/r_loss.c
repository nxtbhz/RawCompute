#include <rc/r_loss.h>

float r_cross_entropy(const RNONNULL RMatrix *matrix, const RNONNULL RMatrix *src)
{
    float total = 0.0f;

    for (size_t i = 0; i < matrix->rows; i++)
    {
        for (size_t j = 0; j < matrix->cols; j++)
        {
            float prediction = matrix->data[RMatrixIDX(i, j, matrix->cols)];
            float correct = src->data[RMatrixIDX(i, j, src->cols)];
            total += correct * log(prediction + EPSILON);
        }
    }

    return -total / src->rows;
}

float r_bin_cross_entropy(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real)
{
    float total = 0.0f;

    for (size_t i = 0; i < real->rows; i++)
    {
        for (size_t j = 0; j < real->cols; j++)
        {
            float currentReal = real->data[RMatrixIDX(i, j, real->cols)];
            float currentPred = pred->data[RMatrixIDX(i, j, pred->cols)];
            total +=
                currentReal * log(currentPred + EPSILON) + (1.0f - currentReal) * log(1.0f - currentPred + EPSILON);
        }
    }

    return -total / (real->cols * real->rows);
}

float r_cat_cross_entropy(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real)
{
    float total = 0.0f;

    for (size_t i = 0; i < real->rows; i++)
    {
        for (size_t j = 0; j < real->cols; j++)
        {
            float prediction = pred->data[RMatrixIDX(i, j, pred->cols)];
            float correct = real->data[RMatrixIDX(i, j, real->cols)];
            total += correct * log(prediction + EPSILON);
        }
    }

    return -total / real->rows;
}

float r_mse_loss(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real)
{
    float result = 0.0f;

    for (size_t i = 0; i < real->rows; i++)
    {
        for (size_t j = 0; j < real->cols; j++)
        {
            float error = (real->data[RMatrixIDX(i, j, real->cols)] - pred->data[RMatrixIDX(i, j, pred->cols)]);
            result += error * error;
        }
    }

    return result / (real->cols * real->rows);
}

float r_mae_loss(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real)
{
    float result = 0.0f;
    for (size_t i = 0; i < real->rows; i++)
    {
        for (size_t j = 0; j < real->cols; j++)
        {
            float error = (real->data[RMatrixIDX(i, j, real->cols)] - pred->data[RMatrixIDX(i, j, pred->cols)]);
            result += fabsf(error);
        }
    }

    return result / (real->cols * real->rows);
}

float r_bin_focal_loss(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, float gamma, float alpha)
{
    float total = 0.0f;

    for (size_t i = 0; i < real->rows; i++)
    {
        for (size_t j = 0; j < real->cols; j++)
        {
            float y = real->data[RMatrixIDX(i, j, real->cols)];
            float p = pred->data[RMatrixIDX(i, j, pred->cols)];

            float term1 = alpha * y * powf(1.0f - p, gamma) * logf(p + EPSILON);

            float term2 = (1.0f - alpha) * (1.0f - y) * powf(p, gamma) * logf(1.0f - p + EPSILON);

            total += term1 + term2;
        }
    }

    return -total / (real->cols * real->rows);
}

float r_cat_focal_loss(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, float gamma)
{
    float total = 0.0f;

    for (size_t i = 0; i < real->rows; i++)
    {
        for (size_t j = 0; j < real->cols; j++)
        {
            float prediction = pred->data[RMatrixIDX(i, j, pred->cols)];
            float correct = real->data[RMatrixIDX(i, j, real->cols)];

            if (correct > 0.0f)
            {
                float mod_factor = powf(1.0f - prediction, gamma);
                total += correct * mod_factor * logf(prediction + EPSILON);
            }
        }
    }

    return -total / real->rows;
}
