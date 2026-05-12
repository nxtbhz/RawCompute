#include <rc/r_loss_grad.h>

void r_cross_entropy_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, RMatrix *grad)
{
    float norm = (float)pred->rows;

    for (size_t i = 0; i < pred->rows; i++)
    {
        for (size_t j = 0; j < pred->cols; j++)
        {
            float p = pred->data[RMatrixIDX(i, j, pred->cols)];
            float y = real->data[RMatrixIDX(i, j, real->cols)];
            grad->data[RMatrixIDX(i, j, grad->cols)] = -y / (norm * (p + EPSILON));
        }
    }
}

void r_bin_cross_entropy_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, RMatrix *grad)
{
    float norm = (float)(pred->rows * pred->cols);

    for (size_t i = 0; i < pred->rows; i++)
    {
        for (size_t j = 0; j < pred->cols; j++)
        {
            float p = pred->data[RMatrixIDX(i, j, pred->cols)];
            float y = real->data[RMatrixIDX(i, j, real->cols)];
            grad->data[RMatrixIDX(i, j, grad->cols)] =
                (-(y / (p + EPSILON)) + ((1.0f - y) / (1.0f - p + EPSILON))) / norm;
        }
    }
}

void r_cat_cross_entropy_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, RMatrix *grad)
{
    float norm = (float)pred->rows;

    for (size_t i = 0; i < pred->rows; i++)
    {
        for (size_t j = 0; j < pred->cols; j++)
        {
            float p = pred->data[RMatrixIDX(i, j, pred->cols)];
            float y = real->data[RMatrixIDX(i, j, real->cols)];
            grad->data[RMatrixIDX(i, j, grad->cols)] = -y / (norm * (p + EPSILON));
        }
    }
}

void r_mse_loss_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, RMatrix *grad)
{
    float norm = (float)(pred->rows * pred->cols);

    for (size_t i = 0; i < pred->rows; i++)
    {
        for (size_t j = 0; j < pred->cols; j++)
        {
            float p = pred->data[RMatrixIDX(i, j, pred->cols)];
            float y = real->data[RMatrixIDX(i, j, real->cols)];
            grad->data[RMatrixIDX(i, j, grad->cols)] = 2.0f * (p - y) / norm;
        }
    }
}

void r_mae_loss_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, RMatrix *grad)
{
    float norm = (float)(pred->rows * pred->cols);

    for (size_t i = 0; i < pred->rows; i++)
    {
        for (size_t j = 0; j < pred->cols; j++)
        {
            float diff = pred->data[RMatrixIDX(i, j, pred->cols)] - real->data[RMatrixIDX(i, j, real->cols)];
            float sign = (diff > 0.0f) ? 1.0f : (diff < 0.0f) ? -1.0f : 0.0f;
            grad->data[RMatrixIDX(i, j, grad->cols)] = sign / norm;
        }
    }
}

void r_bin_focal_loss_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, float gamma, float alpha,
                           RMatrix *grad)
{
    float norm = (float)(pred->rows * pred->cols);

    for (size_t i = 0; i < pred->rows; i++)
    {
        for (size_t j = 0; j < pred->cols; j++)
        {
            float p = pred->data[RMatrixIDX(i, j, pred->cols)];
            float y = real->data[RMatrixIDX(i, j, real->cols)];
            float omp = 1.0f - p;
            float lp = logf(p + EPSILON);
            float lop = logf(omp + EPSILON);

            float d_t1 = alpha * y * (-gamma * powf(omp, gamma - 1.0f) * lp + powf(omp, gamma) / (p + EPSILON));

            float d_t2 =
                (1.0f - alpha) * (1.0f - y) * (gamma * powf(p, gamma - 1.0f) * lop - powf(p, gamma) / (omp + EPSILON));

            grad->data[RMatrixIDX(i, j, grad->cols)] = -(d_t1 + d_t2) / norm;
        }
    }
}

void r_cat_focal_loss_grad(const RNONNULL RMatrix *pred, const RNONNULL RMatrix *real, float gamma, RMatrix *grad)
{
    float norm = (float)pred->rows;

    for (size_t i = 0; i < pred->rows; i++)
    {
        for (size_t j = 0; j < pred->cols; j++)
        {
            float y = real->data[RMatrixIDX(i, j, real->cols)];
            float g = 0.0f;

            if (y > 0.0f)
            {
                float p = pred->data[RMatrixIDX(i, j, pred->cols)];
                float omp = 1.0f - p;
                g = -(y / norm) *
                    (-gamma * powf(omp, gamma - 1.0f) * logf(p + EPSILON) + powf(omp, gamma) / (p + EPSILON));
            }

            grad->data[RMatrixIDX(i, j, grad->cols)] = g;
        }
    }
}
