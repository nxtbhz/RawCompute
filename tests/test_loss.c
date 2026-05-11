#include "common.h"

int main(void)
{
    printf("=== LOSS FUNCTIONS TESTS ===\n\n");

    // Test 1: Cross Entropy Loss
    printf("1. CROSS ENTROPY LOSS\n");
    float predictions_data[2][3] = {{0.7, 0.2, 0.1}, {0.1, 0.8, 0.1}};

    float targets_data[2][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};

    RMatrix *pred_ce = r_create_matrix(2, 3);
    RMatrix *targets_ce = r_create_matrix(2, 3);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            pred_ce->data[RMatrixIDX(i, j, 3)] = predictions_data[i][j];
            targets_ce->data[RMatrixIDX(i, j, 3)] = targets_data[i][j];
        }
    }

    float ce_loss = r_cross_entropy(pred_ce, targets_ce);
    printf("Cross Entropy Loss: %.6f\n", ce_loss);
    printf("Expected: ~0.356207 (computed from: -(log(0.7) + log(0.8))/2)\n\n");

    r_free_matrix(pred_ce);
    r_free_matrix(targets_ce);

    // Test 2: Binary Cross Entropy Loss
    printf("2. BINARY CROSS ENTROPY LOSS\n");
    float predictions_bin[3][2] = {{0.9, 0.1}, {0.2, 0.8}, {0.6, 0.4}};

    float targets_bin[3][2] = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}};

    RMatrix *pred_bce = r_create_matrix(3, 2);
    RMatrix *targets_bce = r_create_matrix(3, 2);

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            pred_bce->data[RMatrixIDX(i, j, 2)] = predictions_bin[i][j];
            targets_bce->data[RMatrixIDX(i, j, 2)] = targets_bin[i][j];
        }
    }

    float bce_loss = r_bin_cross_entropy(pred_bce, targets_bce);
    printf("Binary Cross Entropy Loss: %.6f\n", bce_loss);
    printf("Expected: small value close to 0 (good predictions)\n\n");

    r_free_matrix(pred_bce);
    r_free_matrix(targets_bce);

    // Test 3: Mean Squared Error Loss
    printf("3. MEAN SQUARED ERROR LOSS\n");
    float predictions_mse_data[2][4] = {{2.5, 3.1, 1.9, 4.2}, {1.8, 2.9, 3.2, 2.1}};

    float targets_mse_data[2][4] = {{2.0, 3.0, 2.0, 4.0}, {2.0, 3.0, 3.0, 2.0}};

    RMatrix *pred_mse = r_create_matrix(2, 4);
    RMatrix *targets_mse = r_create_matrix(2, 4);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            pred_mse->data[RMatrixIDX(i, j, 4)] = predictions_mse_data[i][j];
            targets_mse->data[RMatrixIDX(i, j, 4)] = targets_mse_data[i][j];
        }
    }

    float mse_loss = r_mse_loss(pred_mse, targets_mse);
    printf("Mean Squared Error Loss: %.6f\n", mse_loss);
    printf("Expected: ~0.04625 (average of squared differences)\n\n");

    r_free_matrix(pred_mse);
    r_free_matrix(targets_mse);

    // Test 4: Mean Absolute Error Loss
    printf("4. MEAN ABSOLUTE ERROR LOSS\n");
    float predictions_mae_data[2][3] = {{1.5, 2.2, 3.1}, {0.9, 1.8, 2.9}};

    float targets_mae_data[2][3] = {{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}};

    RMatrix *pred_mae = r_create_matrix(2, 3);
    RMatrix *targets_mae = r_create_matrix(2, 3);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            pred_mae->data[RMatrixIDX(i, j, 3)] = predictions_mae_data[i][j];
            targets_mae->data[RMatrixIDX(i, j, 3)] = targets_mae_data[i][j];
        }
    }

    float mae_loss = r_mae_loss(pred_mae, targets_mae);
    printf("Mean Absolute Error Loss: %.6f\n", mae_loss);
    printf("Expected: ~0.216667 (average of absolute differences)\n\n");

    r_free_matrix(pred_mae);
    r_free_matrix(targets_mae);

    printf("=== ALL LOSS TESTS COMPLETED ===\n");

    return 0;
}
