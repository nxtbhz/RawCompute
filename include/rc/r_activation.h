#pragma once

#include <rc/r_types.h>
#include <rc/r_matrix.h>

RMatrix *r_activation_relu(RNONNULL RMatrix *matrix);
RMatrix *r_activation_softmax(RNONNULL RMatrix *matrix);
