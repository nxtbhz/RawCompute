#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>

#define RC_VERSION_MAJOR 0
#define RC_VERSION_MINOR 1
#define RC_VERSION_PATCH 0

#define RC_VERSION_STRING "0.1.0"

#define RC_VERSION_ENCODE(major, minor, patch) \
    ((major) * 10000 + (minor) * 100 + (patch))

#define RC_VERSION \
    RC_VERSION_ENCODE(RC_VERSION_MAJOR, RC_VERSION_MINOR, RC_VERSION_PATCH)


#define RNONNULL
#define RNULLABLE

#define RMatrixIDX(i, j, cols) ((i) * (cols) + (j))

#define EPSILON 1e-7f

