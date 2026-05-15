#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// =========================================================
// FUNCTION PROTOTYPE
// =========================================================
void sparse__multiply(
    int rows,
    int cols,
    const double* A,
    const double* x,
    int* out_nnz,
    double* values,
    int* col_indices,
    int* row_ptrs,
    double* y
);

// =========================================================
// USER IMPLEMENTATION
// =========================================================
/**
 * @brief Converts a dense matrix to CSR format and computes a matrix-vector product.
 *
 * @details This function compresses a dense matrix 'A' into the Compressed Sparse Row (CSR)
 *          format representation (values, col_indices, row_ptrs) while
 *          simultaneously calculating the total number of non-zero elements.
 *          It then immediately uses this sparse data to compute
 *          the matrix-vector multiplication y = A * x.
 *
 * @param[in]  rows         Number of rows in the dense matrix A.
 * @param[in]  cols         Number of columns in the dense matrix A.
 * @param[in]  A            Pointer to the continuous dense input matrix array of size (rows * cols).
 * @param[in]  x            Pointer to the input vector array of size (cols).
 * @param[in,out] out_nnz   Pointer to an integer that tracks and outputs the total number of non-zero elements.
 *                          @note This must be initialized to 0 by the caller before execution.
 * @param[out] values       Pre-allocated array to store the non-zero elements of the matrix.
 * @param[out] col_indices  Pre-allocated array to store the column indices of the non-zero elements.
 * @param[out] row_ptrs     Pre-allocated array of size (rows + 1) to store the row index pointers for CSR.
 * @param[out] y            Pre-allocated output vector array of size (rows) to store the result of the multiplication.
 *
 * @note No additional dynamic memory allocations were made.
 *
 * @pre Pointers must not be null and output arrays must be large enough to hold all non-zero elements.
 * @post The output arrays will be populated with the CSR representation and multiplication result.
 *
 * @author Mahmoud Hashem
 * @date 2026-05-16
 */
void sparse_multiply(
    int rows, int cols, const double* A, const double* x,
    int* out_nnz, double* values, int* col_indices, int* row_ptrs,
    double* y
) {
    row_ptrs[0] = 0;

    for (int r = 0; r < rows; r++) {
        y[r] = 0.0;

        for (int c = 0; c < cols; c++) {
            double nz_elm = A[r * cols + c];

            if (nz_elm) {
                values[*out_nnz] = nz_elm;
                col_indices[*out_nnz] = c;

                y[r] += nz_elm * x[c];

                (*out_nnz)++;
            }
        }
        row_ptrs[r + 1] = *out_nnz;
    }
}

// =========================================================
// TEST HARNESS
// =========================================================
int main(void) {
    srand(time(NULL));
    
    const int num_iterations = 100;
    int passed_count = 0;

    for (int iter = 0; iter < num_iterations; ++iter) {
        int rows = rand() % 41 + 5;
        int cols = rand() % 41 + 5;
        double density = 0.05 + (rand() / (double) RAND_MAX) * 0.35;
        
        size_t mat_sz = (size_t) rows * cols;

        double* A = calloc(mat_sz, sizeof(double));
        for (size_t i = 0; i < mat_sz; ++i) {
            if (((double) rand() / RAND_MAX) < density) {
                A[i] = ((double) rand() / RAND_MAX) * 20.0 - 10.0;
            }
        }

        double* values = malloc(mat_sz * sizeof(double));
        int* col_indices = malloc(mat_sz * sizeof(int));
        int* row_ptrs = malloc((rows + 1) * sizeof(int));
        double* x = malloc(cols * sizeof(double));
        double* y_user = malloc(rows * sizeof(double));
        double* y_ref = calloc(rows, sizeof(double));
        int out_nnz = 0;

        for (int i = 0; i < cols; ++i) {
            x[i] = ((double) rand() / RAND_MAX) * 20.0 - 10.0;
        }

        for (int i = 0; i < rows; ++i) {
            double sum = 0.0;
            for (int j = 0; j < cols; ++j) {
                sum += A[i * cols + j] * x[j];
            }
            y_ref[i] = sum;
        }

        sparse_multiply(rows, cols, A, x, &out_nnz, values, col_indices, row_ptrs, y_user);

        double max_err = 0.0;
        int passed = 1;
        for (int i = 0; i < rows; ++i) {
            double diff = fabs(y_user[i] - y_ref[i]);
            double tol = 1e-7 + 1e-7 * fabs(y_ref[i]); // Mixed absolute/relative tolerance
            if (diff > tol) {
                max_err = fmax(max_err, diff);
                passed = 0;
            }
        }

        if (passed) {
            passed_count++;
        }

        printf(
            "Iter %2d [%3dx%3d, density=%.2f, nnz=%4d]: %s (Max error: %.2e)\n",
            iter, rows, cols, density, out_nnz, passed ? "PASS" : "FAIL", max_err
        );

        free(A);
        free(values);
        free(col_indices);
        free(row_ptrs);
        free(x);
        free(y_user);
        free(y_ref);
    }

    printf(
        "\n%s (%d/%d iterations passed)\n",
        passed_count == num_iterations ? "All tests passed!" : "Some tests failed.",
        passed_count, num_iterations
    );
           
    return passed_count == num_iterations ? 0 : 1;
}
