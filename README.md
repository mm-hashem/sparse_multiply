# RISC-V Sparse Challenge

Implementation of Compressed Sparse Row (CSR) extraction and sparse matrix-vector multiplication without dynamic allocation.

**Time Complexity:** O(rows * cols)

## Approach

The implementation:
1. Scans a dense row-major matrix
2. Extracts non-zero elements into CSR format using caller-provided buffers.
3. Computes the matrix-vector producct y = A * x during extraction
4. Writes the result directly into a caller-provided output buffer during calculations.

The algorithm performs extraction and multiplication in a single pass.

## Build

```bash
gcc -lm -o run challenge.c
./run
```

## Potential future optimizations:
- Separate CSR extraction and SpMV for SIMD/vector acceleration using RVV