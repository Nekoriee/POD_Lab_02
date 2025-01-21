#include <omp.h>
#include <immintrin.h>
#include <thread>
// cols = rows = 0 (mod 8)
#define Cols 256
#define Rows 256

void add_matrix(double* A, const double* B, const double* C, size_t cols, size_t rows) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            A[i + j*rows] = B[i + j*rows] + C[i + j * rows];
        }
    }
}

void add_matrix_256(double* A, const double* B, const double* C, size_t cols, size_t rows) {
    for (size_t i = 0; i < rows; i += 4) {
        for (size_t j = 0; j < cols; j++) {
            __m256d b = _mm256_loadu_pd(&B[i + j * rows]);
            __m256d c = _mm256_loadu_pd(&C[i + j * rows]);
            __m256d a = _mm256_add_pd(b, c);
            _mm256_storeu_pd(&A[i + j * rows], a);
        }
    }
}

#include <iostream>
#include <cstring>
int main(int argc, char** argv)
{
    double* A = (double*)malloc(Cols * Rows * sizeof(double));
    double* B = (double*)malloc(Cols * Rows * sizeof(double));
    double* C = (double*)malloc(Cols * Rows * sizeof(double));
    for (size_t i = 0; i < Rows * Cols; ++i)
        B[i] = -(C[i] = i);
    for (size_t i = 0; i < 2; i++) {
        std::cout << "Thread num: " << i << "\n";
        for (size_t j = 0; j < 20; j++) {
            if (i > 0) {
                double t1 = omp_get_wtime();
                add_matrix_256(A, B, C, Cols, Rows);
                double t2 = omp_get_wtime();
                std::cout << t2 - t1 << "\n";
            }
            else {
                double t1 = omp_get_wtime();
                add_matrix(A, B, C, Cols, Rows);
                double t2 = omp_get_wtime();
                std::cout << t2 - t1 << "\n";
            }

        }
    }
    //for (size_t r = 0; r < Rows; ++r) {
    //    std::cout << "[" << A[r + 0 * Rows];
    //    for (size_t c = 1; c < Cols; ++c)
    //        std::cout << ", " << A[r + c * Rows];
    //    std::cout << "]\n";
    //}
    free(A); free(B); free(C);
    return 0;
}
