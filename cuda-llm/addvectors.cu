
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

__global__
void addVectors_kernel(float* C, const float* A, const float* B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
void addVectors(float* C, const float* A, const float* B, int N) {
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    addVectors_kernel <<<numBlocks, threadsPerBlock >>> (C, A, B, N);
}