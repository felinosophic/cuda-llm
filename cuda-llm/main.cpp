#include "addvectors.h"
#include "helpers.h"
#include "cuda_runtime.h"
#include <iostream>


int main()
{
    int N = 1024;
    float* h_A, * h_B, * h_C;
    h_A = new float[N];
    h_B = new float[N];
    h_C = new float[N];

    for (int i = 0; i < N; ++i)
    {
        h_A[i] = i;
        h_B[i] = 2.0 * i;
    }
    float* d_A, * d_B, * d_C;

    checkCudaErrors(cudaMalloc((void**)&d_A, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_B, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_C, N * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));


    addVectors(d_C, d_A, d_B, N);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));


    for (int i = 0; i < N; ++i)
    {
        if (h_C[i] != i + 2.0 * i) {
            std::cerr << "Failed!" << "\n";
            return 1;
        };

    }
    std::cout << "Success!!" << "\n";


    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}
