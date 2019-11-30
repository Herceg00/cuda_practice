#include <iostream>
#include "cuda_runtime.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cstdlib>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#define BLOCK_SIZE 32
#define BLOCK_DIM 32


__global__ void shared_multiply ( float *a, float *b,  float *c,long n )
{
    int blockx = blockIdx.x;
    int blocky = blockIdx.y;
    int threadx = threadIdx.x;
    int thready = threadIdx.y;
    long upper_bound  = blocky * n * BLOCK_SIZE; //встали на строчку, где находится желаемый для копирования в shared память блок
    long lower_bound  = upper_bound + n - 1; //последняя его строчка
    long left_bound = BLOCK_SIZE * blockx;
    float sum = 0.0;
    __shared__ float a_shared [BLOCK_SIZE][BLOCK_SIZE+1];
    __shared__ float b_shared [BLOCK_SIZE][BLOCK_SIZE+1];
    for (long ia = upper_bound, ib = left_bound; ia <= lower_bound; ia += BLOCK_SIZE, ib += n*BLOCK_SIZE )
    {
        a_shared [threadx][thready] = a [ia + n * thready + threadx];
        b_shared [threadx][thready] = b [ib + n * thready + threadx];
        __syncthreads ();
        for ( int k = 0; k < BLOCK_SIZE; k++ ) {
            sum += a_shared [k][thready] * b_shared [threadx][k];
        }
        __syncthreads ();
    }
    c[upper_bound+left_bound + thready*n + threadx] = sum;
}

template<typename type>
void checkResult(type *host_array, type *device_array, long matrix_size) {
    double epsilon = 1;
    for (long i = 0; i < matrix_size; i++) {
        if (abs(device_array[i] - host_array[i]) > epsilon) {
            std::cout << "ERROR " << i<< "  " << host_array[i] <<" "<<device_array[i];
            return;
        }
    }
    std::cout << "SUCCESS\n";
}
template<typename type>
void multiply_on_host(const type *A, const type *B, type *C, long matrix_size) { //JIK mult
    for (int j = 0; j < matrix_size; j++) {
        for (int i = 0; i < matrix_size; i++) {
            type r = 0;
            for (int k = 0; k < matrix_size; k++) {
                r += A[(i*matrix_size) + k] * B[(k * matrix_size) + j];
            }
            C[(i* matrix_size) + j] = r;
        }
    }
}

int main(int argc, char **argv) {

    long matrix_size = atol(argv[1]);

    thrust::host_vector<float> A_h(matrix_size*matrix_size);
    thrust::host_vector<float> B_h(matrix_size*matrix_size);
    thrust::host_vector<float> C_h(matrix_size*matrix_size);
    thrust::host_vector<float> C_from_kernel(matrix_size*matrix_size);

    thrust::sequence(B_h.begin(), B_h.end());
    thrust::sequence(A_h.begin(), A_h.end());

    //multiply_on_host<float>(thrust::raw_pointer_cast(&A_h[0]),thrust::raw_pointer_cast(&B_h[0]),thrust::raw_pointer_cast(&C_h[0]),matrix_size);

    thrust::device_vector<float> A_d = A_h;
    thrust::device_vector<float> B_d = B_h;

    thrust::device_vector<float> C_d(matrix_size*matrix_size);

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((matrix_size + block.x - 1) / block.x, (matrix_size + block.y - 1) / block.y);

    cudaEvent_t start_multiply, stop_multiply;
    float time_multiply;
    cudaEventCreate(&start_multiply);
    cudaEventCreate(&stop_multiply);
    cudaEventRecord(start_multiply);
    shared_multiply<<<grid,block>>>(thrust::raw_pointer_cast(&A_d[0]), thrust::raw_pointer_cast(&B_d[0]), thrust::raw_pointer_cast(&C_d[0]), matrix_size);
    cudaEventRecord(stop_multiply);
    cudaEventSynchronize(stop_multiply);
    cudaEventElapsedTime(&time_multiply, start_multiply, stop_multiply);
    std::cout << "GPU PERFORMANCE: " << (2 * matrix_size * matrix_size * matrix_size * 1000) / ((double) time_multiply)
              << "FLOPS\n";
    thrust::copy(C_d.begin(), C_d.end(), C_from_kernel.begin());
    //checkResult<float >(thrust::raw_pointer_cast(&C_h[0]),thrust::raw_pointer_cast(&C_from_kernel[0]),matrix_size);

}