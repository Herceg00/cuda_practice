#include <iostream>
#include "cuda_runtime.h"
#include <cublas_v2.h>
#include <curand.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#define BLOCK_DIM 32
#define TILE_DIM 32
#define BLOCK_ROWS 8


void Initialize_Matrix(float* matrix_vector, long vector_size){ //vector_size = N*N
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
    curandGenerateUniform(prng, matrix_vector,vector_size);

}


void cublas_multiply(const float *A, const float *B, float *C, long matrix_size) {
    //int dima = matrix_size,dimB =matrix_size , dimC =matrix_size;
    cudaEvent_t start_multiply, stop_multiply;
    float time_multiply;
    cudaEventCreate(&start_multiply);
    cudaEventCreate(&stop_multiply);
    float alpha = 1.0;
    float betta = 0.0;
    float *alph_point = &alpha;
    float *betta_point = &betta;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEventRecord(start_multiply);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size, matrix_size, matrix_size, alph_point, A, matrix_size, B,
                matrix_size, betta_point, C, matrix_size);
    cudaEventRecord(stop_multiply);
    cudaEventSynchronize(stop_multiply);
    cudaEventElapsedTime(&time_multiply, start_multiply, stop_multiply);
    cublasDestroy(handle);
    std::cout << "GPU PERFORMANCE: " << (matrix_size * matrix_size * matrix_size * 1000) / ((double) time_multiply)
              << "FLOPS\n";
}


int main(int argc, char **argv) {

    long matrix_size = atol(argv[1]);


    //thrust::host_vector<float> A_h(matrix_size*matrix_size);


    //thrust::sequence(A_h.begin(), A_h.end());

    //thrust::host_vector<float> B_h(matrix_size*matrix_size);

    //thrust::sequence(B_h.begin(), B_h.end());

    
    thrust::device_vector<float> A_d(matrix_size*matrix_size);
    thrust::device_vector<float> B_d(matrix_size*matrix_size);
    Initialize_Matrix(thrust::raw_pointer_cast(&A_d[0]),matrix_size*matrix_size);
    Initialize_Matrix(thrust::raw_pointer_cast(&B_d[0]),matrix_size*matrix_size);



    thrust::device_vector<float> C_d(matrix_size*matrix_size);



    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((matrix_size + block.x - 1) / block.x, (matrix_size + block.y - 1) / block.y);

    cublas_multiply(thrust::raw_pointer_cast(&A_d[0]), thrust::raw_pointer_cast(&B_d[0]), thrust::raw_pointer_cast(&C_d[0]), matrix_size);
    cublas_multiply(thrust::raw_pointer_cast(&A_d[0]), thrust::raw_pointer_cast(&B_d[0]), thrust::raw_pointer_cast(&C_d[0]), matrix_size);


}