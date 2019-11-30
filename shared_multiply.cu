#include <iostream>
#include "cuda_runtime.h"
#include <cublas_v2.h>
#include <curand.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <algorithm>
#include <cstdlib>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#define BLOCK_SIZE 32
#define BLOCK_DIM 32

#define TILE_DIM 32
#define BLOCK_ROWS 8



float rand_r(){
    return (float(rand()%1024))/512.0;

}


template<class T>
class MatrixHost {
    T *values;
    long matrix_size;
public:
    explicit MatrixHost(long n) : matrix_size(n) {
        values = new T[matrix_size * matrix_size * sizeof(T)];
    }

    void Initialize_Matrix(void) {
        time_t t;
        srand((unsigned) time(&t));
        for (int j = 0; j < matrix_size * matrix_size; j++) {
            values[j] = (T) (rand() & 0xFF) / 10.0f;
        }
    }

    T *get_values(void) {
        return values;
    }

    long get_size() {
        return sizeof(T) * matrix_size * matrix_size;
    }

    ~MatrixHost() {
        delete[]values;
    }
};


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


__global__ void shared_multiply ( float *a, float *b,  float *c,long n )
{
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    long aBegin = n * BLOCK_SIZE * by, aEnd = aBegin + n - 1;
    long bBegin = BLOCK_SIZE * bx, aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * n;
    float sum = 0.0f;
    __shared__ float as [BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float bs [BLOCK_SIZE][BLOCK_SIZE];
    for ( long ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep )
    {   as [tx][ty] = a [ia + n * ty + tx]; bs [tx][ty] = b [ib + n * ty + tx];
        __syncthreads ();
        for ( int k = 0; k < BLOCK_SIZE; k++ ) sum += as [k][ty] * bs [tx][k];
        __syncthreads ();
    }
    c [aBegin + bBegin + ty * n + tx] = sum;
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

    //std::generate(A_h.begin(), A_h.end(), rand_r());
    //std::generate(B_h.begin(), B_h.end(), rand_r());
    thrust::sequence(B_h.begin(), B_h.end());
    thrust::sequence(A_h.begin(), A_h.end());

    multiply_on_host<float>(thrust::raw_pointer_cast(&A_h[0]),thrust::raw_pointer_cast(&B_h[0]),thrust::raw_pointer_cast(&C_h[0]),matrix_size);

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
    checkResult<float >(thrust::raw_pointer_cast(&C_h[0]),thrust::raw_pointer_cast(&C_from_kernel[0]),matrix_size);

}