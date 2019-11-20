#include <iostream>
#include "cuda_runtime.h"
#define BLOCK_DIM 32

template<class T>
class MatrixHost {
    T *values;
    long matrix_size;
public:
    explicit MatrixHost(long n) : matrix_size(n) {
        values = new T[matrix_size * matrix_size* sizeof(T)];
    }
    void Initialize_Matrix(void){
        time_t t;
        srand((unsigned) time(&t));
        for (int j = 0; j < matrix_size*matrix_size; j++) {
            values[j] = (T) (rand() & 0xFF) / 10.0f;
        }
    }

    T * get_values(void) {
        return values;
    }

    long get_size() {
        return sizeof(T) * matrix_size * matrix_size;
    }

    ~MatrixHost() {
        delete []values;
    }
};

template<class T>
class MatrixDevice{
    T *values;
    long matrix_size;
public:
    explicit MatrixDevice(long n) : matrix_size(n) {
        cudaMalloc((T **) &values, matrix_size* matrix_size * sizeof(T));
    };

    T *get_values(void) {
        return values;
    }

    ~MatrixDevice() {
        cudaFree(values);
    }
};

template <typename type>
__global__ void IJK(const type *A, const type *B,type *C,int matrix_size) {
    int i = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int j = blockIdx.y * BLOCK_DIM + threadIdx.y;
    float temp = 0;
    for (int k = 0; k < matrix_size ; k++) {
        temp += A[i*matrix_size + k] * B[k*matrix_size+ j];
    }
    C[i*matrix_size+ j] = temp;
}

template <typename type>
__global__ void IKJ(const type *A, const type *B,type *C,int matrix_size) {
    int i = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int k = blockIdx.y * BLOCK_DIM + threadIdx.y;
    float reg = A[i*matrix_size+ k];
    for (int j = 0;  j < matrix_size; j++) {
        atomicAdd(&C[i*matrix_size+j],reg * B[k*matrix_size +j]);
    }
}


template <typename type>
__global__ void KIJ(const type *A, const type *B,type *C,int matrix_size) {
    int k = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int i = blockIdx.y * BLOCK_DIM + threadIdx.y;
    float reg = A[i * matrix_size + k];
    for (int j = 0; j < matrix_size; j++) {
        atomicAdd(&C[i*matrix_size + j] , reg *B[k*matrix_size +j]);
    }
}


template <typename type>
__global__ void JIK(const type *A, const type *B,type *C,int matrix_size) {
    int j = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int i = blockIdx.y * BLOCK_DIM + threadIdx.y;
    float temp = 0;
    for (int k = 0; k < matrix_size ; k++) {
        temp += A[i*matrix_size + k] * B[k*matrix_size + j];
    }
    C[i*matrix_size + j] = temp;
}


template <typename type>
__global__ void JKI(const type *A, const type *B,type *C,int matrix_size) {
    int j = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int k = blockIdx.y * BLOCK_DIM + threadIdx.y;
    float reg = B[k*matrix_size + j];
    for (int i = 0; i < matrix_size ; i++) {
        atomicAdd(&C[i*matrix_size+j],A[i*matrix_size + k] * reg);
    }
}

template <typename type>
__global__ void KJI(const type *A, const type *B,type *C,int matrix_size) {
    int k = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int j = blockIdx.y * BLOCK_DIM + threadIdx.y;
    float reg = B[k*matrix_size + j];
    for (int i = 0; i < matrix_size ; i++) {
        atomicAdd(&C[i*matrix_size+j],A[i*matrix_size + k] * reg);
    }

}


template <typename type>
void multiply_on_host(const type *A, const type *B,type *C,int matrix_size) {
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            float r = 0;
            for (int k = 0; k < matrix_size; k++) {
                r += A[i*matrix_size + k] * B[k*matrix_size+ j];
            }
            C[i*matrix_size+ j] = r;
        }
    }
}
template <typename type>
void checkResult(type *host_array, type *device_array, const long matrix_size) {
    double epsilon = 0.1;
    for (long i = 0; i < matrix_size; i++) {
        if (abs(device_array[i] - host_array[i]) > epsilon) {
            std::cout <<i;
        }
    }
    std::cout << "SUCCESS\n";
}

int main(int argc,char** argv) {
    int matrix_size =  atoi(argv[1]);

    MatrixHost<float> A_h (matrix_size);
    MatrixHost<float> B_h (matrix_size);
    MatrixHost<float> C_h(matrix_size);
    MatrixHost<float>C_from_d (matrix_size);



    A_h.Initialize_Matrix();
    B_h.Initialize_Matrix();

    multiply_on_host<float>(A_h.get_values(),B_h.get_values(),C_h.get_values(),matrix_size);

    MatrixDevice <float> A_d(matrix_size);
    MatrixDevice <float> B_d(matrix_size);
    MatrixDevice <float> C_d(matrix_size);

    cudaMemcpy(A_d.get_values(), A_h.get_values(), A_h.get_size(), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d.get_values(), B_h.get_values(), A_h.get_size(), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_DIM,BLOCK_DIM,1);
    dim3 grid ((matrix_size+block.x-1)/block.x,(matrix_size+block.y-1)/block.y,1 );
    KJI<float><<<grid,block>>>(A_d.get_values(),B_d.get_values(),C_d.get_values(),matrix_size);

    cudaMemcpy(C_from_d.get_values(), C_d.get_values(), A_h.get_size(), cudaMemcpyDeviceToHost);


    checkResult<float>(C_h.get_values(),C_from_d.get_values(),matrix_size);

}