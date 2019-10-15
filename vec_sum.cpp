#include <iostream>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cmath>
#include <chrono>


template <class T>
class ArrayHost{
    T * values;
    long elem_numb;
public:
    explicit ArrayHost(long n):elem_numb(n) {
        values = new T[elem_numb* sizeof(T)];
    }
    void Init(void){
        time_t t;
        srand((unsigned) time(&t));
        for (int i = 0; i < elem_numb; i++) {
            values[i] = (T) (rand() & 0xFF) / 10.0f;
        }
    }
    T* get_values(void){
        return values;
    }
    long get_size(){
        return sizeof(T)*elem_numb;
    }
    ~ArrayHost(){
        delete []values;
    }
};


template <class T>
class ArrayDevice{
    T * values;
    long elem_numb;
public:
    explicit ArrayDevice(long n):elem_numb(n){
        cudaMalloc((T**)&values,elem_numb*sizeof(T));


    };
    T* get_values(void){
        return values;
    }
    ~ArrayDevice(){
        cudaFree(values);
    }
};


template <typename type>
void checkResult(type *host_array,type *device_array,const long array_size){
    double epsilon  = 0.1;
    for (long i=0;i<array_size;i++){
        if(device_array[i] - host_array[i] > epsilon){
            std::cout<<"ERROR in %ld position"<<i;
            break;
        }
    }
    printf("SUCCESS");
}

template <typename type>
void sum_on_host(type *A,type *B, type *C,long N){
    for(long i =0;i<N;i++){
        C[i] = A[i] + B[i];
    }
}

template <typename type>
__global__ void sum_on_device(type *A, type *B, type *C) {
    int i = threadIdx.x  + blockIdx.x*blockDim.x;
    C[i] = A[i] + B[i];
}


int main(int argc, char** argv){

    using namespace std;
    //long values_num = atoi(argv[1]);
    long values_num = 1030;
    //cout<<values_num;
    int dev = 0;
    ArrayHost<int> A_h (values_num);
    ArrayHost<int> B_h (values_num);

    A_h.Init();
    B_h.Init();

    ArrayHost<int> HostSum(values_num);
    ArrayHost<int> Device_to_HostSum(values_num);


    sum_on_host(A_h.get_values(),B_h.get_values(),HostSum.get_values(),values_num);


    ArrayDevice<int> A_d(values_num);
    ArrayDevice<int> B_d(values_num);
    ArrayDevice<int> C_d(values_num);

    cudaMemcpy(A_d.get_values(),A_h.get_values(),A_h.get_size(),cudaMemcpyHostToDevice);
    cudaMemcpy(B_d.get_values(),B_h.get_values(),B_h.get_size(),cudaMemcpyHostToDevice);

    int max_threads_for_block = 512;
    dim3 block(max_threads_for_block);
    dim3 grid ((values_num / max_threads_for_block) + 1);

    sum_on_device  <<<grid,block>>>(A_d.get_values(),B_d.get_values(),C_d.get_values());

    cudaMemcpy(Device_to_HostSum.get_values(),C_d.get_values(),A_h.get_size(),cudaMemcpyDeviceToHost);

    checkResult(HostSum.get_values(),Device_to_HostSum.get_values(),values_num);


};