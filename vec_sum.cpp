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
};


template <class T>
class ArrayDevice{
    T * values;
    long elem_numb;
public:
    explicit ArrayDevice(long n):elem_numb(n){
        values = cudaMalloc((T**)&values,elem_numb*sizeof(T));

    };
    T* get_values(void){
        return values;
    }
};


template <class type>
void checkResult(type *host_array,type *device_array,const long array_size){
    double epsilon  = 1.0E-6;
    for (long i=0;i<array_size;i++){
        if(abs(device_array[i] - host_array[i]) > epsilon){
            std::cout<<"ERROR in %ld position"<<i;
            break;
        }
    }
}

template <class type>
void sum_on_host(type *A,type *B, type *C,long N){
    for(long i =0;i<N;i++){
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char** argv){

    using namespace std;
    //long values_num = atoi(argv[1]);
    long values_num = 4;
    //cout<<values_num;
    int dev = 0;
    ArrayHost<int> A (values_num);
    ArrayHost<int> B (values_num);

    A.Init();
    B.Init();

    ArrayHost<int> HostSum(values_num);
    ArrayHost<int> Device_to_HostSum(values_num);

    ArrayDevice<int> A_d(values_num);
    ArrayDevice<int> B_d(values_num);
    ArrayDevice<int> C_d(values_num);

    cudaMemcpy(A_d.get_values(),A.get_values(),A.get_size(),cudaMemcpyHostToDevice);
    cudaMemcpy(B_d.get_values(),B.get_values(),B.get_size(),cudaMemcpyHostToDevice);
    sum_on_host(A.get_values(),B.get_values(),HostSum.get_values(),values_num);

    checkResult(HostSum.get_values(),Device_to_HostSum.get_values(),values_num);

};
