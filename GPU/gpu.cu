#include <cuda_runtime.h>
#include <iostream>

int main(int argc, char** argv) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "SM count: " << prop.multiProcessorCount << std::endl;
    std::cout << "L2 size: " << prop.l2CacheSize / 1024 << "KB" <<std::endl;
    std::cout << "L1 size (per block) :" << prop.sharedMemPerBlock / 1024 << "KB" << std::endl;
    std::cout << "L1 size (per SM) :" << prop.sharedMemPerMultiprocessor / 1024 << "KB" << std::endl;
}