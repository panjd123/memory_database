#include "utils.cuh"
#include "SSB.cuh"
#include "TPCH.cuh"
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    // int device;
    // cudaGetDevice(&device);
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, device);
    // std::cout << "SM count: " << prop.multiProcessorCount << std::endl;
    {
        using namespace TPCH;
        auto args = generate(argc, argv);
        benchmark(args);
    }
}