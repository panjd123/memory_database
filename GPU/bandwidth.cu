#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(call) \
    if ((call) != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(call)); \
        exit(EXIT_FAILURE); \
    }

// 用 float4 读带宽 kernel
__global__ void read_bw_kernel(const float4* __restrict__ d_data, int N, float* __restrict__ d_sink) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        float4 v = d_data[i];
        val += v.x + v.y + v.z + v.w;
    }
    if (tid == 0) d_sink[0] = val;
}

// 用 float4 写带宽 kernel
__global__ void write_bw_kernel(float4* d_data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        float4 v = make_float4(i * 0.5f, i * 0.5f, i * 0.5f, i * 0.5f);
        d_data[i] = v;
    }
}

// 用 float4 读写混合带宽 kernel
__global__ void read_write_bw_kernel(float4* d_data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        float4 v = d_data[i];
        v.x = v.x * 1.1f + 1.0f;
        v.y = v.y * 1.1f + 1.0f;
        v.z = v.z * 1.1f + 1.0f;
        v.w = v.w * 1.1f + 1.0f;
        d_data[i] = v;
    }
}

void test_bandwidth(int num_elements, int mode) {
    // num_elements 表示 float4 个数，不是 float 数
    float4* d_data = nullptr;
    float* d_sink = nullptr;
    CHECK(cudaMalloc(&d_data, sizeof(float4) * num_elements));
    CHECK(cudaMalloc(&d_sink, sizeof(float)));

    CHECK(cudaMemset(d_data, 0, sizeof(float4) * num_elements));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    int threads = 256;
    int blocks = 256;

    // 预热
    switch (mode) {
        case 0: read_bw_kernel<<<blocks, threads>>>(d_data, num_elements, d_sink); break;
        case 1: write_bw_kernel<<<blocks, threads>>>(d_data, num_elements); break;
        case 2: read_write_bw_kernel<<<blocks, threads>>>(d_data, num_elements); break;
    }
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(start));
    switch (mode) {
        case 0: read_bw_kernel<<<blocks, threads>>>(d_data, num_elements, d_sink); break;
        case 1: write_bw_kernel<<<blocks, threads>>>(d_data, num_elements); break;
        case 2: read_write_bw_kernel<<<blocks, threads>>>(d_data, num_elements); break;
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    // 计算访问字节数，float4 每元素16字节
    size_t bytes_accessed = 0;
    if (mode == 0) bytes_accessed = (size_t)num_elements * sizeof(float4);
    else if (mode == 1) bytes_accessed = (size_t)num_elements * sizeof(float4);
    else if (mode == 2) bytes_accessed = (size_t)num_elements * sizeof(float4) * 2;

    float bandwidth = bytes_accessed / (ms / 1000.0f) / (1024.0f * 1024.0f * 1024.0f); // GB/s

    const char* mode_str = (mode == 0) ? "READ" : (mode == 1) ? "WRITE" : "READ+WRITE";

    printf("%s Bandwidth: %.2f GB/s, Data Size: %.2f MB, Time: %.3f ms\n",
           mode_str, bandwidth, bytes_accessed / (1024.0f * 1024.0f), ms);

    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_sink));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
}

int main() {
    // 这里设置 float4 个数，64M * 4float 变成 16M float4，约256MB * 4 = 1GB 访问量
    int num_elements = 64 * 1024 * 1024; // 64M float4, 256MB*4=1GB 内存访问
    printf("Testing GPU memory bandwidth with %d float4 elements (~%.2f MB)\n",
           num_elements, num_elements * sizeof(float4) / (1024.f * 1024.f));
    test_bandwidth(num_elements, 0);
    test_bandwidth(num_elements, 1);
    test_bandwidth(num_elements, 2);
    return 0;
}
