#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(call) \
  if ((call) != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(call)); \
    exit(EXIT_FAILURE); \
  }

__device__ unsigned int lcg_random(unsigned int* state) {
  const unsigned int a = 1664525u;
  const unsigned int c = 1013904223u;
  *state = a * (*state) + c;
  return *state;
}

__global__ void seq_read_float4_kernel(const float4* __restrict__ d_data, int N, float* __restrict__ d_sink) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  float val = 0.f;
  for (int i = tid; i < N; i += stride) {
    float4 v = d_data[i];
    val += v.x + v.y + v.z + v.w;
  }
  if (tid == 0) d_sink[0] = val;
}

__global__ void seq_read_float_kernel(const float* __restrict__ d_data, int N, float* __restrict__ d_sink) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  float val = 0.f;
  for (int i = tid; i < N; i += stride) {
    val += d_data[i];
  }
  if (tid == 0) d_sink[0] = val;
}

__global__ void rand_read_float_lcg_kernel(const float* __restrict__ d_data, int N, float* __restrict__ d_sink) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  unsigned int state = tid + 12345;
  float val = 0.f;
  for (int i = tid; i < N; i += stride) {
    unsigned int idx = lcg_random(&state) % N;
    val += d_data[idx];
  }
  if (tid == 0) d_sink[0] = val;
}

__global__ void rand_read_float4_lcg_kernel(const float4* __restrict__ d_data, int N, float* __restrict__ d_sink) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  unsigned int state = tid + 12345;
  float val = 0.f;
  for (int i = tid; i < N; i += stride) {
    unsigned int idx = lcg_random(&state) % N;
    float4 v = d_data[idx];
    val += v.x + v.y + v.z + v.w;
  }
  if (tid == 0) d_sink[0] = val;
}

int main() {
  const int threads = 256;
  const int blocks = 256;

  const int test_sizes_mb[] = {4, 8, 16, 32, 64, 128, 256, 512};
  const int num_tests = sizeof(test_sizes_mb) / sizeof(test_sizes_mb[0]);

  float4* d_data_float4 = nullptr;
  float* d_data_float = nullptr;
  float* d_sink = nullptr;

  CHECK(cudaMalloc(&d_sink, sizeof(float)));

  printf("DataSizeMB,TestType,BandwidthGBps\n");

  for (int test_i = 0; test_i < num_tests; ++test_i) {
    int mb = test_sizes_mb[test_i];
    size_t bytes_float = (size_t)mb * 1024 * 1024;
    int num_float = (int)(bytes_float / sizeof(float));
    int num_float4 = num_float / 4;

    if (d_data_float4) cudaFree(d_data_float4);
    if (d_data_float) cudaFree(d_data_float);

    CHECK(cudaMalloc(&d_data_float4, sizeof(float4) * num_float4));
    CHECK(cudaMalloc(&d_data_float, sizeof(float) * num_float));

    CHECK(cudaMemset(d_data_float4, 0, sizeof(float4) * num_float4));
    CHECK(cudaMemset(d_data_float, 0, sizeof(float) * num_float));
    CHECK(cudaMemset(d_sink, 0, sizeof(float)));

    seq_read_float4_kernel<<<blocks, threads>>>(d_data_float4, num_float4, d_sink);
    seq_read_float_kernel<<<blocks, threads>>>(d_data_float, num_float, d_sink);
    rand_read_float_lcg_kernel<<<blocks, threads>>>(d_data_float, num_float, d_sink);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    float ms;

    // seq_float4
    CHECK(cudaEventRecord(start));
    seq_read_float4_kernel<<<blocks, threads>>>(d_data_float4, num_float4, d_sink);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    float gbps_seq_float4 = (bytes_float / (1024.f*1024.f*1024.f)) / (ms / 1000.f);
    printf("%d,seq_float4,%.3f\n", mb, gbps_seq_float4);

    // seq_float
    CHECK(cudaEventRecord(start));
    seq_read_float_kernel<<<blocks, threads>>>(d_data_float, num_float, d_sink);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    float gbps_seq_float = (bytes_float / (1024.f*1024.f*1024.f)) / (ms / 1000.f);
    printf("%d,seq_float,%.3f\n", mb, gbps_seq_float);

    // rand_float4
    CHECK(cudaEventRecord(start));
    rand_read_float4_lcg_kernel<<<blocks, threads>>>(d_data_float4, num_float4, d_sink);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    float gbps_rand_float4 = (bytes_float / (1024.f*1024.f*1024.f)) / (ms / 1000.f);
    printf("%d,rand_float4,%.3f\n", mb, gbps_rand_float4);

    // rand_float
    CHECK(cudaEventRecord(start));
    rand_read_float_lcg_kernel<<<blocks, threads>>>(d_data_float, num_float, d_sink);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    float gbps_rand_float = (bytes_float / (1024.f*1024.f*1024.f)) / (ms / 1000.f);
    printf("%d,rand_float,%.3f\n", mb, gbps_rand_float);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
  }

  cudaFree(d_data_float4);
  cudaFree(d_data_float);
  cudaFree(d_sink);

  return 0;
}
