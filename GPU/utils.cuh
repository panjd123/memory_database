#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>
#include <random>
#include <ranges>
#include "argparse.hpp"
#include "tabulate.hpp"

constexpr int8_t DIM_NULL = -1;     // NULL value for dimension vector
constexpr int16_t GROUP_NULL = -1;  // NULL value for group ID

inline double rand_x(double min, double max) {
    return min + static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / (max - min)));
}

template <typename T>
__global__ void generate_rand_x_kernel(T* d_array, T min, T max, int size, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandState state;
    curand_init(seed, id, 0, &state);  // Initialize random state
    for (int i = id; i < size; i += stride) {
        if constexpr (std::is_integral_v<T>) {
            d_array[i] = min + curand(&state) % (max - min);
        } else if constexpr (std::is_floating_point_v<T>) {
            d_array[i] = min + (max - min) * curand_uniform(&state);
        } else {
            // Handle other types if necessary
        }
    }
}

template <typename T>
cudaStream_t generate_rand_x(T* d_array, T min, T max, int size, unsigned long seed) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    int blockSize = 256;  // Number of threads per block
    int n = 8;
    int numBlocks = (size + (blockSize * n) - 1) / (blockSize * n);  // Calculate number of blocks needed
    generate_rand_x_kernel<<<numBlocks, blockSize, 0, stream>>>(d_array, min, max, size, seed);
    return stream;
}

template <typename T = int>
__global__ void generate_bitmap_kernel(T* d_array, float selectRate, int groupNum, int size, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandState state;
    curand_init(seed, id, 0, &state);  // Initialize random state
    for (int i = id; i < size; i += stride) {
        d_array[i] = (curand_uniform(&state) <= selectRate) ? static_cast<T>(curand(&state) % groupNum) : DIM_NULL;  // Generate bitmap value
    }
}

template <typename T = int>
cudaStream_t generate_bitmap(T* d_array, float selectRate, int groupNum, int size, unsigned long seed) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    int blockSize = 256;  // Number of threads per block
    int n = 8;
    int numBlocks = (size + (blockSize * n) - 1) / (blockSize * n);  // Calculate number of blocks needed
    generate_bitmap_kernel<<<numBlocks, blockSize, 0, stream>>>(d_array, selectRate, groupNum, size, seed);
    return stream;
}

constexpr unsigned long seed = 2025;  // Seed for random number generation

template <typename T, typename Func = std::function<T(int)>>
T* Gather(Func func, int count) {
    T* h_array = new T[count];
    T* d_array;
    cudaMalloc(&d_array, count * sizeof(T));
    for (int i = 0; i < count; ++i) {
        h_array[i] = func(i);
    }
    cudaMemcpy(d_array, h_array, count * sizeof(T), cudaMemcpyHostToDevice);
    delete[] h_array;
    return d_array;
}

struct Table {
    float selectRate;
    bool isBitmap;
    int groupNum;
    int8_t* h_dimVec;
    int8_t* d_dimVec;
    int** h_foreignKeys;
    int** h_d_foreignKeys;
    int** d_foreignKeys;
    int foreignKeyNum;
    int size;

    int valueVecNum;
    float** h_valueVecs;
    float** h_d_valueVecs;
    float** d_valueVecs;
    Table() = default;
    Table(float m_selectRate, bool m_isBitmap, int m_groupNum, int m_size, std::vector<int> m_ForeignKeyRange = {}, int m_valueVecNum = 0)
        : selectRate(m_selectRate), isBitmap(m_isBitmap), groupNum(m_groupNum), size(m_size), valueVecNum(m_valueVecNum) {
        if (selectRate >= 0) {
            // h_dimVec = new int8_t[size];
            cudaMalloc(&d_dimVec, size * sizeof(int8_t));
            if (isBitmap) {
                // #pragma omp parallel for
                // for (int i = 0; i < size; ++i) {
                //     h_dimVec[i] = rand_x(0.0, 1.0) <= selectRate ? (int8_t)rand_x(0, groupNum - 1) : DIM_NULL;
                // }
                generate_bitmap<int8_t>(d_dimVec, selectRate, groupNum, size, seed);
                // generate_rand_x<int8_t>(d_dimVec, int8_t(0), int8_t(groupNum - 1), size, seed);
            } else {
                // #pragma omp parallel for
                //                 for (int i = 0; i < size; ++i) {
                //                     h_dimVec[i] = rand_x(0.0, 1.0) <= selectRate ? 0 : DIM_NULL;
                //                 }
                generate_bitmap<int8_t>(d_dimVec, selectRate, 1, size, seed);
            }
            // cudaMemcpy(d_dimVec, h_dimVec, size * sizeof(int8_t), cudaMemcpyHostToDevice);
        }

        foreignKeyNum = m_ForeignKeyRange.size();
        if (foreignKeyNum > 0) {
            h_d_foreignKeys = new int*[foreignKeyNum];                 // 存 cuda 指针
            h_foreignKeys = new int*[foreignKeyNum];                   // 真实数据
            cudaMalloc(&d_foreignKeys, foreignKeyNum * sizeof(int*));  // cuda 二级指针
            for (int i = 0; i < foreignKeyNum; ++i) {
                h_foreignKeys[i] = new int[size];
                cudaMalloc(&h_d_foreignKeys[i], size * sizeof(int));
                // #pragma omp parallel for
                //                 for (int j = 0; j < size; j++) {
                //                     h_foreignKeys[i][j] = rand_x(0, m_ForeignKeyRange[i]);
                //                 }
                // cudaMemcpy(h_d_foreignKeys[i], h_foreignKeys[i], size * sizeof(int), cudaMemcpyHostToDevice);
                generate_rand_x<int>(h_d_foreignKeys[i], int(0), int(m_ForeignKeyRange[i]), size, seed);
            }
            cudaMemcpy(d_foreignKeys, h_d_foreignKeys, foreignKeyNum * sizeof(int*), cudaMemcpyHostToDevice);
        } else {
            h_foreignKeys = nullptr;
            d_foreignKeys = nullptr;
        }

        if (valueVecNum > 0) {
            h_valueVecs = new float*[valueVecNum];
            h_d_valueVecs = new float*[valueVecNum];
            cudaMalloc(&d_valueVecs, valueVecNum * sizeof(float*));
            for (int i = 0; i < valueVecNum; ++i) {
                h_valueVecs[i] = new float[size];
                cudaMalloc(&h_d_valueVecs[i], size * sizeof(float));
                // #pragma omp parallel for
                //                 for (int j = 0; j < size; ++j) {
                //                     h_valueVecs[i][j] = rand_x(0.0, 1.0);
                //                 }
                //                 cudaMemcpy(h_d_valueVecs[i], h_valueVecs[i], size * sizeof(float), cudaMemcpyHostToDevice);
                generate_rand_x<float>(h_d_valueVecs[i], 0.0f, 1.0f, size, seed);
            }
            cudaMemcpy(d_valueVecs, h_d_valueVecs, valueVecNum * sizeof(float*), cudaMemcpyHostToDevice);
        } else {
            h_valueVecs = nullptr;
            d_valueVecs = nullptr;
        }
    }
    ~Table() {
    }
};

void export_to_csv(tabulate::Table& table, const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    size_t num_rows = table.size();
    for (size_t i = 0; i < num_rows; ++i) {
        auto& row = table[i];
        for (size_t j = 0; j < row.size(); ++j) {
            auto cell = row[j].get_text();
            // 如果单元格中含有逗号、引号或换行，按 CSV 规范需要加引号，并转义双引号
            if (cell.find_first_of(",\"\n") != std::string::npos) {
                size_t pos = 0;
                while ((pos = cell.find("\"", pos)) != std::string::npos) {
                    cell.insert(pos, "\"");  // 转义双引号
                    pos += 2;
                }
                cell = "\"" + cell + "\"";
            }
            ofs << cell;
            if (j + 1 < row.size())
                ofs << ",";
        }
        ofs << "\n";
    }
}


#include "params.h"
