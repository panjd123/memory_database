#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>
#include <random>
#include <ranges>
#include "utils.cuh"

namespace SSB {
enum class TABLE_NAME {
    CUSTOMER = 0,
    SUPPLIER,
    PART,
    DATE,
    LINEORDER
};
const size_t LINEORDER_BASE = 6000000;  // lineorder base num
const size_t CUSTOMER_BASE = 30000;     // customer base num
const size_t SUPPLIER_BASE = 2000;      // supplier base num
const size_t PART_BASE = 200000;        // part base num
const size_t DATE_BASE = 7 * 365;       // date base num
inline int size_of_table(const TABLE_NAME& table, const double& SF) {
    switch (table) {
        case TABLE_NAME::CUSTOMER:
            return CUSTOMER_BASE * SF;
        case TABLE_NAME::SUPPLIER:
            return SUPPLIER_BASE * SF;
        case TABLE_NAME::PART:
            return (int)(PART_BASE * (double)(1 + log2(SF)));
        case TABLE_NAME::DATE:
            return DATE_BASE;
        case TABLE_NAME::LINEORDER:
            return LINEORDER_BASE * SF;
    }
    return 0;
}

struct Params {
    int8_t** dimVec_array;
    int** foreignKey_array;
    int* orders;
    int dimVecNum;
    int* factor;
    int size_lineorder;
    float* M1;
    float* M2;
    int groupNum;
    float* ans;

    // reserved for columnwise
    int32_t* OID;
    int16_t* groupID;
};

struct Arguments {
    int SF;
    int nthreads;
    bool IsBitmap[4];
    float selectRate[4];
    int groupNum[4];
    bool verbose;
    std::string testCase;
    Params params;
};

Arguments generate(int argc, char** argv) {
    auto dimVecName = std::vector<std::string>{"customer", "supplier", "part", "date"};

    argparse::ArgumentParser program("SSB benchmark");
    Arguments args;
    program.add_argument("--SF").default_value(1).store_into(args.SF);
    program.add_argument("-j", "--nthreads").store_into(args.nthreads).default_value(8).help("Number of threads to use for the computation");
    program.add_argument("--sqlnum");  // unused
    for (size_t i = 0; i < dimVecName.size(); ++i) {
        program.add_argument("--" + dimVecName[i] + "-bitmap", "--" + dimVecName[i].substr(0, 1) + "-bitmap")
            .default_value(false)
            .store_into(args.IsBitmap[i])
            .help("Use bitmap for " + dimVecName[i] + " dimension vector");
        program.add_argument("--" + dimVecName[i] + "-select-rate", "--" + dimVecName[i].substr(0, 1) + "-sele")
            .default_value(0.1f)
            .store_into(args.selectRate[i])
            .help("Selectivity rate for " + dimVecName[i] + " dimension");
        program.add_argument("--" + dimVecName[i] + "-group-num", "--" + dimVecName[i].substr(0, 1) + "-groups")
            .default_value(3)
            .store_into(args.groupNum[i])
            .help("Number of groups for " + dimVecName[i] + " dimension");
    }
    program.add_argument("--test-case")
        .default_value("SSB_default")
        .store_into(args.testCase)
        .help("Test case's name, used for output file naming");
    program.add_argument("--verbose")
        .default_value(false)
        .implicit_value(true)
        .store_into(args.verbose)
        .help("Enable verbose output");
    program.parse_args(argc, argv);

    args.params.size_lineorder = size_of_table(TABLE_NAME::LINEORDER, args.SF);

    auto start = std::chrono::high_resolution_clock::now();
    Table* dim_tables;
    dim_tables = new Table[dimVecName.size()];
    for (size_t i = 0; i < dimVecName.size(); ++i) {
        Table table(args.selectRate[i], args.IsBitmap[i], args.groupNum[i], size_of_table(static_cast<TABLE_NAME>(i), args.SF), {0});
        dim_tables[i] = table;
    }

    std::vector<int> dimVecSizes;
    for (size_t i = 0; i < dimVecName.size(); ++i) {
        dimVecSizes.push_back(dim_tables[i].size);
    }
    Table fact_table(-1, false, 0, args.params.size_lineorder, dimVecSizes, 2);

    // std::cout << "Generating tables..." << std::endl;
    cudaDeviceSynchronize();

    int dimTableNum = dimVecName.size();

    int* h_orders = new int[dimTableNum];
    args.params.dimVecNum = 0;
    for (int i = 0; i < dimTableNum; ++i) {
        if (args.selectRate[i] != 0) {
            args.params.dimVecNum += 1;
            h_orders[args.params.dimVecNum - 1] = i;
        }
    }
    std::sort(h_orders, h_orders + args.params.dimVecNum, [&args](int a, int b) {
        return args.selectRate[a] < args.selectRate[b];
    });

    args.params.dimVec_array = Gather<int8_t*>([dim_tables](int i) { return dim_tables[i].d_dimVec; }, dimTableNum);
    args.params.foreignKey_array = fact_table.d_foreignKeys;

    args.params.orders = Gather<int>([h_orders](int i) { return h_orders[i]; }, args.params.dimVecNum);

    int* h_factor = new int[args.params.dimVecNum];
    args.params.groupNum = 1;

    for (int i = 0; i < args.params.dimVecNum; i++) {
        args.params.groupNum *= args.groupNum[h_orders[i]];
        h_factor[i] = 1;
        if (!args.IsBitmap[h_orders[i]]) {
            for (int j = i + 1; j < args.params.dimVecNum; j++) {
                h_factor[i] *= args.groupNum[h_orders[j]];
            }
        }
    }
    cudaMalloc(&args.params.factor, args.params.dimVecNum * sizeof(int));
    cudaMemcpy(args.params.factor, h_factor, args.params.dimVecNum * sizeof(int), cudaMemcpyHostToDevice);

    args.params.M1 = fact_table.h_d_valueVecs[0];
    args.params.M2 = fact_table.h_d_valueVecs[1];
    cudaMalloc(&args.params.ans, args.params.groupNum * sizeof(float));

    cudaMalloc(&args.params.OID, args.params.size_lineorder * sizeof(int32_t));
    cudaMalloc(&args.params.groupID, args.params.size_lineorder * sizeof(int16_t));

    delete[] h_orders;
    delete[] h_factor;
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    // std::cout << "Initialization time: " << ms << " ms" << std::endl;
    return args;
}

void init_ans(Params& params) {
    // Initialize the ans array to zero
    cudaMemset(params.ans, 0, params.groupNum * sizeof(float));
}

__global__ void OLAPcore_rowwise_kernel(Params params) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float result_tmp[1000];
    int groupID = 0;
    for (int i = threadIdx.x; i < params.groupNum; i += blockDim.x) {
        result_tmp[i] = 0;
    }
    __syncthreads();
    for (int i = idx; i < params.size_lineorder; i += blockDim.x * gridDim.x) {
        int flag = 1;
        for (int j = 0; j < params.dimVecNum; j++) {
            int table_index = params.orders[j];
            int8_t idx_flag = params.dimVec_array[table_index][params.foreignKey_array[table_index][i]];
            if (idx_flag != DIM_NULL) {
                groupID += idx_flag * params.factor[j];
                continue;
            } else {
                flag = 0;
                groupID = 0;
                break;
            }
        }
        if (flag) {
            float sum = params.M1[i] + params.M2[i];
            atomicAdd(&result_tmp[groupID], sum);
            groupID = 0;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < params.groupNum; i += blockDim.x) {
        atomicAdd(&params.ans[i], result_tmp[i]);
    }
}

template <int TmpWorkspaceSize = 128>
__global__ void OLAPcore_rowwise_register_kernel(Params params) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float result_tmp[TmpWorkspaceSize];
    __shared__ float shared_result[1024];
    for (int i = 0; i < params.groupNum; i++) {
        result_tmp[i] = 0;
    }
    for (int i = idx; i < params.size_lineorder; i += blockDim.x * gridDim.x) {
        int groupID = 0;
        int flag = 1;
        for (int j = 0; j < params.dimVecNum; j++) {
            int table_index = params.orders[j];
            int8_t idx_flag = params.dimVec_array[table_index][params.foreignKey_array[table_index][i]];
            if (idx_flag != DIM_NULL) {
                groupID += idx_flag * params.factor[j];
            } else {
                flag = 0;
                break;
            }
        }
        if (flag) {
            float sum = params.M1[i] + params.M2[i];
            result_tmp[groupID] += sum;  // Use local memory for atomic operations
        }
    }

    int tid = threadIdx.x;
    for (int i = 0; i < params.groupNum; i++) {
        shared_result[tid] = result_tmp[i];
        // atomicAdd(&params.ans[i], result_tmp[i]);  // Use atomic operation to update global memory
        for (int s = blockDim.x / 2; s > 0; s /= 2) {
            if (tid < s) {
                shared_result[tid] += shared_result[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            atomicAdd(&params.ans[i], shared_result[0]);  // Write the final result to global memory
        }
    }
}

void OLAPcore_rowwise(Params& params) {
    dim3 block(1024);
    // dim3 grid((params.size_lineorder + block.x - 1) / block.x);
    dim3 grid(128);
    OLAPcore_rowwise_kernel<<<grid, block>>>(params);
}

void OLAPcore_rowwise_register(Params& params) {
    dim3 block(1024);
    // dim3 grid((params.size_lineorder + block.x - 1) / block.x);
    dim3 grid(128);
    if (params.groupNum <= 64) {
        OLAPcore_rowwise_register_kernel<64><<<grid, block>>>(params);
    } else if (params.groupNum <= 128) {
        OLAPcore_rowwise_register_kernel<128><<<grid, block>>>(params);
    } else {
        OLAPcore_rowwise_kernel<<<grid, block>>>(params);
    }
}

__global__ void OLAPcore_columnwise_dv_kernel(Params params) {
    int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t store_idx = idx;
    int32_t store_end;
    int32_t stride = blockDim.x * gridDim.x;
    for (int j = 0; j < params.dimVecNum; j++) {
        if (!j) {
            for (int k = idx; k < params.size_lineorder; k += stride) {
                int table_index = params.orders[j];
                int8_t idx_flag = params.dimVec_array[table_index][params.foreignKey_array[table_index][k]];
                if (idx_flag != DIM_NULL) {
                    params.OID[store_idx] = k;
                    params.groupID[store_idx] = idx_flag * params.factor[j];
                    store_idx += stride;
                }
            }
        } else {
            store_end = store_idx;
            store_idx = idx;
            for (int k = idx; k < store_end; k += stride) {
                int location = params.OID[k];
                int table_index = params.orders[j];
                int8_t idx_flag = params.dimVec_array[table_index][params.foreignKey_array[table_index][location]];
                if (idx_flag != DIM_NULL) {
                    params.OID[store_idx] = location;
                    params.groupID[store_idx] += idx_flag * params.factor[j];
                    store_idx += stride;
                }
            }
        }
    }
    store_end = store_idx;
    for (store_idx = idx; store_idx < store_end; store_idx += stride) {
        int16_t tmp = params.groupID[store_idx];
        float sum = params.M1[params.OID[store_idx]] + params.M2[params.OID[store_idx]];
        atomicAdd(&params.ans[tmp], sum);
    }
}

__global__ void OLAPcore_columnwise_sv_kernel(Params params) {
    int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t stride = blockDim.x * gridDim.x;
    for (int j = 0; j < params.dimVecNum; j++) {
        if (!j) {
            for (int k = idx; k < params.size_lineorder; k += stride) {
                int table_index = params.orders[j];
                int8_t idx_flag = params.dimVec_array[table_index][params.foreignKey_array[table_index][k]];
                if (idx_flag != DIM_NULL) {
                    params.groupID[k] = idx_flag * params.factor[j];
                } else {
                    params.groupID[k] = GROUP_NULL;
                }
            }
        } else {
            for (int k = idx; k < params.size_lineorder; k += stride) {
                if (params.groupID[k] == GROUP_NULL) {
                    continue;
                }
                int table_index = params.orders[j];
                int8_t idx_flag = params.dimVec_array[table_index][params.foreignKey_array[table_index][k]];
                if (idx_flag != DIM_NULL) {
                    params.groupID[k] += idx_flag * params.factor[j];
                } else {
                    params.groupID[k] = GROUP_NULL;
                }
            }
        }
    }
    for (int k = idx; k < params.size_lineorder; k += stride) {
        int16_t tmp = params.groupID[k];
        if (tmp == GROUP_NULL) {
            continue;  // Skip if groupID is NULL
        }
        float sum = params.M1[k] + params.M2[k];
        atomicAdd(&params.ans[tmp], sum);
    }
}

void OLAPcore_columnwise_dv(Params& params) {
    dim3 block(1024);
    dim3 grid(128);
    OLAPcore_columnwise_dv_kernel<<<grid, block>>>(params);
}

void OLAPcore_columnwise_sv(Params& params) {
    dim3 block(1024);
    dim3 grid(128);
    OLAPcore_columnwise_sv_kernel<<<grid, block>>>(params);
}

template <int VectorSize = 32>
__global__ void OLAPcore_vectorwise_dv_kernel(Params params) {
    int32_t OID[VectorSize];
    int16_t groupID[VectorSize];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int32_t block_start = idx; block_start < params.size_lineorder; block_start += VectorSize * stride) {
        int block_end = min(block_start + VectorSize * stride, params.size_lineorder);
        int store_local_k = 0;
        int store_local_end = 0;
        for (int j = 0; j < params.dimVecNum; j++) {
            if (!j) {
                for (int k = block_start; k < block_end; k += stride) {
                    int table_index = params.orders[j];
                    int8_t idx_flag = params.dimVec_array[table_index][params.foreignKey_array[table_index][k]];
                    if (idx_flag != DIM_NULL) {
                        OID[store_local_k] = k;
                        groupID[store_local_k] = idx_flag * params.factor[j];
                        store_local_k++;
                    }
                }
            } else {
                store_local_end = store_local_k;
                store_local_k = 0;
                for (int last_local_k = 0; last_local_k < store_local_end; last_local_k++) {
                    int location = OID[last_local_k];
                    int table_index = params.orders[j];
                    int8_t idx_flag = params.dimVec_array[table_index][params.foreignKey_array[table_index][location]];
                    if (idx_flag != DIM_NULL) {
                        OID[store_local_k] = location;
                        groupID[store_local_k] += idx_flag * params.factor[j];
                        store_local_k++;
                    }
                }
            }
        }
        store_local_end = store_local_k;
        for (store_local_k = 0; store_local_k < store_local_end; ++store_local_k) {
            int16_t tmp = groupID[store_local_k];
            float sum = params.M1[OID[store_local_k]] + params.M2[OID[store_local_k]];
            atomicAdd(&params.ans[tmp], sum);
        }
    }
}

template <int VectorSize = 32>
__global__ void OLAPcore_vectorwise_sv_kernel(Params params) {
    int16_t groupID[VectorSize];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int32_t block_start = idx; block_start < params.size_lineorder; block_start += VectorSize * stride) {
        int block_end = min(block_start + VectorSize * stride, params.size_lineorder);
        for (int j = 0; j < params.dimVecNum; j++) {
            if (!j) {
                for (int k = block_start, local_k = 0; k < block_end; k += stride, ++local_k) {
                    int table_index = params.orders[j];
                    int8_t idx_flag = params.dimVec_array[table_index][params.foreignKey_array[table_index][k]];
                    if (idx_flag != DIM_NULL) {
                        groupID[local_k] = idx_flag * params.factor[j];
                    } else {
                        groupID[local_k] = GROUP_NULL;
                    }
                }
            } else {
                for (int k = block_start, local_k = 0; k < block_end; k += stride, ++local_k) {
                    if (groupID[local_k] == GROUP_NULL) {
                        continue;
                    }
                    int table_index = params.orders[j];
                    int8_t idx_flag = params.dimVec_array[table_index][params.foreignKey_array[table_index][k]];
                    if (idx_flag != DIM_NULL) {
                        groupID[local_k] += idx_flag * params.factor[j];
                    } else {
                        groupID[local_k] = GROUP_NULL;
                    }
                }
            }
        }
        for (int k = block_start, local_k = 0; k < block_end; k += stride, ++local_k) {
            int16_t tmp = groupID[local_k];
            if (tmp == GROUP_NULL) {
                continue;  // Skip if groupID is NULL
            }
            float sum = params.M1[k] + params.M2[k];
            atomicAdd(&params.ans[tmp], sum);
        }
    }
}

void OLAPcore_vectorwise_dv(Params& params) {
    dim3 block(1024);
    dim3 grid(128);
    OLAPcore_vectorwise_dv_kernel<<<grid, block>>>(params);
}

void OLAPcore_vectorwise_sv(Params& params) {
    dim3 block(1024);
    dim3 grid(128);
    OLAPcore_vectorwise_sv_kernel<<<grid, block>>>(params);
}

std::pair<float, float> timeit(void (*func)(Params&), Params& params) {
    for (int i = 0; i < warmup; i++) {
        func(params);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < runs; i++) {
        init_ans(params);  // Reset ans array before each run
        func(params);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= runs;  // Average time over runs
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    float* h_ans = new float[params.groupNum];
    cudaMemcpy(h_ans, params.ans, params.groupNum * sizeof(float), cudaMemcpyDeviceToHost);
    float sum = 0;
    for (int i = 0; i < params.groupNum; i++) {
        sum += h_ans[i];
    }
    delete[] h_ans;
    return {milliseconds, sum};
}

void benchmark(Arguments& args) {
    tabulate::Table result_table;
    result_table.add_row({"Method", "Time", "TotalSum"});
    auto [time_rowwise, sum_rowwise] = timeit(OLAPcore_rowwise, args.params);
    result_table.add_row(tabulate::RowStream{} << "rowwise" << time_rowwise << sum_rowwise);
    auto [time_rowwise_register, sum_rowwise_register] = timeit(OLAPcore_rowwise_register, args.params);
    result_table.add_row(tabulate::RowStream{} << "rowwise_register" << time_rowwise_register << sum_rowwise_register);
    auto [time_columnwise_dv, sum_columnwise_dv] = timeit(OLAPcore_columnwise_dv, args.params);
    result_table.add_row(tabulate::RowStream{} << "columnwise_dynamic_vector" << time_columnwise_dv << sum_columnwise_dv);
    auto [time_columnwise_sv, sum_columnwise_sv] = timeit(OLAPcore_columnwise_sv, args.params);
    result_table.add_row(tabulate::RowStream{} << "columnwise_static_vector" << time_columnwise_sv << sum_columnwise_sv);
    auto [time_vectorwise_dv, sum_vectorwise_dv] = timeit(OLAPcore_vectorwise_dv, args.params);
    result_table.add_row(tabulate::RowStream{} << "vectorwise_dynamic_vector" << time_vectorwise_dv << sum_vectorwise_dv);
    auto [time_vectorwise_sv, sum_vectorwise_sv] = timeit(OLAPcore_vectorwise_sv, args.params);
    result_table.add_row(tabulate::RowStream{} << "vectorwise_static_vector" << time_vectorwise_sv << sum_vectorwise_sv);
    std::cout << result_table << std::endl;
    export_to_csv(result_table, "benchmark_results_" + args.testCase + ".csv");
}

};  // namespace SSB