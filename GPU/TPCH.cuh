#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>
#include <random>
#include <ranges>
#include <nvtx3/nvToolsExt.h>
#include "argparse.hpp"
#include "tabulate.hpp"
#include "utils.cuh"

namespace TPCH {
enum class TABLE_NAME {
    ORDERS = 0,
    LINEITEM,
    PARTSUPP,  // PS
    CUSTOMER,
    PART,
    SUPPLIER,
    NATION,
    REGION
};

const size_t ORDERS_BASE = 1500000;    // orders base num
const size_t LINEITEM_BASE = 6000000;  // lineitem base num
const size_t PARTSUPP_BASE = 800000;  // partsupp base num
const size_t CUSTOMER_BASE = 150000;   // customer base num
const size_t PART_BASE = 200000;       // part base num
const size_t SUPPLIER_BASE = 10000;    // supplier base num
const size_t NATION_BASE = 25;         // nation base num
const size_t REGION_BASE = 5;          // region base num

inline int size_of_table(const TABLE_NAME& table, const double& SF) {
    switch (table) {
        case TABLE_NAME::ORDERS:
            return ORDERS_BASE * SF;
        case TABLE_NAME::LINEITEM:
            return LINEITEM_BASE * SF;
        case TABLE_NAME::PARTSUPP:
            return PARTSUPP_BASE * SF;
        case TABLE_NAME::CUSTOMER:
            return CUSTOMER_BASE * SF;
        case TABLE_NAME::PART:
            return PART_BASE * SF;
        case TABLE_NAME::SUPPLIER:
            return SUPPLIER_BASE * SF;
        case TABLE_NAME::NATION:
            return NATION_BASE;
        case TABLE_NAME::REGION:
            return REGION_BASE;
        default:
            return 0;
    }
}

/*
L -> O
L -> S -> N
O bitmap
N bitmap
N dimvec
L M1 M2

preS
L -> O
L -> S
O bitmap
S bitmap
S dimVec
*/

struct Params {
    int8_t* o_dimVec;  // O 的 bitmap
    int8_t* s_dimVec;
    int8_t* n_dimVec;
    int* o_foreignKey;
    int* s_foreignKey;
    int* n_foreignKey;
    bool o_first;
    bool pre_s;         // 提前把 N 的 bitmap 和 dimvec 映射到 S 上
    int dimVecNum = 1;  // 对于 Q5 来说，永远为 1
    int* factor;        // unused
    int size_lineitem;
    int size_supplier;
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
    float selectRate[2];  // O, N
    int groupNum[2];      // O, N
    bool verbose;
    std::string testCase;
    Params params;
};

Arguments generate(int argc, char** argv) {
    auto dimVecName = std::vector<std::string>{"orders", "nation"};

    argparse::ArgumentParser program("TPCH Benchmark");
    Arguments args;
    program.add_argument("--SF").default_value(1).store_into(args.SF);
    program.add_argument("-j", "--nthreads").store_into(args.nthreads).default_value(8).help("Number of threads to use for the computation");
    program.add_argument("--sqlnum");  // unused
    for (size_t i = 0; i < dimVecName.size(); ++i) {
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

    args.params.size_lineitem = TPCH::size_of_table(TPCH::TABLE_NAME::LINEITEM, args.SF);
    args.params.size_supplier = TPCH::size_of_table(TPCH::TABLE_NAME::SUPPLIER, args.SF);
    Table o_table(args.selectRate[0], false, 0,
                  size_of_table(TPCH::TABLE_NAME::ORDERS, args.SF), {0});
    Table n_table(args.selectRate[1], true, args.groupNum[1],
                  size_of_table(TPCH::TABLE_NAME::NATION, args.SF), {0});
    Table s_table(1, false, 0, size_of_table(TPCH::TABLE_NAME::SUPPLIER, args.SF),
                  {n_table.size});
    // s selectRate set to 1, reserved dimVec for preS
    Table l_table(-1, false, 0, args.params.size_lineitem,
                  {
                      o_table.size,
                      s_table.size,
                  },
                  2);
    args.params.o_foreignKey = l_table.h_d_foreignKeys[0];
    args.params.s_foreignKey = l_table.h_d_foreignKeys[1];
    args.params.n_foreignKey = s_table.h_d_foreignKeys[0];
    args.params.o_dimVec = o_table.d_dimVec;
    args.params.s_dimVec = s_table.d_dimVec;
    args.params.n_dimVec = n_table.d_dimVec;
    args.params.groupNum = args.groupNum[1];
    args.params.M1 = l_table.h_d_valueVecs[0];
    args.params.M2 = l_table.h_d_valueVecs[1];
    cudaMalloc(&args.params.ans, args.params.groupNum * sizeof(float));
    cudaMalloc(&args.params.OID, args.params.size_lineitem * sizeof(int32_t));
    cudaMalloc(&args.params.groupID, args.params.size_lineitem * sizeof(int16_t));
    return args;
}

void init_ans(Params& params) {
    // Initialize the ans array to zero
    cudaMemset(params.ans, 0, params.groupNum * sizeof(float));
}

template <bool OFirst = true>
__global__ void Q5_rowwise_kernel(Params params) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float result_tmp[1000];
    int groupID = 0;
    for (int i = threadIdx.x; i < params.groupNum; i += blockDim.x) {
        result_tmp[i] = 0;
    }
    __syncthreads();
    for (int i = idx; i < params.size_lineitem; i += blockDim.x * gridDim.x) {
        if constexpr (OFirst) {
            if (params.o_dimVec[params.o_foreignKey[i]] == DIM_NULL)
                continue;
            groupID = params.n_dimVec[params.n_foreignKey[params.s_foreignKey[i]]];
            if (groupID != DIM_NULL) {
                float sum = params.M1[i] * (1 - params.M2[i]);
                atomicAdd(&result_tmp[groupID], sum);
            }
        } else {
            groupID = params.n_dimVec[params.n_foreignKey[params.s_foreignKey[i]]];
            if (groupID == DIM_NULL)
                continue;
            if (params.o_dimVec[params.o_foreignKey[i]] != DIM_NULL) {
                float sum = params.M1[i] * (1 - params.M2[i]);
                atomicAdd(&result_tmp[groupID], sum);
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < params.groupNum; i += blockDim.x) {
        atomicAdd(&params.ans[i], result_tmp[i]);
    }
}

__global__ void prepare_S_dimvec_kernel(Params params) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < params.size_supplier) {
        params.s_dimVec[idx] = params.n_dimVec[params.n_foreignKey[idx]];
    }
}

void prepare_S_dimvec(Params& params) {
    dim3 block(1024);
    dim3 grid((params.size_supplier + block.x - 1) / block.x);
    prepare_S_dimvec_kernel<<<grid, block>>>(params);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in prepare_S_dimvec: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

template <bool OFirst = true>
__global__ void tuple_rowwise_kernel(Params params) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float result_tmp[1000];
    int groupID = 0;
    for (int i = threadIdx.x; i < params.groupNum; i += blockDim.x) {
        result_tmp[i] = 0;
    }
    __syncthreads();
    for (int i = idx; i < params.size_lineitem; i += blockDim.x * gridDim.x) {
        if constexpr (OFirst) {
            if (params.o_dimVec[params.o_foreignKey[i]] == DIM_NULL)
                continue;
            groupID = params.s_dimVec[params.s_foreignKey[i]];
            if (groupID != DIM_NULL) {
                float sum = params.M1[i] * (1 - params.M2[i]);
                atomicAdd(&result_tmp[groupID], sum);
            }
        }else{
            groupID = params.s_dimVec[params.s_foreignKey[i]];
            if (groupID == DIM_NULL)
                continue;
            if (params.o_dimVec[params.o_foreignKey[i]] != DIM_NULL) {
                float sum = params.M1[i] * (1 - params.M2[i]);
                atomicAdd(&result_tmp[groupID], sum);
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < params.groupNum; i += blockDim.x) {
        atomicAdd(&params.ans[i], result_tmp[i]);
    }
}

void Q5_rowwise(Params& params) {
    dim3 block(1024);
    dim3 grid(128);
    if (params.pre_s) {
        prepare_S_dimvec(params);  // Prepare S's dimVec before kernel launch
        if (params.o_first) {
            tuple_rowwise_kernel<true><<<grid, block>>>(params);
        } else {
            tuple_rowwise_kernel<false><<<grid, block>>>(params);
        }
    } else {
        if (params.o_first) {
            Q5_rowwise_kernel<true><<<grid, block>>>(params);
        } else {
            Q5_rowwise_kernel<false><<<grid, block>>>(params);
        }
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in Q5_rowwise: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void Q5_columnwise_dv_kernel(Params params) {
    int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t store_idx = idx;
    int32_t store_end;
    int32_t stride = blockDim.x * gridDim.x;
    __shared__ float result_tmp[1000];
    for (int i = threadIdx.x; i < params.groupNum; i += blockDim.x) {
        result_tmp[i] = 0;
    }
    __syncthreads();
    
    for (int k = idx; k < params.size_lineitem; k += stride) {
        int groupID = params.n_dimVec[params.n_foreignKey[params.s_foreignKey[k]]];
        if (groupID == DIM_NULL) {
            continue;
        }
        params.OID[store_idx] = k;
        params.groupID[store_idx] = groupID;
        store_idx += stride;
    }

    store_end = store_idx;
    store_idx = idx;
    for (int k = idx; k < store_end; k += stride) {
        int location = params.OID[k];
        int8_t idx_flag = params.o_dimVec[params.o_foreignKey[location]];
        if (idx_flag != DIM_NULL) {
            params.OID[store_idx] = location;
            params.groupID[store_idx] = params.groupID[k];
            store_idx += stride;
        }
    }

    store_end = store_idx;
    for (store_idx = idx; store_idx < store_end; store_idx += stride) {
        int16_t tmp = params.groupID[store_idx];
        float sum = params.M1[params.OID[store_idx]] * (1 - params.M2[params.OID[store_idx]]);
        atomicAdd(&result_tmp[tmp], sum);  // Use local memory for atomic operations
    }
    __syncthreads();
    for (int i = threadIdx.x; i < params.groupNum; i += blockDim.x) {
        atomicAdd(&params.ans[i], result_tmp[i]);  // Write the final result to global memory
    }
}

void Q5_columnwise(Params& params) {
    dim3 block(1024);
    dim3 grid(128);
    if (params.pre_s) {
        prepare_S_dimvec(params);  // Prepare S's dimVec before kernel launch
    }
    Q5_columnwise_dv_kernel<<<grid, block>>>(params);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in Q5_columnwise: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

std::pair<float, float> timeit(void (*func)(Params&), Params& params) {
    for (int i = 0; i < warmup; i++) {
        func(params);
    }

    nvtxRangePushA("profile");
    func(params);
    nvtxRangePop();

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
    {
        args.params.pre_s = false;  // No preS for Q5
        args.params.o_first = true;  // O first
        auto [time_rowwise_ofirst, sum_rowwise_ofirst] = timeit(Q5_rowwise, args.params);
        args.params.o_first = false;  // S first
        auto [time_rowwise_sfirst, sum_rowwise_sfirst] = timeit(Q5_rowwise, args.params);
        result_table.add_row({"RowwiseO", std::to_string(time_rowwise_ofirst), std::to_string(sum_rowwise_ofirst)});
        result_table.add_row({"RowwiseS", std::to_string(time_rowwise_sfirst), std::to_string(sum_rowwise_sfirst)});
    }
    {
        args.params.pre_s = true;  // No preS for Q5
        args.params.o_first = true;  // O first
        auto [time_rowwise_ofirst, sum_rowwise_ofirst] = timeit(Q5_rowwise, args.params);
        args.params.o_first = false;  // S first
        auto [time_rowwise_sfirst, sum_rowwise_sfirst] = timeit(Q5_rowwise, args.params);
        result_table.add_row({"RowwiseO_P", std::to_string(time_rowwise_ofirst), std::to_string(sum_rowwise_ofirst)});
        result_table.add_row({"RowwiseS_P", std::to_string(time_rowwise_sfirst), std::to_string(sum_rowwise_sfirst)});
    }
    {
        auto [time_columnwise_dv, sum_columnwise_dv] = timeit(Q5_columnwise, args.params);
        result_table.add_row({"ColumnwiseDV", std::to_string(time_columnwise_dv), std::to_string(sum_columnwise_dv)});
    }
    std::cout << result_table << std::endl;
    export_to_csv(result_table, "TPCH_benchmark_" + args.testCase + ".csv");
}

}  // namespace TPCH