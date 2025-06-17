/**
 * @file test_multi_compute_operator.cu
 * @author ruichenhan (hanruichen@ruc.edu.cn)
 * @brief test OLAPcore algorithms on GPU
 * @version 0.1
 * @date 2023-05-20
 *
 * @copyright Copyright (c) 2022
 *
 */
#include <cuda.h>
#include <getopt.h>   /* getopt */
#include <limits.h>   /* INT_MAX */
#include <malloc.h>   /* malloc */
#include <sched.h>    /* sched_setaffinity */
#include <stdio.h>    /* printf */
#include <stdlib.h>   /* exit */
#include <string.h>   /* strcmp */
#include <sys/time.h> /* gettimeofday */
#include <array>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
#include "../include/gendata_util.hpp"
#include "../include/metadata.h"
#include "../include/statistical_analysis_util.hpp"
#define BLOCK_NUM 128
#define THREAD_NUM 1024
#define COMSIZE 6144
#define BITSIZE 16384
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}
/**
 * @brief command line handling functions
 *
 * @param progname
 * @return void
 */
void print_help(char* progname) {
    printf("Usage: %s [options]\n", progname);

    printf(
        "\
    join configuration options, with default values in [] :             \n\
       -n --nthreads=<N>  Number of threads to use <N> [2]                    \n\
       --SF               Size of dataset                                     \n\
       --d-sele           Selection rate of column on date table              \n\
       --d-bitmap         Flag of  bitmap index on date<1——bitmap 0——vector>  \n\
       --d-groups         Number of groups of columns on date                 \n\
       --s-sele           Selection rate of column on supplier table          \n\
       --s-bitmap         Flag of bitmap index on supplier<1——bitmap 0——vector>\n\
       --s-groups         Number of groups of columns on supplier             \n\
       --p-sele           Selection rate of column on part table              \n\
       --p-bitmap         Flag of  bitmap index on part<1——bitmap 0——vector>  \n\
       --p-groups         Number of groups of columns on part                 \n\
       --c-sele           Selection rate of column on customer table          \n\
       --c-bitmap         Flag of  bitmap index on customer<1——bitmap 0——vector>\n\
       --c-groups         Number of groups of columns on customer             \n\
                                                                              \n\
    Basic user options                                                        \n\
        -h --help         Show this message                                   \n\
    \n");
}
/**
 * @brief command line handling functions
 *
 * @param argc
 * @param argv
 * @param cmd_params
 * @return void
 */
void parse_args(int argc, char** argv, param_t* cmd_params) {
    int c, i, found;
    static int basic_numa_flag;
    while (1) {
        static struct option long_options[] =
            {
                /* These options set a flag. */

                {"help", no_argument, 0, 'h'},
                /* These options don't set a flag.
                   We distinguish them by their indices. */
                {"nthreads", required_argument, 0, 'n'},
                {"basic_numa", no_argument, &basic_numa_flag, 1},
                {"SF", required_argument, 0, 'F'},
                {"sqlnum", required_argument, 0, 'sq'},
                {"d-bitmap", required_argument, 0, 'd'},
                {"s-bitmap", required_argument, 0, 'f'},
                {"p-bitmap", required_argument, 0, 'P'},
                {"c-bitmap", required_argument, 0, 'c'},
                {"d-sele", required_argument, 0, 'D'},
                {"d-groups", required_argument, 0, 'DG'},
                {"s-sele", required_argument, 0, 'ss'},
                {"s-groups", required_argument, 0, 'sg'},
                {"p-sele", required_argument, 0, 'ps'},
                {"p-groups", required_argument, 0, 'pg'},
                {"c-sele", required_argument, 0, 'cs'},
                {"c-groups", required_argument, 0, 'cg'},
                {0, 0, 0, 0}};
        /* getopt_long stores the option index here. */
        int option_index = 0;

        c = getopt_long(argc, argv, "n:hv:F:sq:d:f:P:c:D:DG:ss:sg:ps:pg:cs:cg:dn:sn:pn:cn",
                        long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1)
            break;
        switch (c) {
            case 0:
                /* If this option set a flag, do nothing else now. */
                if (long_options[option_index].flag != 0)
                    break;
                printf("option %s", long_options[option_index].name);
                if (optarg)
                    printf(" with arg %s", optarg);
                printf("\n");
                break;

            case 'h':
            case '?':
                /* getopt_long already printed an error message. */
                print_help(argv[0]);
                exit(EXIT_SUCCESS);
                break;
            case 'n':
                cmd_params->nthreads = atoi(optarg);
                break;
            case 'F':

                cmd_params->sf = atoi(optarg);

                break;
            case 'sq':

                cmd_params->sqlnum = atoi(optarg);

                break;
            case 'd':

                cmd_params->d_bitmap = atoi(optarg);

                break;
            case 'f':

                cmd_params->s_bitmap = atoi(optarg);

                break;
            case 'P':

                cmd_params->p_bitmap = atoi(optarg);

                break;
            case 'c':

                cmd_params->c_bitmap = atoi(optarg);

                break;
            case 'D':

                cmd_params->d_sele = atof(optarg);

                break;
            case 'DG':

                cmd_params->d_groups = atoi(optarg);

                break;
            case 'ss':

                cmd_params->s_sele = atof(optarg);

                break;
            case 'sg':

                cmd_params->s_groups = atoi(optarg);

                break;
            case 'ps':

                cmd_params->p_sele = atof(optarg);

                break;
            case 'pg':

                cmd_params->p_groups = atoi(optarg);

                break;
            case 'cs':

                cmd_params->c_sele = atof(optarg);

                break;
            case 'cg':

                cmd_params->c_groups = atoi(optarg);

                break;
            default:
                break;
        }
    }
    cmd_params->basic_numa = basic_numa_flag;
    /* Print any remaining command line arguments (not options). */
    if (optind < argc) {
        std::cout << "non-option arguments: ";
        while (optind < argc)
            std::cout << argv[optind++] << " ";
        std::cout << std::endl;
    }
}
/**
 * @brief set the join order by select rate
 *
 * @param h_sele_array
 * @param h_orders
 * @param size
 * @return void
 */
void sort_by_rate(double* h_sele_array, int* h_orders, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            if (h_sele_array[i] > h_sele_array[j]) {
                int order_tmp = h_orders[i];
                h_orders[i] = h_orders[j];
                h_orders[j] = order_tmp;
            }
        }
    }
}
__global__ void OLAPcore_rowwise(int8_t** dimvec_array, int32_t** fk_array, int* size_array, int* orders, int* dimvec_nums, int* factor, int* size_lineorder, int* group_nums, int32_t* M1, int32_t* M2, uint32_t* group_vector) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ unsigned long long result_tmp[1000];
    int groupID = 0;
    for (i = threadIdx.x; i < *group_nums; i += blockDim.x) {
        result_tmp[i] = 0;
    }
    __syncthreads();
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < *size_lineorder; i += blockDim.x * gridDim.x) {
        int flag = 1;
        for (int j = 0; j < *dimvec_nums; j++) {
            int table_index = orders[j];
            int8_t idx_flag = dimvec_array[table_index][fk_array[table_index][i]];
            if (idx_flag != DIM_NULL) {
                groupID += idx_flag * factor[j];
                continue;
            } else {
                flag = 0;
                groupID = 0;
                break;
            }
        }
        if (flag) {
            int sum = M1[i] + M2[i];
            atomicAdd(&result_tmp[groupID], sum);

            groupID = 0;
        }
    }
    __syncthreads();

    for (i = threadIdx.x; i < *group_nums; i += blockDim.x) {
        atomicAdd(&group_vector[i], result_tmp[i]);
    }
}
__global__ void OLAPcore_columnwise_dv(int8_t** dimvec_array, int32_t** fk_array, int* size_array, int* orders, int* dimvec_nums, int64_t* OID, int16_t* groupID, int* factor, int* size_lineorder, int* group_nums, int32_t* M1, int32_t* M2, uint32_t* group_vector) {
    int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = 0;
    int16_t tmp = -1;
    int64_t comlength = 0;
    int sum = 0;

    for (int j = 0; j < *dimvec_nums; j++) {
        if (!j) {
            for (int k = threadIdx.x + blockIdx.x * blockDim.x; k < *size_lineorder; k += blockDim.x * gridDim.x) {
                int table_index = orders[j];
                int idx_flag = dimvec_array[table_index][fk_array[table_index][k]];

                OID[i] = k;
                groupID[i] = idx_flag * factor[j];
                i += (int)(idx_flag != DIM_NULL) * (blockDim.x * gridDim.x);
            }
        } else {
            comlength = i;
            i = threadIdx.x + blockIdx.x * blockDim.x;
            for (int k = threadIdx.x + blockIdx.x * blockDim.x; k < comlength; k += blockDim.x * gridDim.x) {
                int location = OID[k];
                int table_index = orders[j];
                int idx_flag = dimvec_array[table_index][fk_array[table_index][location]];
                OID[i] = location;
                groupID[i] += (int)(idx_flag != DIM_NULL) * (idx_flag * factor[j]);
                i += (int)(idx_flag != DIM_NULL) * (blockDim.x * gridDim.x);
            }
        }
    }
    comlength = i;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < comlength; i += blockDim.x * gridDim.x) {
        tmp = groupID[i];
        sum = M1[OID[i]] + M2[OID[i]];
        atomicAdd(&group_vector[tmp], sum);
    }
}
__global__ void OLAPcore_vectorwise_dv(int8_t** dimvec_array, int32_t** fk_array, int* size_array, int* orders, int* dimvec_nums, int* factor, int* size_lineorder, int* group_nums, int32_t* M1, int32_t* M2, uint32_t* group_vector) {
    int64_t i = threadIdx.x;
    int64_t blockLength = (*size_lineorder) / BLOCK_NUM;
    int64_t blockStart = blockIdx.x * blockLength;
    blockLength = (blockIdx.x == BLOCK_NUM - 1) ? *size_lineorder - blockLength * (BLOCK_NUM - 1) : blockLength;
    int64_t blockEnd = blockStart + blockLength;
    int16_t tmp = -1;
    int sum = 0;
    __shared__ int32_t OID[COMSIZE];
    __shared__ int16_t groupID[COMSIZE];
    int index = 0, comlength = 0;
    int table_index = 0;
    int idx_flag = 0;

    while (i < blockEnd) {
        for (int j = 0; j < *dimvec_nums; j++) {
            if (!j) {
                index = threadIdx.x;
                for (i = threadIdx.x + blockStart; i < COMSIZE + blockStart && i < blockEnd; i += blockDim.x) {
                    table_index = orders[j];
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i]];
                    if (idx_flag != DIM_NULL) {
                        OID[index] = i;
                        groupID[index] = idx_flag * factor[j];
                        index += blockDim.x;
                    }
                }
                comlength = index;
            } else {
                index = threadIdx.x;
                for (i = threadIdx.x + blockStart; i < comlength + blockStart && i < blockEnd; i += blockDim.x) {
                    table_index = orders[j];
                    idx_flag = dimvec_array[table_index][fk_array[table_index][OID[i - blockStart]]];
                    if (idx_flag != DIM_NULL) {
                        OID[index] = OID[i - blockStart];
                        groupID[index] = groupID[i - blockStart] + idx_flag * factor[j];
                        index += blockDim.x;
                    }
                }
                comlength = index;
            }
        }

        for (i = threadIdx.x + blockStart; i < comlength + blockStart && i < blockEnd; i += blockDim.x) {
            tmp = groupID[i - blockStart];
            sum = M1[OID[i - blockStart]] + M2[OID[i - blockStart]];
            atomicAdd(&group_vector[tmp], sum);
        }
        blockStart += COMSIZE;
    }
}

__global__ void OLAPcore_vectorwise_sv(int8_t** dimvec_array, int32_t** fk_array, int* size_array, int* orders, int* dimvec_nums, int* factor, int* size_lineorder, int* group_nums, int32_t* M1, int32_t* M2, uint32_t* group_vector) {
    int64_t i = threadIdx.x;
    int64_t blockLength = (*size_lineorder) / BLOCK_NUM;
    int64_t blockStart = blockIdx.x * blockLength;
    blockLength = (blockIdx.x == BLOCK_NUM - 1) ? *size_lineorder - blockLength * (BLOCK_NUM - 1) : blockLength;
    int64_t blockEnd = blockStart + blockLength;
    int16_t tmp = -1;
    int sum = 0;
    __shared__ int16_t groupID[BITSIZE];
    int table_index = 0;
    int idx_flag = 0;

    while (i < blockEnd) {
        for (int j = 0; j < *dimvec_nums; j++) {
            if (!j) {
                for (i = threadIdx.x + blockStart; i < BITSIZE + blockStart && i < blockEnd; i += blockDim.x) {
                    table_index = orders[j];
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i]];
                    if (idx_flag != DIM_NULL)
                        groupID[i - blockStart] = idx_flag * factor[j];
                    else
                        groupID[i - blockStart] = GROUP_NULL;
                }
            } else {
                for (i = threadIdx.x + blockStart; i < BITSIZE + blockStart && i < blockEnd; i += blockDim.x) {
                    table_index = orders[j];
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i]];
                    if ((groupID[i - blockStart] != GROUP_NULL) && (idx_flag != DIM_NULL))
                        groupID[i - blockStart] = groupID[i - blockStart] + idx_flag * factor[j];
                    else
                        groupID[i - blockStart] = GROUP_NULL;
                }
            }
        }
        for (i = threadIdx.x + blockStart; i < BITSIZE + blockStart && i < blockEnd; i += blockDim.x) {
            tmp = groupID[i - blockStart];
            if (tmp != GROUP_NULL) {
                sum = M1[i] + M2[i];
                atomicAdd(&group_vector[tmp], sum);
            }
        }
        blockStart += BITSIZE;
    }
}
__global__ void OLAPcore_vectorwise_sv_register(int8_t** dimvec_array, int32_t** fk_array, int* size_array, int* orders, int* dimvec_nums, int* factor, int* size_lineorder, int* group_nums, int32_t* M1, int32_t* M2, uint32_t* group_vector) {
    int64_t i = threadIdx.x;
    int64_t blockLength = (*size_lineorder) / BLOCK_NUM;
    int64_t blockStart = blockIdx.x * blockLength;
    blockLength = (blockIdx.x == BLOCK_NUM - 1) ? *size_lineorder - blockLength * (BLOCK_NUM - 1) : blockLength;
    int64_t blockEnd = blockStart + blockLength;
    int16_t tmp = -1;
    int sum = 0;
    int16_t groupID0;
    int16_t groupID1;
    int16_t groupID2;
    int16_t groupID3;
    int16_t groupID4;
    int16_t groupID5;
    int16_t groupID6;
    int16_t groupID7;
    int16_t groupID8;
    int16_t groupID9;
    int16_t groupID10;
    int16_t groupID11;
    int16_t groupID12;
    int16_t groupID13;
    int16_t groupID14;
    int16_t groupID15;
    int table_index = 0;
    int idx_flag = 0;
    i = threadIdx.x + blockStart;
    while (i < blockEnd) {
        for (int j = 0; j < *dimvec_nums; j++) {
            if (!j) {
                table_index = orders[j];

                idx_flag = dimvec_array[table_index][fk_array[table_index][i]];
                if (idx_flag != DIM_NULL)
                    groupID0 = idx_flag * factor[j];
                else
                    groupID0 = GROUP_NULL;
                if (i + blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + blockDim.x]];
                    if (idx_flag != DIM_NULL)
                        groupID1 = idx_flag * factor[j];
                    else
                        groupID1 = GROUP_NULL;
                }

                if (i + 2 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 2 * blockDim.x]];
                    if (idx_flag != DIM_NULL)
                        groupID2 = idx_flag * factor[j];
                    else
                        groupID2 = GROUP_NULL;
                }

                if (i + 3 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 3 * blockDim.x]];
                    if (idx_flag != DIM_NULL)
                        groupID3 = idx_flag * factor[j];
                    else
                        groupID3 = GROUP_NULL;
                }
                if (i + 4 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 4 * blockDim.x]];
                    if (idx_flag != DIM_NULL)
                        groupID4 = idx_flag * factor[j];
                    else
                        groupID4 = GROUP_NULL;
                }
                if (i + 5 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 5 * blockDim.x]];
                    if (idx_flag != DIM_NULL)
                        groupID5 = idx_flag * factor[j];
                    else
                        groupID5 = GROUP_NULL;
                }
                if (i + 6 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 6 * blockDim.x]];
                    if (idx_flag != DIM_NULL)
                        groupID6 = idx_flag * factor[j];
                    else
                        groupID6 = GROUP_NULL;
                }
                if (i + 7 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 7 * blockDim.x]];
                    if (idx_flag != DIM_NULL)
                        groupID7 = idx_flag * factor[j];
                    else
                        groupID7 = GROUP_NULL;
                }
                if (i + 8 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 8 * blockDim.x]];
                    if (idx_flag != DIM_NULL)
                        groupID8 = idx_flag * factor[j];
                    else
                        groupID8 = GROUP_NULL;
                }
                if (i + 9 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 9 * blockDim.x]];
                    if (idx_flag != DIM_NULL)
                        groupID9 = idx_flag * factor[j];
                    else
                        groupID9 = GROUP_NULL;
                }
                if (i + 10 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 10 * blockDim.x]];
                    if (idx_flag != DIM_NULL)
                        groupID10 = idx_flag * factor[j];
                    else
                        groupID10 = GROUP_NULL;
                }
                if (i + 11 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 11 * blockDim.x]];
                    if (idx_flag != DIM_NULL)
                        groupID11 = idx_flag * factor[j];
                    else
                        groupID11 = GROUP_NULL;
                }
                if (i + 12 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 12 * blockDim.x]];
                    if (idx_flag != DIM_NULL)
                        groupID12 = idx_flag * factor[j];
                    else
                        groupID12 = GROUP_NULL;
                }
                if (i + 13 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 13 * blockDim.x]];
                    if (idx_flag != DIM_NULL)
                        groupID13 = idx_flag * factor[j];
                    else
                        groupID13 = GROUP_NULL;
                }
                /*if ( i + 14 * blockDim.x < blockEnd)
                {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 14 * blockDim.x]];
                    if (idx_flag != DIM_NULL)
                        groupID14 = idx_flag * factor[j];
                    else
                        groupID14 = GROUP_NULL;
                }
                if ( i + 15 * blockDim.x < blockEnd)
                {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 15 * blockDim.x]];
                    if (idx_flag != DIM_NULL)
                        groupID15 = idx_flag * factor[j];
                    else
                        groupID15 = GROUP_NULL;
                }*/

            } else {
                table_index = orders[j];
                idx_flag = dimvec_array[table_index][fk_array[table_index][i]];
                if ((groupID0 != GROUP_NULL) && (idx_flag != DIM_NULL))
                    groupID0 = groupID0 + idx_flag * factor[j];
                else
                    groupID0 = GROUP_NULL;
                if (i + blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + blockDim.x]];
                    if ((groupID1 != GROUP_NULL) && (idx_flag != DIM_NULL))
                        groupID1 = groupID1 + idx_flag * factor[j];
                    else
                        groupID1 = GROUP_NULL;
                }
                if (i + 2 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 2 * blockDim.x]];
                    if ((groupID2 != GROUP_NULL) && (idx_flag != DIM_NULL))
                        groupID2 = groupID2 + idx_flag * factor[j];
                    else
                        groupID2 = GROUP_NULL;
                }
                if (i + 3 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 3 * blockDim.x]];
                    if ((groupID3 != GROUP_NULL) && (idx_flag != DIM_NULL))
                        groupID3 = groupID3 + idx_flag * factor[j];
                    else
                        groupID3 = GROUP_NULL;
                }
                if (i + 4 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 4 * blockDim.x]];
                    if ((groupID4 != GROUP_NULL) && (idx_flag != DIM_NULL))
                        groupID4 = groupID4 + idx_flag * factor[j];
                    else
                        groupID4 = GROUP_NULL;
                }
                if (i + 5 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 5 * blockDim.x]];
                    if ((groupID5 != GROUP_NULL) && (idx_flag != DIM_NULL))
                        groupID5 = groupID5 + idx_flag * factor[j];
                    else
                        groupID5 = GROUP_NULL;
                }
                if (i + 6 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 6 * blockDim.x]];
                    if ((groupID6 != GROUP_NULL) && (idx_flag != DIM_NULL))
                        groupID6 = groupID6 + idx_flag * factor[j];
                    else
                        groupID6 = GROUP_NULL;
                }
                if (i + 7 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 7 * blockDim.x]];
                    if ((groupID7 != GROUP_NULL) && (idx_flag != DIM_NULL))
                        groupID7 = groupID7 + idx_flag * factor[j];
                    else
                        groupID7 = GROUP_NULL;
                }
                if (i + 8 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 8 * blockDim.x]];
                    if ((groupID8 != GROUP_NULL) && (idx_flag != DIM_NULL))
                        groupID8 = groupID8 + idx_flag * factor[j];
                    else
                        groupID8 = GROUP_NULL;
                }
                if (i + 9 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 9 * blockDim.x]];
                    if ((groupID9 != GROUP_NULL) && (idx_flag != DIM_NULL))
                        groupID9 = groupID9 + idx_flag * factor[j];
                    else
                        groupID9 = GROUP_NULL;
                }
                if (i + 10 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 10 * blockDim.x]];
                    if ((groupID10 != GROUP_NULL) && (idx_flag != DIM_NULL))
                        groupID10 = groupID10 + idx_flag * factor[j];
                    else
                        groupID10 = GROUP_NULL;
                }
                if (i + 11 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 11 * blockDim.x]];
                    if ((groupID11 != GROUP_NULL) && (idx_flag != DIM_NULL))
                        groupID11 = groupID11 + idx_flag * factor[j];
                    else
                        groupID11 = GROUP_NULL;
                }
                if (i + 12 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 12 * blockDim.x]];
                    if ((groupID12 != GROUP_NULL) && (idx_flag != DIM_NULL))
                        groupID12 = groupID12 + idx_flag * factor[j];
                    else
                        groupID12 = GROUP_NULL;
                }
                if (i + 13 * blockDim.x < blockEnd) {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 13 * blockDim.x]];
                    if ((groupID13 != GROUP_NULL) && (idx_flag != DIM_NULL))
                        groupID13 = groupID13 + idx_flag * factor[j];
                    else
                        groupID13 = GROUP_NULL;
                }
                /*if (i + 14 * blockDim.x < blockEnd)
                {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 14 * blockDim.x]];
                    if ((groupID14 != GROUP_NULL) && (idx_flag != DIM_NULL))
                        groupID14 = groupID14 + idx_flag * factor[j];
                    else
                        groupID14 = GROUP_NULL;
                }
                if (i + 15 * blockDim.x < blockEnd)
                {
                    idx_flag = dimvec_array[table_index][fk_array[table_index][i + 15 * blockDim.x]];
                    if ((groupID15 != GROUP_NULL) && (idx_flag != DIM_NULL))
                        groupID15 = groupID15 + idx_flag * factor[j];
                    else
                        groupID15 = GROUP_NULL;
                }*/
            }
        }

        if (groupID0 != GROUP_NULL) {
            sum = M1[i] + M2[i];
            atomicAdd(&group_vector[groupID0], sum);
        }
        if (i + blockDim.x < blockEnd) {
            if (groupID1 != GROUP_NULL) {
                sum = M1[i + blockDim.x] + M2[i + blockDim.x];
                atomicAdd(&group_vector[groupID1], sum);
            }
        }
        if (i + 2 * blockDim.x < blockEnd) {
            if (groupID2 != GROUP_NULL) {
                sum = M1[i + 2 * blockDim.x] + M2[i + 2 * blockDim.x];
                atomicAdd(&group_vector[groupID2], sum);
            }
        }
        if (i + 3 * blockDim.x < blockEnd) {
            if (groupID3 != GROUP_NULL) {
                sum = M1[i + 3 * blockDim.x] + M2[i + 3 * blockDim.x];
                atomicAdd(&group_vector[groupID3], sum);
            }
        }
        if (i + 4 * blockDim.x < blockEnd) {
            if (groupID4 != GROUP_NULL) {
                sum = M1[i + 4 * blockDim.x] + M2[i + 4 * blockDim.x];
                atomicAdd(&group_vector[groupID4], sum);
            }
        }
        if (i + 5 * blockDim.x < blockEnd) {
            if (groupID5 != GROUP_NULL) {
                sum = M1[i + 5 * blockDim.x] + M2[i + 5 * blockDim.x];
                atomicAdd(&group_vector[groupID5], sum);
            }
        }
        if (i + 6 * blockDim.x < blockEnd) {
            if (groupID6 != GROUP_NULL) {
                sum = M1[i + 6 * blockDim.x] + M2[i + 6 * blockDim.x];
                atomicAdd(&group_vector[groupID6], sum);
            }
        }
        if (i + 7 * blockDim.x < blockEnd) {
            if (groupID7 != GROUP_NULL) {
                sum = M1[i + 7 * blockDim.x] + M2[i + 7 * blockDim.x];
                atomicAdd(&group_vector[groupID7], sum);
            }
        }
        if (i + 8 * blockDim.x < blockEnd) {
            if (groupID8 != GROUP_NULL) {
                sum = M1[i + 8 * blockDim.x] + M2[i + 8 * blockDim.x];
                atomicAdd(&group_vector[groupID8], sum);
            }
        }
        if (i + 9 * blockDim.x < blockEnd) {
            if (groupID9 != GROUP_NULL) {
                sum = M1[i + 9 * blockDim.x] + M2[i + 9 * blockDim.x];
                atomicAdd(&group_vector[groupID9], sum);
            }
        }
        if (i + 10 * blockDim.x < blockEnd) {
            if (groupID10 != GROUP_NULL) {
                sum = M1[i + 10 * blockDim.x] + M2[i + 10 * blockDim.x];
                atomicAdd(&group_vector[groupID10], sum);
            }
        }
        if (i + 11 * blockDim.x < blockEnd) {
            if (groupID11 != GROUP_NULL) {
                sum = M1[i + 11 * blockDim.x] + M2[i + 11 * blockDim.x];
                atomicAdd(&group_vector[groupID11], sum);
            }
        }
        if (i + 12 * blockDim.x < blockEnd) {
            if (groupID12 != GROUP_NULL) {
                sum = M1[i + 12 * blockDim.x] + M2[i + 12 * blockDim.x];
                atomicAdd(&group_vector[groupID12], sum);
            }
        }
        if (i + 13 * blockDim.x < blockEnd) {
            if (groupID13 != GROUP_NULL) {
                sum = M1[i + 13 * blockDim.x] + M2[i + 13 * blockDim.x];
                atomicAdd(&group_vector[groupID13], sum);
            }
        }
        /*if (i + 14 * blockDim.x < blockEnd)
        {
            if (groupID14 != GROUP_NULL)
            {
                sum = M1[i + 14 * blockDim.x] + M2[i + 14 * blockDim.x];
                atomicAdd(&group_vector[groupID14], sum);
            }
        }
        if (i + 15 * blockDim.x < blockEnd)
        {
            if (groupID15 != GROUP_NULL)
            {
                sum = M1[i + 15 * blockDim.x] + M2[i + 15 * blockDim.x];
                atomicAdd(&group_vector[groupID15], sum);
            }
        }*/

        i += 15 * blockDim.x;
    }
}
__global__ void OLAPcore_columnwise_sv(int8_t** dimvec_array, int32_t** fk_array, int* size_array, int* orders, int* dimvec_nums, int16_t* groupID, int* factor, int* size_lineorder, int* group_nums, int32_t* M1, int32_t* M2, uint32_t* group_vector) {
    int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = 0;
    int16_t tmp = -1;
    int64_t comlength = 0;
    int sum = 0;

    for (int j = 0; j < *dimvec_nums; j++) {
        if (!j) {
            for (int k = threadIdx.x + blockIdx.x * blockDim.x; k < *size_lineorder; k += blockDim.x * gridDim.x) {
                int table_index = orders[j];
                int idx_flag = dimvec_array[table_index][fk_array[table_index][k]];
                if (idx_flag != DIM_NULL)
                    groupID[k] = idx_flag * factor[j];
                else
                    groupID[k] = GROUP_NULL;
            }
        } else {
            for (int k = threadIdx.x + blockIdx.x * blockDim.x; k < *size_lineorder; k += blockDim.x * gridDim.x) {
                int table_index = orders[j];
                int idx_flag = dimvec_array[table_index][fk_array[table_index][k]];
                if ((groupID[k] != GROUP_NULL) && (idx_flag != DIM_NULL))
                    groupID[k] += idx_flag * factor[j];
                else
                    groupID[k] = GROUP_NULL;
            }
        }
    }
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < *size_lineorder; i += blockDim.x * gridDim.x) {
        tmp = groupID[i];
        if (tmp != GROUP_NULL) {
            sum = M1[i] + M2[i];
            atomicAdd(&group_vector[tmp], sum);
        }
    }
}
/**
 * @brief test for GPU OLAPcore based on Row-wise model
 *
 * @param sele_array
 * @param dimvec_array
 * @param bitmap_array
 * @param fk_array
 * @param M1
 * @param M2
 * @param factor
 * @param orders
 * @param dimvec_nums
 * @param group_nums
 * @return void
 */
void test_OLAPcore_rowwise(double SF, double* sele_array, int8_t** dimvec_array, int* bitmap_array, int32_t** fk_array, int* size_array, int32_t* M1, int32_t* M2, int* factor, int* d_orders, int* h_orders, int* d_dimvec_nums, int& h_dimvec_nums, int* d_group_nums, int& group_nums, int sqlnum, std::ofstream& timefile) {
    std::cout << ">>> Start test GPU OLAPcore using row-wise model" << std::endl;
    cudaEvent_t start, stop;
    double total_rate = 1;
    for (int i = 0; i < h_dimvec_nums; i++)
        total_rate *= sele_array[h_orders[i]];
    int size_lineorder = size_of_table(TABLE_NAME::lineorder, SF);
    int* d_size_lineorder;
    gpuErrchk(cudaMalloc((void**)&d_size_lineorder, sizeof(int)));
    gpuErrchk(cudaMemcpy(d_size_lineorder, &size_lineorder, sizeof(int), cudaMemcpyHostToDevice));
    uint32_t* group_vector = new uint32_t[group_nums];
    memset(group_vector, 0, sizeof(uint32_t) * group_nums);
    uint32_t* d_group_vector;
    gpuErrchk(cudaMalloc((void**)&d_group_vector, sizeof(uint32_t) * (group_nums)));
    gpuErrchk(cudaMemcpy(d_group_vector, group_vector, sizeof(uint32_t) * (group_nums), cudaMemcpyHostToDevice));
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    OLAPcore_rowwise<<<BLOCK_NUM, THREAD_NUM>>>(dimvec_array, fk_array, size_array, d_orders, d_dimvec_nums, factor, d_size_lineorder, d_group_nums, M1, M2, d_group_vector);
    gpuErrchk(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaMemcpy(group_vector, d_group_vector, sizeof(uint32_t) * (group_nums), cudaMemcpyDeviceToHost);
    int count = 0;
    for (int i = 0; i < group_nums; i++)
        count += group_vector[i];
    float ms = 0.0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::cout << "          Result count of total selection rate " << total_rate * 100 << "% is " << count << "/" << size_lineorder << std::endl;
    std::cout << "          total time: " << ms << "ms" << std::endl;
    timefile << sqlnum
             << "\t"
             << "Row-wise query model"
             << "\t"
             << ""
             << "\t"
             << "Row-wise query model"
             << "\t"
             << "total"
             << "\t"
             << ms << std::endl;
}
void test_OLAPcore_cwm_dv(double SF, double* sele_array, int8_t** dimvec_array, int* bitmap_array, int32_t** fk_array, int* size_array, int32_t* M1, int32_t* M2, int* factor, int* d_orders, int* h_orders, int* d_dimvec_nums, int& h_dimvec_nums, int* d_group_nums, int& group_nums, int sqlnum, std::ofstream& timefile) {
    std::cout << ">>> Start test GPU OLAPcore using Column-wise model and dynamic vector" << std::endl;
    cudaEvent_t start, stop;
    double total_rate = 1;
    for (int i = 0; i < h_dimvec_nums; i++)
        total_rate *= sele_array[h_orders[i]];
    int size_lineorder = size_of_table(TABLE_NAME::lineorder, SF);
    int* d_size_lineorder;
    gpuErrchk(cudaMalloc((void**)&d_size_lineorder, sizeof(int)));
    gpuErrchk(cudaMemcpy(d_size_lineorder, &size_lineorder, sizeof(int), cudaMemcpyHostToDevice));
    int64_t* OID;
    int16_t* groupID;
    gpuErrchk(cudaMalloc((void**)&OID, sizeof(int64_t) * size_lineorder));
    gpuErrchk(cudaMalloc((void**)&groupID, sizeof(int16_t) * size_lineorder));
    uint32_t* group_vector = new uint32_t[group_nums];
    memset(group_vector, 0, sizeof(uint32_t) * group_nums);
    uint32_t* d_group_vector;
    gpuErrchk(cudaMalloc((void**)&d_group_vector, sizeof(uint32_t) * (group_nums)));
    gpuErrchk(cudaMemcpy(d_group_vector, group_vector, sizeof(uint32_t) * (group_nums), cudaMemcpyHostToDevice));

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    OLAPcore_columnwise_dv<<<BLOCK_NUM, THREAD_NUM>>>(dimvec_array, fk_array, size_array, d_orders, d_dimvec_nums, OID, groupID, factor, d_size_lineorder, d_group_nums, M1, M2, d_group_vector);
    gpuErrchk(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms = 0.0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(group_vector, d_group_vector, sizeof(uint32_t) * (group_nums), cudaMemcpyDeviceToHost);
    int count = 0;
    for (int i = 0; i < group_nums; i++)
        count += group_vector[i];
    std::cout << "          Result count of total selection rate " << total_rate * 100 << "% is " << count << "/" << size_lineorder << std::endl;
    std::cout << "          Total time: " << ms << "ms" << std::endl;
    timefile << sqlnum
             << "\t"
             << "Column-wise query model"
             << "\t"
             << "dynamic vector"
             << "\t"
             << "Column-wise query model with dynamic vector"
             << "\t"
             << "total"
             << "\t"
             << ms << std::endl;
}
void test_OLAPcore_vwm_dv(double SF, double* sele_array, int8_t** dimvec_array, int* bitmap_array, int32_t** fk_array, int* size_array, int32_t* M1, int32_t* M2, int* factor, int* d_orders, int* h_orders, int* d_dimvec_nums, int& h_dimvec_nums, int* d_group_nums, int& group_nums, int sqlnum, std::ofstream& timefile) {
    std::cout << ">>> Start test GPU OLAPcore using Vector-wise model and dynamic vector" << std::endl;
    cudaEvent_t start, stop;
    double total_rate = 1;
    for (int i = 0; i < h_dimvec_nums; i++)
        total_rate *= sele_array[h_orders[i]];
    int size_lineorder = size_of_table(TABLE_NAME::lineorder, SF);
    int* d_size_lineorder;
    gpuErrchk(cudaMalloc((void**)&d_size_lineorder, sizeof(int)));
    gpuErrchk(cudaMemcpy(d_size_lineorder, &size_lineorder, sizeof(int), cudaMemcpyHostToDevice));
    uint32_t* group_vector = new uint32_t[group_nums];
    memset(group_vector, 0, sizeof(uint32_t) * group_nums);
    uint32_t* d_group_vector;
    gpuErrchk(cudaMalloc((void**)&d_group_vector, sizeof(uint32_t) * (group_nums)));
    gpuErrchk(cudaMemcpy(d_group_vector, group_vector, sizeof(uint32_t) * (group_nums), cudaMemcpyHostToDevice));
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    OLAPcore_vectorwise_dv<<<BLOCK_NUM, THREAD_NUM>>>(dimvec_array, fk_array, size_array, d_orders, d_dimvec_nums, factor, d_size_lineorder, d_group_nums, M1, M2, d_group_vector);
    gpuErrchk(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms = 0.0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(group_vector, d_group_vector, sizeof(uint32_t) * (group_nums), cudaMemcpyDeviceToHost);
    int count = 0;
    for (int i = 0; i < group_nums; i++)
        count += group_vector[i];
    std::cout << "          Result count of total selection rate " << total_rate * 100 << "% is " << count << "/" << size_lineorder << std::endl;
    std::cout << "          Total time: " << ms << "ms" << std::endl;
    timefile << sqlnum
             << "\t"
             << "Vector-wise query model"
             << "\t"
             << "dynamic vector"
             << "\t"
             << "Vector-wise query model with dynamic vector"
             << "\t"
             << "total"
             << "\t"
             << ms << std::endl;
}
void test_OLAPcore_vwm_sv(double SF, double* sele_array, int8_t** dimvec_array, int* bitmap_array, int32_t** fk_array, int* size_array, int32_t* M1, int32_t* M2, int* factor, int* d_orders, int* h_orders, int* d_dimvec_nums, int& h_dimvec_nums, int* d_group_nums, int& group_nums, int sqlnum, std::ofstream& timefile) {
    std::cout << ">>> Start test GPU OLAPcore using Vector-wise model and static vector" << std::endl;
    cudaEvent_t start, stop;
    double total_rate = 1;
    for (int i = 0; i < h_dimvec_nums; i++)
        total_rate *= sele_array[h_orders[i]];
    int size_lineorder = size_of_table(TABLE_NAME::lineorder, SF);
    int* d_size_lineorder;
    gpuErrchk(cudaMalloc((void**)&d_size_lineorder, sizeof(int)));
    gpuErrchk(cudaMemcpy(d_size_lineorder, &size_lineorder, sizeof(int), cudaMemcpyHostToDevice));
    uint32_t* group_vector = new uint32_t[group_nums];
    memset(group_vector, 0, sizeof(uint32_t) * group_nums);
    uint32_t* d_group_vector;
    gpuErrchk(cudaMalloc((void**)&d_group_vector, sizeof(uint32_t) * (group_nums)));
    gpuErrchk(cudaMemcpy(d_group_vector, group_vector, sizeof(uint32_t) * (group_nums), cudaMemcpyHostToDevice));
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    OLAPcore_vectorwise_sv<<<BLOCK_NUM, THREAD_NUM>>>(dimvec_array, fk_array, size_array, d_orders, d_dimvec_nums, factor, d_size_lineorder, d_group_nums, M1, M2, d_group_vector);
    gpuErrchk(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms = 0.0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(group_vector, d_group_vector, sizeof(uint32_t) * (group_nums), cudaMemcpyDeviceToHost);
    int count = 0;
    for (int i = 0; i < group_nums; i++)
        count += group_vector[i];
    std::cout << "          Result count of total selection rate " << total_rate * 100 << "% is " << count << "/" << size_lineorder << std::endl;
    std::cout << "          Total time: " << ms << "ms" << std::endl;
    timefile << sqlnum
             << "\t"
             << "Vector-wise query model"
             << "\t"
             << "static vector"
             << "\t"
             << "Vector-wise query model with static vector"
             << "\t"
             << "total"
             << "\t"
             << ms << std::endl;
}
void test_OLAPcore_cwm_sv(double SF, double* sele_array, int8_t** dimvec_array, int* bitmap_array, int32_t** fk_array, int* size_array, int32_t* M1, int32_t* M2, int* factor, int* d_orders, int* h_orders, int* d_dimvec_nums, int& h_dimvec_nums, int* d_group_nums, int& group_nums, int sqlnum, std::ofstream& timefile) {
    std::cout << ">>> Start test GPU OLAPcore using Column-wise model and static vector" << std::endl;
    cudaEvent_t start, stop;
    double total_rate = 1;
    for (int i = 0; i < h_dimvec_nums; i++)
        total_rate *= sele_array[h_orders[i]];
    int size_lineorder = size_of_table(TABLE_NAME::lineorder, SF);
    int* d_size_lineorder;
    gpuErrchk(cudaMalloc((void**)&d_size_lineorder, sizeof(int)));
    gpuErrchk(cudaMemcpy(d_size_lineorder, &size_lineorder, sizeof(int), cudaMemcpyHostToDevice));
    int16_t* groupID;
    gpuErrchk(cudaMalloc((void**)&groupID, sizeof(int16_t) * size_lineorder));
    uint32_t* group_vector = new uint32_t[group_nums];
    memset(group_vector, 0, sizeof(uint32_t) * group_nums);
    uint32_t* d_group_vector;
    gpuErrchk(cudaMalloc((void**)&d_group_vector, sizeof(uint32_t) * (group_nums)));
    gpuErrchk(cudaMemcpy(d_group_vector, group_vector, sizeof(uint32_t) * (group_nums), cudaMemcpyHostToDevice));
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    OLAPcore_columnwise_sv<<<BLOCK_NUM, THREAD_NUM>>>(dimvec_array, fk_array, size_array, d_orders, d_dimvec_nums, groupID, factor, d_size_lineorder, d_group_nums, M1, M2, d_group_vector);
    gpuErrchk(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms = 0.0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(group_vector, d_group_vector, sizeof(uint32_t) * (group_nums), cudaMemcpyDeviceToHost);
    int count = 0;
    for (int i = 0; i < group_nums; i++)
        count += group_vector[i];
    std::cout << "          Result count of total selection rate " << total_rate * 100 << "% is " << count << "/" << size_lineorder << std::endl;
    std::cout << "          Total time: " << ms << "ms" << std::endl;
    timefile << sqlnum
             << "\t"
             << "Column-wise query model"
             << "\t"
             << "static vector"
             << "\t"
             << "Column-wise query model with static vector"
             << "\t"
             << "total"
             << "\t"
             << ms << std::endl;
}
void test_OLAPcore_columnwise(double SF, double* sele_array, int8_t** dimvec_array, int* bitmap_array, int32_t** fk_array, int* size_array, int32_t* M1, int32_t* M2, int* factor, int* d_orders, int* h_orders, int* d_dimvec_nums, int& h_dimvec_nums, int* d_group_nums, int& group_nums, int sqlnum, std::ofstream& timefile) {
    /* 1. OLAPcore based on Column-wise model and dynamic vector*/
    test_OLAPcore_cwm_dv(SF, sele_array, dimvec_array, bitmap_array, fk_array, size_array,
                         M1, M2, factor, d_orders, h_orders, d_dimvec_nums, h_dimvec_nums, d_group_nums, group_nums, sqlnum, timefile);
    /* 2. OLAPcore based on Column-wise model and static vector*/
    test_OLAPcore_cwm_sv(SF, sele_array, dimvec_array, bitmap_array, fk_array, size_array,
                         M1, M2, factor, d_orders, h_orders, d_dimvec_nums, h_dimvec_nums, d_group_nums, group_nums, sqlnum, timefile);
}
void test_OLAPcore_vectorwise(double SF, double* sele_array, int8_t** dimvec_array, int* bitmap_array, int32_t** fk_array, int* size_array, int32_t* M1, int32_t* M2, int* factor, int* d_orders, int* h_orders, int* d_dimvec_nums, int& h_dimvec_nums, int* d_group_nums, int& group_nums, int sqlnum, std::ofstream& timefile) {
    /* 1. OLAPcore based on Vector-wise model and dynamic vector*/
    test_OLAPcore_vwm_dv(SF, sele_array, dimvec_array, bitmap_array, fk_array, size_array,
                         M1, M2, factor, d_orders, h_orders, d_dimvec_nums, h_dimvec_nums, d_group_nums, group_nums, sqlnum, timefile);
    /* 2. OLAPcore based on Vector-wise model and static vector*/
    test_OLAPcore_vwm_sv(SF, sele_array, dimvec_array, bitmap_array, fk_array, size_array,
                         M1, M2, factor, d_orders, h_orders, d_dimvec_nums, h_dimvec_nums, d_group_nums, group_nums, sqlnum, timefile);
}
int main(int argc, char** argv) {
    /* Command line parameters */
    param_t cmd_params;
    double tim_all[3] = {0};
    struct timeval start, end;
    int i, j;
    int index_num = 0;
    cmd_params.nthreads = 2;
    cmd_params.sqlnum = 0;
    cmd_params.sf = 0.0;
    cmd_params.d_sele = 0.0;
    cmd_params.d_groups = 0;
    cmd_params.s_sele = 0.0;
    cmd_params.s_groups = 0;
    cmd_params.p_sele = 0.0;
    cmd_params.p_groups = 0;
    cmd_params.c_sele = 0.0;
    cmd_params.c_groups = 0;
    cmd_params.d_bitmap = 0;
    cmd_params.s_bitmap = 0;
    cmd_params.p_bitmap = 0;
    cmd_params.c_bitmap = 0;
    parse_args(argc, argv, &cmd_params);

    std::ofstream timefile;
    if (cmd_params.sqlnum == 21) {
        timefile.open(OLAPCORE_TEST_TIME_FILE, std::ios::out | std::ios::trunc);
        timefile << "query number"
                 << "\t"
                 << "query model"
                 << "\t"
                 << "intermediate results"
                 << "\t"
                 << "query model with different intermediate result"
                 << "\t"
                 << "phase type"
                 << "\t"
                 << "runtimes(ms)" << std::endl;
    } else
        timefile.open(OLAPCORE_TEST_TIME_FILE, std::ios::app);
    int8_t *h_dimvec_c, *h_dimvec_s, *h_dimvec_p, *h_dimvec_d;
    int32_t *h_fk_c, *h_fk_s, *h_fk_p, *h_fk_d;
    int32_t *h_M1, *h_M2;
    int h_factor[4] = {1, 1, 1, 1};
    uint32_t groups[4] = {cmd_params.c_groups, cmd_params.s_groups, cmd_params.p_groups, cmd_params.d_groups};
    gen_data(cmd_params.c_sele, cmd_params.s_sele, cmd_params.p_sele, cmd_params.d_sele, cmd_params.sf, cmd_params.c_bitmap, cmd_params.s_bitmap, cmd_params.p_bitmap, cmd_params.d_bitmap,
             cmd_params.c_groups, cmd_params.s_groups, cmd_params.p_groups, cmd_params.d_groups,
             h_dimvec_c, h_dimvec_s, h_dimvec_p, h_dimvec_d,
             h_fk_c, h_fk_s, h_fk_p, h_fk_d,
             h_M1, h_M2);
    int h_dimvec_nums = 0;
    int h_group_nums = 1;
    int* h_size_array = new int[4];
    h_size_array[0] = size_of_table(TABLE_NAME::customer, cmd_params.sf);
    h_size_array[1] = size_of_table(TABLE_NAME::supplier, cmd_params.sf);
    h_size_array[2] = size_of_table(TABLE_NAME::part, cmd_params.sf);
    h_size_array[3] = size_of_table(TABLE_NAME::date, cmd_params.sf);
    int size_lineorder = size_of_table(TABLE_NAME::lineorder, cmd_params.sf);
    double h_sele_array[4];

    int8_t* h_dimvec_array[4];
    int h_bitmap_array[4];
    int32_t* h_fk_array[4];
    int h_orders[4];

    /*customer*/
    h_sele_array[0] = cmd_params.c_sele;
    h_dimvec_array[0] = h_dimvec_c;
    h_bitmap_array[0] = cmd_params.c_bitmap;
    h_fk_array[0] = h_fk_c;
    /*supplier*/
    h_sele_array[1] = cmd_params.s_sele;
    h_dimvec_array[1] = h_dimvec_s;
    h_bitmap_array[1] = cmd_params.s_bitmap;
    h_fk_array[1] = h_fk_s;
    /*part*/
    h_sele_array[2] = cmd_params.p_sele;
    h_dimvec_array[2] = h_dimvec_p;
    h_bitmap_array[2] = cmd_params.p_bitmap;
    h_fk_array[2] = h_fk_p;
    /*date*/
    h_sele_array[3] = cmd_params.d_sele;
    h_dimvec_array[3] = h_dimvec_d;
    h_bitmap_array[3] = cmd_params.d_bitmap;
    h_fk_array[3] = h_fk_d;
    for (int i = 0; i < 4; i++) {
        if (groups[i] == 0)
            continue;
        else {
            h_orders[h_dimvec_nums] = i;
            h_dimvec_nums++;
        }
    }

    // set the join order by select rate
    sort_by_rate(h_sele_array, h_orders, h_dimvec_nums);
    for (int i = 0; i < h_dimvec_nums; i++) {
        h_group_nums *= groups[h_orders[i]];
        for (int j = i + 1; j < h_dimvec_nums; j++) {
            if (h_bitmap_array[h_orders[i]] == 0)
                break;
            h_factor[i] *= groups[h_orders[j]];
        }
    }
    /*GPU data structure definition and declaration*/
    double* d_sele_array;
    int8_t** d_dimvec_array;
    int* d_bitmap_array;
    int32_t** d_fk_array;
    int32_t *d_M1, *d_M2;
    int* d_factor;
    int* d_group_nums;
    int* d_dimvec_nums;
    int* d_orders;
    int* d_size_array;
    cudaMalloc((void**)&d_size_array, sizeof(int) * 4);
    cudaMemcpy(d_size_array, h_size_array, sizeof(int) * 4, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_group_nums, sizeof(int));
    cudaMemcpy(d_group_nums, &h_group_nums, sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_dimvec_nums, sizeof(int));
    cudaMemcpy(d_dimvec_nums, &h_dimvec_nums, sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_factor, sizeof(int) * 4);
    cudaMemcpy(d_factor, h_factor, sizeof(int) * 4, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_sele_array, sizeof(double) * 4);
    cudaMemcpy(d_sele_array, h_sele_array, sizeof(double) * 4, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_bitmap_array, sizeof(int) * 4);
    cudaMemcpy(d_bitmap_array, h_bitmap_array, sizeof(int) * 4, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_orders, sizeof(int) * 4);
    cudaMemcpy(d_orders, h_orders, sizeof(int) * 4, cudaMemcpyHostToDevice);
    /*dimension table */
    cudaMallocManaged(&d_dimvec_array, sizeof(int8_t*) * 4);
    cudaMallocManaged(&d_dimvec_array[0], sizeof(int8_t) * h_size_array[0]);
    cudaMemcpy(d_dimvec_array[0], h_dimvec_array[0], sizeof(int8_t) * h_size_array[0], cudaMemcpyHostToDevice);
    cudaMallocManaged(&d_dimvec_array[1], sizeof(int8_t) * h_size_array[1]);
    cudaMemcpy(d_dimvec_array[1], h_dimvec_array[1], sizeof(int8_t) * h_size_array[1], cudaMemcpyHostToDevice);
    cudaMallocManaged(&d_dimvec_array[2], sizeof(int8_t) * h_size_array[2]);
    cudaMemcpy(d_dimvec_array[2], h_dimvec_array[2], sizeof(int8_t) * h_size_array[2], cudaMemcpyHostToDevice);
    cudaMallocManaged(&d_dimvec_array[3], sizeof(int8_t) * h_size_array[3]);
    cudaMemcpy(d_dimvec_array[3], h_dimvec_array[3], sizeof(int8_t) * h_size_array[3], cudaMemcpyHostToDevice);
    /*fact table*/
    cudaMallocManaged(&d_fk_array, sizeof(int32_t*) * 4);
    cudaMallocManaged(&d_fk_array[0], sizeof(int32_t) * size_lineorder);
    cudaMemcpy(d_fk_array[0], h_fk_array[0], sizeof(int32_t) * size_lineorder, cudaMemcpyHostToDevice);
    cudaMallocManaged(&d_fk_array[1], sizeof(int32_t) * size_lineorder);
    cudaMemcpy(d_fk_array[1], h_fk_array[1], sizeof(int32_t) * size_lineorder, cudaMemcpyHostToDevice);
    cudaMallocManaged(&d_fk_array[2], sizeof(int32_t) * size_lineorder);
    cudaMemcpy(d_fk_array[2], h_fk_array[2], sizeof(int32_t) * size_lineorder, cudaMemcpyHostToDevice);
    cudaMallocManaged(&d_fk_array[3], sizeof(int32_t) * size_lineorder);
    cudaMemcpy(d_fk_array[3], h_fk_array[3], sizeof(int32_t) * size_lineorder, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_M1, sizeof(int32_t) * size_lineorder);
    cudaMemcpy(d_M1, h_M1, sizeof(int32_t) * size_lineorder, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_M2, sizeof(int32_t) * size_lineorder);
    cudaMemcpy(d_M2, h_M2, sizeof(int32_t) * size_lineorder, cudaMemcpyHostToDevice);
    /* 1. GPU OLAPcore based on Row-wise model*/
    test_OLAPcore_rowwise(cmd_params.sf, h_sele_array, d_dimvec_array, d_bitmap_array, d_fk_array, d_size_array,
                          d_M1, d_M2, d_factor, d_orders, h_orders, d_dimvec_nums, h_dimvec_nums, d_group_nums, h_group_nums, cmd_params.sqlnum, timefile);
    /* 2. GPU OLAPcore based on Column-wise model*/
    test_OLAPcore_columnwise(cmd_params.sf, h_sele_array, d_dimvec_array, d_bitmap_array, d_fk_array, d_size_array,
                             d_M1, d_M2, d_factor, d_orders, h_orders, d_dimvec_nums, h_dimvec_nums, d_group_nums, h_group_nums, cmd_params.sqlnum, timefile);
    /* 3. GPU OLAPcore based on Vector-wise model*/
    test_OLAPcore_vectorwise(cmd_params.sf, h_sele_array, d_dimvec_array, d_bitmap_array, d_fk_array, d_size_array,
                             d_M1, d_M2, d_factor, d_orders, h_orders, d_dimvec_nums, h_dimvec_nums, d_group_nums, h_group_nums, cmd_params.sqlnum, timefile);

    return 0;
}
