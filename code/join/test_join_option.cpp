/**
 * @file test_join_option.cpp
 * @author ruichenhan (hanruichen@ruc.edu.cn)
 * @brief test join algorithms
 * @version 0.1
 * @date 2023-05-02
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <sys/time.h>
#include <unordered_map>
#include "../include/statistical_analysis_util.hpp"
#include "../include/metadata.h"
#include "../include/gendata_util.hpp"
/**
 * @brief vecjoin build phase
 * 
 * @param R Table data for building the vector
 * @param vector Intermediate results——vector
 * @return void 
 */
void build(relation_t* R, int * vector)
{
    for (int i = 0; i < R->num_tuples; i++)
    {
        vector[R->key[i]] = R->payload[i];
    }
}
/**
 * @brief hashjoin build phase
 * 
 * @param R Table data for building the vector
 * @param hashtable Intermediate results——hashtable
 * @return void 
 */
void build(relation_t* R, std::unordered_map<int, int> &hashtable)
{
    for (int i = 0; i < R->num_tuples; i++)
    {
        hashtable[R->key[i]] = R->payload[i];
    }
}
/**
 * @brief vecjoin probe phase
 * 
 * @param S Table data for probe the vector
 * @param vector Intermediate results——vector
 * @return void 
 */
int probe(relation_t* S, int * vector)
{
    int count = 0;
    for (int i = 0; i < S->num_tuples; i++)
        count += vector[S->key[i]];
    return count;
}
/**
 * @brief vecjoin probe phase based on prefetching optimization
 * 
 * @param S Table data for probe the vector
 * @param vector Intermediate results——vector
 * @return void 
 */
int probe_prefetch(relation_t* S, int * vector)
{
    int count = 0;
    for (int i = 0; i < S->num_tuples; i++)
    {
        __builtin_prefetch(&S->key[i + 10]);
        count += vector[S->key[i]];
    }
        
    return count;
}
/**
 * @brief hashjoin probe phase
 * 
 * @param S Table data for probe the vector
 * @param hashtable Intermediate results——hashtable
 * @return void 
 */
int probe(relation_t* S, std::unordered_map<int, int> &hashtable)
{
    int count = 0;
    for (int i = 0; i < S->num_tuples; i++)
        count += hashtable[S->key[i]];
    return count;
}

/** 
 * @brief join algorithm based on the vector 
 * 
 * @param R Table data for building the vector
 * @param S Table data for probing the vector
 * @param timefile 
 * @return void 
 */
void test_vecjoin(relation_t* R, 
                  relation_t* S,
                  std::ofstream& timefile,
                  int8_t test_flag)
{
    std::cout << ">>> Start test join by vector" << std::endl;
    
    timeval start, end;
    double ms_build, ms_probe;
    int count;
    if (!test_flag || test_flag == 3)
    {
        int * vector  = (int *)malloc(sizeof(int) * (R->num_tuples + 1));
        
        std::cout << ">>> vecjoin build phase" << std::endl;
        gettimeofday(&start, NULL);
        build(R, vector);
        gettimeofday(&end, NULL);
        ms_build = calc_ms(end, start);
        std::cout << ">>> vecjoin probe phase" << std::endl;
        gettimeofday(&start, NULL);
        count = probe(S, vector);
        gettimeofday(&end, NULL);
        ms_probe = calc_ms(end, start);
    }
    else
    {
        int vector[R->num_tuples];
        std::cout << ">>> vecjoin build phase" << std::endl;
        gettimeofday(&start, NULL);
        build(R, vector);
        gettimeofday(&end, NULL);
        ms_build = calc_ms(end, start);
        std::cout << ">>> vecjoin probe phase" << std::endl;
        gettimeofday(&start, NULL);
        count = probe(S, vector);
        gettimeofday(&end, NULL);
        ms_probe = calc_ms(end, start);
    }
    
    std::cout << "          Result count is "<< count << "/" << S->num_tuples << std::endl;
    std::cout << "          Build phase time: " << ms_build << "ms" << std::endl;
    std::cout << "          Probe phase time: " << ms_probe << "ms" << std::endl;
    std::cout << "          Total time: " << ms_build + ms_probe << "ms" << std::endl;
    switch(test_flag)
    {
        case 0:
            timefile << "vecjoin"
                     << "\t"
                     << R->table_size
                     << "\t"
                     << "build"
                     << "\t"
                     << ms_build<< std::endl;
            timefile << "vecjoin"
                     << "\t"
                     << R->table_size
                     << "\t"
                     << "probe"
                     << "\t"
                     << ms_probe<< std::endl;
            timefile << "vecjoin"
                     << "\t"
                     << R->table_size
                     << "\t"
                     << "total"
                     << "\t"
                     << ms_build + ms_probe<< std::endl;
        break;
        case 1:
            timefile << "vecjoin"
                     << "\t"
                     << "L1"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "build"
                     << "\t"
                     << ms_build << std::endl;
            timefile << "vecjoin"
                     << "\t"
                     << "L1"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "probe"
                     << "\t"
                     <<ms_probe << std::endl;
            timefile << "vecjoin"
                     << "\t"
                     << "L1"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "total"
                     << "\t"
                     <<ms_build + ms_probe << std::endl;
        break;
        case 2:
            timefile << "vecjoin"
                     << "\t"
                     << "L2"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "build"
                     << "\t"
                     << ms_build<< std::endl;
            timefile << "vecjoin"
                     << "\t"
                     << "L2"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "probe"
                     << "\t"
                     << ms_probe<< std::endl;
            timefile << "vecjoin"
                     << "\t"
                     << "L2"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "total"
                     << "\t"
                     << ms_build + ms_probe<< std::endl;
        break;
        case 3:
            timefile << "vecjoin"
                     << "\t"
                     << "L3"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "build"
                     << "\t"
                     << ms_build<< std::endl;
            timefile << "vecjoin"
                     << "\t"
                     << "L3"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "probe"
                     << "\t"
                     << ms_probe<< std::endl;
            timefile << "vecjoin"
                     << "\t"
                     << "L3"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "total"
                     << "\t"
                     << ms_build + ms_probe<< std::endl;
        break;
    }
    
    
}
/** 
 * @brief join algorithm based on the vector  based on prefetching optimization
 * 
 * @param R Table data for building the vector
 * @param S Table data for probing the vector
 * @param timefile 
 * @return void 
 */
void test_vecjoin_prefetch(relation_t* R, 
                  relation_t* S,
                  std::ofstream& timefile,
                  int8_t test_flag)
{
    std::cout << ">>> Start test join by vector based on prefetching optimization" << std::endl;
    
    timeval start, end;
    double ms_build, ms_probe;
    int count;
    if (!test_flag || test_flag == 3)
    {
        int * vector  = (int *)malloc(sizeof(int) * (R->num_tuples + 1));
        
        std::cout << ">>> vecjoin build phase" << std::endl;
        gettimeofday(&start, NULL);
        build(R, vector);
        gettimeofday(&end, NULL);
        ms_build = calc_ms(end, start);
        std::cout << ">>> vecjoin probe phase" << std::endl;
        gettimeofday(&start, NULL);
        count = probe_prefetch(S, vector);
        gettimeofday(&end, NULL);
        ms_probe = calc_ms(end, start);
    }
    else
    {
        int vector[R->num_tuples];
        std::cout << ">>> vecjoin build phase" << std::endl;
        gettimeofday(&start, NULL);
        build(R, vector);
        gettimeofday(&end, NULL);
        ms_build = calc_ms(end, start);
        std::cout << ">>> vecjoin probe phase" << std::endl;
        gettimeofday(&start, NULL);
        count = probe_prefetch(S, vector);
        gettimeofday(&end, NULL);
        ms_probe = calc_ms(end, start);
    }
    
    std::cout << "          Result count is "<< count << "/" << S->num_tuples << std::endl;
    std::cout << "          Build phase time: " << ms_build << "ms" << std::endl;
    std::cout << "          Probe phase time: " << ms_probe << "ms" << std::endl;
    std::cout << "          Total time: " << ms_build + ms_probe << "ms" << std::endl;
    switch(test_flag)
    {
        case 0:
            timefile << "vecjoin based on prefetching optimization"
                     << "\t"
                     << R->table_size
                     << "\t"
                     << "build"
                     << "\t"
                     << ms_build<< std::endl;
            timefile << "vecjoin based on prefetching optimization"
                     << "\t"
                     << R->table_size
                     << "\t"
                     << "probe"
                     << "\t"
                     << ms_probe<< std::endl;
            timefile << "vecjoin based on prefetching optimization"
                     << "\t"
                     << R->table_size
                     << "\t"
                     << "total"
                     << "\t"
                     << ms_build + ms_probe<< std::endl;
        break;
        case 1:
            timefile << "vecjoin based on prefetching optimization"
                     << "\t"
                     << "L1"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "build"
                     << "\t"
                     << ms_build << std::endl;
            timefile << "vecjoin based on prefetching optimization"
                     << "\t"
                     << "L1"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "probe"
                     << "\t"
                     << ms_probe << std::endl;
            timefile << "vecjoin based on prefetching optimization"
                     << "\t"
                     << "L1"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "total"
                     << "\t"
                     << ms_build + ms_probe << std::endl;
        break;
        case 2:
            timefile << "vecjoin based on prefetching optimization"
                     << "\t"
                     << "L2"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "build"
                     << "\t"
                     << ms_build<< std::endl;
            timefile << "vecjoin based on prefetching optimization"
                     << "\t"
                     << "L2"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "probe"
                     << "\t"
                     << ms_probe<< std::endl;
            timefile << "vecjoin based on prefetching optimization"
                     << "\t"
                     << "L2"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "total"
                     << "\t"
                     << ms_build + ms_probe<< std::endl;
        break;
        case 3:
            timefile << "vecjoin based on prefetching optimization"
                     << "\t"
                     << "L3"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "build"
                     << "\t"
                     << ms_build << std::endl;
            timefile << "vecjoin based on prefetching optimization"
                     << "\t"
                     << "L3"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "probe"
                     << "\t"
                     << ms_probe << std::endl;
            timefile << "vecjoin based on prefetching optimization"
                     << "\t"
                     << "L3"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "total"
                     << "\t"
                     << ms_build + ms_probe << std::endl;
        break;
    }
    
    
}
/**
 * @brief join algorithm based on the hashtable
 * 
 * @param R Table data for building the vector
 * @param S Table data for probing the vector
 * @param timefile 
 * @return void 
 */
void test_hashjoin(relation_t* R, 
                  relation_t* S,
                  std::ofstream& timefile,
                  int8_t test_flag)
{
    std::cout << ">>> Start test join by hashtable" << std::endl;
    timeval start, end;
    std::unordered_map<int, int> hashtable;
    hashtable.reserve(R->num_tuples);
    std::cout << ">>> hashjoin build phase" << std::endl;
    gettimeofday(&start, NULL);
    build(R, hashtable);
    gettimeofday(&end, NULL);
    double ms_build = calc_ms(end, start);
    std::cout << ">>> hashjoin probe phase" << std::endl;
    gettimeofday(&start, NULL);
    int count = probe(S, hashtable);
    gettimeofday(&end, NULL);
    double ms_probe = calc_ms(end, start);
    std::cout << "          Result count is "<< count << "/" << S->num_tuples << std::endl;
    std::cout << "          Build phase time: " << ms_build << "ms" << std::endl;
    std::cout << "          Probe phase time: " << ms_probe << "ms" << std::endl;
    std::cout << "          Total time: " << ms_build + ms_probe << "ms" << std::endl;
    switch(test_flag)
    {
        case 0:
            timefile << "hashjoin"
                     << "\t"
                     << R->table_size
                     << "\t"
                     << "build"
                     << "\t"
                     << ms_build<< std::endl;
            timefile << "hashjoin"
                     << "\t"
                     << R->table_size
                     << "\t"
                     << "probe"
                     << "\t"
                     << ms_probe<< std::endl;
            timefile << "hashjoin"
                     << "\t"
                     << R->table_size
                     << "\t"
                     << "total"
                     << "\t"
                     << ms_build + ms_probe<< std::endl;
        break;
        case 1:
            timefile << "hashjoin"
                     << "\t"
                     << "L1"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "build"
                     << "\t"
                     << ms_build<< std::endl;
            timefile << "hashjoin"
                     << "\t"
                     << "L1"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "probe"
                     << "\t"
                     << ms_probe<< std::endl;
            timefile << "hashjoin"
                     << "\t"
                     << "L1"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "total"
                     << "\t"
                     << ms_build + ms_probe<< std::endl;
        break;
        case 2:
            timefile << "hashjoin"
                     << "\t"
                     << "L2"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "build"
                     << "\t"
                     << ms_build << std::endl;
            timefile << "hashjoin"
                     << "\t"
                     << "L2"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "probe"
                     << "\t"
                     << ms_probe << std::endl;
            timefile << "hashjoin"
                     << "\t"
                     << "L2"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "total"
                     << "\t"
                     << ms_build + ms_probe << std::endl;
        break;
        case 3:
            timefile << "hashjoin"
                     << "\t"
                     << "L3"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "build"
                     << "\t"
                     << ms_build<< std::endl;
            timefile << "hashjoin"
                     << "\t"
                     << "L3"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "probe"
                     << "\t"
                     << ms_probe<< std::endl;
            timefile << "hashjoin"
                     << "\t"
                     << "L3"
                     << "\t"
                     << R->payload_rate
                     << "\t"
                     << "total"
                     << "\t"
                     << ms_build + ms_probe<< std::endl;
        break;
    }
    
}
int main()
{
    relation_t *R, *S;
    std::ofstream test1_timefile, test2_timefile;
    test1_timefile.open(JOINALGO_TEST1_TIME_FILE, std::ios::out | std::ios::trunc);
    test2_timefile.open(JOINALGO_TEST2_TIME_FILE, std::ios::out | std::ios::trunc);
    test1_timefile << "join algorithm" 
                   << "\t"
                   << "table R size"
                   << "\t"
                   << "phase type"
                   << "\t"
                   << "running time"
                   << std::endl;
    test2_timefile << "join algorithm" 
                   << "\t"
                   << "table R size"
                   << "\t"
                   << "Cache type"
                   << "\t"
                   << "phase type"
                   << "\t"
                   << "running time"
                   << std::endl;
    R = new relation_t;
    S = new relation_t;
    std::cout<<"test 1: Comparison of Join Algorithm Performance with Different Load Sizes"<< std::endl;
    int R_size = 5;
    int S_size = 30;
    R->table_size = 0;
    R->payload_rate = 0.0;
    for(; R_size <= S_size; R_size ++)
    {
        int maxid = 1 << R_size;
        int S_nums = 1 << S_size;
        R->key  = new int32_t[maxid];
        R->payload = new int32_t[maxid];
        R->table_size = R_size;
        R->num_tuples = maxid;
        S->key = new int32_t[S_nums];
        S->payload = new int32_t[S_nums];
        S->num_tuples = S_nums;
        std::cout << "Generating R table data:2 ^ " << R_size << " lines" << std::endl;
        gen_data(maxid, maxid, R, 48);
        std::cout << "Generating S table data:2 ^ " << S_size << " lines" << std::endl;
        gen_data(S_nums, maxid, S, 48);
        /*Implementation of join algorithm based on vectors*/
        test_vecjoin(R, S, test1_timefile, 0);
        /*Implementation of join algorithm based on vectors based on prefetching optimization*/
        test_vecjoin_prefetch(R, S, test1_timefile, 0);
        /*Implementation of join algorithm based on hashtable*/
        test_hashjoin(R, S, test1_timefile, 0);


        delete[] R->key;
        delete[] R->payload;
        delete[] S->key;
        delete[] S->payload;
    }
    std::cout << "test 2: The correlation between the performance of vector connection algorithms and the size of cache at all levels" <<std::endl;
    std::cout << "Correlation with L1 cache size" << std::endl;
    int cache_size = get_cachesize(L1_cache_file_path);
    double payload_rate = 0.1;
    R->table_size = 0;
    R->payload_rate = 0.0;
    for (; payload_rate <= 1.5; payload_rate+=0.1)
    {
        int maxid = payload_rate * cache_size;
        int S_nums = 1 << S_size;
        R->key  = new int32_t[maxid];
        R->payload = new int32_t[maxid];
        R->payload_rate = payload_rate;
        R->num_tuples = maxid;
        S->key = new int32_t[S_nums];
        S->payload = new int32_t[S_nums];
        S->num_tuples = S_nums;
        std::cout << "Generating R table data:" << maxid << " lines" << std::endl;
        gen_data(maxid, maxid, R, 48);
        std::cout << "Generating S table data:" << S_nums << " lines" << std::endl;
        gen_data(S_nums, maxid, S, 48);
        /*Implementation of join algorithm based on vectors*/
        test_vecjoin(R, S, test2_timefile, 1);
        /*Implementation of join algorithm based on vectors based on prefetching optimization*/
        test_vecjoin_prefetch(R, S, test2_timefile, 1);
        /*Implementation of join algorithm based on hashtable*/
        test_hashjoin(R, S, test2_timefile, 1);


        delete[] R->key;
        delete[] R->payload;
        delete[] S->key;
        delete[] S->payload;
    }
    std::cout << "Correlation with L2 cache size" << std::endl;
    cache_size = get_cachesize(L2_cache_file_path);
    payload_rate = 0.1;
    R->table_size = 0;
    R->payload_rate = 0.0;
    for (; payload_rate <= 1.5; payload_rate+=0.1)
    {
        int maxid = payload_rate * cache_size;
        int S_nums = 1 << S_size;
        R->key  = new int32_t[maxid];
        R->payload = new int32_t[maxid];
        R->num_tuples = maxid;
        R->payload_rate = payload_rate;
        S->key = new int32_t[S_nums];
        S->payload = new int32_t[S_nums];
        S->num_tuples = S_nums;
        std::cout << "Generating R table data:" << maxid << " lines" << std::endl;
        gen_data(maxid, maxid, R, 48);
        std::cout << "Generating S table data:" << S_nums << " lines" << std::endl;
        gen_data(S_nums, maxid, S, 48);
        /*Implementation of join algorithm based on vectors*/
        test_vecjoin(R, S, test2_timefile, 2);
        /*Implementation of join algorithm based on vectors based on prefetching optimization*/
        test_vecjoin_prefetch(R, S, test2_timefile, 2);
        /*Implementation of join algorithm based on hashtable*/
        test_hashjoin(R, S, test2_timefile, 2);
        delete[] R->key;
        delete[] R->payload;
        delete[] S->key;
        delete[] S->payload;
    }
    std::cout << "Correlation with L3 cache size" << std::endl;
    cache_size = get_cachesize(L3_cache_file_path);
    payload_rate =0.1;
    R->table_size = 0;
    R->payload_rate = 0.0;
    for (; payload_rate <= 1.5; payload_rate+=0.1)
    {
        int maxid = payload_rate * cache_size;
        int S_nums = 1 << S_size;
        R->key  = new int32_t[maxid];
        R->payload = new int32_t[maxid];
        R->num_tuples = maxid;
        R->payload_rate = payload_rate;
        S->key = new int32_t[S_nums];
        S->payload = new int32_t[S_nums];
        S->num_tuples = S_nums;
        std::cout << "Generating R table data:" << maxid << " lines" << std::endl;
        gen_data(maxid, maxid, R, 48);
        std::cout << "Generating S table data:" << S_nums << " lines" << std::endl;
        gen_data(S_nums, maxid, S, 48);
        /*Implementation of join algorithm based on vectors*/
        test_vecjoin(R, S, test2_timefile, 3);
        /*Implementation of join algorithm based on vectors based on prefetching optimization*/
        test_vecjoin_prefetch(R, S, test2_timefile, 3);
        /*Implementation of join algorithm based on hashtable*/
        test_hashjoin(R, S, test2_timefile, 3);
        delete[] R->key;
        delete[] R->payload;
        delete[] S->key;
        delete[] S->payload;
    }
    


    return 0;
}