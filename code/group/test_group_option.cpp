/**
 * @file test_group_option.cpp
 * @author ruichenhan (hanruichen@ruc.edu.cn)
 * @brief test of group algorithms
 * @version 0.1
 * @date 2023-05-05
 * 
 * @copyright Copyright (c) 2023
 * 
 */
 /**
 * @brief vector group algorithm
 * 
 * @param size_R 
 * @param vecInx 
 * @param m1 
 * @param m2 
 * @param group_vector 
 * @return int64_t the sum of the group vector result
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <sys/time.h>
#include <string.h>
#include "../include/statistical_analysis_util.hpp"
#include "../include/metadata.h"
#include "../include/gendata_util.hpp"
#include <unordered_map>
 /**
 * @brief vector group algorithm
 * 
 * @param size_R 
 * @param vecInx 
 * @param m1 
 * @param m2 
 * @param group_vector 
 * @return int64_t the sum of the group vector result
 */
int64_t vector_group(const size_t& size_R,
                    const int* vecInx, 
                    const int* m1, 
                    const int* m2, 
                    int* group_vector,
                    const int& group_num){
    int64_t final_result = 0; 
    for(int i = 0; i < size_R; ++i) {
        group_vector[vecInx[i] - 1] += m1[i] + m2[i];
    }

    for(size_t i = 0; i != group_num; ++i) {
        final_result += group_vector[i];
    }

    return final_result;
}
 /**
 * @brief hashtable group algorithm
 * 
 * @param size_R 
 * @param vecInx 
 * @param m1 
 * @param m2 
 * @param group_hashtable 
 * @return int64_t the sum of the group vector result
 */
int64_t hashtable_group(const size_t& size_R,
                    const int* vecInx, 
                    const int* m1, 
                    const int* m2, 
                    std::unordered_map<int, int> &group_hashtable){
    int64_t final_result = 0; 

    for(size_t i = 0; i != size_R; ++i) {

        group_hashtable[vecInx[i] - 1] += m1[i] + m2[i];
    }

    for(size_t i = 0; i != group_hashtable.size(); ++i) {
        final_result += group_hashtable[i];
    }

    return final_result;
}
/**
 * @brief test vector group algorithm
 * 
 * @param size_R 
 * @param group_num 
 * @param vecInx 
 * @param m1 
 * @param m2 
 * @param timefile 
 */
 void test_vector_group(const size_t& size_R, 
                        const size_t& Lg_group_num,
                        const int& group_num, const int* vecInx, 
                        const int* m1, const int* m2, 
                        std::ofstream& timefile) {
                                    
    std::cout << ">>> Start test vector group algorithm, with " << group_num << " groups" << std::endl;  

    
    timeval start, end;
    
    int * group_vector = (int *)malloc(sizeof(int) * (group_num));
    memset(group_vector, 0 , group_num * sizeof(int));
    gettimeofday(&start, NULL);
    
    int64_t final_result =  vector_group(size_R, vecInx, m1, m2, group_vector, group_num);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Sum of " << group_num << " groups:  "  << final_result << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;  
    timefile << "vector group algorithm"
             << "\t"
             << Lg_group_num
             << "\t"
             << ms << std::endl;
    delete [] group_vector;
}
/**
 * @brief test hashtable group algorithm
 * 
 * @param size_R 
 * @param group_num 
 * @param vecInx 
 * @param m1 
 * @param m2 
 * @param timefile 
 */
 void test_hashtable_group(const size_t& size_R,
                        const size_t& Lg_group_num,
                        const int& group_num, const int* vecInx, 
                        const int* m1, const int* m2, 
                        std::ofstream& timefile) {
                                    
    std::cout << ">>> Start test hashtable group algorithm, with " << group_num << " groups" << std::endl;  
    
    
    timeval start, end;
    std::unordered_map<int, int>group_hashtable;
    gettimeofday(&start, NULL);
    int64_t final_result =  hashtable_group(size_R, vecInx, m1, m2, group_hashtable);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Sum of " << group_num << " groups:  "  << final_result << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;  
    timefile << "hashtable group algorithm"
             << "\t"
             << Lg_group_num
             << "\t"
             << ms << std::endl;
}
 int main()
 {
    std::ofstream timefile;
    timefile.open(GROUPALGO_TEST_TIME_FILE, std::ios::out | std::ios::trunc);
    int SF = 100;
    int data_size = DATA_NUM_BASE * SF;
    timefile << "Group algorithm"
             << "\t"
             << "group_num"
             << "\t"
             << "runtimes(ms)" << std::endl;
    for (int i = GROUP_EXP_MIN; i <= GROUP_EXP_MAX; ++i)
    {
        int* vecInx = new int[data_size];
        int* m1 = new int[data_size];
        int* m2 = new int[data_size];
        int group_num = 1 << i;
        gen_data(data_size, group_num, vecInx, m1, m2);
        /*Group algorithm based on vector*/
        test_vector_group(data_size, i, group_num, vecInx, m1, m2, timefile);
        /*Group algorithm based on hashtable*/
        test_hashtable_group(data_size, i, group_num, vecInx, m1, m2, timefile);
        delete[] vecInx;
        delete[] m1;
        delete[] m2;
    }
    timefile.close();
 }