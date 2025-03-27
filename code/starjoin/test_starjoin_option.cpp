/**
 * @file test_startjoin_option.cpp
 * @author Ruichen Han (hanruichen@ruc.edu.cn)
 * @brief test star join algorithms
 * @version 0.1
 * @date 2023-05-09
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
#include "../include/gendata_util.hpp"
#include "../include/statistical_analysis_util.hpp"
#include "../include/metadata.h"
#include <string.h>
/**
 * @brief starjoin using Row-wise model
 * 
 * @param size_lineorder 
 * @param dimvec_c 
 * @param dimvec_s 
 * @param dimvec_p 
 * @param dimvec_d 
 * @param fk_c 
 * @param fk_s 
 * @param fk_p 
 * @param fk_d 
 * @return int 
 */
int starjoinalgo_rowwise(const int& size_lineorder,
                       const int8_t* dimvec_c, const int8_t* dimvec_s, 
                       const int8_t* dimvec_p, const int8_t* dimvec_d, 
                       const int32_t* fk_c, const int32_t* fk_s, 
                       const int32_t* fk_p, const int32_t* fk_d){
    // join on four tables
    int count = 0;
    for(int i = 0; i != size_lineorder; ++i) {
        if(dimvec_c[fk_c[i]] != DIM_NULL && dimvec_s[fk_s[i]] != DIM_NULL && dimvec_p[fk_p[i]] != DIM_NULL && dimvec_d[fk_d[i]] != DIM_NULL) {
           count ++;
        } 
    }
    return count;
}
/**
 * @brief starjoin using Column-wise model by static vector
 * 
 * @param size_lineorder 
 * @param dimvec_c 
 * @param dimvec_s 
 * @param dimvec_p 
 * @param dimvec_d 
 * @param fk_c 
 * @param fk_s 
 * @param fk_p 
 * @param fk_d 
 * @param result 
 * @return int 
 */
int starjoinalgo_cwm_sv(const int& size_lineorder,
                       const int8_t* dimvec_c, const int8_t* dimvec_s, 
                       const int8_t* dimvec_p, const int8_t* dimvec_d, 
                       const int32_t* fk_c, const int32_t* fk_s, 
                       const int32_t* fk_p, const int32_t* fk_d,
                       int * result){
    // join on customer table
    for (int i = 0; i != size_lineorder; i++)
    {
        if (dimvec_c[fk_c[i]] != DIM_NULL)
            result[i] += ((int)dimvec_c[fk_c[i]]) << (GROUP_BITS_TABLE * 3);
        else
            result[i] = GROUP_NULL;
    }
    //join on supplier table
    for (int i = 0; i != size_lineorder; i++)
    {
        
        if (result[i] != GROUP_NULL)
        {
            if (dimvec_s[fk_s[i]] != DIM_NULL)
                result[i] += ((int)dimvec_s[fk_s[i]]) << (GROUP_BITS_TABLE * 2);
            else
                result[i] = GROUP_NULL;
        }
    }
    //join on part table
    for (int i = 0; i != size_lineorder; i++)
    {
        if (result[i] != GROUP_NULL)
        {
            if (dimvec_p[fk_p[i]] != DIM_NULL)
                result[i] += ((int)dimvec_p[fk_p[i]]) << (GROUP_BITS_TABLE);
            else
                result[i] = GROUP_NULL;
        }
    }
    //join on date table
    for (int i = 0; i != size_lineorder; i++)
    {
        if (result[i] != GROUP_NULL)
        {
            if (dimvec_d[fk_d[i]] != DIM_NULL)
                result[i] += (int)dimvec_d[fk_d[i]];
            else
                result[i] = GROUP_NULL;
        }
    }
    int count = 0;
    for (int i = 0; i != size_lineorder; i++)
    {
        if (result[i] != GROUP_NULL)
            ++count;
    }
    return count;
}
/**
 * @brief starjoin using Column-wise model by dynamic vector
 * 
 * @param size_lineorder 
 * @param dimvec_c 
 * @param dimvec_s 
 * @param dimvec_p 
 * @param dimvec_d 
 * @param fk_c 
 * @param fk_s 
 * @param fk_p 
 * @param fk_d 
 * @param result 
 * @return int 
 */
int starjoinalgo_cwm_dv(const int& size_lineorder,
                       const int8_t* dimvec_c, const int8_t* dimvec_s, 
                       const int8_t* dimvec_p, const int8_t* dimvec_d, 
                       const int32_t* fk_c, const int32_t* fk_s, 
                       const int32_t* fk_p, const int32_t* fk_d,
                       std::vector<std::pair<int, int> > result){
    size_t read_idx, cur_size, write_idx;
    // join on customer table
    for (read_idx = 0, write_idx = 0; read_idx != size_lineorder; ++read_idx)
    {
        if (dimvec_c[fk_c[read_idx]] != DIM_NULL)
        {
            result.emplace_back(read_idx, (((int)dimvec_c[fk_c[read_idx]]) << (GROUP_BITS_TABLE * 3))); // (pos, group)
            ++write_idx;
        }
    }
    //join on supplier table
    cur_size = write_idx;
    for (read_idx = 0, write_idx = 0; read_idx != cur_size; ++read_idx)
    {
        auto cur_pos = result[read_idx].first;
        if (dimvec_s[fk_s[cur_pos]] != DIM_NULL)
        {
            result[write_idx].first = cur_pos;    // (pos)
            result[write_idx].second = result[read_idx].second + (((int)dimvec_s[fk_s[cur_pos]]) << (GROUP_BITS_TABLE * 2));  // (group)
            ++write_idx;
        }
    }
    //join on part table
    cur_size = write_idx;
    for (read_idx = 0, write_idx = 0; read_idx != cur_size; ++read_idx)
    {
        auto cur_pos = result[read_idx].first;
       if (dimvec_p[fk_p[cur_pos]] != DIM_NULL){
            result[write_idx].first = cur_pos;    // (pos)
            result[write_idx].second = result[read_idx].second + (((int)dimvec_p[fk_p[cur_pos]]) << GROUP_BITS_TABLE);  // (group)
            ++write_idx;
        }
    }
    //join on date table
    cur_size = write_idx;
    for (read_idx = 0, write_idx = 0; read_idx != cur_size; ++read_idx)
    {
        auto cur_pos = result[read_idx].first;
        if (dimvec_d[fk_d[cur_pos]] != DIM_NULL)
        {
            result[write_idx].first = cur_pos;    // (pos)
            result[write_idx].second = result[read_idx].second + (int)dimvec_d[fk_d[cur_pos]];  // (group)
            ++write_idx;
        }
    }
    return write_idx;
}
/**
 * @brief test for starjoin using Column-wise model by static vector
 * 
 * @param SF 
 * @param rate 
 * @param dimvec_c 
 * @param dimvec_s 
 * @param dimvec_p 
 * @param dimvec_d 
 * @param fk_c 
 * @param fk_s 
 * @param fk_p 
 * @param fk_d
 * @param timefile
 * @return void 
 */
void test_starjoinalgo_cwm_sv(const double& SF, const double& rate, 
                            const int8_t* dimvec_c, const int8_t* dimvec_s, 
                            const int8_t* dimvec_p, const int8_t* dimvec_d, 
                            const int32_t* fk_c, const int32_t* fk_s, 
                            const int32_t* fk_p, const int32_t* fk_d, 
                            std::ofstream& timefile)
{
    std::cout << ">>> Start test join using Column-wise model by static vector" << std::endl;
    std::cout << "      dim selection rate " << rate * 100 << 
                    "%, total selection rate " << pow(rate, 4) * 100 << "%" << std::endl;
    timeval start, end;
    int size_lineorder = size_of_table(TABLE_NAME::lineorder, SF);
    int *result = new int[size_lineorder];
    gettimeofday(&start, NULL);
    int count =  starjoinalgo_cwm_sv(size_lineorder, dimvec_c, dimvec_s, dimvec_p, dimvec_d, 
                                    fk_c, fk_s, fk_p, fk_d, result);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << pow(rate, 4) * 100 << "% is " << count << "/" << size_lineorder << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    timefile << "Column-wise model"
             << "\t"
             << "static vector"
             << "\t"
             << "Column-wise model with static vector"
             << "\t"
             << log2(rate)
             << "\t"
             << ms << std::endl;
}
/**
 * @brief test for starjoin using Vector-wise model by static vector
 * 
 * @param SF 
 * @param rate 
 * @param dimvec_c 
 * @param dimvec_s 
 * @param dimvec_p 
 * @param dimvec_d 
 * @param fk_c 
 * @param fk_s 
 * @param fk_p 
 * @param fk_d
 * @param timefile
 * @return void 
 */
void test_starjoinalgo_vwm_sv(const double& SF, const double& rate, 
                            const int8_t* dimvec_c, const int8_t* dimvec_s, 
                            const int8_t* dimvec_p, const int8_t* dimvec_d, 
                            const int32_t* fk_c, const int32_t* fk_s, 
                            const int32_t* fk_p, const int32_t* fk_d, 
                            std::ofstream& timefile)
{
    std::cout << ">>> Start test join using Vector-wise model by static vector" << std::endl;
    std::cout << "      dim selection rate " << rate * 100 << 
                    "%, total selection rate " << pow(rate, 4) * 100 << "%" << std::endl;
    timeval start, end;
    int size_lineorder = size_of_table(TABLE_NAME::lineorder, SF);
    
    int vec_num =size_lineorder / size_v;
    int count = 0;
    gettimeofday(&start, NULL);
    for (int i = 0; i <= vec_num; i++)
    {
        int *result = new int[size_v];
        int nums = (i != vec_num) ? size_v : size_lineorder - i  * size_v;
        count +=  starjoinalgo_cwm_sv(nums, dimvec_c, dimvec_s, dimvec_p, dimvec_d, 
                                    fk_c + i * size_v, fk_s + i * size_v, fk_p + i * size_v, fk_d + i * size_v, result);
    }
    
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << pow(rate, 4) * 100 << "% is " << count << "/" << size_lineorder << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    timefile << "Vector-wise model"
             << "\t"
             << "static vector"
             << "\t"
             << "Vector-wise model with static vector"
             << "\t"
             << log2(rate)
             << "\t"
             << ms << std::endl;
}
/**
 * @brief test for starjoin using Column-wise model by dynamic vector
 * 
 * @param SF 
 * @param rate 
 * @param dimvec_c 
 * @param dimvec_s 
 * @param dimvec_p 
 * @param dimvec_d 
 * @param fk_c 
 * @param fk_s 
 * @param fk_p 
 * @param fk_d
 * @param timefile
 * @return void 
 */
void test_starjoinalgo_cwm_dv(const double& SF, const double& rate, 
                            const int8_t* dimvec_c, const int8_t* dimvec_s, 
                            const int8_t* dimvec_p, const int8_t* dimvec_d, 
                            const int32_t* fk_c, const int32_t* fk_s, 
                            const int32_t* fk_p, const int32_t* fk_d, 
                            std::ofstream& timefile)
{
    std::cout << ">>> Start test join using Column-wise model by dynamic vector" << std::endl;
    std::cout << "      dim selection rate " << rate * 100 << 
                    "%, total selection rate " << pow(rate, 4) * 100 << "%" << std::endl;
    timeval start, end;
    int size_lineorder = size_of_table(TABLE_NAME::lineorder, SF);
    std::vector<std::pair<int, int> > result;
    gettimeofday(&start, NULL);
    int count =  starjoinalgo_cwm_dv(size_lineorder, dimvec_c, dimvec_s, dimvec_p, dimvec_d, 
                                    fk_c, fk_s, fk_p, fk_d, result);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << pow(rate, 4) * 100 << "% is " << count << "/" << size_lineorder << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    timefile << "Column-wise model"
             << "\t"
             << "dynamic vector"
             << "\t"
             << "Column-wise model with dynamic vector"
             << "\t"
             << log2(rate)
             << "\t"
             << ms << std::endl;
}

/**
 * @brief test for starjoin using Vector-wise model by dynamic vector
 * 
 * @param SF 
 * @param rate 
 * @param dimvec_c 
 * @param dimvec_s 
 * @param dimvec_p 
 * @param dimvec_d 
 * @param fk_c 
 * @param fk_s 
 * @param fk_p 
 * @param fk_d
 * @param timefile
 * @return void 
 */
void test_starjoinalgo_vwm_dv(const double& SF, const double& rate, 
                            const int8_t* dimvec_c, const int8_t* dimvec_s, 
                            const int8_t* dimvec_p, const int8_t* dimvec_d, 
                            const int32_t* fk_c, const int32_t* fk_s, 
                            const int32_t* fk_p, const int32_t* fk_d, 
                            std::ofstream& timefile)
{
    std::cout << ">>> Start test join using Vector-wise model by dynamic vector" << std::endl;
    std::cout << "      dim selection rate " << rate * 100 << 
                    "%, total selection rate " << pow(rate, 4) * 100 << "%" << std::endl;
    timeval start, end;
    int size_lineorder = size_of_table(TABLE_NAME::lineorder, SF);
    std::vector<std::pair<int, int> > result;
    int vec_num = size_lineorder / size_v;
    int count = 0;
    gettimeofday(&start, NULL);
    for (int i = 0; i <= vec_num; i++)
    {
        int nums = (i != vec_num) ? size_v : size_lineorder - i  * size_v;
        count += starjoinalgo_cwm_dv(nums, dimvec_c, dimvec_s, dimvec_p, dimvec_d, 
                                    fk_c + i * size_v, fk_s + i * size_v, fk_p + i * size_v, fk_d + i * size_v, result);
        result.clear();
    }
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << pow(rate, 4) * 100 << "% is " << count << "/" << size_lineorder << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    timefile << "Vector-wise model"
             << "\t"
             << "dynamic vector"
             << "\t"
             << "Vector-wise model with dynamic vector"
             << "\t"
             << log2(rate)
             << "\t"
             << ms << std::endl;
}
/**
 * @brief test for starjoin using Row-wise model
 * 
 * @param SF 
 * @param rate 
 * @param dimvec_c 
 * @param dimvec_s 
 * @param dimvec_p 
 * @param dimvec_d 
 * @param fk_c 
 * @param fk_s 
 * @param fk_p 
 * @param fk_d
 * @param timefile
 * @return void 
 */
void test_starjoinalgo_rowwise(const double& SF, const double& rate, 
                            const int8_t* dimvec_c, const int8_t* dimvec_s, 
                            const int8_t* dimvec_p, const int8_t* dimvec_d, 
                            const int32_t* fk_c, const int32_t* fk_s, 
                            const int32_t* fk_p, const int32_t* fk_d, 
                            std::ofstream& timefile)
{
    std::cout << ">>> Start test starjoin by Row-wise model" << std::endl;
    std::cout << "      dim selection rate " << rate * 100 << 
                    "%, total selection rate " << pow(rate, 4) * 100 << "%" << std::endl;
    timeval start, end;
    int size_lineorder = size_of_table(TABLE_NAME::lineorder, SF);
    gettimeofday(&start, NULL);
    int count =  starjoinalgo_rowwise(size_lineorder, dimvec_c, dimvec_s, dimvec_p, dimvec_d, 
                                    fk_c, fk_s, fk_p, fk_d);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << pow(rate, 4) * 100 << "% is " << count << "/" << size_lineorder << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    timefile << "Row-wise model"
             << "\t"
             << ""
             << "\t"
             << "Row-wise model"
             << "\t"
             << log2(rate)
             << "\t"
             << ms << std::endl;
}
/**
 * @brief test for starjoin using Column-wise model
 * 
 * @param SF 
 * @param rate 
 * @param dimvec_c 
 * @param dimvec_s 
 * @param dimvec_p 
 * @param dimvec_d 
 * @param fk_c 
 * @param fk_s 
 * @param fk_p 
 * @param fk_d
 * @param timefile
 * @return void 
 */
void test_starjoinalgo_columnwise(const double& SF, const double& rate, 
                            const int8_t* dimvec_c, const int8_t* dimvec_s, 
                            const int8_t* dimvec_p, const int8_t* dimvec_d, 
                            const int32_t* fk_c, const int32_t* fk_s, 
                            const int32_t* fk_p, const int32_t* fk_d, 
                            std::ofstream& timefile)
{
    // 1. starjoin using Column-wise model by static vector
    test_starjoinalgo_cwm_sv(SF, rate, dimvec_c, dimvec_s, dimvec_p, dimvec_d, 
                                    fk_c, fk_s, fk_p, fk_d, 
                                    timefile);
    // 2. starjoin using Column-wise model by dynamic vector
    test_starjoinalgo_cwm_dv(SF, rate, dimvec_c, dimvec_s, dimvec_p, dimvec_d, 
                                    fk_c, fk_s, fk_p, fk_d, 
                                    timefile);
}
/**
 * @brief test for starjoin using Vector-wise model
 * 
 * @param SF 
 * @param rate 
 * @param dimvec_c 
 * @param dimvec_s 
 * @param dimvec_p 
 * @param dimvec_d 
 * @param fk_c 
 * @param fk_s 
 * @param fk_p 
 * @param fk_d
 * @param timefile
 * @return void 
 */
void test_starjoinalgo_vectorwise(const double& SF, const double& rate, 
                            const int8_t* dimvec_c, const int8_t* dimvec_s, 
                            const int8_t* dimvec_p, const int8_t* dimvec_d, 
                            const int32_t* fk_c, const int32_t* fk_s, 
                            const int32_t* fk_p, const int32_t* fk_d, 
                            std::ofstream& timefile)
{
    // 1. starjoin using Column-wise model by static vector
    test_starjoinalgo_vwm_sv(SF, rate, dimvec_c, dimvec_s, dimvec_p, dimvec_d, 
                                    fk_c, fk_s, fk_p, fk_d, 
                                    timefile);
    // 2. starjoin using Column-wise model by dynamic vector
    test_starjoinalgo_vwm_dv(SF, rate, dimvec_c, dimvec_s, dimvec_p, dimvec_d, 
                                    fk_c, fk_s, fk_p, fk_d, 
                                    timefile);
}
int main() {
    std::ofstream timefile;
    timefile.open(STARJOINALGO_TEST_TIME_FILE, std::ios::out | std::ios::trunc);
    timefile << "query model"
             << "\t"
             << "Intermediate results"
             << "\t"
             << "query model with different Intermediate results"
             << "\t"
             << "selection rate"
             << "\t"
             << "Rumtimes(ms)" << std::endl;
    std::vector<double> rates = {pow(2, 0), pow(2, -1), pow(2, -2), pow(2, -3), pow(2, -4)};
    double SF = 10;
    int8_t *dimvec_c, *dimvec_s, *dimvec_p, *dimvec_d;
    int32_t *fk_c, *fk_s, *fk_p, *fk_d;
    for(const auto& rate : rates){
        gen_data(rate, SF, dimvec_c, dimvec_s, dimvec_p, dimvec_d, 
                     fk_c, fk_s, fk_p, fk_d);
        // 1. test starjoin using Row-wise model
        test_starjoinalgo_rowwise(SF, rate, dimvec_c, dimvec_s, dimvec_p, dimvec_d, 
                                    fk_c, fk_s, fk_p, fk_d, 
                                    timefile);
        // 2. test starjoin using Column-wise model
        test_starjoinalgo_columnwise(SF, rate, dimvec_c, dimvec_s, dimvec_p, dimvec_d, 
                                    fk_c, fk_s, fk_p, fk_d, 
                                    timefile);
        // 3. test starjoin using Vector-wise model
        test_starjoinalgo_vectorwise(SF, rate, dimvec_c, dimvec_s, dimvec_p, dimvec_d, 
                                    fk_c, fk_s, fk_p, fk_d, 
                                    timefile);
    }   
    

    delete[] dimvec_c;
    delete[] dimvec_s;
    delete[] dimvec_p;
    delete[] dimvec_d;

    delete[] fk_c;
    delete[] fk_s;
    delete[] fk_p;
    delete[] fk_d;
    timefile.close();
}