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
#include <immintrin.h>

#ifdef __AVX512F__
#define USE_AVX512
#endif

/** 
 * @brief aggregation algorithm using Row-wise model 
 * 
 * @param size_R 
 * @param l_tax 
 * @param l_quantity 
 * @param l_extendedprice 
 * @return void 
 */
double aggalgo_rowwise(const idx & size_R,
                          const double * l_tax,
                          const double * l_quantity,
                          const double * l_extendedprice)
{
    double result = 0.0;
    for (int i = 0; i < size_R; i++)
        result += (l_extendedprice[i] * l_quantity[i]) * l_tax[i] + l_extendedprice[i] * l_quantity[i];
    return result;
}
/** 
 * @brief aggregation algorithm using Column-wise model 
 * 
 * @param size_R 
 * @param l_tax 
 * @param l_quantity 
 * @param l_extendedprice 
 * @param netto_value 
 * @param tax_value
 * @param total_value
 * @return void 
 */
double aggalgo_columnwise(const idx & size_R,
                          const double * l_tax,
                          const double * l_quantity,
                          const double * l_extendedprice,
                          double * netto_value,
                          double * tax_value,
                          double * total_value)
{
    double result = 0.0;
    /*l_extendedprice * l_quantity*/
    for (int i = 0; i < size_R; i++)
        netto_value[i] = l_extendedprice[i] * l_quantity[i];
    /*netto_value * l_tax*/
    for (int i = 0; i < size_R; i++)
        tax_value[i] = netto_value[i] * l_tax[i];
    /*netto_value + tax_value*/
    for (int i = 0; i < size_R; i++)
        result += netto_value[i] + tax_value[i];
    return result;
}
/** 
 * @brief aggregation algorithm using Column-wise model with SIMD
 * 
 * @param size_R 
 * @param l_tax 
 * @param l_quantity 
 * @param l_extendedprice 
 * @param netto_value 
 * @param tax_value
 * @param total_value
 * @return void 
 */
double aggalgo_columnwise_simd(const idx & size_R,
                          const double * l_tax,
                          const double * l_quantity,
                          const double * l_extendedprice)
{
    double result = 0.0;
    __m512d netto_value, tax_value, total_value, l_tax_simd, l_quantity_simd, l_extendedprice_simd, result_simd;
    double * result_back = (double *)_mm_malloc(8 * sizeof(double), 64);
    for (int i = 0; i < size_R; i+=8)
    {
        l_tax_simd = _mm512_load_pd(l_tax + i);
        l_quantity_simd = _mm512_load_pd(l_quantity + i);
        l_extendedprice_simd = _mm512_load_pd(l_extendedprice + i);
        /*l_extendedprice * l_quantity*/
        netto_value = _mm512_mul_pd(l_extendedprice_simd, l_quantity_simd);
        /*netto_value * l_tax*/
        tax_value = _mm512_mul_pd(netto_value, l_tax_simd);
        /*netto_value + tax_value*/
        result_simd = _mm512_add_pd(netto_value, tax_value);
        _mm512_store_pd(result_back, result_simd);
        for (int j = 0; j < 8; j++)result += result_back[j];

    }
    
    return result;
}
/** 
 * @brief Test for aggregation algorithm using Row-wise model
 * 
 * @param size_R 
 * @param l_tax 
 * @param l_quantity 
 * @param l_extendedprice 
 * @param timefile
 * @return void 
 */
void test_aggalgo_rowwise(const idx & size_R,
                          const double * l_tax,
                          const double * l_quantity,
                          const double * l_extendedprice,
                          std::ofstream& timefile)
{
    std::cout << ">>> Start test aggregation using Row-wise model" << std::endl;
    timeval start, end;
    gettimeofday(&start, NULL);
    double result = aggalgo_rowwise(size_R, l_tax, l_quantity, l_extendedprice);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count is "<< result << "/" << size_R << std::endl;
    std::cout << "          Runing time: " << ms << "ms" << std::endl;
    timefile << "Row-wise model" 
             << "\t"
             << ms << std::endl;
}
/** 
 * @brief Test for aggregation algorithm using Column-wise model 
 * 
 * @param size_R 
 * @param l_tax 
 * @param l_quantity 
 * @param l_extendedprice 
 * @param timefile
 * @return void 
 */
void test_aggalgo_columnwise(const idx & size_R,
                          const double * l_tax,
                          const double * l_quantity,
                          const double * l_extendedprice,
                          std::ofstream& timefile)
{
    std::cout << ">>> Start test aggregation using Column-wise model" << std::endl;
    timeval start, end;
    double *netto_value = new double[size_R];
    double *tax_value = new double[size_R];
    double *total_value = new double[size_R];
    gettimeofday(&start, NULL);
    double result = aggalgo_columnwise(size_R, l_tax, l_quantity, l_extendedprice, netto_value, tax_value, total_value);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count is "<< result << "/" << size_R << std::endl;
    std::cout << "          Runing time: " << ms << "ms" << std::endl;
    timefile << "Column-wise model" 
             << "\t"
             << ms << std::endl;
}
#ifdef USE_AVX512
/** 
 * @brief Test for aggregation algorithm using Column-wise model with SIMD
 * 
 * @param size_R 
 * @param l_tax 
 * @param l_quantity 
 * @param l_extendedprice 
 * @param timefile
 * @return void 
 */
void test_aggalgo_columnwise_simd(const idx & size_R,
                          const double * l_tax,
                          const double * l_quantity,
                          const double * l_extendedprice,
                          std::ofstream& timefile)
{
    std::cout << ">>> Start test aggregation using Column-wise model with SIMD" << std::endl;
    timeval start, end;
    gettimeofday(&start, NULL);
    double result = aggalgo_columnwise_simd(size_R, l_tax, l_quantity, l_extendedprice);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count is "<< result << "/" << size_R << std::endl;
    std::cout << "          Runing time: " << ms << "ms" << std::endl;
    timefile << "Column-wise model with SIMD"
             << "\t"
             << ms << std::endl;
}
#endif
/** 
 * @brief Test for aggregation algorithm using Vector-wise model 
 * 
 * @param size_R 
 * @param l_tax 
 * @param l_quantity 
 * @param l_extendedprice 
 * @param timefile
 * @return void 
 */
void test_aggalgo_vectorwise(const idx & size_R,
                          const int & vec_size,
                          const double * l_tax,
                          const double * l_quantity,
                          const double * l_extendedprice,
                          std::ofstream& timefile,
                          std::ofstream& vec_timefile)
{
    std::cout << ">>> Start test aggregation using Vector-wise model whose vec_size is " << vec_size << std::endl;
    timeval start, end;
    int vec_num = size_R / vec_size;
    double result = 0.0;
    gettimeofday(&start, NULL);
    for (int i = 0; i < vec_num; i++)
    {
        double *netto_value = new double[vec_size];
        double *tax_value = new double[vec_size];
        double *total_value = new double[vec_size];
        result += aggalgo_columnwise(vec_size, l_tax + i * vec_size, l_quantity + i * vec_size, l_extendedprice + i * vec_size, netto_value, tax_value, total_value);
    
    }
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count is "<< result << "/" << size_R << std::endl;
    std::cout << "          Runing time: " << ms << "ms" << std::endl;
    if (vec_size == 1024)
    {
        timefile << "Vector-wise model"
                 << "\t"
                 << ms << std::endl;
    }
    vec_timefile << "Vector-wise model"
                 << "\t"
                 << vec_size
                 << "\t"
                 << ms << std::endl;
}
#ifdef USE_AVX512
/** 
 * @brief Test for aggregation algorithm using Vector-wise model with SIMD
 * 
 * @param size_R 
 * @param l_tax 
 * @param l_quantity 
 * @param l_extendedprice 
 * @param timefile
 * @return void 
 */
void test_aggalgo_vectorwise_simd(const idx & size_R,
                          const int & vec_size,
                          const double * l_tax,
                          const double * l_quantity,
                          const double * l_extendedprice,
                          std::ofstream& timefile,
                          std::ofstream& vec_timefile)
{
    std::cout << ">>> Start test aggregation using Vector-wise model with SIMD whose vec_size is " << vec_size << std::endl;
    timeval start, end;
    int vec_num = size_R / vec_size;
    double result = 0.0;
    gettimeofday(&start, NULL);
    for (int i = 0; i < vec_num; i++)
    {
        result += aggalgo_columnwise_simd(vec_size, l_tax + i * vec_size, l_quantity + i * vec_size, l_extendedprice + i * vec_size);
    
    }
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count is "<< result << "/" << size_R << std::endl;
    std::cout << "          Runing time: " << ms << "ms" << std::endl;
    if (vec_size == 1024)
    {
        timefile << "Vector-wise model with SIMD"
                 << "\t"
                 << ms << std::endl;
    }
    vec_timefile << "Vector-wise model with SIMD"
                 << "\t"
                 << vec_size
                 << "\t"
                 << ms << std::endl;
}
#endif
int main()
{
    std::ofstream timefile, vec_timefile;
    timefile.open(AGGALGO_TEST_TIME_FILE, std::ios::out | std::ios::trunc);
    vec_timefile.open(AGGALGO_VEC_TEST_TIME_FILE, std::ios::out | std::ios::trunc);
    timefile << "Aggregation algorithm"
             << "\t"
             << "Runtimes(ms)" << std::endl;
    vec_timefile << "Aggregation algorithm"
                 << "\t"
                 << "vec_size"
                 << "\t"
                 << "Runtimes(ms)" << std::endl;
    idx size_R = 1 << 24;
#ifdef USE_AVX512
    double * l_tax = (double *)_mm_malloc(size_R * sizeof(double), 64);
    double * l_quantity = (double *)_mm_malloc(size_R * sizeof(double), 64);
    double * l_extendedprice = (double *)_mm_malloc(size_R * sizeof(double), 64);
#else
    double * l_tax = new double[size_R];
    double * l_quantity = new double[size_R];
    double * l_extendedprice = new double[size_R];
#endif
    gen_data(size_R, l_tax, l_quantity, l_extendedprice);
    /*1. test aggregation using Row-wise model*/
    test_aggalgo_rowwise(size_R, l_tax, l_quantity, l_extendedprice, timefile);
    /*2. test aggregation using Column-wise model*/
    test_aggalgo_columnwise(size_R, l_tax, l_quantity, l_extendedprice, timefile);
    for (int i = 0; i <=VEC_SIZE_MAX; i++)
    {
        int vec_size = 1 << i;
        /*3. test aggregation using Vector-wise model*/
        test_aggalgo_vectorwise(size_R, vec_size, l_tax, l_quantity, l_extendedprice, timefile, vec_timefile);
    }
#ifdef USE_AVX512
    /*4. test aggregation using Column-wise model with SIMD*/
    test_aggalgo_columnwise_simd(size_R, l_tax, l_quantity, l_extendedprice, timefile);
    for (int i = 3; i <=VEC_SIZE_MAX; i++)
    {
        int vec_size = 1 << i;
        /*5. test aggregation using Vector-wise model with SIMD*/
        test_aggalgo_vectorwise_simd(size_R, vec_size, l_tax, l_quantity, l_extendedprice, timefile, vec_timefile);
    }
#endif
}