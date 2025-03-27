#ifndef Metadata_H
#define Metadata_H
#include <string>
typedef int idx;
typedef int T;
#define STACKSIZE 8388608
idx DATA_NUM = 1 << 26;
const int DATA_MAX = 1000;     //max for low selection rate test
const idx CONST_TEST_NUM = 10;   // 10 tests with const change of selection rate stride
const idx CONST_TEST_NUM_LSR = 6; // 9 tests with change for  low selection rate 
const double LSR_BASE = 0.000001; //low selection rate base selection rate
const int LSR_STRIDE= 10;           //stride of low selection rate
const float CONST_BASE = 0.1;       // const stride base selection rate
const float CONST_STRIDE = 0.1;     // stride of selection rate
const char* SELALGO_TIME_FILE = "./log/select/selalgo_test.tsv"; // the result time file for select algorithm implementation and testing
const char* SELALGO_MODEL_TIME_FILE = "./log/select/selalgo_model_test.tsv"; // the result time file for select algorithm implementation and query model testing
const char* SELALGO_MODEL_LSR_TIME_FILE = "./log/select/selalgo_model_lsr_test.tsv"; // the result time file for select algorithm implementation and low selection rate testing
const char* CASESTUDY_TIME_FILE = "./log/select/casestudy_test.tsv"; // the result time file
const char* CASESTUDY_LSR_TIME_FILE = "./log/select/casestudy_lsr_test.tsv"; // the result time file
const char* PROALGO_TIME_FILE = "./log/project/proalgo_test.tsv";
const char* PROALGO_LSR_TIME_FILE = "./log/project/proalgo_lsr_test.tsv";
const char* JOINALGO_TEST1_TIME_FILE = "./log/join/joinalgo_test1.tsv"; //the result time file for join test 1
const char* JOINALGO_TEST2_TIME_FILE = "./log/join/joinalgo_test2.tsv"; //the result time file for join test 2
const char* GROUPALGO_TEST_TIME_FILE = "./log/group/groupalgo_test.tsv"; //the result time file for group test
const char* AGGALGO_TEST_TIME_FILE = "./log/aggregation/aggalgo_test.tsv"; //the result time file for aggregation test
const char* AGGALGO_VEC_TEST_TIME_FILE = "./log/aggregation/aggalgo_vec_test.tsv"; //the result time file for aggregation vector test
const char* STARJOINALGO_TEST_TIME_FILE = "./log/starjoin/starjoin_test.tsv"; //the result time file for starjoin test  
const char* OLAPCORE_TEST_TIME_FILE = "./log/multi_compute_operator/olapcore_test.tsv"; //the result time file for OLAPcore test 
const char* GPUOLAPCORE_TEST_TIME_FILE = "./log/gpu_multi_compute_operator/gpuolapcore_test.tsv"; //the result time file for OLAPcore test 
const char* lineitem_data_dir = "./dbgen/lineitem.csv"; //TPCH lineitem data_dir
const char* partsupp_data_dir = "./dbgen/partsupp.csv"; //TPCH partsupp data_dir
const char* orders_data_dir = "./dbgen/orders.csv"; //TPCH orders data_dir
const char* part_data_dir = "./dbgen/part.csv"; //TPCH part data_dir
const char* supplier_data_dir = "./dbgen/supplier.csv"; //TPCH supplier data_dir
const char* customer_data_dir = "./dbgen/customer.csv"; //TPCH customer data_dir
const char* nation_data_dir = "./dbgen/nation.csv"; //TPCH nation data_dir
const char* region_data_dir = "./dbgen/region.csv"; //TPCH region data_dir
const char* lineitem_tbl_data_dir = "./dbgen/lineitem.tbl"; //TPCH lineitem data_dir
const char* partsupp_tbl_data_dir = "./dbgen/partsupp.tbl"; //TPCH partsupp data_dir
const char* orders_tbl_data_dir = "./dbgen/orders.tbl"; //TPCH orders data_dir
const char* part_tbl_data_dir = "./dbgen/part.tbl"; //TPCH part data_dir
const char* supplier_tbl_data_dir = "./dbgen/supplier.tbl"; //TPCH supplier data_dir
const char* customer_tbl_data_dir = "./dbgen/customer.tbl"; //TPCH customer data_dir
const char* nation_tbl_data_dir = "./dbgen/nation.tbl"; //TPCH nation data_dir
const char* region_tbl_data_dir = "./dbgen/region.tbl"; //TPCH region data_dir
const char* lineitem_arrow_dir = "./dbgen/lineitem.arrow"; //TPCH lineitem data_dir
const char* partsupp_arrow_dir = "./dbgen/partsupp.arrow"; //TPCH partsupp data_dir
const char* orders_arrow_dir = "./dbgen/orders.arrow"; //TPCH orders data_dir
const char* part_arrow_dir = "./dbgen/part.arrow"; //TPCH part data_dir
const char* supplier_arrow_dir = "./dbgen/supplier.arrow"; //TPCH supplier data_dir
const char* customer_arrow_dir = "./dbgen/customer.arrow"; //TPCH customer data_dir
const char* nation_arrow_dir = "./dbgen/nation.arrow"; //TPCH nation data_dir
const char* region_arrow_dir = "./dbgen/region.arrow"; //TPCH region data_dir
const int M_VALUE = 5;                          // value of M1 and M2
const int DATA_NUM_BASE = 6000000;
const int GROUP_EXP_MIN = 5;                    // group num min 2^5
const int GROUP_EXP_MAX = 26;                   // group num max 2^26
const int L_TAX_MIN = 1;                   // tax value min 0.01
const int L_TAX_MAX = 20;                   // tax value min 0.01
const int L_QUANTITY_MIN = 1;                   // quantity value min 0.01
const int L_QUANTITY_MAX = 10;                   // quantity value min 0.01
const int L_EXPTENDEDPRICE_MIN = 5;                   // extendedprice value min 0.01
const int L_EXPTENDEDPRICE_MAX = 20;                   // extendedprice value min 0.01
const int VEC_SIZE_MAX = 24;                   // vec_size max 2 ^ 24
const char* L1_cache_file_path = "/sys/devices/system/cpu/cpu0/cache/index0/size"; // File path for storing L1cache size
const char* L2_cache_file_path = "/sys/devices/system/cpu/cpu0/cache/index2/size"; // File path for storing L2cache size
const char* L3_cache_file_path = "/sys/devices/system/cpu/cpu0/cache/index3/size"; // File path for storing L3cache size
const idx size_v = 1024;//Vector length
constexpr int8_t DIM_NULL = INT8_MAX;
constexpr int GROUP_NULL = INT16_MAX;
const size_t LINEORDER_BASE = 6000000;                // lineorder base num
const size_t CUSTOMER_BASE = 30000;                   // customer base num
const size_t SUPPLIER_BASE = 2000;                    // supplier base num
const size_t PART_BASE = 200000;                      // part base num
const size_t DATE_BASE = 7 * 365;                     // date base num
const int GROUP_BITS_TABLE = 4;                   // group num on each table

#define NATION_COUNT 25
enum Selalgo_Branch {
    BRANCH_ONE_TWO_THREE = 0,
    BRANCH_ONE_TWO,
    BRANCH_ONE,
    NON_BRANCH
};
struct Select_Node;
struct Select_Data
{
  const void *sel_col1;
  const void *sel_col2;
  int8_t op;        // Filter symbols
  int8_t col2_flag; // Single value or column
  int8_t select_flag;
  int8_t  count;
  int *pre_bmp;
  int *res_bmp;
  void* (*select)(void *);
};
struct Select_Node
{
  int select_num;
  Select_Data select_data[5];
  int8_t logic;//&& OR ||
  int col_length;
  std::string tablename;
};
struct Group_Data_gt
{
  void *gro_col;
  void *com_dic_t;
  int location;
  int dic_location;
  std::string tablename;

};
struct Group_Data
{
  int colnum;
  int *group_count;
  int *res_vec;
  void * gro_col[5];
  void * com_dic_t[5];
  std::string colname[10];
  int location;
  int dic_location;
  std::string tablename;
  int (*group[5])(Group_Data_gt);
  void (*group_assignment[5])(Group_Data_gt);
  int table_size;
};
struct Group_Node
{
  int tablenum;
  int *group_total_num;
  Group_Data group_data[5];
};
struct Join_Node
{
  const void *join_col[5];
  int *pre_vec[5];
  const void *join_col_cross[5];
  int *pre_vec_cross[5];
  void *(*join[5])(void *);
  int32_t *OID;
  int16_t *groupID;
  int table_size;
  int factor[5];
  int join_col_num;
  int index[300];
};
struct Agg_Node
{
  int agg_num;
  int table_size;
  const void *agg_col1[5];
  const void *agg_col2[5];
  int32_t *OID;
  int16_t *groupID;
  int *index;
  int group_num;
  double *res_vec[300];
  double *pre_res[300];
  void *(* agg[5])(void *);
};
struct Project_Data
{
  void *res_array;
  std::string name_array;
  const void *pro_sel;
  int group_count;
  int factor;
  int32_t * OID;
  int fk_num;
  const int *FK_sel[5];

};

struct Project_Node
{
  int group_total_num;
  Project_Data project_data[5];
  void (*project[5])(Project_Data &, int);
  void (*write[5])(Project_Data &, int, std::ofstream &);
  int colnum;

};

struct pth_st
{
  pthread_barrier_t *barrier;
  const void *sel_col1;
  const void *sel_col2;
  std::string tablename;
  int8_t op;
  int8_t col2_flag;
  int8_t select_flag;
  int *pre_bmp;
  int *res_bmp;
  int8_t logic;
  int startindex;
  int comline; // 维表行数
};
struct pth_gt
{
  int colnum;
  int *group_count;
  int startindex;
  int comline; // 维表行数
  pthread_mutex_t *mut;
  pthread_barrier_t *barrier;
  std::string tablename;
  std::string *colname;
  void **gro_col;
  void **com_dic_t;
  int (**group)(Group_Data_gt);
  void (**group_assignment)(Group_Data_gt);
  int *res_vec;
};
struct pth_jt
{
  int num_tuples;
  int start;
  int join_id;
  const void *join_col;
  int *pre_vec;
  const void *join_col_cross;
  int *pre_vec_cross;
  int32_t *OID;
  int16_t *groupID;
  int factor;
  int *index;
};
struct pth_at
{
  int num_tuples;
  int start;
  const void *agg_col1;
  const void *agg_col2;
  int32_t *OID;
  int16_t *groupID;
  int *index;
  double *pre_res;
  double *res_vec;
};

struct fixed_arrays
{
  idx *pos_value1;
  idx *value2;
  idx array_size = 0;
};
struct row_store_min
{
  idx Ra;
  char Rb[52];
  idx Rc;
};
struct row_store_max
{
  idx Ra;
  char Rb[64];
  idx Rc;
};
struct relation_t
{
  int32_t *key;
  int32_t *payload;
  uint64_t num_tuples;
  double payload_rate;
  int table_size;
  
};
struct Dimvec_array_numa
{
  int8_t *dimvec[4];
};
struct Fk_array_numa
{
  int32_t *fk[4];
};
struct create_arg_t {
    relation_t          rel;
    int64_t             firstkey;
    int64_t             maxid;
    uint64_t            ridstart;
    pthread_barrier_t * barrier;
    int tid;
};
enum TABLE_NAME {
    customer, 
    supplier,
    part, 
    date, 
    lineorder
};
 struct param_t
{
    uint32_t nthreads;
    double sf;
    static bool is_lsr;
    uint32_t d_groups;
    uint32_t s_groups;
    uint32_t p_groups;
    uint32_t c_groups;
    double d_sele;
    double s_sele;
    double p_sele;
    double c_sele;
    int d_bitmap;
    int s_bitmap;
    int p_bitmap;
    int c_bitmap;
    int basic_numa;
    int sqlnum;
};
struct pth_rowolapcoret
{
  int join_id;
  int64_t start;
  int64_t num_tuples;
  int8_t **dimvec_array;
  int32_t **fk_array;
  int dimvec_nums;
  const int *orders;
  uint32_t *group_vector;
  int32_t * M1;
  int32_t * M2;
  const int * factor;
};
struct pth_vwmolapcoret
{
  int join_id;
  int64_t start;
  int64_t num_tuples;
  int8_t **dimvec_array;
  int32_t **fk_array;
  int dimvec_nums;
  int *orders;
  int64_t * OID;
  int16_t * groupID;
  uint32_t *group_vector;
  int32_t * M1;
  int32_t * M2;
  int * index;
  int * factor;
};
struct pth_vwmolapcoret_numa
{
  int join_id;
  int64_t start;
  int64_t num_tuples;
  Dimvec_array_numa *dimvec_array_numa;
  Fk_array_numa *fk_array_numa;
  int dimvec_nums;
  int *orders;
  int64_t * OID;
  int16_t * groupID;
  uint32_t *group_vector;
  int32_t * M1;
  int32_t * M2;
  int * index;
  int * factor;
};
struct pth_cwmjoint
{
  int join_id;
  int64_t start;
  int64_t num_tuples;
  int8_t *dimvec;
  int32_t *fk;
  int64_t * OID;
  int16_t * groupID;
  int *index;
  int factor;
  int tid;
};
struct pth_cwmaggt
{
  int64_t start;
  int64_t num_tuples;
  int64_t * OID;
  int16_t * groupID;
  int * index;
  int32_t * M1;
  int32_t * M2;
  uint32_t * group_vector;
};
extern row_store_min row_min[67108864];
extern row_store_max row_max[67108864];

const Selalgo_Branch SELALGO_BRANCH[] = {BRANCH_ONE_TWO_THREE, NON_BRANCH};

const Selalgo_Branch CASE_COMBINED_BRANCH[] = {BRANCH_ONE_TWO_THREE, BRANCH_ONE_TWO, BRANCH_ONE, NON_BRANCH};

const Selalgo_Branch CASE_MULTIPASS_BRANCH[] = {BRANCH_ONE_TWO, BRANCH_ONE};
#endif