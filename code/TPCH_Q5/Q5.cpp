#include <arrow/api.h>
#include <unistd.h>
#include <arrow/table.h>
#include <cstdint>
#include <arrow/csv/api.h>
#include <arrow/csv/writer.h>
#include <arrow/io/api.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/pretty_print.h>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <cstring>
#include <pthread.h>
#include <fstream>
#include <arrow/ipc/api.h>
#include <sstream>
#include "../include/gendata_util.hpp"
#include "../include/statistical_analysis_util.hpp"
#include "../include/metadata.h"
std::shared_ptr<arrow::RecordBatch> lineitem_t;
std::shared_ptr<arrow::RecordBatch> partsupp_t;
std::shared_ptr<arrow::RecordBatch> orders_t;
std::shared_ptr<arrow::RecordBatch> part_t;
std::shared_ptr<arrow::RecordBatch> supplier_t;
std::shared_ptr<arrow::RecordBatch> customer_t;
std::shared_ptr<arrow::RecordBatch> nation_t;
std::shared_ptr<arrow::RecordBatch> region_t;

std::map<std::string, int> metadata_tablename;
std::map<std::string, int> metadata_lineitem;
std::map<std::string, int> metadata_partsupp;
std::map<std::string, int> metadata_orders;
std::map<std::string, int> metadata_part;
std::map<std::string, int> metadata_supplier;
std::map<std::string, int> metadata_customer;
std::map<std::string, int> metadata_nation;
std::map<std::string, int> metadata_region;
struct pth_dst
{
  int64_t comline;
  int64_t start;
  int *bitmap;
  int *bitmap_S;
  const int *n_regionkey;
  const int *s_nationkey;
};
struct pth_dot
{
  int64_t comline;
  int64_t start;
  int *bitmap;
  int *bitmap_o;
  int DATE1;
  int DATE2;
  const int *n_regionkey;
  const int *c_nationkey;
  const int *o_custkey;
  const int *o_orderdate;
};
struct pth_ttjt
{
  int64_t comline;
  int64_t start;
  int *bitmap_S;
  int *bitmap_o;
  const int *l_suppkey;
  const int *l_orderkey_new;
  const double *l_extendedprice;
  const double *l_discount;
  double *GrpVex;
};


arrow::Status load()
{
  arrow::MemoryPool *pool = arrow::default_memory_pool();
  std::cout << "load lineitem" << std::endl;
  std::ifstream lineitem_tbl;
  lineitem_tbl.open(lineitem_tbl_data_dir, std::ios::in);
  int l_orderkey;
  int l_partkey;
  int l_suppkey;
  int l_linenumber;
  double l_quantity;
  double l_extendedprice;
  double l_discount;
  double l_tax;
  std::string l_returnflag;
  std::string l_linestatus;
  int l_shipdate;
  int l_commitdate;
  int l_receiptdate;
  std::string l_shipinstruct;
  std::string l_shipmode;
  std::string l_comment;
  arrow::Int32Builder l_orderkey_builder(pool);
  arrow::Int32Builder l_partkey_builder(pool);
  arrow::Int32Builder l_suppkey_builder(pool);
  arrow::Int32Builder l_linenumber_builder(pool);
  arrow::DoubleBuilder l_quantity_builder(pool);
  arrow::DoubleBuilder l_extendedprice_builder(pool);
  arrow::DoubleBuilder l_discount_builder(pool);
  arrow::DoubleBuilder l_tax_builder(pool);
  arrow::StringBuilder l_returnflag_builder(pool);
  arrow::StringBuilder l_linestatus_builder(pool);
  arrow::Int32Builder l_shipdate_builder(pool);
  arrow::Int32Builder l_commitdate_builder(pool);
  arrow::Int32Builder l_receiptdate_builder(pool);
  arrow::StringBuilder l_shipinstruct_builder(pool);
  arrow::StringBuilder l_shipmode_builder(pool);
  arrow::StringBuilder l_comment_builder(pool);
  if (lineitem_tbl.is_open())
  {
    std::string str;
    while (getline(lineitem_tbl, str))
    {
      std::stringstream line(str);
      std::string tmp;
      int count = 0;
      while (getline(line, tmp, '|'))
      {
        switch (count)
        {
        case 0:
          l_orderkey = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_orderkey_builder.Append(l_orderkey));
          break;
        case 1:
          l_partkey = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_partkey_builder.Append(l_partkey));
          break;
        case 2:
          l_suppkey = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_suppkey_builder.Append(l_suppkey));
          break;
        case 3:
          l_linenumber = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_linenumber_builder.Append(l_linenumber));
          break;
        case 4:
          l_quantity = atof(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_quantity_builder.Append(l_quantity));
          break;
        case 5:
          l_extendedprice = atof(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_extendedprice_builder.Append(l_extendedprice));
          break;
        case 6:
          l_discount = atof(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_discount_builder.Append(l_discount));
          break;
        case 7:
          l_tax = atof(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_tax_builder.Append(l_tax));
          break;
        case 8:
          l_returnflag = tmp;
          ARROW_RETURN_NOT_OK(l_returnflag_builder.Append(l_returnflag.c_str()));
          break;
        case 9:
          l_linestatus = tmp;
          ARROW_RETURN_NOT_OK(l_linestatus_builder.Append(l_linestatus.c_str()));
          break;
        case 10:
          l_shipdate = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_shipdate_builder.Append(l_shipdate));
          break;
        case 11:
          l_commitdate = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_commitdate_builder.Append(l_commitdate));
          break;
        case 12:
          l_receiptdate = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_receiptdate_builder.Append(l_receiptdate));
          break;
        case 13:
          l_shipinstruct = tmp;
          ARROW_RETURN_NOT_OK(l_shipinstruct_builder.Append(l_shipinstruct.c_str()));
          break;
        case 14:
          l_shipmode = tmp;
          ARROW_RETURN_NOT_OK(l_shipmode_builder.Append(l_shipmode.c_str()));
          break;
        case 15:
          l_comment = tmp;
          ARROW_RETURN_NOT_OK(l_comment_builder.Append(l_comment.c_str()));
          break;
        }
        count++;
      }
    }
    std::shared_ptr<arrow::Array> l_orderkey_array;
    ARROW_RETURN_NOT_OK(l_orderkey_builder.Finish(&l_orderkey_array));
    std::shared_ptr<arrow::Array> l_partkey_array;
    ARROW_RETURN_NOT_OK(l_partkey_builder.Finish(&l_partkey_array));
    std::shared_ptr<arrow::Array> l_suppkey_array;
    ARROW_RETURN_NOT_OK(l_suppkey_builder.Finish(&l_suppkey_array));
    std::shared_ptr<arrow::Array> l_linenumber_array;
    ARROW_RETURN_NOT_OK(l_linenumber_builder.Finish(&l_linenumber_array));
    std::shared_ptr<arrow::Array> l_quantity_array;
    ARROW_RETURN_NOT_OK(l_quantity_builder.Finish(&l_quantity_array));
    std::shared_ptr<arrow::Array> l_extendedprice_array;
    ARROW_RETURN_NOT_OK(l_extendedprice_builder.Finish(&l_extendedprice_array));
    std::shared_ptr<arrow::Array> l_discount_array;
    ARROW_RETURN_NOT_OK(l_discount_builder.Finish(&l_discount_array));
    std::shared_ptr<arrow::Array> l_tax_array;
    ARROW_RETURN_NOT_OK(l_tax_builder.Finish(&l_tax_array));
    std::shared_ptr<arrow::Array> l_returnflag_array;
    ARROW_RETURN_NOT_OK(l_returnflag_builder.Finish(&l_returnflag_array));
    std::shared_ptr<arrow::Array> l_linestatus_array;
    ARROW_RETURN_NOT_OK(l_linestatus_builder.Finish(&l_linestatus_array));
    std::shared_ptr<arrow::Array> l_shipdate_array;
    ARROW_RETURN_NOT_OK(l_shipdate_builder.Finish(&l_shipdate_array));
    std::shared_ptr<arrow::Array> l_commitdate_array;
    ARROW_RETURN_NOT_OK(l_commitdate_builder.Finish(&l_commitdate_array));
    std::shared_ptr<arrow::Array> l_receiptdate_array;
    ARROW_RETURN_NOT_OK(l_receiptdate_builder.Finish(&l_receiptdate_array));
    std::shared_ptr<arrow::Array> l_shipinstruct_array;
    ARROW_RETURN_NOT_OK(l_shipinstruct_builder.Finish(&l_shipinstruct_array));
    std::shared_ptr<arrow::Array> l_shipmode_array;
    ARROW_RETURN_NOT_OK(l_shipmode_builder.Finish(&l_shipmode_array));
    std::shared_ptr<arrow::Array> l_comment_array;
    ARROW_RETURN_NOT_OK(l_comment_builder.Finish(&l_comment_array));
    std::vector<std::shared_ptr<arrow::Field>> lineitem_schema_vector = {
        arrow::field("l_orderkey", arrow::int32()),
        arrow::field("l_partkey", arrow::int32()),
        arrow::field("l_suppkey", arrow::int32()),
        arrow::field("l_linenumber", arrow::int32()),
        arrow::field("l_quantity", arrow::float64()),
        arrow::field("l_extendedprice", arrow::float64()),
        arrow::field("l_discount", arrow::float64()),
        arrow::field("l_tax", arrow::float64()),
        arrow::field("l_returnflag", arrow::utf8()),
        arrow::field("l_linestatus", arrow::utf8()),
        arrow::field("l_shipdate", arrow::int32()),
        arrow::field("l_commitdate", arrow::int32()),
        arrow::field("l_receiptdate", arrow::int32()),
        arrow::field("l_shipinstruct", arrow::utf8()),
        arrow::field("l_shipmode", arrow::utf8()),
        arrow::field("l_comment", arrow::utf8())};
    auto lineitem_schema = std::make_shared<arrow::Schema>(lineitem_schema_vector);
    lineitem_t = arrow::RecordBatch::Make(lineitem_schema, l_orderkey_array->length(), {l_orderkey_array, l_partkey_array, l_suppkey_array, l_linenumber_array, l_quantity_array, l_extendedprice_array, l_discount_array, l_tax_array, l_returnflag_array, l_linestatus_array, l_shipdate_array, l_commitdate_array, l_receiptdate_array, l_shipinstruct_array, l_shipmode_array, l_comment_array});
  }
  std::cout << "load partsupp" << std::endl;
  std::ifstream partsupp_tbl;
  partsupp_tbl.open(partsupp_tbl_data_dir, std::ios::in);
  int ps_partkey;
  int ps_suppkey;
  int ps_availqty;
  double ps_supplycost;
  std::string ps_comment;
  arrow::Int32Builder ps_partkey_builder(pool);
  arrow::Int32Builder ps_suppkey_builder(pool);
  arrow::Int32Builder ps_availqty_builder(pool);
  arrow::DoubleBuilder ps_supplycost_builder(pool);
  arrow::StringBuilder ps_comment_builder(pool);
  if (partsupp_tbl.is_open())
  {
    std::string str;
    while (getline(partsupp_tbl, str))
    {
      std::stringstream line(str);
      std::string tmp;
      int count = 0;
      while (getline(line, tmp, '|'))
      {
        switch (count)
        {
        case 0:
          ps_partkey = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(ps_partkey_builder.Append(ps_partkey));
          break;
        case 1:
          ps_suppkey = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(ps_suppkey_builder.Append(ps_suppkey));
          break;
        case 2:
          ps_availqty = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(ps_availqty_builder.Append(ps_availqty));
          break;
        case 3:
          ps_supplycost = atof(tmp.c_str());
          ARROW_RETURN_NOT_OK(ps_supplycost_builder.Append(ps_supplycost));
          break;
        case 4:
          ps_comment = tmp;
          ARROW_RETURN_NOT_OK(ps_comment_builder.Append(ps_comment.c_str()));
          break;
        }
        count++;
      }
    }
    std::shared_ptr<arrow::Array> ps_partkey_array;
    ARROW_RETURN_NOT_OK(ps_partkey_builder.Finish(&ps_partkey_array));
    std::shared_ptr<arrow::Array> ps_suppkey_array;
    ARROW_RETURN_NOT_OK(ps_suppkey_builder.Finish(&ps_suppkey_array));
    std::shared_ptr<arrow::Array> ps_availqty_array;
    ARROW_RETURN_NOT_OK(ps_availqty_builder.Finish(&ps_availqty_array));
    std::shared_ptr<arrow::Array> ps_supplycost_array;
    ARROW_RETURN_NOT_OK(ps_supplycost_builder.Finish(&ps_supplycost_array));
    std::shared_ptr<arrow::Array> ps_comment_array;
    ARROW_RETURN_NOT_OK(ps_comment_builder.Finish(&ps_comment_array));

    std::vector<std::shared_ptr<arrow::Field>> partsupp_schema_vector = {
        arrow::field("ps_partkey", arrow::int32()),
        arrow::field("ps_suppkey", arrow::int32()),
        arrow::field("ps_availqty", arrow::int32()),
        arrow::field("ps_supplycost", arrow::float64()),
        arrow::field("ps_comment", arrow::utf8())};
    auto partsupp_schema = std::make_shared<arrow::Schema>(partsupp_schema_vector);
    partsupp_t = arrow::RecordBatch::Make(partsupp_schema, ps_partkey_array->length(), {ps_partkey_array, ps_suppkey_array, ps_availqty_array, ps_supplycost_array, ps_comment_array});
    // partsupp_t = partsupp;
  }
  std::cout << "load orders" << std::endl;
  std::ifstream orders_tbl;
  orders_tbl.open(orders_tbl_data_dir, std::ios::in);

  int o_orderkey;
  int o_custkey;
  std::string o_orderstatus;
  double o_totalprice;
  int o_orderdate;
  std::string o_orderpriority;
  std::string o_clerk;
  int o_shippriority;
  std::string o_comment;

  arrow::Int32Builder o_orderkey_builder(pool);
  arrow::Int32Builder o_custkey_builder(pool);
  arrow::StringBuilder o_orderstatus_builder(pool);
  arrow::DoubleBuilder o_totalprice_builder(pool);
  arrow::Int32Builder o_orderdate_builder(pool);
  arrow::StringBuilder o_orderpriority_builder(pool);
  arrow::StringBuilder o_clerk_builder(pool);
  arrow::Int32Builder o_shippriority_builder(pool);
  arrow::StringBuilder o_comment_builder(pool);
  if (orders_tbl.is_open())
  {
    std::string str;
    while (getline(orders_tbl, str))
    {
      std::stringstream line(str);
      std::string tmp;
      int count = 0;
      while (getline(line, tmp, '|'))
      {
        switch (count)
        {
        case 0:
          o_orderkey = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(o_orderkey_builder.Append(o_orderkey));
          break;
        case 1:
          o_custkey = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(o_custkey_builder.Append(o_custkey));
          break;
        case 2:
          o_orderstatus = tmp;
          ARROW_RETURN_NOT_OK(o_orderstatus_builder.Append(o_orderstatus.c_str()));
          break;
        case 3:
          o_totalprice = atof(tmp.c_str());
          ARROW_RETURN_NOT_OK(o_totalprice_builder.Append(o_totalprice));
          break;
        case 4:
          tmp.erase(std::remove(tmp.begin(), tmp.end(), '-'), tmp.end());

          o_orderdate = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(o_orderdate_builder.Append(o_orderdate));
          break;
        case 5:
          o_orderpriority = tmp;
          ARROW_RETURN_NOT_OK(o_orderpriority_builder.Append(o_orderpriority.c_str()));
          break;
        case 6:
          o_clerk = tmp;
          ARROW_RETURN_NOT_OK(o_clerk_builder.Append(o_clerk.c_str()));
          break;
        case 7:
          o_shippriority = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(o_shippriority_builder.Append(o_shippriority));
          break;
        case 8:
          o_comment = tmp;
          ARROW_RETURN_NOT_OK(o_comment_builder.Append(o_comment.c_str()));
          break;
        }
        count++;
      }
    }

    std::shared_ptr<arrow::Array> o_orderkey_array;
    ARROW_RETURN_NOT_OK(o_orderkey_builder.Finish(&o_orderkey_array));
    std::shared_ptr<arrow::Array> o_custkey_array;
    ARROW_RETURN_NOT_OK(o_custkey_builder.Finish(&o_custkey_array));
    std::shared_ptr<arrow::Array> o_orderstatus_array;
    ARROW_RETURN_NOT_OK(o_orderstatus_builder.Finish(&o_orderstatus_array));
    std::shared_ptr<arrow::Array> o_totalprice_array;
    ARROW_RETURN_NOT_OK(o_totalprice_builder.Finish(&o_totalprice_array));
    std::shared_ptr<arrow::Array> o_orderdate_array;
    ARROW_RETURN_NOT_OK(o_orderdate_builder.Finish(&o_orderdate_array));
    std::shared_ptr<arrow::Array> o_orderpriority_array;
    ARROW_RETURN_NOT_OK(o_orderpriority_builder.Finish(&o_orderpriority_array));
    std::shared_ptr<arrow::Array> o_clerk_array;
    ARROW_RETURN_NOT_OK(o_clerk_builder.Finish(&o_clerk_array));
    std::shared_ptr<arrow::Array> o_shippriority_array;
    ARROW_RETURN_NOT_OK(o_shippriority_builder.Finish(&o_shippriority_array));
    std::shared_ptr<arrow::Array> o_comment_array;
    ARROW_RETURN_NOT_OK(o_comment_builder.Finish(&o_comment_array));

    std::vector<std::shared_ptr<arrow::Field>> orders_schema_vector = {

        arrow::field("o_orderkey", arrow::int32()),
        arrow::field("o_custkey", arrow::int32()),
        arrow::field("o_orderstatus", arrow::utf8()),
        arrow::field("o_totalprice", arrow::float64()),
        arrow::field("o_orderdate", arrow::int32()),
        arrow::field("o_orderpriority", arrow::utf8()),
        arrow::field("o_clerk", arrow::utf8()),
        arrow::field("o_shippriority", arrow::int32()),
        arrow::field("o_comment", arrow::utf8())};
    auto orders_schema = std::make_shared<arrow::Schema>(orders_schema_vector);
    orders_t = arrow::RecordBatch::Make(orders_schema, o_orderkey_array->length(), {o_orderkey_array, o_custkey_array, o_orderstatus_array, o_totalprice_array, o_orderdate_array, o_orderpriority_array, o_clerk_array, o_shippriority_array, o_comment_array});
  }
  std::cout << "load part" << std::endl;
  std::ifstream part_tbl;
  part_tbl.open(part_tbl_data_dir, std::ios::in);
  int p_partkey;
  std::string p_name;
  std::string p_mfgr;
  std::string p_brand;
  std::string p_type;
  int p_size;
  std::string p_container;
  double p_retailprice;
  std::string p_comment;

  arrow::Int32Builder p_partkey_builder(pool);
  arrow::StringBuilder p_name_builder(pool);
  arrow::StringBuilder p_mfgr_builder(pool);
  arrow::StringBuilder p_brand_builder(pool);
  arrow::StringBuilder p_type_builder(pool);
  arrow::Int32Builder p_size_builder(pool);
  arrow::StringBuilder p_container_builder(pool);
  arrow::DoubleBuilder p_retailprice_builder(pool);
  arrow::StringBuilder p_comment_builder(pool);
  if (part_tbl.is_open())
  {
    std::string str;
    while (getline(part_tbl, str))
    {
      std::stringstream line(str);
      std::string tmp;
      int count = 0;
      while (getline(line, tmp, '|'))
      {
        switch (count)
        {
        case 0:
          p_partkey = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(p_partkey_builder.Append(p_partkey));
          break;
        case 1:
          p_name = tmp;
          ARROW_RETURN_NOT_OK(p_name_builder.Append(p_name.c_str()));
          break;
        case 2:
          p_mfgr = tmp;
          ARROW_RETURN_NOT_OK(p_mfgr_builder.Append(p_mfgr.c_str()));
          break;
        case 3:
          p_brand = tmp;
          ARROW_RETURN_NOT_OK(p_brand_builder.Append(p_brand.c_str()));
          break;
        case 4:
          p_type = tmp;
          ARROW_RETURN_NOT_OK(p_type_builder.Append(p_type.c_str()));
          break;
        case 5:
          p_size = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(p_size_builder.Append(p_size));
          break;
        case 6:
          p_container = tmp;
          ARROW_RETURN_NOT_OK(p_container_builder.Append(p_container.c_str()));
          break;
        case 7:
          p_retailprice = atof(tmp.c_str());
          ARROW_RETURN_NOT_OK(p_retailprice_builder.Append(p_retailprice));
          break;
        case 8:
          p_comment = tmp;
          ARROW_RETURN_NOT_OK(p_comment_builder.Append(p_comment.c_str()));
          break;
        }
        count++;
      }
    }
    std::shared_ptr<arrow::Array> p_partkey_array;
    ARROW_RETURN_NOT_OK(p_partkey_builder.Finish(&p_partkey_array));
    std::shared_ptr<arrow::Array> p_name_array;
    ARROW_RETURN_NOT_OK(p_name_builder.Finish(&p_name_array));
    std::shared_ptr<arrow::Array> p_mfgr_array;
    ARROW_RETURN_NOT_OK(p_mfgr_builder.Finish(&p_mfgr_array));
    std::shared_ptr<arrow::Array> p_brand_array;
    ARROW_RETURN_NOT_OK(p_brand_builder.Finish(&p_brand_array));
    std::shared_ptr<arrow::Array> p_type_array;
    ARROW_RETURN_NOT_OK(p_type_builder.Finish(&p_type_array));
    std::shared_ptr<arrow::Array> p_size_array;
    ARROW_RETURN_NOT_OK(p_size_builder.Finish(&p_size_array));
    std::shared_ptr<arrow::Array> p_container_array;
    ARROW_RETURN_NOT_OK(p_container_builder.Finish(&p_container_array));
    std::shared_ptr<arrow::Array> p_retailprice_array;
    ARROW_RETURN_NOT_OK(p_retailprice_builder.Finish(&p_retailprice_array));
    std::shared_ptr<arrow::Array> p_comment_array;
    ARROW_RETURN_NOT_OK(p_comment_builder.Finish(&p_comment_array));

    std::vector<std::shared_ptr<arrow::Field>> part_schema_vector = {
        arrow::field("p_partkey", arrow::int32()),
        arrow::field("p_name", arrow::utf8()),
        arrow::field("p_mfgr", arrow::utf8()),
        arrow::field("p_brand", arrow::utf8()),
        arrow::field("p_type", arrow::utf8()),
        arrow::field("p_size", arrow::int32()),
        arrow::field("p_container", arrow::utf8()),
        arrow::field("p_retailprice", arrow::float64()),
        arrow::field("p_comment", arrow::utf8())};
    auto part_schema = std::make_shared<arrow::Schema>(part_schema_vector);
    part_t = arrow::RecordBatch::Make(part_schema, p_partkey_array->length(), {p_partkey_array, p_name_array, p_mfgr_array, p_brand_array, p_type_array, p_size_array, p_container_array, p_retailprice_array, p_comment_array});
  }
  std::cout << "load supplier" << std::endl;
  std::ifstream supplier_tbl;
  supplier_tbl.open(supplier_tbl_data_dir, std::ios::in);
  int s_suppkey;
  std::string s_name;
  std::string s_address;
  int s_nationkey;
  std::string s_phone;
  double s_acctbal;
  std::string s_comment;

  arrow::Int32Builder s_suppkey_builder(pool);
  arrow::StringBuilder s_name_builder(pool);
  arrow::StringBuilder s_address_builder(pool);
  arrow::Int32Builder s_nationkey_builder(pool);
  arrow::StringBuilder s_phone_builder(pool);
  arrow::DoubleBuilder s_acctbal_builder(pool);
  arrow::StringBuilder s_comment_builder(pool);
  if (supplier_tbl.is_open())
  {
    std::string str;
    while (getline(supplier_tbl, str))
    {
      std::stringstream line(str);
      std::string tmp;
      int count = 0;
      while (getline(line, tmp, '|'))
      {
        switch (count)
        {
        case 0:
          s_suppkey = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(s_suppkey_builder.Append(s_suppkey));
          break;
        case 1:
          s_name = tmp;
          ARROW_RETURN_NOT_OK(s_name_builder.Append(s_name.c_str()));
          break;
        case 2:
          s_address = tmp;
          ARROW_RETURN_NOT_OK(s_address_builder.Append(s_address.c_str()));
          break;
        case 3:
          s_nationkey = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(s_nationkey_builder.Append(s_nationkey));
          break;
        case 4:
          s_phone = tmp;
          ARROW_RETURN_NOT_OK(s_phone_builder.Append(s_phone.c_str()));
          break;
        case 5:
          s_acctbal = atof(tmp.c_str());
          ARROW_RETURN_NOT_OK(s_acctbal_builder.Append(s_acctbal));
          break;
        case 6:
          s_comment = tmp;
          ARROW_RETURN_NOT_OK(s_comment_builder.Append(s_comment.c_str()));
          break;
        }
        count++;
      }
    }
    std::shared_ptr<arrow::Array> s_suppkey_array;
    ARROW_RETURN_NOT_OK(s_suppkey_builder.Finish(&s_suppkey_array));
    std::shared_ptr<arrow::Array> s_name_array;
    ARROW_RETURN_NOT_OK(s_name_builder.Finish(&s_name_array));
    std::shared_ptr<arrow::Array> s_address_array;
    ARROW_RETURN_NOT_OK(s_address_builder.Finish(&s_address_array));
    std::shared_ptr<arrow::Array> s_nationkey_array;
    ARROW_RETURN_NOT_OK(s_nationkey_builder.Finish(&s_nationkey_array));
    std::shared_ptr<arrow::Array> s_phone_array;
    ARROW_RETURN_NOT_OK(s_phone_builder.Finish(&s_phone_array));
    std::shared_ptr<arrow::Array> s_acctbal_array;
    ARROW_RETURN_NOT_OK(s_acctbal_builder.Finish(&s_acctbal_array));
    std::shared_ptr<arrow::Array> s_comment_array;
    ARROW_RETURN_NOT_OK(s_comment_builder.Finish(&s_comment_array));

    std::vector<std::shared_ptr<arrow::Field>> supplier_schema_vector = {
        arrow::field("s_suppkey", arrow::int32()),
        arrow::field("s_name", arrow::utf8()),
        arrow::field("s_address", arrow::utf8()),
        arrow::field("s_nationkey", arrow::int32()),
        arrow::field("s_phone", arrow::utf8()),
        arrow::field("s_acctbal", arrow::float64()),
        arrow::field("s_comment", arrow::utf8())};
    auto supplier_schema = std::make_shared<arrow::Schema>(supplier_schema_vector);
    supplier_t = arrow::RecordBatch::Make(supplier_schema, s_suppkey_array->length(), {s_suppkey_array, s_name_array, s_address_array, s_nationkey_array, s_phone_array, s_acctbal_array, s_comment_array});
  }
  std::cout << "load customer" << std::endl;
  std::ifstream customer_tbl;
  customer_tbl.open(customer_tbl_data_dir, std::ios::in);
  int c_custkey;
  std::string c_name;
  std::string c_address;
  int c_nationkey;
  std::string c_phone;
  double c_acctbal;
  std::string c_mktsegment;
  std::string c_comment;
  arrow::Int32Builder c_custkey_builder(pool);
  arrow::StringBuilder c_name_builder(pool);
  arrow::StringBuilder c_address_builder(pool);
  arrow::Int32Builder c_nationkey_builder(pool);
  arrow::StringBuilder c_phone_builder(pool);
  arrow::DoubleBuilder c_acctbal_builder(pool);
  arrow::StringBuilder c_mktsegment_builder(pool);
  arrow::StringBuilder c_comment_builder(pool);

  if (customer_tbl.is_open())
  {
    std::string str;
    while (getline(customer_tbl, str))
    {
      std::stringstream line(str);
      std::string tmp;
      int count = 0;
      while (getline(line, tmp, '|'))
      {
        switch (count)
        {
        case 0:
          c_custkey = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(c_custkey_builder.Append(c_custkey));
          break;
        case 1:
          c_name = tmp;
          ARROW_RETURN_NOT_OK(c_name_builder.Append(c_name.c_str()));
          break;
        case 2:
          c_address = tmp;
          ARROW_RETURN_NOT_OK(c_address_builder.Append(c_address.c_str()));
          break;
        case 3:
          c_nationkey = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(c_nationkey_builder.Append(c_nationkey));
          break;
        case 4:
          c_phone = tmp;
          ARROW_RETURN_NOT_OK(c_phone_builder.Append(c_phone.c_str()));
          break;
        case 5:
          c_acctbal = atof(tmp.c_str());
          ARROW_RETURN_NOT_OK(c_acctbal_builder.Append(c_acctbal));
          break;
        case 6:
          c_mktsegment = tmp;
          ARROW_RETURN_NOT_OK(c_mktsegment_builder.Append(c_mktsegment.c_str()));
          break;
        case 7:
          c_comment = tmp;
          ARROW_RETURN_NOT_OK(c_comment_builder.Append(c_comment.c_str()));
          break;
        }
        count++;
      }
    }

    std::shared_ptr<arrow::Array> c_custkey_array;
    ARROW_RETURN_NOT_OK(c_custkey_builder.Finish(&c_custkey_array));
    std::shared_ptr<arrow::Array> c_name_array;
    ARROW_RETURN_NOT_OK(c_name_builder.Finish(&c_name_array));
    std::shared_ptr<arrow::Array> c_address_array;
    ARROW_RETURN_NOT_OK(c_address_builder.Finish(&c_address_array));
    std::shared_ptr<arrow::Array> c_nationkey_array;
    ARROW_RETURN_NOT_OK(c_nationkey_builder.Finish(&c_nationkey_array));
    std::shared_ptr<arrow::Array> c_phone_array;
    ARROW_RETURN_NOT_OK(c_phone_builder.Finish(&c_phone_array));
    std::shared_ptr<arrow::Array> c_acctbal_array;
    ARROW_RETURN_NOT_OK(c_acctbal_builder.Finish(&c_acctbal_array));
    std::shared_ptr<arrow::Array> c_mktsegment_array;
    ARROW_RETURN_NOT_OK(c_mktsegment_builder.Finish(&c_mktsegment_array));
    std::shared_ptr<arrow::Array> c_comment_array;
    ARROW_RETURN_NOT_OK(c_comment_builder.Finish(&c_comment_array));
    std::vector<std::shared_ptr<arrow::Field>> customer_schema_vector = {
        arrow::field("c_custkey", arrow::int32()),
        arrow::field("c_name", arrow::utf8()),
        arrow::field("c_address", arrow::utf8()),
        arrow::field("c_nationkey", arrow::int32()),
        arrow::field("c_phone", arrow::utf8()),
        arrow::field("c_acctbal", arrow::float64()),
        arrow::field("c_mktsegment", arrow::utf8()),
        arrow::field("c_comment", arrow::utf8())};
    auto customer_schema = std::make_shared<arrow::Schema>(customer_schema_vector);
    customer_t = arrow::RecordBatch::Make(customer_schema, c_custkey_array->length(), {c_custkey_array, c_name_array, c_address_array, c_nationkey_array, c_phone_array, c_acctbal_array, c_mktsegment_array, c_comment_array});
  }
  std::cout << "load nation" << std::endl;
  std::ifstream nation_tbl;
  nation_tbl.open(nation_tbl_data_dir, std::ios::in);
  int n_nationkey;
  std::string n_name;
  int n_regionkey;
  std::string n_comment;

  arrow::Int32Builder n_nationkey_builder(pool);
  arrow::StringBuilder n_name_builder(pool);
  arrow::Int32Builder n_regionkey_builder(pool);
  arrow::StringBuilder n_comment_builder(pool);

  if (nation_tbl.is_open())
  {
    std::string str;
    while (getline(nation_tbl, str))
    {
      std::stringstream line(str);
      std::string tmp;
      int count = 0;
      while (getline(line, tmp, '|'))
      {
        switch (count)
        {
        case 0:
          n_nationkey = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(n_nationkey_builder.Append(n_nationkey));
          break;
        case 1:
          n_name = tmp;
          ARROW_RETURN_NOT_OK(n_name_builder.Append(n_name.c_str()));
          break;
        case 2:
          n_regionkey = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(n_regionkey_builder.Append(n_regionkey));
          break;
        case 3:
          n_comment = tmp;
          ARROW_RETURN_NOT_OK(n_comment_builder.Append(n_comment.c_str()));
          break;
        }
        count++;
      }
    }
    std::shared_ptr<arrow::Array> n_nationkey_array;
    ARROW_RETURN_NOT_OK(n_nationkey_builder.Finish(&n_nationkey_array));
    std::shared_ptr<arrow::Array> n_name_array;
    ARROW_RETURN_NOT_OK(n_name_builder.Finish(&n_name_array));
    std::shared_ptr<arrow::Array> n_regionkey_array;
    ARROW_RETURN_NOT_OK(n_regionkey_builder.Finish(&n_regionkey_array));
    std::shared_ptr<arrow::Array> n_comment_array;
    ARROW_RETURN_NOT_OK(n_comment_builder.Finish(&n_comment_array));
    std::vector<std::shared_ptr<arrow::Field>> nation_schema_vector = {
        arrow::field("n_nationkey", arrow::int32()),
        arrow::field("n_name", arrow::utf8()),
        arrow::field("n_regionkey", arrow::int32()),
        arrow::field("n_comment", arrow::utf8())};
    auto nation_schema = std::make_shared<arrow::Schema>(nation_schema_vector);
    nation_t = arrow::RecordBatch::Make(nation_schema, n_nationkey_array->length(), {n_nationkey_array, n_name_array, n_regionkey_array, n_comment_array});
  }
  std::cout << "load region" << std::endl;
  std::ifstream region_tbl;
  region_tbl.open(region_tbl_data_dir, std::ios::in);
  int r_regionkey;
  std::string r_name;
  std::string r_comment;

  arrow::Int32Builder r_regionkey_builder(pool);
  arrow::StringBuilder r_name_builder(pool);
  arrow::StringBuilder r_comment_builder(pool);
  if (region_tbl.is_open())
  {
    std::string str;
    while (getline(region_tbl, str))
    {
      std::stringstream line(str);
      std::string tmp;
      int count = 0;
      while (getline(line, tmp, '|'))
      {
        switch (count)
        {
        case 0:
          r_regionkey = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(r_regionkey_builder.Append(r_regionkey));
          break;
        case 1:
          r_name = tmp;
          ARROW_RETURN_NOT_OK(r_name_builder.Append(r_name.c_str()));
          break;
        case 2:
          r_comment = tmp;
          ARROW_RETURN_NOT_OK(r_comment_builder.Append(r_comment.c_str()));
          break;
        }
        count++;
      }
    }
    std::shared_ptr<arrow::Array> r_regionkey_array;
    ARROW_RETURN_NOT_OK(r_regionkey_builder.Finish(&r_regionkey_array));
    std::shared_ptr<arrow::Array> r_name_array;
    ARROW_RETURN_NOT_OK(r_name_builder.Finish(&r_name_array));
    std::shared_ptr<arrow::Array> r_comment_array;
    ARROW_RETURN_NOT_OK(r_comment_builder.Finish(&r_comment_array));

    std::vector<std::shared_ptr<arrow::Field>> region_schema_vector = {
        arrow::field("r_regionkey", arrow::int32()),
        arrow::field("r_name", arrow::utf8()),
        arrow::field("r_comment", arrow::utf8())};
    auto region_schema = std::make_shared<arrow::Schema>(region_schema_vector);
    region_t = arrow::RecordBatch::Make(region_schema, r_regionkey_array->length(), {r_regionkey_array, r_name_array, r_comment_array});
  }
  return arrow::Status::OK();
}
arrow::Status preproc()
{
  arrow::MemoryPool *pool = arrow::default_memory_pool();
  std::cout << "pre-processing for table partsupp" << std::endl;
  auto ps_partkey_tmp = std::static_pointer_cast<arrow::Int32Array>(partsupp_t->column(metadata_partsupp["ps_partkey"]));
  auto ps_partkey = ps_partkey_tmp->raw_values();
  auto ps_suppkey_tmp = std::static_pointer_cast<arrow::Int32Array>(partsupp_t->column(metadata_partsupp["ps_suppkey"]));
  auto ps_suppkey = ps_suppkey_tmp->raw_values();
  arrow::Int32Builder ps_key_builder(pool);
  std::map<std::pair<int, int>, int> ps_key_map;

  int count = 0;
  for (int i = 0; i < partsupp_t->num_rows(); i++)
  {
    auto ps_partkey_suppkey = std::make_pair(ps_partkey[i], ps_suppkey[i]);
    if (ps_key_map.find(ps_partkey_suppkey) != ps_key_map.end())
    {
      ARROW_RETURN_NOT_OK(ps_key_builder.Append(ps_key_map[ps_partkey_suppkey]));
    }
    else
    {
      ps_key_map[ps_partkey_suppkey] = count;
      ARROW_RETURN_NOT_OK(ps_key_builder.Append(count));
      count++;
    }
  }
  std::shared_ptr<arrow::Array> ps_key_array;
  ARROW_RETURN_NOT_OK(ps_key_builder.Finish(&ps_key_array));
  auto ps_key_field = arrow::field("ps_key", arrow::int32());

  partsupp_t = partsupp_t->AddColumn(partsupp_t->num_columns(), ps_key_field, ps_key_array).ValueOrDie();
  ;

  std::cout << "pre-processing for table orders" << std::endl;
  auto o_orderkey_tmp = std::static_pointer_cast<arrow::Int32Array>(orders_t->column(metadata_orders["o_orderkey"]));
  auto o_orderkey = o_orderkey_tmp->raw_values();
  arrow::Int32Builder o_orderkey_new_builder(pool);
  std::map<int, int> o_orderkey_new_map;
  count = 0;
  for (int i = 0; i < orders_t->num_rows(); i++)
  {
    ARROW_RETURN_NOT_OK(o_orderkey_new_builder.Append(count));
    o_orderkey_new_map[o_orderkey[i]] = count;
    count++;
  }
  std::shared_ptr<arrow::Array> o_orderkey_new_array;
  ARROW_RETURN_NOT_OK(o_orderkey_new_builder.Finish(&o_orderkey_new_array));
  auto o_orderkey_new_field = arrow::field("o_orderkey_new", arrow::int32());

  orders_t = orders_t->AddColumn(orders_t->num_columns(), o_orderkey_new_field, o_orderkey_new_array).ValueOrDie();
  std::cout << "pre-processing for table lineitem" << std::endl;
  auto l_orderkey_tmp = std::static_pointer_cast<arrow::Int32Array>(lineitem_t->column(metadata_lineitem["l_orderkey"]));
  auto l_orderkey = l_orderkey_tmp->raw_values();
  auto l_partkey_tmp = std::static_pointer_cast<arrow::Int32Array>(lineitem_t->column(metadata_lineitem["l_partkey"]));
  auto l_partkey = l_partkey_tmp->raw_values();
  auto l_suppkey_tmp = std::static_pointer_cast<arrow::Int32Array>(lineitem_t->column(metadata_lineitem["l_suppkey"]));
  auto l_suppkey = l_suppkey_tmp->raw_values();
  arrow::Int32Builder l_pskey_builder(pool);
  arrow::Int32Builder l_orderkey_new_builder(pool);
  for (int i = 0; i < lineitem_t->num_rows(); i++)
  {
    auto l_partkey_suppkey = std::make_pair(l_partkey[i], l_suppkey[i]);
    ARROW_RETURN_NOT_OK(l_pskey_builder.Append(ps_key_map[l_partkey_suppkey]));

    ARROW_RETURN_NOT_OK(l_orderkey_new_builder.Append(o_orderkey_new_map[l_orderkey[i]]));
  }
  std::shared_ptr<arrow::Array> l_pskey_array;
  ARROW_RETURN_NOT_OK(l_pskey_builder.Finish(&l_pskey_array));
  auto l_pskey_field = arrow::field("l_pskey", arrow::int32());

  lineitem_t = lineitem_t->AddColumn(lineitem_t->num_columns(), l_pskey_field, l_pskey_array).ValueOrDie();

  std::shared_ptr<arrow::Array> l_orderkey_new_array;
  ARROW_RETURN_NOT_OK(l_orderkey_new_builder.Finish(&l_orderkey_new_array));
  auto l_orderkey_new_field = arrow::field("l_orderkey_new", arrow::int32());

  lineitem_t = lineitem_t->AddColumn(lineitem_t->num_columns(), l_orderkey_new_field, l_orderkey_new_array).ValueOrDie();
  return arrow::Status::OK();
}
void set_bitmap(int *bitmap, int REGION, const int *r_regionkey)
{
  for (int i = 0; i < region_t->num_rows(); i++)
  {
    if (r_regionkey[i] == REGION)
      bitmap[i] = 1;
  }
}
void *DimVec_S_thread(void *param)
{
  pth_dst *argst = (pth_dst *)param;
  for (int i = 0; i < argst->comline; i++)
  {
    int location = i + argst->start;
    if (argst->bitmap[argst->n_regionkey[argst->s_nationkey[location]]] == 1)
      argst->bitmap_S[location] = argst->s_nationkey[location];
  }
  return nullptr;
}
void *DimVec_o_thread(void *param)
{
  pth_dot *argst = (pth_dot *)param;
  for (int i = 0; i < argst->comline; i++)
  {
    int location = i + argst->start;
    if (argst->bitmap[argst->n_regionkey[argst->c_nationkey[argst->o_custkey[location] - 1]]] == 1 && argst->o_orderdate[location] >= argst->DATE1 && argst->o_orderdate[location] < argst->DATE2)
    {
      argst->bitmap_o[location] = argst->c_nationkey[argst->o_custkey[location] - 1];
    }
  }
  return nullptr;
}
void *Tri_table_join_vector_thread(void *param)
{
  pth_ttjt *argst = (pth_ttjt *)param;
  int size_v = 1024;
  int nblock = argst->comline / size_v;
  int iter = 0;
  int groupID[size_v];
  int sum = 0;
  while (iter <= nblock)
  {
    int64_t length = (iter == nblock) ? argst->comline % size_v : size_v;
    for (int j = 0; j < length; j++)
    {
      int location = argst->start + iter * size_v + j;
      if (argst->bitmap_S[argst->l_suppkey[location] - 1] != -1 && argst->bitmap_S[argst->l_suppkey[location] - 1] == argst->bitmap_o[argst->l_orderkey_new[location]])
        groupID[j] = argst->bitmap_S[argst->l_suppkey[location] - 1];
      else
        groupID[j] = -1;
    }
    for (int i = 0; i < length; i++)
    {
      int16_t tmp = groupID[i];
      int location = argst->start + iter * size_v + i;

      if (tmp != -1)
      {
        sum++;
        argst->GrpVex[tmp] += argst->l_extendedprice[location] * (1 - argst->l_discount[location]);
      }
    }
    iter++;
  }
  return nullptr;
}
void DimVec_S(int *bitmap, int *bitmap_S, const int *s_nationkey, const int *n_regionkey)
{
  int nthreads = 16;
  pth_dst argst[nthreads];
  int64_t numS, numSthr;
  int j;
  int rv;

  cpu_set_t set;
  pthread_t tid[nthreads];
  pthread_attr_t attr;
  pthread_barrier_t barrier;
  numS = supplier_t->num_rows();
  numSthr = numS / nthreads;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, STACKSIZE);
  for (j = 0; j < nthreads; j++)
  {
    int cpu_idx = j;
    CPU_ZERO(&set);
    CPU_SET(cpu_idx, &set);
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);
    argst[j].comline = (j == (nthreads - 1)) ? numS : numSthr;
    argst[j].start = numSthr * j;
    numS -= numSthr;
    argst[j].bitmap = bitmap;
    argst[j].bitmap_S = bitmap_S;
    argst[j].n_regionkey = n_regionkey;
    argst[j].s_nationkey = s_nationkey;
    rv = pthread_create(&tid[j], &attr, DimVec_S_thread, (void *)&argst[j]);
    if (rv)
    {
      printf("ERROR; return code from pthread_create() is %d\n", rv);
      exit(-1);
    }
  }
  for (j = 0; j < nthreads; j++)
  {
    pthread_join(tid[j], NULL);
  }
  return;
}
void DimVec_o(int *bitmap, int *bitmap_o, const int *o_orderdate, const int *o_custkey, const int *c_nationkey, const int *n_regionkey, int DATE1, int DATE2)
{
  int nthreads = 16;
  pth_dot argst[nthreads];
  int64_t numS, numSthr;
  int j;
  int rv;

  cpu_set_t set;
  pthread_t tid[nthreads];
  pthread_attr_t attr;
  pthread_barrier_t barrier;
  numS = orders_t->num_rows();
  numSthr = numS / nthreads;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, STACKSIZE);
  for (j = 0; j < nthreads; j++)
  {
    int cpu_idx = j;
    CPU_ZERO(&set);
    CPU_SET(cpu_idx, &set);
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);
    argst[j].comline = (j == (nthreads - 1)) ? numS : numSthr;
    argst[j].start = numSthr * j;
    numS -= numSthr;
    argst[j].bitmap = bitmap;
    argst[j].bitmap_o = bitmap_o;
    argst[j].DATE1 = DATE1;
    argst[j].DATE2 = DATE2;
    argst[j].n_regionkey = n_regionkey;
    argst[j].c_nationkey = c_nationkey;
    argst[j].o_custkey = o_custkey;
    argst[j].o_orderdate = o_orderdate;
    rv = pthread_create(&tid[j], &attr, DimVec_o_thread, (void *)&argst[j]);
    if (rv)
    {
      printf("ERROR; return code from pthread_create() is %d\n", rv);
      exit(-1);
    }
  }
  for (j = 0; j < nthreads; j++)
  {
    pthread_join(tid[j], NULL);
  }
  return;
}
void Tri_table_join(int *bitmap_S, int *bitmap_o, const int *l_suppkey, const int *l_orderkey_new, const double *l_extendedprice, const double *l_discount, double *GrpVex)
{
  int nthreads = 16;
  pth_ttjt argst[nthreads];
  int64_t numS, numSthr;
  int j;
  int rv;

  cpu_set_t set;
  pthread_t tid[nthreads];
  pthread_attr_t attr;
  pthread_barrier_t barrier;
  numS = lineitem_t->num_rows();
  numSthr = numS / nthreads;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, STACKSIZE);
  for (j = 0; j < nthreads; j++)
  {
    int cpu_idx = j;
    CPU_ZERO(&set);
    CPU_SET(cpu_idx, &set);
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);
    argst[j].comline = (j == (nthreads - 1)) ? numS : numSthr;
    argst[j].start = numSthr * j;
    numS -= numSthr;
    argst[j].bitmap_S = bitmap_S;
    argst[j].bitmap_o = bitmap_o;
    argst[j].l_suppkey = l_suppkey;
    argst[j].l_orderkey_new = l_orderkey_new;
    argst[j].l_extendedprice = l_extendedprice;
    argst[j].l_discount = l_discount;
    argst[j].GrpVex = new double[25];
    memset(argst[j].GrpVex, 0.0, 25 * sizeof(double));
    rv = pthread_create(&tid[j], &attr, Tri_table_join_vector_thread, (void *)&argst[j]);
    if (rv)
    {
      printf("ERROR; return code from pthread_create() is %d\n", rv);
      exit(-1);
    }
  }
  for (j = 0; j < nthreads; j++)
  {
    pthread_join(tid[j], NULL);
  }
  for (j = 0; j < nthreads; j++)
  {
    for (int i = 0; i < 25; i++)
      GrpVex[i] +=argst[j].GrpVex[i];
  }
  return;
}
int main()
{
  metadata_tablename["lineitem"] = 0;
  metadata_tablename["partsupp"] = 1;
  metadata_tablename["orders"] = 2;
  metadata_tablename["part"] = 3;
  metadata_tablename["supplier"] = 4;
  metadata_tablename["customer"] = 5;
  metadata_tablename["nation"] = 6;
  metadata_tablename["region"] = 7;
  /*init the metadata of lineitem*/

  metadata_lineitem["l_orderkey"] = 0;
  metadata_lineitem["l_partkey"] = 1;
  metadata_lineitem["l_suppkey"] = 2;
  metadata_lineitem["l_linenumber"] = 3;
  metadata_lineitem["l_quantity"] = 4;
  metadata_lineitem["l_extendedprice"] = 5;
  metadata_lineitem["l_discount"] = 6;
  metadata_lineitem["l_tax"] = 7;
  metadata_lineitem["l_returnflag"] = 8;
  metadata_lineitem["l_linestatus"] = 9;
  metadata_lineitem["l_shipdate"] = 10;
  metadata_lineitem["l_commitdate"] = 11;
  metadata_lineitem["l_receiptdate"] = 12;
  metadata_lineitem["l_shipinstruct"] = 13;
  metadata_lineitem["l_shipmode"] = 14;
  metadata_lineitem["l_comment"] = 15;
  metadata_lineitem["l_pskey"] = 16;
  metadata_lineitem["l_orderkey_new"] = 17;
  /*init the metadata of partsupp*/
  metadata_partsupp["ps_key"] = 5;
  metadata_partsupp["ps_partkey"] = 0;
  metadata_partsupp["ps_suppkey"] = 1;
  metadata_partsupp["ps_availqty"] = 2;
  metadata_partsupp["ps_supplycost"] = 3;
  metadata_partsupp["ps_comment"] = 4;
  /*init the metadata of orders*/
  metadata_orders["o_orderkey_new"] = 9;
  metadata_orders["o_orderkey"] = 0;
  metadata_orders["o_custkey"] = 1;
  metadata_orders["o_orderstatus"] = 2;
  metadata_orders["o_totalprice"] = 3;
  metadata_orders["o_orderdate"] = 4;
  metadata_orders["o_orderpriority"] = 5;
  metadata_orders["o_clerk"] = 6;
  metadata_orders["o_shippriority"] = 7;
  metadata_orders["o_comment"] = 8;
  /*init the metadata of part*/
  metadata_part["p_partkey"] = 0;
  metadata_part["p_name"] = 1;
  metadata_part["p_mfgr"] = 2;
  metadata_part["p_brand"] = 3;
  metadata_part["p_type"] = 4;
  metadata_part["p_size"] = 5;
  metadata_part["p_container"] = 6;
  metadata_part["p_retailprice"] = 7;
  metadata_part["p_comment"] = 8;
  /*init the metadata of supplier*/
  metadata_supplier["s_suppkey"] = 0;
  metadata_supplier["s_name"] = 1;
  metadata_supplier["s_address"] = 2;
  metadata_supplier["s_nationkey"] = 3;
  metadata_supplier["s_phone"] = 4;
  metadata_supplier["s_acctbal"] = 5;
  metadata_supplier["s_comment"] = 6;
  /*init the metadata of customer*/
  metadata_customer["c_custkey"] = 0;
  metadata_customer["c_name"] = 1;
  metadata_customer["c_address"] = 2;
  metadata_customer["c_nationkey"] = 3;
  metadata_customer["c_phone"] = 4;
  metadata_customer["c_acctbal"] = 5;
  metadata_customer["c_mktsegment"] = 6;
  metadata_customer["c_comment"] = 7;
  /*init the metadata of nation*/
  metadata_nation["n_nationkey"] = 0;
  metadata_nation["n_name"] = 1;
  metadata_nation["n_regionkey"] = 2;
  metadata_nation["n_comment"] = 3;
  /*init the metadata of region*/
  metadata_region["r_regionkey"] = 0;
  metadata_region["r_name"] = 1;
  metadata_region["r_comment"] = 2;
  std::cout << "Start load()" << std::endl;
  auto load_flag = load();
  std::cout << "Start preproc()" << std::endl;
  auto preproc_flag = preproc();

  int REGION = 2;
  int DATE = 19950101;
  int *bitmap = new int[region_t->num_rows()];
  memset(bitmap, 0xff, region_t->num_rows() * sizeof(int));
  int *bitmap_S = new int[supplier_t->num_rows()];
  memset(bitmap_S, 0xff, supplier_t->num_rows() * sizeof(int));
  int *bitmap_o = new int[orders_t->num_rows()];
  memset(bitmap_o, 0xff, orders_t->num_rows() * sizeof(int));
  double *GrpVex = new double[25];
  memset(GrpVex, 0.0, 25 * sizeof(double));
  auto r_regionkey_tmp = std::static_pointer_cast<arrow::Int32Array>(region_t->column(metadata_region["r_regionkey"]));
  const int *r_regionkey = r_regionkey_tmp->raw_values();
  auto s_nationkey_tmp = std::static_pointer_cast<arrow::Int32Array>(supplier_t->column(metadata_supplier["s_nationkey"]));
  const int *s_nationkey = s_nationkey_tmp->raw_values();
  auto n_regionkey_tmp = std::static_pointer_cast<arrow::Int32Array>(nation_t->column(metadata_nation["n_regionkey"]));
  const int *n_regionkey = n_regionkey_tmp->raw_values();
  auto o_orderdate_tmp = std::static_pointer_cast<arrow::Int32Array>(orders_t->column(metadata_orders["o_orderdate"]));
  const int *o_orderdate = o_orderdate_tmp->raw_values();
  auto o_custkey_tmp = std::static_pointer_cast<arrow::Int32Array>(orders_t->column(metadata_orders["o_custkey"]));
  const int *o_custkey = o_custkey_tmp->raw_values();
  auto c_nationkey_tmp = std::static_pointer_cast<arrow::Int32Array>(customer_t->column(metadata_customer["c_nationkey"]));
  const int *c_nationkey = c_nationkey_tmp->raw_values();
  auto l_suppkey_tmp = std::static_pointer_cast<arrow::Int32Array>(lineitem_t->column(metadata_lineitem["l_suppkey"]));
  const int *l_suppkey = l_suppkey_tmp->raw_values();
  auto l_orderkey_new_tmp = std::static_pointer_cast<arrow::Int32Array>(lineitem_t->column(metadata_lineitem["l_orderkey_new"]));
  const int *l_orderkey_new = l_orderkey_new_tmp->raw_values();
  auto l_extendedprice_tmp = std::static_pointer_cast<arrow::DoubleArray>(lineitem_t->column(metadata_lineitem["l_extendedprice"]));
  const double *l_extendedprice = l_extendedprice_tmp->raw_values();
  auto l_discount_tmp = std::static_pointer_cast<arrow::DoubleArray>(lineitem_t->column(metadata_lineitem["l_discount"]));
  const double *l_discount = l_discount_tmp->raw_values();

  auto n_name = std::static_pointer_cast<arrow::StringArray>(nation_t->column(metadata_nation["n_name"]));
  auto r_name = std::static_pointer_cast<arrow::StringArray>(region_t->column(metadata_region["r_name"]));
  int flag = 1;
  std::string region_name;
  while (flag)
  {
    std::cout << "  input 0: quit " << std::endl;
    std::cout << "  input 1: enter the next test " << std::endl;
    std::cout << "  please input:  ";
    std::cin >> flag;
    if (flag == 0)
      break;
    std::cout << "  Input parameter: REGION  ";
    std::cin >> region_name;
    std::cout << "  Input parameter: DATE  ";
    std::cin >> DATE;
    for (int i = 0; i < region_t->num_rows(); i++)
      if (r_name->GetString(i) == region_name)
        REGION = i;
    set_bitmap(bitmap, REGION, r_regionkey);
    DimVec_S(bitmap, bitmap_S, s_nationkey, n_regionkey);

    DimVec_o(bitmap, bitmap_o, o_orderdate, o_custkey, c_nationkey, n_regionkey, DATE, DATE + 10000);

    Tri_table_join(bitmap_S, bitmap_o, l_suppkey, l_orderkey_new, l_extendedprice, l_discount, GrpVex);
    // 验证查询结果
    for (int i = 0; i < 25; i++)
    {
      if (GrpVex[i]!= 0.0)
      {
        std::cout << n_name->GetString(i) << "  " << GrpVex[i] << std::endl;
      }
    }
  }
  return 0;
}
