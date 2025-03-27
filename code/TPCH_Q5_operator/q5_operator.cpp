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

struct Column_store
{
  /*supplier*/
  const int *s_suppkey;
  std::shared_ptr<arrow::StringArray> s_name;
  std::shared_ptr<arrow::StringArray> s_address;
  const int *s_nationkey;
  std::shared_ptr<arrow::StringArray> s_phone;
  const double *s_acctbal;
  std::shared_ptr<arrow::StringArray> s_comment;

  /*customer*/
  const int *c_custkey;
  std::shared_ptr<arrow::StringArray> c_name;
  std::shared_ptr<arrow::StringArray> c_address;
  const int *c_nationkey;
  std::shared_ptr<arrow::StringArray> c_phone;
  const double *c_acctbal;
  std::shared_ptr<arrow::StringArray> c_mktsegment;
  std::shared_ptr<arrow::StringArray> c_comment;

  /*part*/
  const int *p_partkey;
  std::shared_ptr<arrow::StringArray> p_name;
  std::shared_ptr<arrow::StringArray> p_mfgr;
  std::shared_ptr<arrow::StringArray> p_brand;
  std::shared_ptr<arrow::StringArray> p_type;
  const int *p_size;
  std::shared_ptr<arrow::StringArray> p_container;
  const double *p_retailprice;
  std::shared_ptr<arrow::StringArray> p_comment;

  /*nation*/
  const int *n_nationkey;
  std::shared_ptr<arrow::StringArray> n_name;
  const int *n_regionkey;
  std::shared_ptr<arrow::StringArray> n_comment;

  /*region*/
  const int *r_regionkey;
  std::shared_ptr<arrow::StringArray> r_name;
  std::shared_ptr<arrow::StringArray> r_comment;

  /*orders*/
  const int *o_orderkey_1;
  const int *o_orderkey;
  const int *o_custkey;
  std::shared_ptr<arrow::StringArray> o_orderstatus;
  const double *o_totalprice;
  const int *o_orderdate;
  std::shared_ptr<arrow::StringArray> o_orderpriority;
  std::shared_ptr<arrow::StringArray> o_clerk;
  const int *o_shippriority;
  std::shared_ptr<arrow::StringArray> o_comment;

  /*partsupp*/
  const int *ps_key;
  const int *ps_partkey;
  const int *ps_suppkey;
  const int *ps_availqty;
  const double *ps_supplycost;
  std::shared_ptr<arrow::StringArray> ps_comment;

  /*lineitem*/
  const int *l_pskey;
  const int *l_orderkey_1;
  const int *l_orderkey;
  const int *l_partkey;
  const int *l_suppkey;
  const int *l_linenumber;
  const double *l_quantity;
  const double *l_extendedprice;
  const double *l_discount;
  const double *l_tax;
  std::shared_ptr<arrow::StringArray> l_returnflag;
  std::shared_ptr<arrow::StringArray> l_linestatus;
  const int *l_shipdate;
  const int *l_commitdate;
  const int *l_receiptdate;
  std::shared_ptr<arrow::StringArray> l_shipinstruct;
  std::shared_ptr<arrow::StringArray> l_shipmode;
  std::shared_ptr<arrow::StringArray> l_comment;
};
Column_store column_store;

const int *Load_int_column_from_table(std::string tablename, std::string colname)
{
  switch (metadata_tablename[tablename.c_str()])
  {
  case 0:
    switch (metadata_lineitem[colname.c_str()])
    {
    case 0:
      return column_store.l_pskey;
    case 1:
      return column_store.l_orderkey_1;
    case 2:
      return column_store.l_orderkey;
    case 3:
      return column_store.l_partkey;
    case 4:
      return column_store.l_suppkey;
    case 5:
      return column_store.l_linenumber;
    case 12:
      return column_store.l_shipdate;
    case 13:
      return column_store.l_commitdate;
    case 14:
      return column_store.l_receiptdate;
    }
  case 1:
    switch (metadata_partsupp[colname.c_str()])
    {
    case 0:
      return column_store.ps_key;
    case 1:
      return column_store.ps_partkey;
    case 2:
      return column_store.ps_suppkey;
    case 3:
      return column_store.ps_availqty;
    }
  case 2:
    switch (metadata_orders[colname.c_str()])
    {
    case 0:
      return column_store.o_orderkey_1;
    case 1:
      return column_store.o_orderkey;
    case 2:
      return column_store.o_custkey;
    case 5:
      return column_store.o_orderdate;
    case 8:
      return column_store.o_shippriority;
    }
  case 3:
    switch (metadata_part[colname.c_str()])
    {
    case 0:
      return column_store.p_partkey;
    case 5:
      return column_store.p_size;
    }
  case 4:
    switch (metadata_supplier[colname.c_str()])
    {
    case 0:
      return column_store.s_suppkey;
    case 3:
      return column_store.s_nationkey;
    }
  case 5:
    switch (metadata_customer[colname.c_str()])
    {
    case 0:
      return column_store.c_custkey;
    case 3:
      return column_store.c_nationkey;
    }
  case 6:
    switch (metadata_nation[colname.c_str()])
    {
    case 0:
      return column_store.n_nationkey;
    case 2:
      return column_store.n_regionkey;
    }
  case 7:
    switch (metadata_region[colname.c_str()])
    {
    case 0:
      return column_store.r_regionkey;
    }
  }
  return nullptr;
}
const double *Load_double_column_from_table(std::string tablename, std::string colname)
{
  switch (metadata_tablename[tablename.c_str()])
  {
  case 0:
    switch (metadata_lineitem[colname.c_str()])
    {
    case 6:
      return column_store.l_quantity;
    case 7:
      return column_store.l_extendedprice;
    case 8:
      return column_store.l_discount;
    case 9:
      return column_store.l_tax;
    }
  case 1:
    switch (metadata_partsupp[colname.c_str()])
    {
    case 4:
      return column_store.ps_supplycost;
    }
  case 2:
    switch (metadata_orders[colname.c_str()])
    {
    case 4:
      return column_store.o_totalprice;
    }
  case 3:
    switch (metadata_part[colname.c_str()])
    {
    case 7:
      return column_store.p_retailprice;
    }
  case 4:
    switch (metadata_supplier[colname.c_str()])
    {
    case 5:
      return column_store.s_acctbal;
    }
  case 5:
    switch (metadata_customer[colname.c_str()])
    {
    case 5:
      return column_store.c_acctbal;
    }
  }
  return nullptr;
}
std::string_view Load_string_value_from_table(std::string tablename, std::string colname, int location)
{
  switch (metadata_tablename[tablename.c_str()])
  {
  case 0:
    switch (metadata_lineitem[colname.c_str()])
    {
    case 10:
      return column_store.l_returnflag->Value(location);
    case 11:
      return column_store.l_linestatus->Value(location);
    case 15:
      return column_store.l_shipinstruct->Value(location);
    case 16:
      return column_store.l_shipmode->Value(location);
    case 17:
      return column_store.l_comment->Value(location);
    }
  case 1:
    switch (metadata_partsupp[colname.c_str()])
    {
    case 5:
      return column_store.ps_comment->Value(location);
    }
  case 2:
    switch (metadata_orders[colname.c_str()])
    {
    case 3:
      return column_store.o_orderstatus->Value(location);
    case 6:
      return column_store.o_orderpriority->Value(location);
    case 7:
      return column_store.o_clerk->Value(location);
    case 9:
      return column_store.o_comment->Value(location);
    }
  case 3:
    switch (metadata_part[colname.c_str()])
    {
    case 1:
      return column_store.p_name->Value(location);
    case 2:
      return column_store.p_mfgr->Value(location);
    case 3:
      return column_store.p_brand->Value(location);
    case 4:
      return column_store.p_type->Value(location);
    case 6:
      return column_store.p_container->Value(location);
    case 8:
      return column_store.p_comment->Value(location);
    }
  case 4:
    switch (metadata_supplier[colname.c_str()])
    {
    case 1:
      return column_store.s_name->Value(location);
    case 2:
      return column_store.s_address->Value(location);
    case 4:
      return column_store.s_phone->Value(location);
    case 7:
      return column_store.s_comment->Value(location);
    }
  case 5:
    switch (metadata_customer[colname.c_str()])
    {
    case 1:
      return column_store.c_name->Value(location);
    case 2:
      return column_store.c_address->Value(location);
    case 4:
      return column_store.c_phone->Value(location);
    case 6:
      return column_store.c_mktsegment->Value(location);
    case 7:
      return column_store.c_comment->Value(location);
    }
  case 6:
    switch (metadata_nation[colname.c_str()])
    {
    case 1:
      return column_store.n_name->Value(location);
    case 3:
      return column_store.n_comment->Value(location);
    }
  case 7:
    switch (metadata_region[colname.c_str()])
    {
    case 1:
      return column_store.r_name->Value(location);
    case 2:
      return column_store.r_comment->Value(location);
    }
  }
  return "";
}

/**
 * @brief Import table data from the CSV format files
 *
 * @param lineitem_t
 * @param partsupp_t
 * @param orders_t
 * @param part_t
 * @param supplier_t
 * @param customer_t
 * @param nation_t
 * @param region_t
 * @return arrow::Status
 */
arrow::Status Load_From_CSV_To_IPC()
{

  arrow::MemoryPool *pool = arrow::default_memory_pool();
  std::cout << "convert lineitem" << std::endl;
  std::ifstream lineitem_csv;
  lineitem_csv.open(lineitem_data_dir, std::ios::in);
  int l_pskey;
  int l_orderkey_1;
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
  arrow::Int32Builder l_pskey_builder(pool);
  arrow::Int32Builder l_orderkey_1_builder(pool);
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

  if (lineitem_csv.is_open())
  {
    std::string str;
    while (getline(lineitem_csv, str))
    {
      std::stringstream line(str);
      std::string tmp;
      int count = 0;
      while (getline(line, tmp, '|'))
      {
        switch (count)
        {
        case 0:
          l_pskey = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_pskey_builder.Append(l_pskey));
          break;
        case 1:
          l_orderkey_1 = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_orderkey_1_builder.Append(l_orderkey_1));
          break;
        case 2:
          l_orderkey = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_orderkey_builder.Append(l_orderkey));
          break;
        case 3:
          l_partkey = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_partkey_builder.Append(l_partkey));
          break;
        case 4:
          l_suppkey = atoi(tmp.c_str()) - 1;
          ARROW_RETURN_NOT_OK(l_suppkey_builder.Append(l_suppkey));
          break;
        case 5:
          l_linenumber = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_linenumber_builder.Append(l_linenumber));
          break;
        case 6:
          l_quantity = atof(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_quantity_builder.Append(l_quantity));
          break;
        case 7:
          l_extendedprice = atof(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_extendedprice_builder.Append(l_extendedprice));
          break;
        case 8:
          l_discount = atof(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_discount_builder.Append(l_discount));
          break;
        case 9:
          l_tax = atof(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_tax_builder.Append(l_tax));
          break;
        case 10:
          l_returnflag = tmp;
          ARROW_RETURN_NOT_OK(l_returnflag_builder.Append(l_returnflag.c_str()));
          break;
        case 11:
          l_linestatus = tmp;
          ARROW_RETURN_NOT_OK(l_linestatus_builder.Append(l_linestatus.c_str()));
          break;
        case 12:
          l_shipdate = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_shipdate_builder.Append(l_shipdate));
          break;
        case 13:
          l_commitdate = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_commitdate_builder.Append(l_commitdate));
          break;
        case 14:
          l_receiptdate = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(l_receiptdate_builder.Append(l_receiptdate));
          break;
        case 15:
          l_shipinstruct = tmp;
          ARROW_RETURN_NOT_OK(l_shipinstruct_builder.Append(l_shipinstruct.c_str()));
          break;
        case 16:
          l_shipmode = tmp;
          ARROW_RETURN_NOT_OK(l_shipmode_builder.Append(l_shipmode.c_str()));
          break;
        case 17:
          l_comment = tmp;
          ARROW_RETURN_NOT_OK(l_comment_builder.Append(l_comment.c_str()));
          break;
        }
        count++;
      }
    }
    std::shared_ptr<arrow::Array> l_pskey_array;
    ARROW_RETURN_NOT_OK(l_pskey_builder.Finish(&l_pskey_array));
    std::shared_ptr<arrow::Array> l_orderkey_1_array;
    ARROW_RETURN_NOT_OK(l_orderkey_1_builder.Finish(&l_orderkey_1_array));
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
        arrow::field("l_pskey", arrow::int32()),
        arrow::field("l_orderkey_1", arrow::int32()),
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
    auto lineitem = arrow::Table::Make(lineitem_schema, {l_pskey_array, l_orderkey_1_array, l_orderkey_array, l_partkey_array, l_suppkey_array,
                                                         l_linenumber_array, l_quantity_array, l_extendedprice_array, l_discount_array, l_tax_array, l_returnflag_array, l_linestatus_array,
                                                         l_shipdate_array, l_commitdate_array, l_receiptdate_array, l_shipinstruct_array, l_shipmode_array, l_comment_array});
    ARROW_ASSIGN_OR_RAISE(auto lineitem_output_file,
                          arrow::io::FileOutputStream::Open("./dbgen/lineitem.arrow"));
    ARROW_ASSIGN_OR_RAISE(auto lineitem_batch_writer,
                          arrow::ipc::MakeFileWriter(lineitem_output_file, lineitem->schema()));
    ARROW_RETURN_NOT_OK(lineitem_batch_writer->WriteTable(*lineitem));
    ARROW_RETURN_NOT_OK(lineitem_batch_writer->Close());
  }
  std::cout << "convert partsupp" << std::endl;
  std::ifstream partsupp_csv;
  partsupp_csv.open(partsupp_data_dir, std::ios::in);
  int ps_key;
  int ps_partkey;
  int ps_suppkey;
  int ps_availqty;
  double ps_supplycost;
  std::string ps_comment;
  arrow::Int32Builder ps_key_builder(pool);
  arrow::Int32Builder ps_partkey_builder(pool);
  arrow::Int32Builder ps_suppkey_builder(pool);
  arrow::Int32Builder ps_availqty_builder(pool);
  arrow::DoubleBuilder ps_supplycost_builder(pool);
  arrow::StringBuilder ps_comment_builder(pool);
  if (partsupp_csv.is_open())
  {
    std::string str;
    while (getline(partsupp_csv, str))
    {
      std::stringstream line(str);
      std::string tmp;
      int count = 0;
      while (getline(line, tmp, '|'))
      {
        switch (count)
        {
        case 0:
          ps_key = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(ps_key_builder.Append(ps_key));
          break;
        case 1:
          ps_partkey = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(ps_partkey_builder.Append(ps_partkey));
          break;
        case 2:
          ps_suppkey = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(ps_suppkey_builder.Append(ps_suppkey));
          break;
        case 3:
          ps_availqty = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(ps_availqty_builder.Append(ps_availqty));
          break;
        case 4:
          ps_supplycost = atof(tmp.c_str());
          ARROW_RETURN_NOT_OK(ps_supplycost_builder.Append(ps_supplycost));
          break;
        case 5:
          ps_comment = tmp;
          ARROW_RETURN_NOT_OK(ps_comment_builder.Append(ps_comment.c_str()));
          break;
        }
        count++;
      }
    }
    std::shared_ptr<arrow::Array> ps_key_array;
    ARROW_RETURN_NOT_OK(ps_key_builder.Finish(&ps_key_array));
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
        arrow::field("ps_key", arrow::int32()),
        arrow::field("ps_partkey", arrow::int32()),
        arrow::field("ps_suppkey", arrow::int32()),
        arrow::field("ps_availqty", arrow::int32()),
        arrow::field("ps_supplycost", arrow::float64()),
        arrow::field("ps_comment", arrow::utf8())};
    auto partsupp_schema = std::make_shared<arrow::Schema>(partsupp_schema_vector);
    auto partsupp = arrow::Table::Make(partsupp_schema, {ps_key_array, ps_partkey_array, ps_suppkey_array, ps_availqty_array,
                                                         ps_supplycost_array, ps_comment_array});
    ARROW_ASSIGN_OR_RAISE(auto partsupp_output_file,
                          arrow::io::FileOutputStream::Open("./dbgen/partsupp.arrow"));
    ARROW_ASSIGN_OR_RAISE(auto partsupp_batch_writer,
                          arrow::ipc::MakeFileWriter(partsupp_output_file, partsupp->schema()));
    ARROW_RETURN_NOT_OK(partsupp_batch_writer->WriteTable(*partsupp));
    ARROW_RETURN_NOT_OK(partsupp_batch_writer->Close());
  }
  std::cout << "convert orders" << std::endl;
  std::ifstream orders_csv;
  orders_csv.open(orders_data_dir, std::ios::in);
  int o_orderkey_1;
  int o_orderkey;
  int o_custkey;
  std::string o_orderstatus;
  double o_totalprice;
  int o_orderdate;
  std::string o_orderpriority;
  std::string o_clerk;
  int o_shippriority;
  std::string o_comment;
  arrow::Int32Builder o_orderkey_1_builder(pool);
  arrow::Int32Builder o_orderkey_builder(pool);
  arrow::Int32Builder o_custkey_builder(pool);
  arrow::StringBuilder o_orderstatus_builder(pool);
  arrow::DoubleBuilder o_totalprice_builder(pool);
  arrow::Int32Builder o_orderdate_builder(pool);
  arrow::StringBuilder o_orderpriority_builder(pool);
  arrow::StringBuilder o_clerk_builder(pool);
  arrow::Int32Builder o_shippriority_builder(pool);
  arrow::StringBuilder o_comment_builder(pool);
  if (orders_csv.is_open())
  {
    std::string str;
    while (getline(orders_csv, str))
    {
      std::stringstream line(str);
      std::string tmp;
      int count = 0;
      while (getline(line, tmp, '|'))
      {
        switch (count)
        {
        case 0:
          o_orderkey_1 = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(o_orderkey_1_builder.Append(o_orderkey_1));
          break;
        case 1:
          o_orderkey = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(o_orderkey_builder.Append(o_orderkey));
          break;
        case 2:
          o_custkey = atoi(tmp.c_str()) - 1;
          ARROW_RETURN_NOT_OK(o_custkey_builder.Append(o_custkey));
          break;
        case 3:
          o_orderstatus = tmp;
          ARROW_RETURN_NOT_OK(o_orderstatus_builder.Append(o_orderstatus.c_str()));
          break;
        case 4:
          o_totalprice = atof(tmp.c_str());
          ARROW_RETURN_NOT_OK(o_totalprice_builder.Append(o_totalprice));
          break;
        case 5:
          o_orderdate = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(o_orderdate_builder.Append(o_orderdate));
          break;
        case 6:
          o_orderpriority = tmp;
          ARROW_RETURN_NOT_OK(o_orderpriority_builder.Append(o_orderpriority.c_str()));
          break;
        case 7:
          o_clerk = tmp;
          ARROW_RETURN_NOT_OK(o_clerk_builder.Append(o_clerk.c_str()));
          break;
        case 8:
          o_shippriority = atoi(tmp.c_str());
          ARROW_RETURN_NOT_OK(o_shippriority_builder.Append(o_shippriority));
          break;
        case 9:
          o_comment = tmp;
          ARROW_RETURN_NOT_OK(o_comment_builder.Append(o_comment.c_str()));
          break;
        }
        count++;
      }
    }
    std::shared_ptr<arrow::Array> o_orderkey_1_array;
    ARROW_RETURN_NOT_OK(o_orderkey_1_builder.Finish(&o_orderkey_1_array));
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
        arrow::field("o_orderkey_1", arrow::int32()),
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
    auto orders = arrow::Table::Make(orders_schema, {o_orderkey_1_array, o_orderkey_array, o_custkey_array, o_orderstatus_array,
                                                     o_totalprice_array, o_orderdate_array, o_orderpriority_array, o_clerk_array, o_shippriority_array, o_comment_array});
    ARROW_ASSIGN_OR_RAISE(auto orders_output_file,
                          arrow::io::FileOutputStream::Open("./dbgen/orders.arrow"));
    ARROW_ASSIGN_OR_RAISE(auto orders_batch_writer,
                          arrow::ipc::MakeFileWriter(orders_output_file, orders->schema()));
    ARROW_RETURN_NOT_OK(orders_batch_writer->WriteTable(*orders));
    ARROW_RETURN_NOT_OK(orders_batch_writer->Close());
  }
  std::cout << "convert part" << std::endl;
  std::ifstream part_csv;
  part_csv.open(part_data_dir, std::ios::in);
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
  if (part_csv.is_open())
  {
    std::string str;
    while (getline(part_csv, str))
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
    auto part = arrow::Table::Make(part_schema, {p_partkey_array, p_name_array, p_mfgr_array, p_brand_array, p_type_array,
                                                 p_size_array, p_container_array, p_retailprice_array, p_comment_array});
    ARROW_ASSIGN_OR_RAISE(auto part_output_file,
                          arrow::io::FileOutputStream::Open("./dbgen/part.arrow"));
    ARROW_ASSIGN_OR_RAISE(auto part_batch_writer,
                          arrow::ipc::MakeFileWriter(part_output_file, part->schema()));
    ARROW_RETURN_NOT_OK(part_batch_writer->WriteTable(*part));
    ARROW_RETURN_NOT_OK(part_batch_writer->Close());
  }
  std::cout << "convert supplier" << std::endl;
  std::ifstream supplier_csv;
  supplier_csv.open(supplier_data_dir, std::ios::in);
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
  if (supplier_csv.is_open())
  {
    std::string str;
    while (getline(supplier_csv, str))
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
    auto supplier = arrow::Table::Make(supplier_schema, {s_suppkey_array, s_name_array, s_address_array, s_nationkey_array,
                                                         s_phone_array, s_acctbal_array, s_comment_array});
    ARROW_ASSIGN_OR_RAISE(auto supplier_output_file,
                          arrow::io::FileOutputStream::Open("./dbgen/supplier.arrow"));
    ARROW_ASSIGN_OR_RAISE(auto supplier_batch_writer,
                          arrow::ipc::MakeFileWriter(supplier_output_file, supplier->schema()));
    ARROW_RETURN_NOT_OK(supplier_batch_writer->WriteTable(*supplier));
    ARROW_RETURN_NOT_OK(supplier_batch_writer->Close());
  }
  std::cout << "convert customer" << std::endl;
  std::ifstream customer_csv;
  customer_csv.open(customer_data_dir, std::ios::in);
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

  if (customer_csv.is_open())
  {
    std::string str;
    while (getline(customer_csv, str))
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
    auto customer = arrow::Table::Make(customer_schema, {c_custkey_array, c_name_array, c_address_array, c_nationkey_array,
                                                         c_phone_array, c_acctbal_array, c_mktsegment_array, c_comment_array});
    ARROW_ASSIGN_OR_RAISE(auto customer_output_file,
                          arrow::io::FileOutputStream::Open("./dbgen/customer.arrow"));
    ARROW_ASSIGN_OR_RAISE(auto customer_batch_writer,
                          arrow::ipc::MakeFileWriter(customer_output_file, customer->schema()));
    ARROW_RETURN_NOT_OK(customer_batch_writer->WriteTable(*customer));
    ARROW_RETURN_NOT_OK(customer_batch_writer->Close());
  }
  std::cout << "convert nation" << std::endl;
  std::ifstream nation_csv;
  nation_csv.open(nation_data_dir, std::ios::in);
  int n_nationkey;
  std::string n_name;
  int n_regionkey;
  std::string n_comment;

  arrow::Int32Builder n_nationkey_builder(pool);
  arrow::StringBuilder n_name_builder(pool);
  arrow::Int32Builder n_regionkey_builder(pool);
  arrow::StringBuilder n_comment_builder(pool);

  if (nation_csv.is_open())
  {
    std::string str;
    while (getline(nation_csv, str))
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
    auto nation = arrow::Table::Make(nation_schema, {n_nationkey_array, n_name_array, n_regionkey_array, n_comment_array});
    ARROW_ASSIGN_OR_RAISE(auto nation_output_file,
                          arrow::io::FileOutputStream::Open("./dbgen/nation.arrow"));
    ARROW_ASSIGN_OR_RAISE(auto nation_batch_writer,
                          arrow::ipc::MakeFileWriter(nation_output_file, nation->schema()));
    ARROW_RETURN_NOT_OK(nation_batch_writer->WriteTable(*nation));
    ARROW_RETURN_NOT_OK(nation_batch_writer->Close());
  }

  std::cout << "convert region" << std::endl;
  std::ifstream region_csv;
  region_csv.open(region_data_dir, std::ios::in);
  int r_regionkey;
  std::string r_name;
  std::string r_comment;

  arrow::Int32Builder r_regionkey_builder(pool);
  arrow::StringBuilder r_name_builder(pool);
  arrow::StringBuilder r_comment_builder(pool);
  if (region_csv.is_open())
  {
    std::string str;
    while (getline(region_csv, str))
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
    auto region = arrow::Table::Make(region_schema, {r_regionkey_array, r_name_array, r_comment_array});
    ARROW_ASSIGN_OR_RAISE(auto region_output_file,
                          arrow::io::FileOutputStream::Open("./dbgen/region.arrow"));
    ARROW_ASSIGN_OR_RAISE(auto region_batch_writer,
                          arrow::ipc::MakeFileWriter(region_output_file, region->schema()));
    ARROW_RETURN_NOT_OK(region_batch_writer->WriteTable(*region));
    ARROW_RETURN_NOT_OK(region_batch_writer->Close());
  }
  return arrow::Status::OK();
}
arrow::Status Load_From_IPC_To_ARROW(std::shared_ptr<arrow::RecordBatch> &lineitem_t, std::shared_ptr<arrow::RecordBatch> &partsupp_t,
                                     std::shared_ptr<arrow::RecordBatch> &orders_t, std::shared_ptr<arrow::RecordBatch> &part_t,
                                     std::shared_ptr<arrow::RecordBatch> &supplier_t, std::shared_ptr<arrow::RecordBatch> &customer_t,
                                     std::shared_ptr<arrow::RecordBatch> &nation_t, std::shared_ptr<arrow::RecordBatch> &region_t)
{
  std::cout << "Load lineitem" << std::endl;
  std::shared_ptr<arrow::io::ReadableFile> lineitem_infile;
  ARROW_ASSIGN_OR_RAISE(lineitem_infile, arrow::io::ReadableFile::Open(
                                             lineitem_arrow_dir, arrow::default_memory_pool()));
  ARROW_ASSIGN_OR_RAISE(auto lineitem_ipc_reader, arrow::ipc::RecordBatchFileReader::Open(lineitem_infile));
  ARROW_ASSIGN_OR_RAISE(lineitem_t, lineitem_ipc_reader->ReadRecordBatch(0));

  std::cout << "Load partsupp" << std::endl;
  std::shared_ptr<arrow::io::ReadableFile> partsupp_infile;
  ARROW_ASSIGN_OR_RAISE(partsupp_infile, arrow::io::ReadableFile::Open(
                                             partsupp_arrow_dir, arrow::default_memory_pool()));
  ARROW_ASSIGN_OR_RAISE(auto partsupp_ipc_reader, arrow::ipc::RecordBatchFileReader::Open(partsupp_infile));
  ARROW_ASSIGN_OR_RAISE(partsupp_t, partsupp_ipc_reader->ReadRecordBatch(0));

  std::cout << "Load orders" << std::endl;
  std::shared_ptr<arrow::io::ReadableFile> orders_infile;
  ARROW_ASSIGN_OR_RAISE(orders_infile, arrow::io::ReadableFile::Open(
                                           orders_arrow_dir, arrow::default_memory_pool()));
  ARROW_ASSIGN_OR_RAISE(auto orders_ipc_reader, arrow::ipc::RecordBatchFileReader::Open(orders_infile));
  ARROW_ASSIGN_OR_RAISE(orders_t, orders_ipc_reader->ReadRecordBatch(0));

  std::cout << "Load part" << std::endl;
  std::shared_ptr<arrow::io::ReadableFile> part_infile;
  ARROW_ASSIGN_OR_RAISE(part_infile, arrow::io::ReadableFile::Open(
                                         part_arrow_dir, arrow::default_memory_pool()));
  ARROW_ASSIGN_OR_RAISE(auto part_ipc_reader, arrow::ipc::RecordBatchFileReader::Open(part_infile));
  ARROW_ASSIGN_OR_RAISE(part_t, part_ipc_reader->ReadRecordBatch(0));

  std::cout << "Load supplier" << std::endl;
  std::shared_ptr<arrow::io::ReadableFile> supplier_infile;
  ARROW_ASSIGN_OR_RAISE(supplier_infile, arrow::io::ReadableFile::Open(
                                             supplier_arrow_dir, arrow::default_memory_pool()));
  ARROW_ASSIGN_OR_RAISE(auto supplier_ipc_reader, arrow::ipc::RecordBatchFileReader::Open(supplier_infile));
  ARROW_ASSIGN_OR_RAISE(supplier_t, supplier_ipc_reader->ReadRecordBatch(0));

  std::cout << "Load customer" << std::endl;
  std::shared_ptr<arrow::io::ReadableFile> customer_infile;
  ARROW_ASSIGN_OR_RAISE(customer_infile, arrow::io::ReadableFile::Open(
                                             customer_arrow_dir, arrow::default_memory_pool()));
  ARROW_ASSIGN_OR_RAISE(auto customer_ipc_reader, arrow::ipc::RecordBatchFileReader::Open(customer_infile));
  ARROW_ASSIGN_OR_RAISE(customer_t, customer_ipc_reader->ReadRecordBatch(0));

  std::cout << "Load nation" << std::endl;
  std::shared_ptr<arrow::io::ReadableFile> nation_infile;
  ARROW_ASSIGN_OR_RAISE(nation_infile, arrow::io::ReadableFile::Open(
                                           nation_arrow_dir, arrow::default_memory_pool()));
  ARROW_ASSIGN_OR_RAISE(auto nation_ipc_reader, arrow::ipc::RecordBatchFileReader::Open(nation_infile));
  ARROW_ASSIGN_OR_RAISE(nation_t, nation_ipc_reader->ReadRecordBatch(0));

  std::cout << "Load region" << std::endl;
  std::shared_ptr<arrow::io::ReadableFile> region_infile;
  ARROW_ASSIGN_OR_RAISE(region_infile, arrow::io::ReadableFile::Open(
                                           region_arrow_dir, arrow::default_memory_pool()));
  ARROW_ASSIGN_OR_RAISE(auto region_ipc_reader, arrow::ipc::RecordBatchFileReader::Open(region_infile));

  ARROW_ASSIGN_OR_RAISE(region_t, region_ipc_reader->ReadRecordBatch(0));

  return arrow::Status::OK();
}
void *select_thread_less_value_AND(void *param)
{
  pth_st *argst = (pth_st *)param;

  for (int i = 0; i < argst->comline; i++)
  {
    int location = i + argst->startindex;
    if (*((int *)argst->sel_col1 + location) < *((int *)argst->sel_col2))
      argst->res_bmp[location] = *((int *)argst->sel_col1 + location);

    if (argst->select_flag != 0)
      if ((argst->res_bmp[location] != -1) && (argst->pre_bmp[location] == -1))
        argst->res_bmp[location] = -1;
  }
  return NULL;
}
void *select_thread_less_equal_value_AND(void *param)
{
  pth_st *argst = (pth_st *)param;

  for (int i = 0; i < argst->comline; i++)
  {
    int location = i + argst->startindex;
    if (*((int *)argst->sel_col1 + location) <= *((int *)argst->sel_col2))
      argst->res_bmp[location] = *((int *)argst->sel_col1 + location);

    if (argst->select_flag != 0)
      if ((argst->res_bmp[location] != -1) && (argst->pre_bmp[location] == -1))
        argst->res_bmp[location] = -1;
  }
  return NULL;
}
void *select_thread_less_value_OR(void *param)
{
  pth_st *argst = (pth_st *)param;

  for (int i = 0; i < argst->comline; i++)
  {
    int location = i + argst->startindex;
    if (*((int *)argst->sel_col1 + location) < *((int *)argst->sel_col2))
      argst->res_bmp[location] = *((int *)argst->sel_col1 + location);
    if (argst->select_flag != 0)
      if (argst->res_bmp[location] == -1 && argst->pre_bmp[location] == 0)
        argst->res_bmp[location] = *((int *)argst->sel_col1 + location);
  }
  return NULL;
}
void *select_thread_less_equal_value_OR(void *param)
{
  pth_st *argst = (pth_st *)param;

  for (int i = 0; i < argst->comline; i++)
  {
    int location = i + argst->startindex;
    if (*((int *)argst->sel_col1 + location) <= *((int *)argst->sel_col2))
      argst->res_bmp[location] = *((int *)argst->sel_col1 + location);
    if (argst->select_flag != 0)
      if (argst->res_bmp[location] == -1 && argst->pre_bmp[location] == 0)
        argst->res_bmp[location] = *((int *)argst->sel_col1 + location);
  }
  return NULL;
}
void *select_thread_more_equal_value_AND(void *param)
{
  pth_st *argst = (pth_st *)param;

  for (int i = 0; i < argst->comline; i++)
  {
    int location = i + argst->startindex;
    if (*((int *)argst->sel_col1 + location) >= *((int *)argst->sel_col2))
      argst->res_bmp[location] = *((int *)argst->sel_col1 + location);

    if (argst->select_flag != 0)
      if (argst->res_bmp[location] != -1 && argst->pre_bmp[location] == -1)
        argst->res_bmp[location] = -1;
  }
  return NULL;
}
void *select_thread_more_value_AND(void *param)
{
  pth_st *argst = (pth_st *)param;

  for (int i = 0; i < argst->comline; i++)
  {
    int location = i + argst->startindex;
    if (*((int *)argst->sel_col1 + location) > *((int *)argst->sel_col2))
      argst->res_bmp[location] = *((int *)argst->sel_col1 + location);

    if (argst->select_flag != 0)
      if (argst->res_bmp[location] != -1 && argst->pre_bmp[location] == -1)
        argst->res_bmp[location] = -1;
  }
  return NULL;
}
void *select_thread_more_equal_value_OR(void *param)
{
  pth_st *argst = (pth_st *)param;

  for (int i = 0; i < argst->comline; i++)
  {
    int location = i + argst->startindex;
    if (*((int *)argst->sel_col1 + location) >= *((int *)argst->sel_col2))
      argst->res_bmp[location] = *((int *)argst->sel_col1 + location);
    if (argst->select_flag != 0)
      if (argst->res_bmp[location] == -1 && argst->pre_bmp[location] == 0)
        argst->res_bmp[location] = *((int *)argst->sel_col1 + location);
  }
  return NULL;
}
void *select_thread_more_value_OR(void *param)
{
  pth_st *argst = (pth_st *)param;

  for (int i = 0; i < argst->comline; i++)
  {
    int location = i + argst->startindex;
    if (*((int *)argst->sel_col1 + location) > *((int *)argst->sel_col2))
      argst->res_bmp[location] = *((int *)argst->sel_col1 + location);
    if (argst->select_flag != 0)
      if (argst->res_bmp[location] == -1 && argst->pre_bmp[location] == 0)
        argst->res_bmp[location] = *((int *)argst->sel_col1 + location);
  }
  return NULL;
}
void *select_thread_equal_value_AND(void *param)
{
  pth_st *argst = (pth_st *)param;

  for (int i = 0; i < argst->comline; i++)
  {
    int location = i + argst->startindex;
    if (*((int *)argst->sel_col1 + location) == *((int *)argst->sel_col2))
      argst->res_bmp[location] = *((int *)argst->sel_col1 + location);
    if (argst->select_flag != 0)
      if (argst->res_bmp[location] != -1 && argst->pre_bmp[location] == -1)
        argst->res_bmp[location] = -1;
  }
  return NULL;
}
void *select_thread_non_equal_value_AND(void *param)
{
  pth_st *argst = (pth_st *)param;

  for (int i = 0; i < argst->comline; i++)
  {
    int location = i + argst->startindex;
    if (*((int *)argst->sel_col1 + location) != *((int *)argst->sel_col2))
      argst->res_bmp[location] = *((int *)argst->sel_col1 + location);
    if (argst->select_flag != 0)
      if (argst->res_bmp[location] != -1 && argst->pre_bmp[location] == -1)
        argst->res_bmp[location] = -1;
  }
  return NULL;
}
void *select_thread_equal_value_OR(void *param)
{
  pth_st *argst = (pth_st *)param;

  for (int i = 0; i < argst->comline; i++)
  {
    int location = i + argst->startindex;
    if (*((int *)argst->sel_col1 + location) == *((int *)argst->sel_col2))
      argst->res_bmp[location] = *((int *)argst->sel_col1 + location);
    if (argst->select_flag != 0)
      if (argst->res_bmp[location] == -1 && argst->pre_bmp[location] == 0)
        argst->res_bmp[location] = *((int *)argst->sel_col1 + location);
  }
  return NULL;
}
void *select_thread_non_equal_value_OR(void *param)
{
  pth_st *argst = (pth_st *)param;

  for (int i = 0; i < argst->comline; i++)
  {
    int location = i + argst->startindex;
    if (*((int *)argst->sel_col1 + location) != *((int *)argst->sel_col2))
      argst->res_bmp[location] = *((int *)argst->sel_col1 + location);
    if (argst->select_flag != 0)
      if (argst->res_bmp[location] == -1 && argst->pre_bmp[location] == 0)
        argst->res_bmp[location] = *((int *)argst->sel_col1 + location);
  }
  return NULL;
}
void *select_thread_equal_col_AND(void *param)
{
  pth_st *argst = (pth_st *)param;

  for (int i = 0; i < argst->comline; i++)
  {
    int location = i + argst->startindex;
    if (*((int *)argst->sel_col2 + *((int *)argst->sel_col1 + location)) != -1)
      argst->res_bmp[location] = *((int *)argst->sel_col1 + location);
    if (argst->select_flag != 0)
      if (argst->res_bmp[location] != -1 && argst->pre_bmp[location] == -1)
        argst->res_bmp[location] = -1;
  }
  return NULL;
}
void *select_thread_equal_col_storecol2_AND(void *param)
{
  pth_st *argst = (pth_st *)param;
  for (int i = 0; i < argst->comline; i++)
  {
    int location = i + argst->startindex;
    if (*((int *)argst->sel_col2 + *((int *)argst->sel_col1 + location)) != -1)
    {
      argst->res_bmp[location] = *((int *)argst->sel_col2 + *((int *)argst->sel_col1 + location));
    }


    if (argst->select_flag != 0)
      if (argst->res_bmp[location] != -1 && argst->pre_bmp[location] == -1)
        {
          argst->res_bmp[location] = -1;
        }
  }
  return NULL;
}
void *select_thread_equal_col_OR(void *param)
{
  pth_st *argst = (pth_st *)param;

  for (int i = 0; i < argst->comline; i++)
  {
    int location = i + argst->startindex;
    if (*((int *)argst->sel_col2 + *((int *)argst->sel_col1 + location)) != -1)
      argst->res_bmp[location] = *((int *)argst->sel_col1 + location);
    if (argst->select_flag != 0)
      if (argst->res_bmp[location] == -1 && argst->pre_bmp[location] == 0)
        argst->res_bmp[location] = *((int *)argst->sel_col1 + location);
  }
  return NULL;
}

void *select_thread_string_equal_value_AND(void *param)
{
  pth_st *argst = (pth_st *)param;
  for (int i = 0; i < argst->comline; i++)
  {
    int location = i + argst->startindex;
    if ((Load_string_value_from_table(argst->tablename, *((std::string *)argst->sel_col1), location).compare((*((std::string *)argst->sel_col2)).c_str())) == 0)
      argst->res_bmp[location] = 0;
    if (argst->select_flag != 0)
      if (argst->res_bmp[location] != -1 && argst->pre_bmp[location] == -1)
        argst->res_bmp[location] = -1;
  }
  return NULL;
}
void *select_thread_string_equal_value_OR(void *param)
{
  pth_st *argst = (pth_st *)param;
  for (int i = 0; i < argst->comline; i++)
  {
    int location = i + argst->startindex;
    if ((Load_string_value_from_table(argst->tablename, *((std::string *)argst->sel_col1), location).compare((*((std::string *)argst->sel_col2)).c_str())) == 0)
      argst->res_bmp[location] = 0;
    if (argst->select_flag != 0)
      if (argst->res_bmp[location] == -1 && argst->pre_bmp[location] == 0)
        argst->res_bmp[location] = 0;
  }
  return NULL;
}
void Select_Option(Select_Node &select_node, int nthreads)
{

  for (int i = 0; i < select_node.select_num; i++)
  {
    pthread_t tid[nthreads];
    pthread_attr_t att;
    cpu_set_t set;
    pthread_barrier_t barrier;
    int rv;
    pthread_attr_init(&att);
    pthread_attr_setstacksize(&att, STACKSIZE);

    pth_st argst[nthreads];
    int nummea = select_node.col_length;
    int numper = nummea / nthreads;

    for (int j = 0; j < nthreads; j++)
    {
      int cpu_idx = j;
      CPU_ZERO(&set);
      CPU_SET(cpu_idx, &set);
      pthread_attr_setaffinity_np(&att, sizeof(cpu_set_t), &set);
      argst[j].sel_col1 = select_node.select_data[i].sel_col1;
      argst[j].sel_col2 = select_node.select_data[i].sel_col2;
      argst[j].select_flag = select_node.select_data[i].select_flag;
      argst[j].pre_bmp = select_node.select_data[i].pre_bmp;
      argst[j].res_bmp = select_node.select_data[i].res_bmp;
      argst[j].tablename = select_node.tablename;
      argst[j].comline = (j == nthreads - 1) ? nummea : numper;
      argst[j].startindex = j * numper;
      nummea -= numper;
      rv = pthread_create(&tid[j], &att, select_node.select_data[i].select, (void *)&argst[j]);
      if (rv)
      {
        printf("ERROR; return code from pthread_create() is %d\n", rv);
        exit(-1);
      }
    }
    for (int j = 0; j < nthreads; j++)
    {
      pthread_join(tid[j], NULL);
    }
  }
  return;
}
int group_thread_string(Group_Data_gt group_data)
{

  std::string *gro_col_string = (std::string *)group_data.gro_col;
  std::string *com_dic_t_string = (std::string *)group_data.com_dic_t;
  if (Load_string_value_from_table(group_data.tablename, *((std::string *)group_data.gro_col), group_data.location).compare((*((std::string *)group_data.com_dic_t + group_data.dic_location)).c_str()) == 0)
    return 1;
  else
    return 0;
}
int group_thread_int(Group_Data_gt group_data)
{
  if ((*((int *)group_data.gro_col + group_data.location)) != (*((int *)group_data.com_dic_t + group_data.dic_location)))
    return 1;
  else
    return 0;
}
int group_thread_double(Group_Data_gt group_data)
{
  if ((*((double *)group_data.gro_col + group_data.location)) != (*((double *)group_data.com_dic_t + group_data.dic_location)))
    return 1;
  else
    return 0;
}
void group_assignment_string(Group_Data_gt group_data)
{
  std::string *gro_col_string = (std::string *)group_data.gro_col;
  std::string *com_dic_t_string = (std::string *)group_data.com_dic_t;
  com_dic_t_string[group_data.dic_location] = Load_string_value_from_table(group_data.tablename, *((std::string *)group_data.gro_col), group_data.location);
}
void group_assignment_int(Group_Data_gt group_data)
{
  int *gro_col_int = (int *)group_data.gro_col;
  int *com_dic_t_int = (int *)group_data.com_dic_t;
  com_dic_t_int[group_data.dic_location] = (*((int *)group_data.gro_col + group_data.location));
}
void group_assignment_double(Group_Data_gt group_data)
{
  double *gro_col_double = (double *)group_data.gro_col;
  double *com_dic_t_double = (double *)group_data.com_dic_t;
  com_dic_t_double[group_data.dic_location] = (*((double *)group_data.gro_col + group_data.location));
}
void *group_thread(void *param)
{
  int i, j, k;
  pth_gt *argst = (pth_gt *)param;
  Group_Data_gt group_data;
  for (i = 0; i < argst->comline; i++)
  {

    int location = i + argst->startindex;
    int dic_location = 0;
    int pre_location = 0;
    if (argst->res_vec[location] != -1)
    {
      pthread_mutex_lock(argst->mut);
      for (j = 0; j < (*(argst->group_count)); j++)
      {
        int flag = 0;
        for (k = 0; k < argst->colnum; k++)
        {
          group_data.gro_col = (std::string *)argst->gro_col[k];
          group_data.com_dic_t = (std::string *)argst->com_dic_t[k];
          group_data.location = location;
          group_data.dic_location = j;
          group_data.tablename = argst->tablename;
          flag = argst->group[k](group_data);
          if (!flag)
            break;
        }
        if ((k == argst->colnum - 1) && flag)
          break;
      }
      if (j == (*(argst->group_count)))
      {
        for (k = 0; k < argst->colnum; k++)
        {

          group_data.gro_col = (std::string *)argst->gro_col[k];
          group_data.com_dic_t = (std::string *)argst->com_dic_t[k];
          group_data.location = location;
          group_data.dic_location = j;
          group_data.tablename = argst->tablename;
          argst->res_vec[location] = (*(argst->group_count));
          argst->group_assignment[k](group_data);
        }

        (*(argst->group_count))++;
      }
      else
        argst->res_vec[location] = j;
      pthread_mutex_unlock(argst->mut);
    }
  }
  return NULL;
}
void Group_Option(Group_Node &group_node, int nthreads)
{

  for (int j = 0; j < group_node.tablenum; j++)
  {
    pthread_t tid[nthreads];
    pthread_attr_t att;
    cpu_set_t set;
    pthread_barrier_t barrier;
    pthread_mutex_t mut;
    pthread_mutex_init(&mut, NULL);
    int rv, i;
    int nummea = group_node.group_data[j].table_size;
    int numper = nummea / nthreads;
    int r = pthread_barrier_init(&barrier, NULL, nthreads);
    pthread_attr_init(&att);
    pthread_attr_setstacksize(&att, STACKSIZE);
    pth_gt argst[nthreads];
    for (i = 0; i < nthreads; i++)
    {
      int cpu_idx = i;
      CPU_ZERO(&set);
      CPU_SET(cpu_idx, &set);
      pthread_attr_setaffinity_np(&att, sizeof(cpu_set_t), &set);
      argst[i].colnum = group_node.group_data[j].colnum;
      argst[i].group_count = group_node.group_data[j].group_count;
      argst[i].startindex = i * numper;
      argst[i].comline = (i == nthreads - 1) ? nummea : numper;
      argst[i].mut = &mut;
      argst[i].barrier = &barrier;
      argst[i].gro_col = group_node.group_data[j].gro_col;
      argst[i].com_dic_t = group_node.group_data[j].com_dic_t;
      argst[i].res_vec = group_node.group_data[j].res_vec;
      argst[i].colname = group_node.group_data[j].colname;
      argst[i].tablename = group_node.group_data[j].tablename;
      argst[i].group = group_node.group_data[j].group;
      argst[i].group_assignment = group_node.group_data[j].group_assignment;
      nummea -= numper;
      rv = pthread_create(&tid[i], &att, group_thread, (void *)&argst[i]);
      if (rv)
      {
        printf("ERROR; return code from pthread_create() is %d\n", rv);
        exit(-1);
      }
    }
    for (int i = 0; i < nthreads; i++)
    {
      pthread_join(tid[i], NULL);
    }
  }
  return;
}
void *join_cwm_dv_cross_groupby_thread(void *param)
{
  pth_jt *arg = (pth_jt *)param;
  if (!arg->join_id)
  {
    *(arg->index) = 0;
    for (int i = 0; i < arg->num_tuples; i++)
    {

      int location = arg->start + i;
      int idx_flag = *((int *)arg->pre_vec + *((int *)arg->join_col + location));
      int idx_flag_cross = *((int *)arg->pre_vec_cross + *((int *)arg->join_col_cross + location));
      if ((idx_flag != -1) && (idx_flag == idx_flag_cross))
      {
        arg->OID[*(arg->index) + arg->start] = location;
        arg->groupID[*(arg->index) + arg->start] += idx_flag * arg->factor;
        (*(arg->index))++;
      }
    }
  }
  else
  {
    int comlength = *(arg->index);
    *(arg->index) = 0;
    for (int i = 0; i < comlength; i++)
    {
      int location = arg->OID[i + arg->start];
      int idx_flag = *((int *)arg->pre_vec + *((int *)arg->join_col + location));
      int idx_flag_cross = *((int *)arg->pre_vec_cross + *((int *)arg->join_col_cross + location));
      if ((idx_flag != -1) && (idx_flag == idx_flag_cross))
      {
        arg->OID[*(arg->index) + arg->start] = location;
        arg->groupID[*(arg->index) + arg->start] += arg->groupID[i + arg->start] + idx_flag * arg->factor;
        (*(arg->index))++;
      }
    }
  }
  return NULL;
}

void Join_Option(Join_Node &join_node, int nthreads)
{
  int i, j;
  for (i = 0; i < join_node.join_col_num; i++)
  {
    pth_jt argst[nthreads];
    int64_t numS, numSthr;
    int rv;
    cpu_set_t set;
    pthread_t tid[nthreads];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, STACKSIZE);
    pthread_barrier_t barrier;
    numS = join_node.table_size;
    numSthr = numS / nthreads;
    for (j = 0; j < nthreads; j++)
    {
      int cpu_idx = j;
      CPU_ZERO(&set);
      CPU_SET(cpu_idx, &set);
      pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);
      argst[j].num_tuples = (j == (nthreads - 1)) ? numS : numSthr;
      argst[j].start = numSthr * j;
      argst[j].join_id = i;
      numS -= numSthr;
      argst[j].join_col = join_node.join_col[i];
      argst[j].pre_vec = join_node.pre_vec[i];
      argst[j].join_col_cross = join_node.join_col_cross[i];
      argst[j].pre_vec_cross = join_node.pre_vec_cross[i];
      argst[j].OID = join_node.OID;
      argst[j].groupID = join_node.groupID;
      argst[j].factor = join_node.factor[i];
      argst[j].index = &join_node.index[j];
      rv = pthread_create(&tid[j], &attr, join_node.join[i], (void *)&argst[j]);
      if (rv)
      {
        printf("ERROR; return code from pthread_create() is %d\n", rv);
        exit(-1);
      }
    }
    for (int j = 0; j < nthreads; j++)
    {
      pthread_join(tid[j], NULL);
    }
  }
  return;
}
void *agg_value_reduce_col_thread(void *param)
{
  pth_at *arg = (pth_at *)param;
  for (int i = 0; i < *(arg->index); i++)
  {
    *((double *)arg->pre_res + arg->OID[i + arg->start]) = *((double *)arg->agg_col1) - *((double *)arg->agg_col2 + arg->OID[i + arg->start]);
  }
  return NULL;
}
void *agg_col_mul_col_last_thread(void *param)
{
  pth_at *arg = (pth_at *)param;
  for (int i = 0; i < *(arg->index); i++)
  {
    int16_t tmp = arg->groupID[i + arg->start];
    *((double *)arg->res_vec + tmp) += *((double *)arg->agg_col1 + arg->OID[i + arg->start]) * *((double *)arg->agg_col2 + arg->OID[i + arg->start]);
  }
  return NULL;
}
void Agg_Option(Agg_Node &agg_node, int nthreads)
{
  int i, j;
  for (i = 0; i < agg_node.agg_num; i++)
  {
    int64_t numS, numSthr;
    int rv;
    cpu_set_t set;
    pth_at argst[nthreads];
    pthread_t tid[nthreads];
    pthread_attr_t attr;
    pthread_barrier_t barrier;
    numS = agg_node.table_size;
    numSthr = numS / nthreads;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, STACKSIZE);
    rv = pthread_barrier_init(&barrier, NULL, nthreads);
    for (j = 0; j < nthreads; j++)
    {
      int cpu_idx = j;
      CPU_ZERO(&set);
      CPU_SET(cpu_idx, &set);
      pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);
      argst[j].num_tuples = (j == (nthreads - 1)) ? numS : numSthr;
      argst[j].start = numSthr * j;
      numS -= numSthr;
      argst[j].agg_col1 = agg_node.agg_col1[i];
      argst[j].agg_col2 = agg_node.agg_col2[i];
      argst[j].OID = agg_node.OID;
      argst[j].groupID = agg_node.groupID;
      argst[j].index = &agg_node.index[j];
      argst[j].pre_res = agg_node.pre_res[i];
      argst[j].res_vec = agg_node.res_vec[j];
      rv = pthread_create(&tid[j], &attr, agg_node.agg[i], (void *)&argst[j]);
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
  }

  for (i = 1; i < nthreads; i++)
    for (j = 0; j < agg_node.group_num; j++)
      (*((double *)agg_node.res_vec[0] + j)) += (*((double *)agg_node.res_vec[i] + j));
  for (i = 0; i < agg_node.group_num; i++)
    std::cout << (*((double *)agg_node.res_vec[0] + i)) << std::endl;
  return;
}
void write_string(Project_Data &project_data, int location, std::ofstream &resfile)
{
  resfile << *((std::string *)project_data.res_array + location);
  return;
}
void write_double(Project_Data &project_data, int location, std::ofstream &resfile)
{
  resfile << *((double *)project_data.res_array + location);
  return;
}
void write_int(Project_Data &project_data, int location, std::ofstream &resfile)
{
  resfile << *((int *)project_data.res_array + location);
  return;
}
void project_groupby_string(Project_Data &project_data, int group_total_num)
{
  for (int k = 0; k < group_total_num; k += project_data.group_count * project_data.factor)
    for (int i = 0; i < project_data.group_count; i++)
      for (int j = i * project_data.factor; j < (i + 1) * project_data.factor; j++)
        *((std::string *)project_data.res_array + k + j) = *((std::string *)project_data.pro_sel + i);
  return;
}
void project_groupby_double(Project_Data &project_data, int group_total_num)
{
  for (int k = 0; k < group_total_num; k += project_data.group_count * project_data.factor)
    for (int i = 0; i < project_data.group_count; i++)
      for (int j = i * project_data.factor; j < (i + 1) * project_data.factor; j++)
        *((double *)project_data.res_array + k + j) = (*((double *)project_data.pro_sel + i));
  return;
}
void project_groupby_int(Project_Data &project_data, int group_total_num)
{
  for (int k = 0; k < group_total_num; k += project_data.group_count * project_data.factor)
    for (int i = 0; i < project_data.group_count; i++)
      for (int j = i * project_data.factor; j < (i + 1) * project_data.factor; j++)
        *((int *)project_data.res_array + k + j) = (*((int *)project_data.pro_sel + i));
  return;
}

void project_int(Project_Data &project_data, int group_total_num)
{

  for (int i = 0; i < project_data.group_count; i++)
  {
    int location = project_data.OID[i];
    for (int j = 0; j < project_data.fk_num; j++)
      location = project_data.FK_sel[j][location];
    *((int *)project_data.res_array + i) = (*((int *)project_data.pro_sel + location));
  }
  return;
}
void project_string(Project_Data &project_data, int group_total_num)
{

  for (int i = 0; i < project_data.group_count; i++)
  {

    int location = project_data.OID[i];
    for (int j = 0; j < project_data.fk_num; j++)
      location = project_data.FK_sel[j][location];
    *((std::string *)project_data.res_array + i) = (*((std::string *)project_data.pro_sel + location));
  }
  return;
}
void project_double(Project_Data &project_data, int group_total_num)
{

  for (int i = 0; i < project_data.group_count; i++)
  {
    int location = project_data.OID[i];
    for (int j = 0; j < project_data.fk_num; j++)
      location = project_data.FK_sel[j][location];
    *((double *)project_data.res_array + i) = (*((double *)project_data.pro_sel + location));
  }
  return;
}
void Project_Option(Project_Node &project_node)
{
  std::ofstream resfile;

  for (int i = 0; i < project_node.colnum; i++)
    project_node.project[i](project_node.project_data[i], project_node.group_total_num);
  resfile.open("./log/TPCH_Q5_operator/TPCH_Q5_operator.txt", std::ios::out | std::ios::trunc);
  for (int i = 0; i < project_node.colnum; i++)
  {
    resfile << project_node.project_data[i].name_array;
    if (i != project_node.colnum - 1)
      resfile << "\t";
  }
  resfile << std::endl;
  for (int i = 0; i < project_node.group_total_num; i++)
  {
    for (int j = 0; j < project_node.colnum; j++)
    {
      project_node.write[j](project_node.project_data[j], i, resfile);
      if (j != project_node.colnum - 1)
        resfile << "\t";
    }
    resfile << std::endl;
  }
  return;
}
int main()
{
  if (access(orders_arrow_dir, F_OK) != 0)
  {
    /*Load TPCH Table data*/
    auto arrow_flag =  Load_From_CSV_To_IPC();
  }
  auto arrow_flag = Load_From_IPC_To_ARROW(lineitem_t, partsupp_t, orders_t, part_t, supplier_t, customer_t, nation_t, region_t);

  /*init the metadata of tablename*/
  metadata_tablename["lineitem"] = 0;
  metadata_tablename["partsupp"] = 1;
  metadata_tablename["orders"] = 2;
  metadata_tablename["part"] = 3;
  metadata_tablename["supplier"] = 4;
  metadata_tablename["customer"] = 5;
  metadata_tablename["nation"] = 6;
  metadata_tablename["region"] = 7;
  /*init the metadata of lineitem*/
  metadata_lineitem["l_pskey"] = 0;
  metadata_lineitem["l_orderkey_1"] = 1;
  metadata_lineitem["l_orderkey"] = 2;
  metadata_lineitem["l_partkey"] = 3;
  metadata_lineitem["l_suppkey"] = 4;
  metadata_lineitem["l_linenumber"] = 5;
  metadata_lineitem["l_quantity"] = 6;
  metadata_lineitem["l_extendedprice"] = 7;
  metadata_lineitem["l_discount"] = 8;
  metadata_lineitem["l_tax"] = 9;
  metadata_lineitem["l_returnflag"] = 10;
  metadata_lineitem["l_linestatus"] = 11;
  metadata_lineitem["l_shipdate"] = 12;
  metadata_lineitem["l_commitdate"] = 13;
  metadata_lineitem["l_receiptdate"] = 14;
  metadata_lineitem["l_shipinstruct"] = 15;
  metadata_lineitem["l_shipmode"] = 16;
  metadata_lineitem["l_comment"] = 17;
  /*init the metadata of partsupp*/
  metadata_partsupp["ps_key"] = 0;
  metadata_partsupp["ps_partkey"] = 1;
  metadata_partsupp["ps_suppkey"] = 2;
  metadata_partsupp["ps_availqty"] = 3;
  metadata_partsupp["ps_supplycost"] = 4;
  metadata_partsupp["ps_comment"] = 5;
  /*init the metadata of orders*/
  metadata_orders["o_orderkey_1"] = 0;
  metadata_orders["o_orderkey"] = 1;
  metadata_orders["o_custkey"] = 2;
  metadata_orders["o_orderstatus"] = 3;
  metadata_orders["o_totalprice"] = 4;
  metadata_orders["o_orderdate"] = 5;
  metadata_orders["o_orderpriority"] = 6;
  metadata_orders["o_clerk"] = 7;
  metadata_orders["o_shippriority"] = 8;
  metadata_orders["o_comment"] = 9;
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
  /*generate the select plan node*/
  Select_Node select_node[5];
  std::string R_NAME = "ASIA";
  int O_ORDERDATE_less = 19950101;
  int O_ORDERDATE_more = 19960101;
  /*select_node of region*/
  /*Load column r_name*/
  auto r_name_tmp = std::static_pointer_cast<arrow::StringArray>(region_t->column(metadata_region["r_name"]));
  column_store.r_name = r_name_tmp;
  std::string region_col_tmp = "r_name";
  select_node[0].select_num = 1;
  select_node[0].tablename = "region";
  select_node[0].select_data[0].sel_col1 = &region_col_tmp;
  select_node[0].select_data[0].sel_col2 = &R_NAME;
  select_node[0].select_data[0].select_flag = 0;
  select_node[0].select_data[0].select = select_thread_string_equal_value_AND;
  select_node[0].select_data[0].pre_bmp = NULL;
  select_node[0].select_data[0].res_bmp = new int[region_t->num_rows()];
  memset(select_node[0].select_data[0].res_bmp, 0xff, sizeof(int) * region_t->num_rows());
  select_node[0].col_length = region_t->num_rows();
  /*select_node on nation*/
  /*Load column n_regionkey*/
  auto n_regionkey_tmp = std::static_pointer_cast<arrow::Int32Array>(nation_t->column(metadata_nation["n_regionkey"]));
  column_store.n_regionkey = n_regionkey_tmp->raw_values();
  select_node[1].select_num = 1;
  select_node[1].tablename = "nation";
  select_node[1].select_data[0].sel_col1 = Load_int_column_from_table("nation", "n_regionkey");
  select_node[1].select_data[0].sel_col2 = select_node[0].select_data[0].res_bmp;
  select_node[1].select_data[0].select_flag = 0;
  select_node[1].select_data[0].select = select_thread_equal_col_AND;
  select_node[1].select_data[0].pre_bmp = NULL;
  select_node[1].select_data[0].res_bmp = new int[nation_t->num_rows()];
  memset(select_node[1].select_data[0].res_bmp, 0xff, sizeof(int) * nation_t->num_rows());
  select_node[1].col_length = nation_t->num_rows();
  /*select_node of supplier*/
  /*Load column s_nationkey*/
  auto s_nationkey_tmp = std::static_pointer_cast<arrow::Int32Array>(supplier_t->column(metadata_supplier["s_nationkey"]));
  column_store.s_nationkey = s_nationkey_tmp->raw_values();

  select_node[2].select_num = 1;
  select_node[2].tablename = "supplier";
  select_node[2].select_data[0].sel_col1 = Load_int_column_from_table("supplier", "s_nationkey");
  select_node[2].select_data[0].sel_col2 = select_node[1].select_data[0].res_bmp;
  select_node[2].select_data[0].select_flag = 0;
  select_node[2].select_data[0].select = select_thread_equal_col_storecol2_AND;
  select_node[2].select_data[0].pre_bmp = NULL;
  select_node[2].select_data[0].res_bmp = new int[supplier_t->num_rows()];
  memset(select_node[2].select_data[0].res_bmp, 0xff, sizeof(int) * supplier_t->num_rows());
  select_node[2].col_length = supplier_t->num_rows();
  /*select_node of customer*/
  /*Load column c_nationkey*/
  auto c_nationkey_tmp = std::static_pointer_cast<arrow::Int32Array>(customer_t->column(metadata_customer["c_nationkey"]));
  column_store.c_nationkey = c_nationkey_tmp->raw_values();

  select_node[3].select_num = 1;
  select_node[3].tablename = "customer";
  select_node[3].select_data[0].sel_col1 = Load_int_column_from_table("customer", "c_nationkey");
  select_node[3].select_data[0].sel_col2 = select_node[1].select_data[0].res_bmp;
  select_node[3].select_data[0].select_flag = 0;
  select_node[3].select_data[0].select = select_thread_equal_col_storecol2_AND;
  select_node[3].select_data[0].pre_bmp = NULL;
  select_node[3].select_data[0].res_bmp = new int[customer_t->num_rows()];
  memset(select_node[3].select_data[0].res_bmp, 0xff, sizeof(int) * customer_t->num_rows());
  select_node[3].col_length = customer_t->num_rows();
  /*select_node of orders*/
  /*Load column o_orderdate*/
  auto o_orderdate_tmp = std::static_pointer_cast<arrow::Int32Array>(orders_t->column(metadata_orders["o_orderdate"]));
  column_store.o_orderdate = o_orderdate_tmp->raw_values();

  auto o_custkey_tmp = std::static_pointer_cast<arrow::Int32Array>(orders_t->column(metadata_orders["o_custkey"]));
  column_store.o_custkey = o_custkey_tmp->raw_values();
  select_node[4].select_num = 3;
  select_node[4].tablename = "orders";
  select_node[4].select_data[0].sel_col1 = Load_int_column_from_table("orders", "o_orderdate");
  select_node[4].select_data[0].sel_col2 = &O_ORDERDATE_less;
  select_node[4].select_data[0].select_flag = 0;
  select_node[4].select_data[0].select = select_thread_more_equal_value_AND;
  select_node[4].select_data[0].pre_bmp = NULL;
  select_node[4].select_data[0].res_bmp = new int[orders_t->num_rows()];
  memset(select_node[4].select_data[0].res_bmp, 0xff, sizeof(int) * orders_t->num_rows());
  select_node[4].col_length = orders_t->num_rows();
  select_node[4].select_data[1].sel_col1 = Load_int_column_from_table("orders", "o_orderdate");
  select_node[4].select_data[1].sel_col2 = &O_ORDERDATE_more;
  select_node[4].select_data[1].select_flag = 1;
  select_node[4].select_data[1].select = select_thread_less_value_AND;
  select_node[4].select_data[1].pre_bmp = select_node[4].select_data[0].res_bmp;
  select_node[4].select_data[1].res_bmp = new int[orders_t->num_rows()];
  memset(select_node[4].select_data[1].res_bmp, 0xff, sizeof(int) * orders_t->num_rows());

  select_node[4].select_data[2].sel_col1 = Load_int_column_from_table("orders", "o_custkey");
  select_node[4].select_data[2].sel_col2 = select_node[3].select_data[0].res_bmp;
  select_node[4].select_data[2].select_flag = 1;
  select_node[4].select_data[2].select = select_thread_equal_col_storecol2_AND;
  select_node[4].select_data[2].pre_bmp = select_node[4].select_data[1].res_bmp;
  select_node[4].select_data[2].res_bmp = new int[orders_t->num_rows()];
  memset(select_node[4].select_data[2].res_bmp, 0xff, sizeof(int) * orders_t->num_rows());
  /*Group option*/
  Group_Node group_node;
  std::string N_NAME = "n_name";
  /*Group_node of nation*/
  auto n_name_tmp = std::static_pointer_cast<arrow::StringArray>(nation_t->column(metadata_nation["n_name"]));
  column_store.n_name = n_name_tmp;
  std::string com_dic_t_string[10000];
  group_node.tablenum = 1;
  group_node.group_total_num = new int[1];
  *(group_node.group_total_num) = 1;

  group_node.group_data[0].colnum = 1;
  group_node.group_data[0].group_count = new int[1];
  *(group_node.group_data[0].group_count) = 0;
  group_node.group_data[0].gro_col[0] = &N_NAME;
  group_node.group_data[0].com_dic_t[0] = (std::string *)com_dic_t_string;
  group_node.group_data[0].res_vec = select_node[1].select_data[0].res_bmp;
  group_node.group_data[0].colname[0] = N_NAME;
  group_node.group_data[0].tablename = "nation";
  group_node.group_data[0].group[0] = group_thread_string;
  group_node.group_data[0].group_assignment[0] = group_assignment_string;
  group_node.group_data[0].table_size = nation_t->num_rows();

  Join_Node join_node;
  /*Join on lineitem and orders*/
  auto l_orderkey_1_tmp = std::static_pointer_cast<arrow::Int32Array>(lineitem_t->column(metadata_lineitem["l_orderkey_1"]));
  column_store.l_orderkey_1 = l_orderkey_1_tmp->raw_values();
  auto l_suppkey_tmp = std::static_pointer_cast<arrow::Int32Array>(lineitem_t->column(metadata_lineitem["l_suppkey"]));
  column_store.l_suppkey = l_suppkey_tmp->raw_values();
  join_node.join_col[0] = Load_int_column_from_table("lineitem", "l_orderkey_1");
  join_node.pre_vec[0] = select_node[4].select_data[2].res_bmp;
  join_node.join_col_cross[0] = Load_int_column_from_table("lineitem", "l_suppkey");
  join_node.pre_vec_cross[0] = select_node[2].select_data[0].res_bmp;
  join_node.join[0] = join_cwm_dv_cross_groupby_thread;
  join_node.OID = new int32_t[lineitem_t->num_rows()];
  join_node.groupID = new int16_t[lineitem_t->num_rows()];
  join_node.table_size = lineitem_t->num_rows();
  join_node.factor[0] = 1;
  join_node.join_col_num = 1;
  memset(join_node.OID, 0, sizeof(int32_t) * lineitem_t->num_rows());
  memset(join_node.groupID, 0, sizeof(int16_t) * lineitem_t->num_rows());

  /*select option on region*/

  Select_Option(select_node[0], 16);

  /*select option on nation*/
  Select_Option(select_node[1], 16);

  /*group option on nation*/
  Group_Option(group_node, 16);

  /*select option on supplier*/
  Select_Option(select_node[2], 16);

  /*select option on customer*/
  Select_Option(select_node[3], 16);

  /*select option on orders*/
  Select_Option(select_node[4], 16);
  for (int i = 0; i < group_node.tablenum; i++)
  {
     *(group_node.group_total_num) *= *(group_node.group_data[i].group_count);
     for (int j = 1; j < group_node.tablenum; j++)
        join_node.factor[i] *=  (*(group_node.group_data[j].group_count));
  }
  /*join option*/
  Join_Option(join_node, 16);
  auto l_discount_tmp = std::static_pointer_cast<arrow::DoubleArray>(lineitem_t->column(metadata_lineitem["l_discount"]));
  column_store.l_discount = l_discount_tmp->raw_values();
  auto l_extendedprice_tmp = std::static_pointer_cast<arrow::DoubleArray>(lineitem_t->column(metadata_lineitem["l_extendedprice"]));
  column_store.l_extendedprice = l_extendedprice_tmp->raw_values();
  double constant = 1.0;
  Agg_Node agg_node;
  agg_node.agg_num = 2;
  agg_node.table_size = lineitem_t->num_rows();
  agg_node.agg_col1[0] = &constant;
  agg_node.agg_col2[0] = Load_double_column_from_table("lineitem", "l_discount");
  agg_node.OID = join_node.OID;
  agg_node.groupID = join_node.groupID;
  agg_node.index = join_node.index;
  agg_node.group_num = *(group_node.group_total_num);
  for (int i = 0; i < 16; i++)
  {
    agg_node.res_vec[i] = new double[(*(group_node.group_total_num))];
    memset(agg_node.res_vec[i], 0.0, sizeof(double) * (*(group_node.group_total_num)));
  }
  int sum = 0;
  for (int i = 0; i < 16; i++)
   sum+=agg_node.index[i];
  agg_node.pre_res[0] = new double[lineitem_t->num_rows()];
  agg_node.agg[0] = agg_value_reduce_col_thread;

  agg_node.agg_col1[1] = Load_double_column_from_table("lineitem", "l_extendedprice");
  agg_node.agg_col2[1] = agg_node.pre_res[0];
  agg_node.pre_res[1] = new double[lineitem_t->num_rows()];
  agg_node.agg[1] = agg_col_mul_col_last_thread;
  Agg_Option(agg_node, 16);
  /*Project Option*/
  Project_Node project_node;
  project_node.group_total_num = (*(group_node.group_total_num));
  project_node.project_data[0].res_array = new std::string[(*(group_node.group_total_num))];
  project_node.project_data[0].name_array = group_node.group_data[0].colname[0];
  project_node.project[0] = project_groupby_string;
  project_node.write[0] = write_string;
  project_node.colnum = 2;
  project_node.project_data[0].pro_sel = group_node.group_data[0].com_dic_t[0];
  project_node.project_data[0].group_count = (*(group_node.group_data[0].group_count));
  project_node.project_data[0].factor = join_node.factor[0];

  project_node.project_data[1].res_array = new double[(*(group_node.group_total_num))];
  project_node.project_data[1].name_array = "revenue";
  project_node.project[1] = project_groupby_double;
  project_node.write[1] = write_double;
  project_node.project_data[1].pro_sel = agg_node.res_vec[0];
  project_node.project_data[1].group_count = (*(group_node.group_total_num));
  project_node.project_data[1].factor = 1;
  int nums = lineitem_t->num_rows()/16;
  for (int i = 1; i < 16; i++)
  {
    memcpy(join_node.OID + join_node.index[0], join_node.OID + i * nums, join_node.index[i] * sizeof(int32_t));
    join_node.index[0] += join_node.index[i];
  }
  Project_Option(project_node);

  return 0;
}