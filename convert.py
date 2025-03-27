import pandas as pd
if __name__ == '__main__':
    data_dir = "./dbgen/"
    # process PARTSUPP
    partsupp_df = pd.read_table(data_dir + 'partsupp.tbl', header=None, names=['ps_partkey', 'ps_suppkey', 'ps_availqty', 'ps_supplycost', 'ps_comment'], sep='|', index_col=False, encoding='utf-8')
    partsupp_df['ps_key'] = range(0,len(partsupp_df))
    partsupp_df = partsupp_df[['ps_key','ps_partkey','ps_suppkey','ps_availqty', 'ps_supplycost', 'ps_comment']]
    partsupp_df.to_csv(data_dir + "partsupp.csv", header=None, index=False, sep='|')
    lineitem_df = pd.read_table(data_dir + 'lineitem.tbl', header=None, names=['o_orderkey', 'ps_partkey', 'ps_suppkey', 'l_linenumber', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus', 'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_shipinstruct', 'l_shipmode', 'l_comment'], sep='|', index_col=False, encoding='utf-8')
    lineitem_df = lineitem_df.merge(partsupp_df[['ps_key','ps_partkey','ps_suppkey']], on = ['ps_partkey', 'ps_suppkey'], how='left', sort=False)
    lineitem_df = lineitem_df[['ps_key', 'o_orderkey', 'ps_partkey', 'ps_suppkey', 'l_linenumber', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus', 'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_shipinstruct', 'l_shipmode', 'l_comment']]
    orders_df = pd.read_table(data_dir + 'orders.tbl', header=None, names=['o_orderkey', 'o_custkey', 'o_orderstatus', 'o_totalprice', 'o_orderdate', 'o_orderpriority', 'o_clerk', 'o_shippriority', 'o_comment'], sep='|', index_col=False, encoding='utf-8')
    orders_df['o_orderkey_1'] = range(0,len(orders_df))
    lineitem_df = lineitem_df.merge(orders_df[['o_orderkey_1','o_orderkey']], on = ['o_orderkey'], how='left', sort=False)
    lineitem_df = lineitem_df[['ps_key', 'o_orderkey_1', 'o_orderkey', 'ps_partkey', 'ps_suppkey', 'l_linenumber', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus', 'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_shipinstruct', 'l_shipmode', 'l_comment']]
    orders_df = orders_df[['o_orderkey_1', 'o_orderkey','o_custkey', 'o_orderstatus', 'o_totalprice', 'o_orderdate', 'o_orderpriority', 'o_clerk', 'o_shippriority', 'o_comment']]
    lineitem_df['l_shipdate'] = lineitem_df['l_shipdate'].str.replace('-','')
    lineitem_df['l_commitdate'] = lineitem_df['l_commitdate'].str.replace('-','')
    lineitem_df['l_receiptdate'] = lineitem_df['l_receiptdate'].str.replace('-','')
    orders_df['o_orderdate'] = orders_df['o_orderdate'].str.replace('-','')
    #print(lineitem_df)
    #print(orders_df)
    lineitem_df.to_csv(data_dir + "lineitem.csv", header=None, index=False, sep='|')
    orders_df.to_csv(data_dir + "orders.csv", header=None, index=False, sep='|')
    #process CUSTOMER
    customer_df = pd.read_table(data_dir + 'customer.tbl', header = None, names = ['c_custkey', 'c_name', 'c_address', 'c_nationkey', 'c_phone', 'c_acctbal', 'c_mktsegment', 'c_comment'], sep = '|', index_col = False, encoding = 'utf-8')
    customer_df.to_csv(data_dir + "customer.csv", header=None, index = False, sep = '|')
    #process PART
    part_df = pd.read_table(data_dir + 'part.tbl', header = None, names = ['p_partkey', 'p_name', 'p_mfgr', 'p_brand', 'p_type', 'p_size', 'p_container', 'p_retailprice', 'p_comment'], sep = '|', index_col = False, encoding = 'utf-8')
    part_df.to_csv(data_dir + "part.csv", header=None, index = False, sep = '|')
    #process SUPPLIER
    supplier_df = pd.read_table(data_dir + 'supplier.tbl', header = None, names = ['s_suppkey', 's_name', 's_address', 's_nationkey', 's_phone', 's_acctbal', 's_comment'], sep = '|', index_col = False, encoding = 'utf-8')
    supplier_df.to_csv(data_dir + "supplier.csv", header=None, index = False, sep = '|')
    #process nation
    nation_df = pd.read_table(data_dir + 'nation.tbl', header = None, names = ['n_nationkey', 'n_name', 'n_regionkey', 'n_comment'], sep = '|', index_col = False, encoding = 'utf-8')
    nation_df.to_csv(data_dir + "nation.csv", header=None, index = False, sep = '|')
    #process region
    region_df = pd.read_table(data_dir + 'region.tbl', header = None, names = ['r_regionkey', 'r_name', 'r_comment'], sep = '|', index_col = False, encoding = 'utf-8')
    region_df.to_csv(data_dir + "region.csv", header=None, index = False, sep = '|')
