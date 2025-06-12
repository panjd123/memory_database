import os
import glob
import pandas as pd
import re

# 查找所有 benchmark_results_*.csv 文件
csv_files = glob.glob("**/benchmark_results_*.csv", recursive=True)

dfs = []
for path in csv_files:
    # 提取版本号，比如 3.1
    match = re.search(r"benchmark_results_([0-9.]+)\.csv", os.path.basename(path))
    if not match:
        continue
    test_case = match.group(1)

    # 读取 CSV，并加上 version 列
    df = pd.read_csv(path)
    df["TestCase"] = test_case
    dfs.append(df)

# 合并所有 dataframe
if dfs:
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv("merged_benchmark_results.csv", index=False)
    print(f"合并完成，共 {len(merged)} 条记录，输出为 merged_benchmark_results.csv")
else:
    print("没有找到匹配的 CSV 文件")
