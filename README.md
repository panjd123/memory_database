# Memory Database Operator Test

这是一个基于C++\CUDA的内存数据库算子测试的项目，包括选择(select)、投影(project)等多个算子模块

## 目录

- [项目结构](#项目结构)
- [依赖](#依赖)
- [编译](#编译)
- [模块说明](#模块说明)
- [使用方法](#使用方法)
- [清理](#清理)

## 项目结构

项目的主要结构如下：
 ```bash
├── code/
│   ├── select/
│   ├── project/
│   ├── join/
│   ├── group/
│   ├── aggregation/
│   ├── starjoin/
│   ├── OLAPcore/
|   |—— GPU_OLAPcore/
│   ├── TPCH_Q5/
│   └── TPCH_Q5_operator/
├── include/
├── obj/
|—— log/
└── makefile

## 依赖
本项目依赖以下库和工具：
- GCC (g++) 版本10.3.0或更高
- NVIDIA CUDA Compiler (nvcc) 版本12.4或更高
- pthread
- numa
- Apache Arrow C++19.0(Stable)及以上
- AVX512 指令集支持

确保您的系统已安装这些依赖项，其中编译Apache Arrow C++的指令为：
cmake -DCMAKE_INSTALL_PREFIX=/path/hrc2/to/arrow -DARROW_PARQUET=ON -DARROW_CSV=ON   .. 并将makefile里的ARROW_PATH替换为/path/hrc2/to/arrow

## 编译

使用提供的Makefile来编译项目：

1. 编译所有模块：
   ```bash
   make ALL
2. 编译特定模块（例如select_test）：
    ```bash
   make select_test

## 模块说明
项目包含以下主要模块：
1. select_test: select算子单元测试
2. project_test: project算子单元测试
3. join_test: join算子单元测试
4. group_test: group算子单元测试
5. aggregation_test: aggregation算子单元测试
6. starjoin_test: 星型连接算子单元测试
7. OLAPcore_test: OLAP核心功能模块测试
9. GPU_OLAPcore: GPU_OLAP核心功能模块测试
10. TPCH_Q5: TPC-H基准中的Q5查询综合测试
11. TPCH_Q5_operator: TPC-H基准中的Q5查询综合测试
每个模块都有其对应的源文件和目标文件。

## 使用方法
编译完成后，可以运行相应的可执行文件来测试各个模块。
- select
./select_test 或 ./select_test --is_lsr(
其中is_lsr代表低选择率测试，测试完成后的日志文件在./log/select中 可更换./POWER_BI/select_test中的pbix的源数据文件来可视化展示select算子性能
- project
./project_test
测试完成后的日志文件在./log/project中 可更换./POWER_BI/project_test中的pbix的源数据文件来可视化展示选择project性能
- join
./join_test
测试完成后的日志文件在./log/join中 可更换./POWER_BI/join_test中的pbix的源数据文件来可视化展示join算子性能
- group
./group_test
测试完成后的日志文件在./log/group中 可更换./POWER_BI/group_test中的pbix的源数据文件来可视化展示group算子性能
- aggregate
./aggregation_test
测试完成后的日志文件在./log/aggregation中 可更换./POWER_BI/aggregation_test中的pbix的源数据文件来可视化展示aggregation算子性能
- starjoin
./starjoin_test
测试完成后的日志文件在./log/starjoin中 可更换./POWER_BI/starjoin_test中的pbix的源数据文件来可视化展示星型连接算子性能
- OLAPcore_test
sh OLAPcore.sh
测试完成后的日志文件在./log/OLAPcore中 可更换./POWER_BI/OLAPcore_test中的pbix的源数据文件来可视化展示OLAP核心功能性能
- GPU_OLAPcore
sh GPU_OLAPcore.sh
测试完成后的日志文件在./log/GPU_OLAPcore中 可更换./POWER_BI/GPU_OLAPcore_test中的pbix的源数据文件来可视化展示GPU_OLAP核心功能性能
- TPCH_Q5
cd ./dbgen && ./dbgen -vf -s 1 && cd .. && python convert.py（生成和处理数据）
./TPCH_Q5_test 参数示例参照实验报告
测试结束后打印测试结果
- TPCH_Q5_operator
./TPCH_Q5_operator_test
测试结果的日志文件存储在./log/TPCH_Q5_operator中

## 清理
要清理所有编译生成的文件，运行：
    make clean
这将删除所有目标文件和可执行文件。

## 注意事项
1. 本项目使用了高级CPU指令集（如AVX512），请确保您的硬件支持这些特性。
2. 编译时使用了 -g -O0 标志，这适合调试。对于生产环境，您可能需要调整优化级别。
3. 项目使用C++17标准，请确保您的编译器支持此版本。

## 贡献
如果您想为这个项目做出贡献，请遵循以下步骤：
1. Fork 该仓库
2. 创建您的特性分支 (git checkout -b feature/AmazingFeature)
3. 提交您的更改 (git commit -m 'Add some AmazingFeature')
4. 推送到分支 (git push origin feature/AmazingFeature)
5. 打开一个 Pull Request
