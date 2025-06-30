# GPU 分组聚合算子

## 编译

```bash
make
```

## 运行

运行参数见

```bash
./ssb_benchmark -h
./tpch_benchmark -h
```

## 复现报告中的结果

1. 复现 SSB 不同 testCase：`bash ssb_benchmarks.sh`
2. 复现 SSB 3.1 TestCase 在不同 SF 下的性能：`bash ssb_benchmarks_SF.sh`
3. 测试不同选择率下的顺序随机读取速度：`bash ssb_read_benchmarks.sh`
4. 复现 TPCH：`bash tpch_benchmarks.sh`
