all: ssb_benchmark tpch_benchmark

ssb_benchmark: ssb_benchmark.cu SSB.cuh TPCH.cuh utils.cuh
	nvcc -O3 -o ssb_benchmark ssb_benchmark.cu -Wno-deprecated-gpu-targets -std=c++20 -lineinfo -Xcompiler -fopenmp

tpch_benchmark: tpch_benchmark.cu TPCH.cuh utils.cuh
	nvcc -O0 -g -o tpch_benchmark tpch_benchmark.cu -Wno-deprecated-gpu-targets -std=c++20 -lineinfo -Xcompiler -fopenmp



run: tpch_benchmark
	tpch_benchmark --test-case default --SF=100 --o-sele=1 --o-groups=0 --n-sele=0.08 --n-groups=2

benchmark: tpch_benchmark
	bash tpch_benchmarks.sh

gdb: tpch_benchmark
	cuda-gdb tpch_benchmark

# memcheck: tpch_benchmark
# 	compute-sanitizer --tool memcheck tpch_benchmark



# run: ssb_benchmark
# 	ssb_benchmark --SF=100 --d-sele=1 --d-bitmap=1 --d-groups=7 --s-sele=0.2 --s-bitmap=0 --s-groups=1 --p-sele=0.008 --p-bitmap=1 --p-groups=8 --c-sele=0 --c-bitmap=0 --c-groups=0 --nthreads=48 --sqlnum=22

# benchmark: ssb_benchmark
# 	cd .. && bash ssb_benchmarks.sh

# gdb: ssb_benchmark
# 	cuda-gdb ssb_benchmark

# memcheck: ssb_benchmark
# 	compute-sanitizer --tool memcheck ssb_benchmark