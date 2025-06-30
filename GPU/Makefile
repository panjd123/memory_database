all: ssb_benchmark tpch_benchmark gpu

ssb_benchmark: ssb_benchmark.cu SSB.cuh utils.cuh params.h
	nvcc -O3 -lineinfo -o ssb_benchmark ssb_benchmark.cu -Wno-deprecated-gpu-targets -std=c++20 -Xcompiler -fopenmp

tpch_benchmark: tpch_benchmark.cu TPCH.cuh utils.cuh params.h
	nvcc -O3 -lineinfo -g -o tpch_benchmark tpch_benchmark.cu -Wno-deprecated-gpu-targets -std=c++20 -Xcompiler -fopenmp

gpu: gpu.cu
	nvcc -O3 -o gpu gpu.cu -Wno-deprecated-gpu-targets -std=c++20 -Xcompiler -fopenmp

read: read.cu
	nvcc -O3 -o read read.cu -Wno-deprecated-gpu-targets -std=c++20 -Xcompiler -fopenmp

bandwidth: bandwidth.cu
	nvcc -O3 -o bandwidth bandwidth.cu -Wno-deprecated-gpu-targets -std=c++20 -Xcompiler -fopenmp

# run: tpch_benchmark
# 	./tpch_benchmark --test-case default --SF=100 --o-sele=0.1 --o-groups=0 --n-sele=0.1 --n-groups=2

# benchmark: tpch_benchmark
# 	bash tpch_benchmarks.sh

# gdb: tpch_benchmark
# 	cuda-gdb tpch_benchmark

# memcheck: tpch_benchmark
# 	compute-sanitizer --tool memcheck tpch_benchmark

# profile: tpch_benchmark
# 	# sudo /usr/local/cuda-12.8/bin/ncu --nvtx --nvtx-include "profile/" --set full --target-processes all --import-source yes -f -o tpch_report ./tpch_benchmark --test-case 0.01 --SF=100 --o-sele=0.5 --o-groups=0 --n-sele=0.5 --n-groups=2
# 	sudo /usr/local/cuda-12.8/bin/ncu --nvtx --nvtx-include "profile/" --set full --target-processes all --import-source yes -f -o tpch_report ./tpch_benchmark --test-case 0.01 --SF=1 --o-sele=0.5 --o-groups=0 --n-sele=0.5 --n-groups=2

run: ssb_benchmark
	./ssb_benchmark --test-case 3.1 --SF=100 --d-sele=0.857 --d-bitmap=1 --d-groups=6 --s-sele=0.2 --s-bitmap=1 --s-groups=5 --p-sele=0 --p-bitmap=0 --p-groups=0 --c-sele=0.2 --c-bitmap=1 --c-groups=5 --nthreads=48 --sqlnum=31

nrun: ssb_benchmark
	./ssb_benchmark --test-case 3.1 --SF=100 --d-sele=0.5 --d-bitmap=1 --d-groups=2 --s-sele=0.5 --s-bitmap=1 --s-groups=2 --p-sele=0.5 --p-bitmap=1 --p-groups=2 --c-sele=0.5 --c-bitmap=1 --c-groups=2 --nthreads=48 --sqlnum=31

# benchmark: ssb_benchmark
# 	cd .. && bash ssb_benchmarks.sh

# gdb: ssb_benchmark
# 	cuda-gdb ssb_benchmark

# memcheck: ssb_benchmark
# 	compute-sanitizer --tool memcheck ssb_benchmark

# profile: ssb_benchmark
# 	sudo /usr/local/cuda-12.8/bin/ncu --nvtx --nvtx-include "profile/" --set full --target-processes all --import-source yes -f -o ssb_report ./ssb_benchmark --test-case 3.1 --SF=50 --d-sele=0.857 --d-bitmap=1 --d-groups=6 --s-sele=0.2 --s-bitmap=1 --s-groups=5 --p-sele=0 --p-bitmap=0 --p-groups=0 --c-sele=0.2 --c-bitmap=1 --c-groups=5 --nthreads=48 --sqlnum=31

# rprofile: ssb_benchmark
# 	sudo /usr/local/cuda-12.8/bin/ncu --nvtx --nvtx-include "profile/" --set full --target-processes all --import-source yes -f -o 0.9_ssb_report ./ssb_benchmark --test-case 0.9 --SF=50  --c-sele=0.9 --c-bitmap=1 --c-groups=10 --d-sele=0 --d-bitmap=0 --d-groups=0 --s-sele=0 --s-bitmap=0 --s-groups=0 --p-sele=0 --p-bitmap=0 --p-groups=0

clean:
	rm -f ssb_benchmark tpch_benchmark