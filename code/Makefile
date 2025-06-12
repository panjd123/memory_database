GPU_OLAPcore_test: GPU_OLAPcore/test_multi_compute_operator.cu
	nvcc -O3 -o ../GPU_OLAPcore_test GPU_OLAPcore/test_multi_compute_operator.cu -lnuma -I.

test_join: join/test_join_option.cpp
	g++ -O3 -o ../test_join join/test_join_option.cpp -I. -I../include -lnuma -lpthread