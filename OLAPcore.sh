#!/bin/bash
rm ./log/multi_compute_operator/olapcore_test.tsv
./OLAPcore --SF=100 --d-sele=1 --d-bitmap=1 --d-groups=7 --s-sele=0.2 --s-bitmap=0 --s-groups=1 --p-sele=0.04 --p-bitmap=1 --p-groups=40 --c-sele=0 --c-bitmap=0 --c-groups=0 --nthreads=48 --sqlnum=21
./OLAPcore --SF=100 --d-sele=1 --d-bitmap=1 --d-groups=7 --s-sele=0.2 --s-bitmap=0 --s-groups=1 --p-sele=0.008 --p-bitmap=1 --p-groups=8 --c-sele=0 --c-bitmap=0 --c-groups=0 --nthreads=48 --sqlnum=22
./OLAPcore --SF=100 --d-sele=1 --d-bitmap=1 --d-groups=7 --s-sele=0.2 --s-bitmap=0 --s-groups=1 --p-sele=0.001 --p-bitmap=1 --p-groups=1 --c-sele=0 --c-bitmap=0 --c-groups=0 --nthreads=48 --sqlnum=23
./OLAPcore --SF=100 --d-sele=0.857 --d-bitmap=1 --d-groups=6 --s-sele=0.2 --s-bitmap=1 --s-groups=5 --p-sele=0 --p-bitmap=0 --p-groups=0 --c-sele=0.2 --c-bitmap=1 --c-groups=5 --nthreads=48 --sqlnum=31
./OLAPcore --SF=100 --d-sele=0.857 --d-bitmap=1 --d-groups=6 --s-sele=0.04 --s-bitmap=1 --s-groups=10 --p-sele=0 --p-bitmap=0 --p-groups=0 --c-sele=0.04 --c-bitmap=1 --c-groups=10 --nthreads=48 --sqlnum=32
./OLAPcore --SF=100 --d-sele=0.857 --d-bitmap=1 --d-groups=6 --s-sele=0.008 --s-bitmap=1 --s-groups=2 --p-sele=0 --p-bitmap=0 --p-groups=0 --c-sele=0.008 --c-bitmap=1 --c-groups=2 --nthreads=48 --sqlnum=33
./OLAPcore --SF=100 --d-sele=0.012 --d-bitmap=1 --d-groups=1 --s-sele=0.008 --s-bitmap=1 --s-groups=2 --p-sele=0 --p-bitmap=0 --p-groups=0 --c-sele=0.008 --c-bitmap=1 --c-groups=2 --nthreads=48 --sqlnum=34
./OLAPcore --SF=100 --d-sele=1 --d-bitmap=1 --d-groups=7 --s-sele=0.2 --s-bitmap=0 --s-groups=1 --p-sele=0.4 --p-bitmap=0 --p-groups=1 --c-sele=0.2 --c-bitmap=1 --c-groups=5 --nthreads=48 --sqlnum=41
./OLAPcore --SF=100 --d-sele=0.286 --d-bitmap=1 --d-groups=2 --s-sele=0.2 --s-bitmap=1 --s-groups=5 --p-sele=0.4 --p-bitmap=1 --p-groups=10 --c-sele=0.2 --c-bitmap=0 --c-groups=1 --nthreads=48 --sqlnum=42
./OLAPcore --SF=100 --d-sele=0.286 --d-bitmap=1 --d-groups=2 --s-sele=0.04 --s-bitmap=1 --s-groups=10 --p-sele=0.04 --p-bitmap=1 --p-groups=40 --c-sele=0.2 --c-bitmap=0 --c-groups=1 --nthreads=48 --sqlnum=43