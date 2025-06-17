#!/bin/bash
./ssb_benchmark --test-case 5 --SF=5 --s-sele=1 --s-bitmap=1 --s-groups=5 --c-sele=1 --c-bitmap=1 --c-groups=5 --d-sele=0 --d-bitmap=0 --d-groups=0 --p-sele=0 --p-bitmap=0 --p-groups=0
./ssb_benchmark --test-case 10 --SF=10 --s-sele=1 --s-bitmap=1 --s-groups=5 --c-sele=1 --c-bitmap=1 --c-groups=5 --d-sele=0 --d-bitmap=0 --d-groups=0 --p-sele=0 --p-bitmap=0 --p-groups=0
./ssb_benchmark --test-case 25 --SF=25 --s-sele=1 --s-bitmap=1 --s-groups=5 --c-sele=1 --c-bitmap=1 --c-groups=5 --d-sele=0 --d-bitmap=0 --d-groups=0 --p-sele=0 --p-bitmap=0 --p-groups=0
./ssb_benchmark --test-case 50 --SF=50 --s-sele=1 --s-bitmap=1 --s-groups=5 --c-sele=1 --c-bitmap=1 --c-groups=5 --d-sele=0 --d-bitmap=0 --d-groups=0 --p-sele=0 --p-bitmap=0 --p-groups=0
./ssb_benchmark --test-case 100 --SF=100 --s-sele=1 --s-bitmap=1 --s-groups=5 --c-sele=1 --c-bitmap=1 --c-groups=5 --d-sele=0 --d-bitmap=0 --d-groups=0 --p-sele=0 --p-bitmap=0 --p-groups=0