#!/bin/bash
./tpch_benchmark --test-case 1 --SF=100 --o-sele=1 --o-groups=0 --n-sele=0.1 --n-groups=2
./tpch_benchmark --test-case 0.5 --SF=100 --o-sele=0.5 --o-groups=0 --n-sele=0.1 --n-groups=2
./tpch_benchmark --test-case 0.1 --SF=100 --o-sele=0.1 --o-groups=0 --n-sele=0.1 --n-groups=2
./tpch_benchmark --test-case 0.01 --SF=100 --o-sele=0.01 --o-groups=0 --n-sele=0.1 --n-groups=2
../tpch_benchmark --test-case 0.01 --SF=100 --o-sele=1 --o-groups=0 --n-sele=1 --n-groups=2
../tpch_benchmark --test-case 0.01 --SF=100 --o-sele=0.5 --o-groups=0 --n-sele=0.5 --n-groups=2