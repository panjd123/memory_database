#!/bin/bash

./tpch_benchmark --test-case o_1.0 --SF=100 --o-sele=1 --o-groups=0 --n-sele=1 --n-groups=5
./tpch_benchmark --test-case o_0.5 --SF=100 --o-sele=0.5 --o-groups=0 --n-sele=1 --n-groups=5
./tpch_benchmark --test-case o_0.1 --SF=100 --o-sele=0.1 --o-groups=0 --n-sele=1 --n-groups=5

./tpch_benchmark --test-case n_1.0 --SF=100 --o-sele=1 --o-groups=0 --n-sele=1 --n-groups=5
./tpch_benchmark --test-case n_0.5 --SF=100 --o-sele=1 --o-groups=0 --n-sele=0.5 --n-groups=5
./tpch_benchmark --test-case n_0.1 --SF=100 --o-sele=1 --o-groups=0 --n-sele=0.1 --n-groups=5

# ./tpch_benchmark --test-case o_1. --SF=1 --o-sele=1 --o-groups=0 --n-sele=1 --n-groups=5
# ./tpch_benchmark --test-case o_.1 --SF=1 --o-sele=0.1 --o-groups=0 --n-sele=1 --n-groups=5
# ./tpch_benchmark --test-case n_.1 --SF=1 --o-sele=1 --o-groups=0 --n-sele=0.1 --n-groups=5

# ./tpch_benchmark --test-case o_1. --SF=8 --o-sele=1 --o-groups=0 --n-sele=1 --n-groups=5
# ./tpch_benchmark --test-case o_.1 --SF=8 --o-sele=0.1 --o-groups=0 --n-sele=1 --n-groups=5
# ./tpch_benchmark --test-case n_.1 --SF=8 --o-sele=1 --o-groups=0 --n-sele=0.1 --n-groups=5

# ./tpch_benchmark --test-case o_1. --SF=32 --o-sele=1 --o-groups=0 --n-sele=1 --n-groups=5
# ./tpch_benchmark --test-case o_.1 --SF=32 --o-sele=0.1 --o-groups=0 --n-sele=1 --n-groups=5
# ./tpch_benchmark --test-case n_.1 --SF=32 --o-sele=1 --o-groups=0 --n-sele=0.1 --n-groups=5

# ./tpch_benchmark --test-case o_1. --SF=64 --o-sele=1 --o-groups=0 --n-sele=1 --n-groups=5
# ./tpch_benchmark --test-case o_.1 --SF=64 --o-sele=0.1 --o-groups=0 --n-sele=1 --n-groups=5
# ./tpch_benchmark --test-case n_.1 --SF=64 --o-sele=1 --o-groups=0 --n-sele=0.1 --n-groups=5

# ./tpch_benchmark --test-case o_1. --SF=100 --o-sele=1 --o-groups=0 --n-sele=1 --n-groups=5
# ./tpch_benchmark --test-case o_.1 --SF=100 --o-sele=0.1 --o-groups=0 --n-sele=1 --n-groups=5
# ./tpch_benchmark --test-case n_.1 --SF=100 --o-sele=1 --o-groups=0 --n-sele=0.1 --n-groups=5