#!/bin/bash
# Copyright 2023 The FLash-LLM Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# M=(21504  7168   28672  7168   27648  9216   36864  9216   36864  12288  49152  12288)
# K=(7168   7168   7168   28672  9216   9216   9216   36864  12288  12288  12288  49152)
# SplitK=(5      7      7      7      2      6      3      6      3      9      9     9)
# N=(8 16 32 64)
# Sparsity=(70 80 90)


# for BS in ${N[@]} 
# do
#     #echo "BS=${BS}"
#     for ((i=0;i<${#M[@]};i++)) 
#     do
#         #echo "Processing Shape ${i}..."
#         for S in ${Sparsity[@]} 
#         do
#             #echo "Sparsity = $S"
#                 ./spmm_test ${M[i]} ${K[i]} ${BS} ${S} ${SplitK[i]}
#         done    
#     done
# done

#NOTE: 复现图10配置
# 四种 M×K 大小（OPT-66B 四个“Skinny MatMul”）.见图2
# M=3H H 4H H
# k=H H H 4H
M=(27648  9216  36864   9216)
K=( 9216  9216   9216  36864)
# 对应的 SplitK 参数（可根据硬件调优结果调整）
SplitK=(    2     6      3      6)

BS=(16 32)
# 只测试 90% 稀疏率
Sparsity=(90)

for bs in "${BS[@]}"; do
  echo -e "\n==== Batch Size = $bs ===="
  for i in "${!M[@]}"; do
    m=${M[i]}
    k=${K[i]}
    sk=${SplitK[i]}
    echo "-> Shape ${m}×${k}, sparsity=90%, splitK=${sk}"
    ./kernel_utilization "$m" "$k" "$bs" "90" "$sk"
  done
done

# ./spmm_test 36864 9216 32 70 3
# ./spmm_test 1024 1024 8 70 5

# ./spmm_test 27648 9216 16 90 5
# ./kernel_utilization 27648 9216 16 90 5