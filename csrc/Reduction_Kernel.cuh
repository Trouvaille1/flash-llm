/***************************************************************************
 * Copyright 2023 The FLash-LLM Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/

// used for the reduction of result matrix if Split-K is used
// Reduction_Workspace:     (M_Global, N_Global, Split_K),  column major
// C:                       (M_Global, N_Global),           column major
// Each thread deals with 8 output elements, each elements is the sum of Split_K elements
// Each Warp: 32 threads_per_warp * 8 half_per_threads -> 256 half_per_warp
// Each GPU: 108 SM -> 108 warp -> 108*256 = 27648
// GridSize = (M_Global*N_Global) / 256

#define ELEMENT_PER_THREADBLOCK 256 // 每个线程块处理的元素数量

__global__ void SplitK_Reduction(half* C, half* Reduction_Workspace, int M_Global, int N_Global, int Split_K) // SplitK归约的全局函数，将多个部分结果合并
{
    // return;
    half* C_BasePTR_ThisBlock = C + ELEMENT_PER_THREADBLOCK * blockIdx.x; // 计算当前线程块在最终输出矩阵C中的基地址
    half* R_BasePTR_ThisBlock = Reduction_Workspace + ELEMENT_PER_THREADBLOCK * blockIdx.x; // 计算当前线程块在归约工作空间中的基地址
    //
    float Results[HALF_PER_128B]; // 用于累积的float类型数组，大小为HALF_PER_128B
//
#pragma unroll
    for (int j = 0; j < HALF_PER_128B; j++) // 遍历每个线程处理的元素
        Results[j] = 0.0f; // 初始化累积结果为0
    //
    for (int i = 0; i < Split_K; i++) { // 遍历所有SplitK的部分结果
#pragma unroll
        for (int j = 0; j < HALF_PER_128B; j++) // 遍历每个线程处理的元素
            Results[j] += __half2float(R_BasePTR_ThisBlock[threadIdx.x * HALF_PER_128B + j]); // 将half精度转换为float并累加到结果中
        R_BasePTR_ThisBlock += M_Global * N_Global; // 移动到下一个SplitK部分结果的位置
    }
#pragma unroll
    for (int j = 0; j < HALF_PER_128B; j++) // 遍历每个线程处理的元素
        C_BasePTR_ThisBlock[threadIdx.x * HALF_PER_128B + j] = __float2half_rn(Results[j]); // 将float结果转换为half精度并写入最终输出矩阵C
}
