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
#ifndef TILINGCONFIG_H
#define TILINGCONFIG_H

//#define DEBUG_MODE 1
#define RELEASE_MODE 1

// Fixed Parameters
// 固定参数
// GPU Tensor Core 的基本计算单位,每个MMA指令执行 16×16×16 的矩阵乘法,对应CUDA的 mma.sync.aligned.m16n16k16 指令
#define MMA_M 16  // MMA指令处理的矩阵A行数
#define MMA_N 16  // MMA指令处理的矩阵B列数
#define MMA_K 16  // MMA指令处理的K维度大小
#define WARP_SIZE 32  // 每个warp包含32个线程（GPU硬件固定）
// Unchangable
// 不可更改的参数
//
#define WARP_ROW_TENSORS 4  // 每个warp处理的矩阵A中的4个tensor行，共4*16=64维m
#define BLOCK_K_TENSORS 4  // 每个block处理的4个K维度的tensor，共4*16=64维k
#define TILE_K (MMA_K * BLOCK_K_TENSORS)  // 64
// Parameters for copying A_TILE & B_TILE & C_TILE
// 用于复制A_TILE、B_TILE和C_TILE的参数
#define COPY_UNIT_FP16_ROWS 8
#define COPY_UNIT_FP16_COLS 64
#define HALF_PER_128B 8           // LDS.128 -> 8 * FP16
                                  // LDS.128指令加载128bit数据，相当于8个FP16
#define REG_PER_C_TENSOR_16_16 8  // 8 for FP32 Accumulation; 4 for FP16 Accumulation
                                  // FP32累加使用8个寄存器；FP16累加使用4个寄存器.详细推导见ipad。一个16*16的fp32的C tile矩阵需要8个寄存器

#define PADDING_SHARED_MEM_FOR_C 4  // Padding 8/2 float each column to eliminating bank-conflict in C fragments
                                    // 每列填充8/2个float来消除C片段中的bank冲突

// NOTE: 对于一个block来说，对于不同的计算负载及优化,内部的warp形状不一样。所以这里只定义了整体的warp数量
template<int BLOCK_ROW_WARPS_, int BLOCK_COL_WARPS_, int WARP_COL_TENSORS_, int N8_ = 0>
struct TilingConfig {
    static constexpr int BLOCK_ROW_WARPS  = BLOCK_ROW_WARPS_;
    static constexpr int BLOCK_COL_WARPS  = BLOCK_COL_WARPS_;
    static constexpr int WARP_COL_TENSORS = WARP_COL_TENSORS_;
    // Sanity checks on the template arguments.
    // 对模板参数进行合理性检查
    // static_assert((BLOCK_ROW_WARPS * BLOCK_COL_WARPS) == 4,
    //               "The number of WARPS per threadblock must be 4.");
    // Derived Parameters
    // 派生参数
    static constexpr int TILE_M        = MMA_M * (WARP_ROW_TENSORS * BLOCK_ROW_WARPS);
    static constexpr int TILE_N        = MMA_N * (WARP_COL_TENSORS * BLOCK_COL_WARPS);
    static constexpr int BLOCK_WARPS   = BLOCK_ROW_WARPS * BLOCK_COL_WARPS;
    static constexpr int BLOCK_THREADS = BLOCK_WARPS * WARP_SIZE;
    // temporary implementation to support N=8
    // 支持N=8的临时实现
    static constexpr int N8      = N8_;
    static constexpr int TILE_N2 = N8 ? 8 : TILE_N;
};

template<int NUM_REG_FOR_SPARSE_KERNEL_ = 64>
struct SparseKernelConfig {
    static constexpr int NUM_REG_FOR_SPARSE_KERNEL    = NUM_REG_FOR_SPARSE_KERNEL_;//每个线程分配的寄存器数量
    static constexpr int VECTOR_SIZE                  = 4;
    static constexpr int PADDING_SIZE_FOR_TILEOFFSETS = 2;  // (N+1 offsets) + 1 padding
                                                            // (N+1个偏移量) + 1个填充
    // Sanity checks on the template arguments.
    // 对模板参数进行合理性检查
    // static_assert((BLOCK_ROW_WARPS * BLOCK_COL_WARPS) == 4,
    //               "The number of WARPS per threadblock must be 4.");
    // Derived Parameters
    // 派生参数
    // static_assert((BLOCK_ROW_WARPS * BLOCK_COL_WARPS) == 4,
    //               "The number of WARPS per threadblock must be 4.");
    // Derived Parameters
    // 派生参数
    // static constexpr int TILE_M = MMA_M * (WARP_ROW_TENSORS * BLOCK_ROW_WARPS);
};

#endif