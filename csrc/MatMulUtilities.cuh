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
#ifndef MatMulUtilities_H
#define MatMulUtilities_H
//在cuBLAS中，ABC都是col major。
//这里只是flash-llm约定A是row major，B是col major，C是col major。实际上AB无所谓，通过cublasGemmEx的CUBLAS_OP_T参数就可以调整是否需要转置。
// NOTE: 无论是cuBlas还是flash-llm，结果矩阵C一定是col major
// C = A*B
// C: col major
// A: row major
// B: col major

#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "AsyncCopy_PTX.cuh"
#include "MMA_PTX.cuh"
#include "TilingConfig.h"

int cuda_CheckError()
{
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    return 0;
}

// New features: Copy size is X * 64, X can be any multiple to 8
// 新特性：复制大小为X * 64，X可以是8的任意倍数。即复制X*64大小的tile矩阵
//该kernel的总数据量为一个block中的数据（多个copy_unit）。以warp为单位遍历：每个warp处理一个copy_unit
//NOTE: 注意TILE_K=64
template<int NumOfRowsToCopy, typename TilingConfig>  // NumOfRowsToCopy must be multiple to COPY_UNIT_FP16_ROWS
                                                      // NumOfRowsToCopy必须是COPY_UNIT_FP16_ROWS的倍数
__device__ __forceinline__ void CopyTileFromGlobalToShared_X_64(half* __restrict__ SharedPTR,  // 共享内存目标地址
                                                                const half* GlobalPTR,         // 全局内存源地址
                                                                const int   GlobalStride,      // 全局内存行跨度，为K_Global
                                                                bool        Pred = true)       // 谓词条件，控制是否执行复制
{
    //一个Copy Unit为8*64=512个half，用一个warp来处理。
    int lane_id       = threadIdx.x % 32;  // 计算warp内的lane ID
    int col           = lane_id % 8;       // bigCol.计算大列索引（每8个half为一列）
    int row1          = lane_id / 8;       // 计算第一行索引
    int row2          = lane_id / 8 + 4;   // 计算第二行索引（偏移4行）
    int store_column1 = col ^ row1;        // 使用XOR避免bank conflicts的存储列索引1
    int store_column2 = col ^ row2;        // 使用XOR避免bank conflicts的存储列索引2
    //
    int       warp_id            = threadIdx.x / 32;  // 计算warp ID
    int       TotalNumOfCopyUnit = NumOfRowsToCopy / COPY_UNIT_FP16_ROWS;  // 计算总的复制单元数量
    //ceil。⌈a/b⌉ = (a - 1) / b + 1 = （a+b-1）/b
    const int MaxIteration =  // 实际上是MaxIterationPerWarp.计算每个warp的最大迭代次数，确保所有warp都能参与复制
        (TotalNumOfCopyUnit - 1) / (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + 1;
//
#pragma unroll
    for (int i = 0; i < MaxIteration; i++) {//warp迭代去组成block
        // NOTE: 遍历所有迭代。线程的迭代实际上就是warp的迭代，warp的迭代组成一个block需要处理的数据量。
        // NOTE: 设计思路是：对于一个block的数据量，先判断出一个warp所要处理的数据量，然后假设只有一个warp，设计for循环去迭代这个warp，写for循环内的代码时按照分配给每个thread的数据量去写
        // NOTE: copy_unit的个数大于一个block中warp的个数，所以需要遍历所有迭代
        int  COPY_UNIT_I        = (i * (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + warp_id);  // COPY_UNIT_Index.当前warp负责的复制单元索引
        bool AsyncCopyPredictor = COPY_UNIT_I < TotalNumOfCopyUnit && Pred;  ///// Bug, too hard to find this bug, 5555
                                                                             // 异步复制谓词：检查是否在有效范围内且满足谓词条件
        const half* GlobalPTR_Unit        = GlobalPTR + COPY_UNIT_I * COPY_UNIT_FP16_ROWS * GlobalStride;  // 计算全局内存中当前复制单元的起始地址
        half* __restrict__ SharedPTR_Unit = SharedPTR + COPY_UNIT_I * COPY_UNIT_FP16_ROWS * TILE_K;       // 计算共享内存中当前复制单元的起始地址
        //每个线程处理2*16Byte=16个half
        //复制8个half到共享内存
        cp_async<16>(SharedPTR_Unit + store_column1 * HALF_PER_128B + row1 * TILE_K,  // 异步复制第一行数据到共享内存
                     GlobalPTR_Unit + col * HALF_PER_128B + row1 * GlobalStride,      // 从全局内存第一行读取
                     AsyncCopyPredictor);                                              // 使用谓词控制是否执行
        //复制8个half到共享内存
        cp_async<16>(SharedPTR_Unit + store_column2 * HALF_PER_128B + row2 * TILE_K,  // 异步复制第二行数据到共享内存
                     GlobalPTR_Unit + col * HALF_PER_128B + row2 * GlobalStride,      // 从全局内存第二行读取
                     AsyncCopyPredictor);                                              // 使用谓词控制是否执行
        // cp_async_test_only<16>( SharedPTR_Unit + store_column1*HALF_PER_128B + row1 * TILE_K , GlobalPTR_Unit +
        // col*HALF_PER_128B + row1*GlobalStride, AsyncCopyPredictor ); cp_async_test_only<16>( SharedPTR_Unit +
        // store_column2*HALF_PER_128B + row2 * TILE_K , GlobalPTR_Unit + col*HALF_PER_128B + row2*GlobalStride,
        // AsyncCopyPredictor );
        // 注释掉的测试代码：用于调试的异步复制测试函数调用
    }
}

//NOTE: 由于要用到tensorcore的mma指令，所以写每个thread代码时，都相当于写一个thread完成一个warp的操作。由于这里所有warp刚好能组成一个block，所以不需要像CopyTileFromGlobalToShared_X_64()一样去迭代warp操作
template<typename TilingConfig> // 模板函数，接受TilingConfig作为模板参数
__device__ __forceinline__ void PipelinedCoreComputations(float c[][REG_PER_C_TENSOR_16_16], // 结果矩阵C的寄存器数组
                                                          uint32_t __restrict__ a[][4], // A矩阵数据的寄存器数组
                                                          uint32_t __restrict__ b[][4], // B矩阵数据的寄存器数组
                                                          half* __restrict__ SharedMemoryPTR, // 共享内存指针
                                                          int warp_start_row, // warp在C矩阵tile中的起始行
                                                          int warp_start_col) // warp在C矩阵tile中的起始列
{
    uint32_t(*c_uint32_t)[REG_PER_C_TENSOR_16_16] = reinterpret_cast<uint32_t(*)[REG_PER_C_TENSOR_16_16]>(c); // 将float类型的C矩阵寄存器重新解释为uint32_t类型，以便进行MMA指令操作
    // First Register Loading
    // 第一次寄存器加载
    FragLoadFromSharedToRegisters<WARP_ROW_TENSORS>(a, SharedMemoryPTR, warp_start_row, 0); // 从共享内存加载A矩阵数据到寄存器，偏移量为0
    B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>( // 从共享内存加载B矩阵数据到寄存器
        b, SharedMemoryPTR + TilingConfig::TILE_M * TILE_K, warp_start_col, 0); // B矩阵起始地址偏移了A矩阵tile的大小
// Sencond loading & first computation, so on
// 第二次加载和第一次计算，依此类推
#pragma unroll
    for (int k = 0; k < BLOCK_K_TENSORS; k++) { // 遍历K维度的所有tensor块
        uint32_t __restrict__(*a_read)[4]  = a; // 指向当前读取的A矩阵寄存器
        uint32_t __restrict__(*b_read)[4]  = b; // 指向当前读取的B矩阵寄存器
        uint32_t __restrict__(*a_write)[4] = a; // 指向下一次写入的A矩阵寄存器
        uint32_t __restrict__(*b_write)[4] = b; // 指向下一次写入的B矩阵寄存器
        a_read += ((k) % 2) * WARP_ROW_TENSORS; // 使用双缓冲机制，根据k的奇偶性选择A矩阵读取缓冲区
        b_read += ((k) % 2) * TilingConfig::WARP_COL_TENSORS; // 使用双缓冲机制，根据k的奇偶性选择B矩阵读取缓冲区
        a_write += ((k + 1) % 2) * WARP_ROW_TENSORS; // 使用双缓冲机制，根据k+1的奇偶性选择A矩阵写入缓冲区
        b_write += ((k + 1) % 2) * TilingConfig::WARP_COL_TENSORS; // 使用双缓冲机制，根据k+1的奇偶性选择B矩阵写入缓冲区
        // data loading
        // 数据预加载
        if (k + 1 < BLOCK_K_TENSORS) { // 如果不是最后一次迭代，预加载下一轮的数据
            // ldmatrix.x4指令（加载4个8*8矩阵，即一个16*16矩阵）
            // 对于A矩阵，从共享内存加载WARP_ROW_TENSORS=4个16*16FP16矩阵到16个寄存器
            // 对于B矩阵，从共享内存加载WARP_COL_TENSORS个16*16FP16矩阵到寄存器
            FragLoadFromSharedToRegisters<WARP_ROW_TENSORS>(a_write, SharedMemoryPTR, warp_start_row, (k + 1) * MMA_K); // 预加载下一轮A矩阵数据到写入缓冲区
            B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>( // 预加载下一轮B矩阵数据到写入缓冲区
                b_write, SharedMemoryPTR + TilingConfig::TILE_M * TILE_K, warp_start_col, (k + 1) * MMA_K);
        }
// computations
// 计算
#pragma unroll
        for (int i = 0; i < WARP_ROW_TENSORS; i++) // 遍历warp负责的行方向tensor数量
#pragma unroll
            for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) { // 遍历warp负责的列方向tensor数量
                // MMA_FP16_M16N16K16( c_uint32_t[i + j*WARP_ROW_TENSORS], a_read[i], b_read[j] );//Ampere架构不支持16*16*16的MMA指令
                // 通过两次16×8×16的MMA指令来实现一个完整的16×16×16矩阵乘法
                MMA_FP16_M16N8K16(c_uint32_t[i + j * WARP_ROW_TENSORS], a_read[i], b_read[j]); // 执行16x8x16的MMA指令，计算矩阵乘法并累加到结果寄存器
                if (!TilingConfig::N8) // 如果不是N8模式（即N=16模式）
                    MMA_FP16_M16N8K16(c_uint32_t[i + j * WARP_ROW_TENSORS] + 4, a_read[i], b_read[j] + 2);  // c+4; b+2
                    // 执行第二个8x8的MMA指令，处理16x16tensor的右半部分
            }
        //// only used for pipeline analysis
        //// 仅用于流水线分析
        //#pragma unroll
        // for (int i = 0; i < WARP_ROW_TENSORS; i++)
        //{
        //  int j=0;
        //  MMA_FP16_M16N8K16( c_uint32_t[i + j*WARP_ROW_TENSORS], a_read[i], b_read[j] );
        //}
        //#pragma unroll
        // for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++)
        //{
        //  int i=0;
        //  if(!TilingConfig::N8)
        //    MMA_FP16_M16N8K16( c_uint32_t[i + j*WARP_ROW_TENSORS]+4 , a_read[i], b_read[j]+2 );    // c+4; b+2
        //}
    }
}

// 将C片段存储到共享内存。warp级操作函数
template<typename TilingConfig> // 模板函数，接受TilingConfig作为模板参数
__device__ __forceinline__ void
StoreToSharedMemoryFromRegister(float (*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C], // 指向共享内存C片段的指针，每行有TILE_M+PADDING个元素
                                float c[][REG_PER_C_TENSOR_16_16]) // 寄存器中的C矩阵数据，每个tensor有8个寄存器
{ 
    const unsigned int warpId        = threadIdx.x / WARP_SIZE; // 计算当前线程所属的warp ID
    int                Warp_i        = warpId / TilingConfig::BLOCK_COL_WARPS; // 计算warp在行方向的索引
    int                Warp_j        = warpId % TilingConfig::BLOCK_COL_WARPS; // 计算warp在列方向的索引
    int                Warp_i_offset = Warp_i * (MMA_M * WARP_ROW_TENSORS); // 计算warp在共享内存中的行偏移量（half元素级）
    int                Warp_j_offset = Warp_j * (MMA_N * TilingConfig::WARP_COL_TENSORS); // 计算warp在共享内存中的列偏移量（half元素级）
    //
    int lane_id = threadIdx.x % WARP_SIZE; // 计算当前线程在warp中的lane ID
//
#pragma unroll
    for (int i = 0; i < WARP_ROW_TENSORS; i++) { // 遍历warp负责的行方向tensor数量（4个）
#pragma unroll
        for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) { // 遍历warp负责的列方向tensor数量
            // Dealing with one 16*16 Tensor
            // 处理一个16×16的tensor
            int RegSetID        = j * WARP_ROW_TENSORS + i; // 计算当前tensor在寄存器数组中的索引
            int Tensor_i_offset = Warp_i_offset + i * MMA_M; // 计算当前tensor在共享内存中的行起始位置（half元素级）
            int Tensor_j_offset = Warp_j_offset + j * MMA_N; // 计算当前tensor在共享内存中的列起始位置（half元素级）
#pragma unroll
            for (int r = 0; r < REG_PER_C_TENSOR_16_16; r++) { // 遍历每个16×16tensor的8个寄存器
                int row_offset = lane_id / 4; // 计算当前线程负责的行偏移（每4个线程处理一行）
                int col_offset = (lane_id % 4) * 2; // 计算当前线程负责的列偏移（每个线程处理2列）
                //
                if (r % 2 > 0) // 如果是奇数寄存器
                    col_offset += 1; // 列偏移加1，处理右侧元素
                //
                if (r % 4 >= 2) // 如果寄存器索引的低2位>=2
                    row_offset += 8; // 行偏移加8，处理下半部分
                if (r >= 4) // 如果寄存器索引>=4
                    col_offset += 8; // 列偏移加8，处理右半部分
                //
                (*(smem_CFrag + Tensor_j_offset + col_offset))[Tensor_i_offset + row_offset] = c[RegSetID][r]; // 将寄存器值存储到共享内存的对应位置
            }
        }
    }
}

#endif