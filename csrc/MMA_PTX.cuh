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
#include "TilingConfig.h"
#include <cuda_fp16.h>

//总数据量为一个thread的寄存器
//一个warp在行方向上加载4个tensors
template<int NumOfTensors> // 模板函数，NumOfTensors指定要加载的tensor数量。定值，为NumOfTensors==WARP_ROW_TENSORS==4
__device__ __forceinline__ void FragLoadFromSharedToRegisters(uint32_t __restrict__ Registers[][4], // 目标寄存器数组，每个tensor占用4个32位寄存器
                                                              half* __restrict__ smem, // 共享内存指针
                                                              int warp_start_row, // warp在矩阵tile中的起始行
                                                              int k_offset) // K维度的偏移量
{
    //
    int lane_id = threadIdx.x % 32; // 计算当前线程在warp中的lane ID（0-31）
    int i       = lane_id % MMA_M; // 计算当前线程负责的行索引（i = lane_id % 16）
    int j       = lane_id / MMA_M; // 计算当前线程负责的列块索引（j = lane_id / 16）
    //
    smem += TILE_K * (warp_start_row + i) + (k_offset + j * HALF_PER_128B); // 计算当前线程要读取的共享内存地址
    uint32_t __restrict__ smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem)); // 将通用指针转换为共享内存地址
    // Row Permutation to eliminating bank-conflict
    // 行置换以消除bank冲突
    uint32_t RowLane_RowPermutation = i % COPY_UNIT_FP16_ROWS; // 计算行置换索引（i % 8）
    uint32_t Mask_RowPermutation    = RowLane_RowPermutation << 4; // 生成置换掩码，左移4位
    smem_local_ptr                  = smem_local_ptr ^ Mask_RowPermutation; // 对共享内存地址进行XOR操作，实现行置换
//
#pragma unroll
    for (int i = 0; i < NumOfTensors; i++) { // 一个warp要加载NumOfTensors个tensor，每个tensor为4*8*8=16*16=256个half（见图：https://pic2.zhimg.com/v2-205f6f878472640583513fceba625b07_r.jpg）
        // PTX内联汇编：ldmatrix指令，从共享内存加载4个8x8FP16=1个16*16矩阵到4个寄存器
        // 输出约束：将结果写入4个32位寄存器，输入约束：共享内存地址
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(Registers[i][0]), "=r"(Registers[i][1]), "=r"(Registers[i][2]), "=r"(Registers[i][3])
                     : "r"(smem_local_ptr));

        smem_local_ptr += TILE_K * MMA_M * sizeof(half); // 移动到下一个tensor的共享内存地址
    }
}

template<int NumOfTensors, int N8>
__device__ __forceinline__ void B_FragLoadFromSharedToRegisters(uint32_t __restrict__ Registers[][4],
                                                                half* __restrict__ smem,
                                                                int warp_start_row,
                                                                int k_offset)
{
    //
    int      lane_id             = threadIdx.x % 32;
    int      i                   = lane_id % 8;
    uint32_t Mask_RowPermutation = i << 4;

    if (lane_id > 15)
        i += 8;
    int j = (lane_id % 16) >= 8 ? 1 : 0;
    //
    smem += TILE_K * (warp_start_row + i) + (k_offset + j * HALF_PER_128B);
    uint32_t __restrict__ smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    // Row Permutation to eliminating bank-conflict

    smem_local_ptr = smem_local_ptr ^ Mask_RowPermutation;
//
#pragma unroll
    for (int i = 0; i < NumOfTensors; i++) {
        // if(N8)
        //  asm volatile ("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        //              : "=r"(Registers[i][0]), "=r"(Registers[i][1])
        //              : "r"(smem_local_ptr));
        // else
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(Registers[i][0]), "=r"(Registers[i][1]), "=r"(Registers[i][2]), "=r"(Registers[i][3])
                     : "r"(smem_local_ptr));

        smem_local_ptr += TILE_K * MMA_N * sizeof(half);
    }
}

//注意：mma指令中，A和C是row-major，B是column-major
__device__ __forceinline__ void
MMA_FP16_M16N8K16(uint32_t __restrict__ c[], uint32_t __restrict__* a, uint32_t __restrict__* b)
{
    // C=A*B+C。累加（必须累加，因为是外积）。
    // A矩阵大小为16*16=256个half，每个线程有4个寄存器，由一个warp的所有寄存器组成；
    // B矩阵大小为16*8=128个half，每个线程有2个寄存器，由一个warp的所有寄存器组成；
    // C矩阵大小为16*8=128个fp32，每个线程有4个寄存器，由一个warp的所有寄存器组成.
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                 "{ %0, %1, %2, %3},"
                 "{ %4, %5, %6, %7 },"
                 "{ %8, %9 },"
                 "{ %10, %11, %12, %13 };"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]),
                   "r"(a[1]),
                   "r"(a[2]),
                   "r"(a[3]),
                   "r"(b[0]),
                   "r"(b[1]),  /////////////// for column-major B
                   "r"(c[0]),
                   "r"(c[1]),
                   "r"(c[2]),
                   "r"(c[3]));
}