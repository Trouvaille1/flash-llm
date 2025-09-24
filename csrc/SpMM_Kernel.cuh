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

#include "MatMulUtilities.cuh"
#include <vector>

//复制一个tile中的稀疏（非零）数据到寄存器
//总数据量为一个block（目的是把GPTR中的数据复制到reg中），用thread操作组成
template<typename TilingConfig, typename SparseKernelConfig>
__device__ __forceinline__ void SpMM_CopyFromGlobalToReg(uint32_t*    Registers_GlobalToShared1,  // 第一部分寄存器缓冲区
                                                         uint32_t*    NNZ_VECTOR_ThreadLocal1,    // 输出：第一部分每个线程的非零向量数量
                                                         const uint4* GlobalPTR1,                 // 第一部分全局内存指针
                                                         int          NNZ_VECTOR_ThisTile1,       // 第一部分此tile的非零向量总数
                                                         uint32_t*    Registers_GlobalToShared2,  // 第二部分寄存器缓冲区
                                                         uint32_t*    NNZ_VECTOR_ThreadLocal2,    // 第二部分每个线程的非零向量数量
                                                         const uint4* GlobalPTR2,                 // 第二部分全局内存指针
                                                         int          NNZ_VECTOR_ThisTile2)       // 第二部分此tile的非零向量总数
{
    // Load Global to registers
    // 从全局内存加载到寄存器
    int Num_NNZ_Vector1 = NNZ_VECTOR_ThisTile1 / (WARP_SIZE * TilingConfig::BLOCK_WARPS);  // 计算第一部分每个线程平均分配的非零向量数
    if (threadIdx.x < (NNZ_VECTOR_ThisTile1 % (WARP_SIZE * TilingConfig::BLOCK_WARPS)))   // 如果当前线程需要处理余数部分
        Num_NNZ_Vector1++;  // 增加一个向量的处理量
    *NNZ_VECTOR_ThreadLocal1 = Num_NNZ_Vector1;  // 存储第一部分线程本地非零向量数量.每个线程需要复制多少个vector到寄存器
    if (TilingConfig::TILE_M == 256) {  // 当tile大小为256时需要处理第二部分
        int Num_NNZ_Vector2 = NNZ_VECTOR_ThisTile2 / (WARP_SIZE * TilingConfig::BLOCK_WARPS);  // 计算第二部分每个线程平均分配的非零向量数
        if (threadIdx.x < (NNZ_VECTOR_ThisTile2 % (WARP_SIZE * TilingConfig::BLOCK_WARPS)))   // 如果当前线程需要处理余数部分
            Num_NNZ_Vector2++;  // 增加一个向量的处理量
        *NNZ_VECTOR_ThreadLocal2 = Num_NNZ_Vector2;  // 存储第二部分线程本地非零向量数量
    }
    //NOTE: 一个Block处理一个tile数据，当前tile需要处理的数据量个数为Max_NNZ_VECTOR_ThisTile
    int Max_NNZ_VECTOR_ThisTile =  // 计算两部分中较大的非零向量数量.每个线程需要复制多少个vector到寄存器
        (TilingConfig::TILE_M == 256) ? max(NNZ_VECTOR_ThisTile1, NNZ_VECTOR_ThisTile2) : NNZ_VECTOR_ThisTile1;
#pragma unroll
    for (int i = 0; i < SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / SparseKernelConfig::VECTOR_SIZE; i++) {  // 复制轮数
        //每轮处理GPTR中的一个元素(uint4)，复制到reg中
        int index = threadIdx.x + i * (WARP_SIZE * (TilingConfig::BLOCK_WARPS));  // 当前线程在第i次迭代中要处理的GlobalPTR数据索引（在一个tile中）
        if (index >= Max_NNZ_VECTOR_ThisTile)  // 如果索引超出范围
            break;  // 退出循环
        if (index < NNZ_VECTOR_ThisTile1  // 如果索引在第一部分范围内
            || TilingConfig::TILE_M != 256)  // if TILE_M!=256, not need to compare since we have break();
                                            // 如果TILE_M!=256，不需要比较因为已经有break
        {
            Registers_GlobalToShared1[i * 4 + 0] = GlobalPTR1[index].x;  // 复制uint4的x分量到寄存器
            Registers_GlobalToShared1[i * 4 + 1] = GlobalPTR1[index].y;  // 复制uint4的y分量到寄存器
            Registers_GlobalToShared1[i * 4 + 2] = GlobalPTR1[index].z;  // 复制uint4的z分量到寄存器
            Registers_GlobalToShared1[i * 4 + 3] = GlobalPTR1[index].w;  // 复制uint4的w分量到寄存器
        }
        if (TilingConfig::TILE_M == 256)  // 当tile大小为256时
            if (index < NNZ_VECTOR_ThisTile2) {  // 如果索引在第二部分范围内
                Registers_GlobalToShared2[i * 4 + 0] = GlobalPTR2[index].x;  // 复制uint4的x分量到第二部分寄存器
                Registers_GlobalToShared2[i * 4 + 1] = GlobalPTR2[index].y;  // 复制uint4的y分量到第二部分寄存器
                Registers_GlobalToShared2[i * 4 + 2] = GlobalPTR2[index].z;  // 复制uint4的z分量到第二部分寄存器
                Registers_GlobalToShared2[i * 4 + 3] = GlobalPTR2[index].w;  // 复制uint4的w分量到第二部分寄存器
            }
    }
}

// Only used for kernel pipeline analysis, to make sure the global load for sparse encoding is not optimied by NVCC, we
// have to store the data loaded from GMem stored in SMem
template<typename TilingConfig, typename SparseKernelConfig>
__device__ __forceinline__ void SpMM_CopyFromGlobalToShared(int          tid,
                                                            half*        smem,
                                                            uint32_t*    Registers_GlobalToShared1,
                                                            uint32_t*    NNZ_VECTOR_ThreadLocal1,
                                                            const uint4* GlobalPTR1,
                                                            int          NNZ_VECTOR_ThisTile1,
                                                            uint32_t*    Registers_GlobalToShared2,
                                                            uint32_t*    NNZ_VECTOR_ThreadLocal2,
                                                            const uint4* GlobalPTR2,
                                                            int          NNZ_VECTOR_ThisTile2)
{
    uint32_t*    smem_int_ptr = reinterpret_cast<uint32_t*>(smem);
    unsigned int tmp1         = 0;
    unsigned int tmp2         = 0;
    // Load Global to registers
    int Num_NNZ_Vector1 = NNZ_VECTOR_ThisTile1 / (WARP_SIZE * TilingConfig::BLOCK_WARPS);
    if (threadIdx.x < (NNZ_VECTOR_ThisTile1 % (WARP_SIZE * TilingConfig::BLOCK_WARPS)))
        Num_NNZ_Vector1++;
    *NNZ_VECTOR_ThreadLocal1 = Num_NNZ_Vector1;
    if (TilingConfig::TILE_M == 256) {
        int Num_NNZ_Vector2 = NNZ_VECTOR_ThisTile2 / (WARP_SIZE * TilingConfig::BLOCK_WARPS);
        if (threadIdx.x < (NNZ_VECTOR_ThisTile2 % (WARP_SIZE * TilingConfig::BLOCK_WARPS)))
            Num_NNZ_Vector2++;
        *NNZ_VECTOR_ThreadLocal2 = Num_NNZ_Vector2;
    }
    //
    int Max_NNZ_VECTOR_ThisTile =
        (TilingConfig::TILE_M == 256) ? max(NNZ_VECTOR_ThisTile1, NNZ_VECTOR_ThisTile2) : NNZ_VECTOR_ThisTile1;
#pragma unroll
    for (int i = 0; i < SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / SparseKernelConfig::VECTOR_SIZE; i++) {
        int index = threadIdx.x + i * (WARP_SIZE * (TilingConfig::BLOCK_WARPS));
        if (index >= Max_NNZ_VECTOR_ThisTile)
            break;
        if (index < NNZ_VECTOR_ThisTile1
            || TilingConfig::TILE_M != 256)  // if TILE_M!=256, not need to compare since we have break();
        {
            tmp1 = GlobalPTR1[index].x + GlobalPTR1[index].y + GlobalPTR1[index].z + GlobalPTR1[index].w;
        }
        if (TilingConfig::TILE_M == 256)
            if (index < NNZ_VECTOR_ThisTile2) {
                tmp2 = GlobalPTR2[index].x + GlobalPTR2[index].y + GlobalPTR2[index].z + GlobalPTR2[index].w;
            }
    }
    smem_int_ptr[tid] = tmp1 + tmp2;
}

// Init Shared Memory to 0
// 初始化共享内存为零
//数据量是一个block，用thread操作组成
template<typename TilingConfig>
__device__ __forceinline__ void SpMM_InitSharedMemory(half* __restrict__ SharedPTR)
{
    int lane_id = threadIdx.x % WARP_SIZE;  // 计算当前线程在warp内的lane ID
    int warp_id = threadIdx.x / WARP_SIZE;  // 计算当前线程所属的warp ID
    //定值assert使用static_assert
    static_assert(TilingConfig::TILE_M % TilingConfig::BLOCK_WARPS == 0,
                  "TILE_M must be an integer multiple to BLOCK_WARPS");
    //constexpr：编译时常量（用于模板参数计算），在编译时计算
    constexpr int RowsPerWarp = TilingConfig::TILE_M / TilingConfig::BLOCK_WARPS;  // 计算每个 warp负责的行数
    //定值assert使用static_assert
    static_assert(TILE_K == 64, "For now, TILE_K is assumed to be 64.\n");
                                // 目前假设TILE_K为64
    const int StartRowNum         = warp_id * RowsPerWarp;  // 计算当前warp负责的起始行号
    half*     SharedPTR_PerThread = SharedPTR + StartRowNum * TILE_K + HALF_PER_128B * lane_id;  // 计算当前线程负责的共享内存起始地址
    //
    static_assert(RowsPerWarp % (WARP_SIZE * HALF_PER_128B / TILE_K) == 0,
                  "RowsPerWarp%(WARP_SIZE*HALF_PER_128B/TILE_K) should be 0\n");
                  // RowsPerWarp必须能被(WARP_SIZE*HALF_PER_128B/TILE_K)整除
    constexpr int ITERATIONS_PER_THREAD = RowsPerWarp / (WARP_SIZE * HALF_PER_128B / TILE_K);  // 计算每个线程需要的迭代次数。这里用HALF_PER_128B是为了提高每个thread处理数据的个数
#pragma unroll
    for (int i = 0; i < ITERATIONS_PER_THREAD; i++) {  // 遍历每个线程分配的迭代次数
        cp_async_ignore_src<16>(SharedPTR_PerThread, (half*)NULL);  // 使用异步复制将NULL源复制到共享内存，实现清零
        SharedPTR_PerThread += WARP_SIZE * HALF_PER_128B;  // 移动到下一个处理位置
    }
}

//NOTE: 对应论文中的extract
//该函数需要处理的总数据量是每个thread自己的寄存器数组，长度为NUM_REG_FOR_SPARSE_KERNEL，所以不涉及block级别的数据处理
template<typename TilingConfig, typename SparseKernelConfig>
__device__ __forceinline__ void SpMM_DecompressFromRegisterToShared(half* __restrict__ SharedPTR1,         // 第一部分共享内存目标地址
                                                                    uint32_t* Registers_For_SparseTiles1,  // 第一部分寄存器中的压缩稀疏数据
                                                                    uint32_t  NNZ_ThreadLocal1,            // 第一部分当前线程的非零元素数量
                                                                    half* __restrict__ SharedPTR2,         // 第二部分共享内存目标地址
                                                                    uint32_t* Registers_For_SparseTiles2,  // 第二部分寄存器中的压缩稀疏数据
                                                                    uint32_t  NNZ_ThreadLocal2)            // 第二部分当前线程的非零元素数量
{
    int Max_NNZ_ThreadLocal =  // 计算两部分中较大的非零元素数量，用于循环边界控制
        (TilingConfig::TILE_M == 256) ? max(NNZ_ThreadLocal1, NNZ_ThreadLocal2) : NNZ_ThreadLocal1;
#pragma unroll
    for (int i = 0; i < SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / SparseKernelConfig::VECTOR_SIZE; i++) {  // 遍历当前线程分配的寄存器向量数量
        if (i >= Max_NNZ_ThreadLocal)  // 如果超出当前线程的数据范围
            break;  // 提前退出循环

        if (i < NNZ_ThreadLocal1  // 如果索引在第一部分范围内
            || (TilingConfig::TILE_M != 256))  // if TILE_M!=256, not need to compare since we have break();
                                              // 如果TILE_M!=256，不需要比较因为已经有break
#pragma unroll
            for (int j = 0; j < SparseKernelConfig::VECTOR_SIZE; j++) {  // 遍历向量中的每个元素
                half* half_ptr =  // 获取32位寄存器中前16位的指针（存储half值）
                    reinterpret_cast<half*>(&(Registers_For_SparseTiles1[i * SparseKernelConfig::VECTOR_SIZE + j]));
                short* short_ptr  = reinterpret_cast<short*>(half_ptr + 1);  // 获取32位寄存器中后16位的指针（存储索引）
                half   value      = *half_ptr;   // 读取压缩数据中的数值部分
                short  index      = *short_ptr;  // 读取压缩数据中的索引部分
                SharedPTR1[index] = value;       // 将数值写入共享内存的对应索引位置，实现解压
            }

        if (TilingConfig::TILE_M == 256)  // 当tile大小为256时需要处理第二部分
            if (i < NNZ_ThreadLocal2)  // 如果索引在第二部分范围内
#pragma unroll
                for (int j = 0; j < SparseKernelConfig::VECTOR_SIZE; j++) {  // 遍历向量中的每个元素
                    half* half_ptr =  // 获取32位寄存器中前16位的指针（存储half值）
                        reinterpret_cast<half*>(&(Registers_For_SparseTiles2[i * SparseKernelConfig::VECTOR_SIZE + j]));
                    short* short_ptr  = reinterpret_cast<short*>(half_ptr + 1);  // 获取32位寄存器中后16位的指针（存储索引）
                    half   value      = *half_ptr;   // 读取压缩数据中的数值部分
                    short  index      = *short_ptr;  // 读取压缩数据中的索引部分
                    SharedPTR2[index] = value;       // 将数值写入共享内存的对应索引位置，实现解压
                }
    }
}

//NOTE: 对应论文中的算法1
// NOTE: ！！！特别注意！！！每个线程都会执行函数内的所有子函数，但是必须要时刻牢记：每个线程只能完成每个子函数的一部分（与C++最大的不同），必须组合起来才能完成函数名对应的功能。所以，写代码的时候去写每个线程完成的子功能即可
//该kernel计算量是全局矩阵乘法，以block为单位，迭代计算
template<typename TilingConfig, typename SparseKernelConfig>
__global__ void SpMM_Kernel(const half*  A,
                            const uint4* Compressed_A,
                            const int*   TileOffsets,
                            const half*  B,
                            half*        Reduction_Workspace,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            int          Split_K)
{
    //blockIdx.x值域：[0,N/TN-1]
    //blockIdx.y值域：[0,M/TM*Split_K-1]
    const int BatchID     = blockIdx.y / (M_Global / TilingConfig::TILE_M); // 计算当前批次ID.
    const int IsLastBatch = (BatchID == (Split_K - 1)); // 判断是否为最后一个批次
    // 输出矩阵C的tile在一个batch内的坐标(x,y)
    // NOTE: 这里的变量x向下，变量y向右.xy是对于C矩阵来说，是指在一个batch内的C tile的坐标
    const int x           = blockIdx.x; // X方向的block索引，对应N维度
    const int y           = blockIdx.y % (M_Global / TilingConfig::TILE_M); // Y方向的block索引在批次内的相对位置，对应M维度
    //
    const int NumKBlock        = K_Global / TILE_K;  // assert (K_Global%TILE_K==0);
                                                     // K维度总的block数量，假设K_Global能被TILE_K整除
    const int AverageNumKBlock = (NumKBlock - 1) / Split_K + 1; // 每个SplitK批次平均分配的K block数量
    const int RoundedKBlock    = AverageNumKBlock * Split_K; // 向上取整后的总K block数量
    const int PaddingKBlock    = RoundedKBlock - NumKBlock; // 需要填充的K block数量
    int       NumIter          = 0; // 当前批次需要处理的迭代次数（当前批次有多少个block）
    if (IsLastBatch)
        NumIter = AverageNumKBlock - PaddingKBlock; // 最后一个批次减去填充的block数量
    else
        NumIter = AverageNumKBlock; // 非最后批次使用平均分配的数量
    //
    const int* TileOffsets_ThisBlock1 = nullptr; // 当前block第一部分的tile偏移数组指针
    const int* TileOffsets_ThisBlock2 = nullptr; // 当前block第二部分的tile偏移数组指针（用于TILE_M=256的情况）
    if (TilingConfig::TILE_M == 256) {
        TileOffsets_ThisBlock1 =
            TileOffsets + K_Global / TILE_K * y * 2
            + BatchID * AverageNumKBlock;  // Address for matrix A, taking SplitK into consideration
                                          // 矩阵A的地址，考虑SplitK的情况，第一部分
        TileOffsets_ThisBlock2 =
            TileOffsets + K_Global / TILE_K * (y * 2 + 1)
            + BatchID * AverageNumKBlock;  // Address for matrix A, taking SplitK into consideration
                                          // 矩阵A的地址，考虑SplitK的情况，第二部分
    } else {  // 非256情况下的地址计算
        //找到本block的tile偏移数组
        //tileOffsets是对于矩阵A来说的
        TileOffsets_ThisBlock1 = TileOffsets + K_Global / TILE_K * y + BatchID * AverageNumKBlock;  //对于矩阵Ablock的y行，BatchID * AverageNumKBlock列
        TileOffsets_ThisBlock2 = TileOffsets_ThisBlock1;  // otherwise will cause problem when passing
                                                          // TileOffsets_ThisBlock2[0] to SpMM_CopyFromGlobalToReg()
                                                          // 否则传递TileOffsets_ThisBlock2[0]给SpMM_CopyFromGlobalToReg()时会出问题
    }
    //
    uint32_t Registers_GlobalToShared[SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL]; // 用于存储稀疏数据的中间寄存器数组。64
    uint32_t NNZ_ThreadLocal1 = 0; // 线程本地的非零元素数量（第一部分）
    uint32_t NNZ_ThreadLocal2 = 0; // 线程本地的非零元素数量（第二部分）
    //动态分配smem。为了不影响Occupancy，通过cudaFuncSetAttribute设置共享内存大小
    extern __shared__ __align__(128) half smem[];  // at least be 128 Bytes aligned
                                                   // 至少128字节对齐的共享内存
    // Warp and lane identification.
    // Warp和lane的识别
    const unsigned int warpId       = threadIdx.x / WARP_SIZE; // 当前线程所属的warp ID
    const int          Tile_Start_M = y * TilingConfig::TILE_M; // 在一个batch内，当前tile在M维度的起始位置
    const int          Tile_Start_N = x * TilingConfig::TILE_N; // 在一个batch内，当前tile在N维度的起始位置
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Compute a grid of C matrix tiles in each warp.
    // 在每个warp中计算C矩阵tile的网格
    int Warp_i         = warpId / TilingConfig::BLOCK_COL_WARPS; // warp在行方向的索引
    int Warp_j         = warpId % TilingConfig::BLOCK_COL_WARPS; // warp在列方向的索引
    int warp_start_row = WARP_ROW_TENSORS * MMA_M * Warp_i; // warp在当前block内的起始行（在C tile中）
    int warp_start_col = TilingConfig::WARP_COL_TENSORS * MMA_N * Warp_j; // warp在当前block内的起始列（在C tile中）
    //寄存器分配 (每个thread)
    //第一维是WARP_ROW_TENSORS * 2，因为A矩阵需要两个缓冲区；第二维是4，是为了ldmatrix.x4指令。a和b都用x4指令加载
    uint32_t __restrict__ a[WARP_ROW_TENSORS * 2][4]; // 存储A矩阵数据的寄存器数组
    uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS * 2][4]; // 存储B矩阵数据的寄存器数组
    // copying B tile from GlobalMemory to SharedMemory
    // 从全局内存复制B tile到共享内存
    const half* BTileGlobalPTR =
        B + Tile_Start_N * K_Global
        + BatchID * AverageNumKBlock * TILE_K;  // Address for matrix B, taking SplitK into consideration
                                               // 矩阵B的地址，考虑SplitK的情况
    //
    int NNZ_ThisTile1 = TileOffsets_ThisBlock1[1] - TileOffsets_ThisBlock1[0]; // 第一个tile的非零元素数量
    int NNZ_ThisTile2 = 0; // 第二个tile的非零元素数量初始化为0
    if (TilingConfig::TILE_M == 256)
        NNZ_ThisTile2 = TileOffsets_ThisBlock2[1] - TileOffsets_ThisBlock2[0]; // 当TILE_M=256时计算第二个tile的非零元素数量
    // printf("NNZ_ThisTile: %d ", NNZ_ThisTile);
    // NOTE: 对应论文中的gmem2reg（数据量为一个block，用thread操作组成）
    SpMM_CopyFromGlobalToReg<TilingConfig, SparseKernelConfig>(Registers_GlobalToShared,
                                                               &NNZ_ThreadLocal1,
                                                               Compressed_A + TileOffsets_ThisBlock1[0],
                                                               NNZ_ThisTile1,
                                                               Registers_GlobalToShared
                                                                   + SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / 2,
                                                               &NNZ_ThreadLocal2,
                                                               Compressed_A + TileOffsets_ThisBlock2[0],
                                                               NNZ_ThisTile2); // 从全局内存复制压缩的A矩阵数据到寄存器
    // NOTE: 对应论文中的rst_smem（数据量为一个block，用thread操作组成）
    SpMM_InitSharedMemory<TilingConfig>(smem); // 初始化共享内存
    cp_async_group_commit(); // 提交异步复制组
    // NOTE: 对应论文中的ld_dense.从gmem复制到smem
    CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
        smem + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global); // 复制B矩阵tile从全局内存到共享内存
    cp_async_group_commit(); // 提交异步复制组
    // Initilazing C Matrix to Zeros
    // 将C矩阵初始化为零
    float c[WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16]; // warp级别C tile矩阵的寄存器数组
    for (int i = 0; i < WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS; i++)
        for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
            c[i][j] = 0.0f; // 初始化C矩阵为0
    // NOTE: 对应论文中的barrier:initSharedMem()
    cp_async_wait_group<1>(); // 等待异步复制组完成，保留1个正在进行的组
    __syncthreads(); // 线程块同步
    // NOTE: 对应论文中的extract
    SpMM_DecompressFromRegisterToShared<TilingConfig, SparseKernelConfig>(
        smem,
        Registers_GlobalToShared,
        NNZ_ThreadLocal1,
        smem + TilingConfig::TILE_M * TILE_K / 2,
        Registers_GlobalToShared + SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / 2,
        NNZ_ThreadLocal2); // 从寄存器解压稀疏数据到共享内存
    // NOTE: 对应论文中的barrier:copyGlobal2Shared()
    cp_async_wait_group<0>(); // 等待所有异步复制组完成
    __syncthreads(); // 线程块同步
    // Prefetch to reduce stall_long_sb
    // 预取以减少stall_long_sb
    int StartIndex_SparseTiles_Prefetch1 = TileOffsets_ThisBlock1[0 + 1]; // 第一部分稀疏tile的预取起始索引
    int NNZ_ThisTile_Prefetch1           = TileOffsets_ThisBlock1[0 + 2] - TileOffsets_ThisBlock1[0 + 1]; // 第一部分预取tile的非零元素数量
    int StartIndex_SparseTiles_Prefetch2 = 0; // 第二部分稀疏tile的预取起始索引初始化为0
    int NNZ_ThisTile_Prefetch2           = 0; // 第二部分预取tile的非零元素数量初始化为0
    if (TilingConfig::TILE_M == 256) {
        StartIndex_SparseTiles_Prefetch2 = TileOffsets_ThisBlock2[0 + 1]; // 当TILE_M=256时设置第二部分预取起始索引
        NNZ_ThisTile_Prefetch2           = TileOffsets_ThisBlock2[0 + 2] - TileOffsets_ThisBlock2[0 + 1]; // 当TILE_M=256时计算第二部分预取非零元素数量
    }
// Debug
// printf("NNZ_ThisTile_Prefetch: %d ", NNZ_ThisTile_Prefetch);
//
// Go through the global K dimension by a fixed step at a time.
// 以固定步长遍历全局K维度
// write buffer[1] first, read buffer[0] first
// 先写buffer[1]，先读buffer[0]
#pragma unroll(1)
    for (int tile_id_k = 0; tile_id_k < NumIter; tile_id_k++) { // 遍历K维度的每个tile
        //以block为粒度，迭代计算
        // Using the previous prefetched value
        // 使用之前预取的值
        int StartIndex_SparseTiles1 = StartIndex_SparseTiles_Prefetch1; // 第一部分稀疏tile的起始索引
        int NNZ_ThisTile1           = NNZ_ThisTile_Prefetch1; // 第一部分tile的非零元素数量
        int StartIndex_SparseTiles2 = 0; // 第二部分稀疏tile的起始索引初始化为0
        int NNZ_ThisTile2           = 0; // 第二部分tile的非零元素数量初始化为0
        if (TilingConfig::TILE_M == 256) {
            StartIndex_SparseTiles2 = StartIndex_SparseTiles_Prefetch2; // 当TILE_M=256时设置第二部分起始索引
            NNZ_ThisTile2           = NNZ_ThisTile_Prefetch2; // 当TILE_M=256时设置第二部分非零元素数量
        }
        //
        StartIndex_SparseTiles_Prefetch1 = TileOffsets_ThisBlock1[tile_id_k + 1 + 1]; // 预取下一个第一部分tile的起始索引
        NNZ_ThisTile_Prefetch1 = TileOffsets_ThisBlock1[tile_id_k + 1 + 2] - TileOffsets_ThisBlock1[tile_id_k + 1 + 1]; // 预取下一个第一部分tile的非零元素数量
        if (TilingConfig::TILE_M == 256) {
            StartIndex_SparseTiles_Prefetch2 = TileOffsets_ThisBlock2[tile_id_k + 1 + 1]; // 当TILE_M=256时预取下一个第二部分tile的起始索引
            NNZ_ThisTile_Prefetch2 =
                TileOffsets_ThisBlock2[tile_id_k + 1 + 2] - TileOffsets_ThisBlock2[tile_id_k + 1 + 1]; // 当TILE_M=256时预取下一个第二部分tile的非零元素数量
        }
        // copying B tile from GlobalMemory to SharedMemory
        // 从全局内存复制B tile到共享内存
        BTileGlobalPTR = B + Tile_Start_N * K_Global + BatchID * AverageNumKBlock * TILE_K + ((tile_id_k + 1) * TILE_K); // 更新B矩阵tile的全局指针
        // double buffer
        // 双缓冲
        half* __restrict__ smem_write_PTR = smem; // 共享内存写指针
        half* __restrict__ smem_read_PTR  = smem; // 共享内存读指针
        smem_write_PTR = smem + ((tile_id_k + 1) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N); // 设置写缓冲区指针
        smem_read_PTR  = smem + ((tile_id_k) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N); // 设置读缓冲区指针
        //
        bool GlobalCopy = (tile_id_k + 1) < NumIter; // 判断是否需要进行全局复制
        // NOTE: 对应论文中的rst_smem
        SpMM_InitSharedMemory<TilingConfig>(smem_write_PTR); // 初始化写缓冲区的共享内存
        cp_async_group_commit(); // 提交异步复制组
        // NOTE: 对应论文中的gmem2reg
        SpMM_CopyFromGlobalToReg<TilingConfig, SparseKernelConfig>(
            Registers_GlobalToShared,
            &NNZ_ThreadLocal1,
            Compressed_A + StartIndex_SparseTiles1,
            NNZ_ThisTile1,
            Registers_GlobalToShared + SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / 2,
            &NNZ_ThreadLocal2,
            Compressed_A + StartIndex_SparseTiles2,
            NNZ_ThisTile2); // 从全局内存复制压缩的A矩阵数据到寄存器

        // Copying B Tile
        // 复制B Tile
        // NOTE: 对应论文中的ld_dense.从gmem复制到smem
        CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
            smem_write_PTR + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global, GlobalCopy); // 复制B矩阵tile到共享内存
        cp_async_group_commit(); // 提交异步复制组

        // only used for kernel pipeline analysis
        // 仅用于内核流水线分析
        // SpMM_CopyFromGlobalToShared<TilingConfig, SparseKernelConfig>
        //               ( threadIdx.x,
        //                 smem_write_PTR,
        //                 Registers_GlobalToShared,
        //                 &NNZ_ThreadLocal1,
        //                 Compressed_A+StartIndex_SparseTiles1,
        //                 NNZ_ThisTile1,
        //                 Registers_GlobalToShared+SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL/2,
        //                 &NNZ_ThreadLocal2,
        //                 Compressed_A+StartIndex_SparseTiles2,
        //                 NNZ_ThisTile2);

        PipelinedCoreComputations<TilingConfig>(c, a, b, smem_read_PTR, warp_start_row, warp_start_col); // 执行流水线核心计算
        //
        // NOTE: 对应论文中的barrier:initSharedMem()
        cp_async_wait_group<1>(); // 等待异步复制组完成，保留1个正在进行的组
        __syncthreads();  // Sync to ensure the completion of stage 2, but the asyncopy of Tile_B may not finished yet
                         // 同步以确保阶段2的完成，但Tile_B的异步复制可能尚未完成
        // NOTE: 对应论文中的extract
        SpMM_DecompressFromRegisterToShared<TilingConfig, SparseKernelConfig>(
            smem_write_PTR,
            Registers_GlobalToShared,
            NNZ_ThreadLocal1,
            smem_write_PTR + TilingConfig::TILE_M * TILE_K / 2,
            Registers_GlobalToShared + SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / 2,
            NNZ_ThreadLocal2); // 从寄存器解压稀疏数据到共享内存
        // NOTE: 对应论文中的barrier:copyGlobal2Shared()
        cp_async_wait_group<0>();  // Sync to ensure the completion of Loading B to shared memory
                                  // 同步以确保B矩阵加载到共享内存的完成
        __syncthreads(); // 线程块同步
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Store the C fragments to shared memory.
    // 将C片段存储到共享内存
    float(*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C] =
        reinterpret_cast<float(*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C]>(smem); // 将共享内存重新解释为C片段数组
    StoreToSharedMemoryFromRegister<TilingConfig>(smem_CFrag, c); // 以col major从寄存器存储到共享内存
    __syncthreads(); // 线程块同步
    // Now that shared memory contains all the D tiles, stream them to global memory.
    // 现在共享内存包含了所有的D tiles，将它们流式传输到全局内存
    half* BlockGlobalPTR =
        Reduction_Workspace + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global; // 当前block在全局reduction workspace中的指针
#pragma unroll
    for (int i = warpId; i < TilingConfig::TILE_N2; i += TilingConfig::BLOCK_WARPS)  // i-th column
                                                                                    // 第i列
#pragma unroll
        for (int j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M; j += WARP_SIZE)  // j-th row
                                                                                         // 第j行
            BlockGlobalPTR[j + i * M_Global] = __float2half_rn((*(smem_CFrag + i))[j]); // 将float类型的结果转换为half类型并存储到全局内存
}

/*
  // Debug: Input Sanity Check
  if(false)
  {
    if( (blockIdx.x==0) && (blockIdx.y==0) && (blockIdx.z==0) && (threadIdx.x==0) && (threadIdx.y==0) &&
  (threadIdx.z==0) )
    {
      printf("SpMM_SplitK_Kernel() debugging...\n");
      printf("TILE_M: %d\n", TilingConfig::TILE_M);
      printf("(K_Global/Split_K/TILE_K): %d  (M_Global/TilingConfig::TILE_M): %d\n", (K_Global/Split_K/TILE_K),
  (M_Global/TilingConfig::TILE_M)); int NumOffsets = (K_Global/Split_K/TILE_K)*(M_Global/TilingConfig::TILE_M) + 2; int
  tmp = 0; for(int i=0; i<NumOffsets; i++)
      {
        tmp += TileOffsets[i];
        printf("TileOffsets[%d] = %d.\n", i, TileOffsets[i]);
      }
      printf("Array TileOffsets do has %d Items, sum is %d.\n", NumOffsets, tmp);
    }
    //return;
  }
*/

/*
// Dense baseline: load-as-dense compute-as-dense
 template<typename TilingConfig, typename SparseKernelConfig>
__global__ void SpMM_Kernel(const half *A, const uint4* Compressed_A, const int *TileOffsets,
                                  const half *B,
                                  half* Reduction_Workspace,
                                  const int M_Global, const int N_Global, const int K_Global,
                                  int Split_K
                                  )
{
  //
  const int BatchID = blockIdx.y / (M_Global/TilingConfig::TILE_M);
  const int IsLastBatch = (BatchID == (Split_K-1) );
  const int x = blockIdx.x;
  const int y = blockIdx.y % (M_Global/TilingConfig::TILE_M);
  //
  const int NumKBlock = K_Global/TILE_K;    //assert (K_Global%TILE_K==0);
  const int AverageNumKBlock = (NumKBlock-1)/Split_K + 1;
  const int RoundedKBlock = AverageNumKBlock * Split_K;
  const int PaddingKBlock = RoundedKBlock - NumKBlock;
  int NumIter = 0;
  if(IsLastBatch)
    NumIter = AverageNumKBlock - PaddingKBlock;
  else
    NumIter = AverageNumKBlock;
  //
  extern __shared__ __align__(128) half smem[];   // at least be 128 Bytes aligned
  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const int Tile_Start_M = y * TilingConfig::TILE_M;
  const int Tile_Start_N = x * TilingConfig::TILE_N;
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Compute a grid of C matrix tiles in each warp.
  int Warp_i = warpId / TilingConfig::BLOCK_COL_WARPS;
  int Warp_j = warpId % TilingConfig::BLOCK_COL_WARPS;
  int warp_start_row = WARP_ROW_TENSORS * MMA_M * Warp_i;
  int warp_start_col = TilingConfig::WARP_COL_TENSORS * MMA_N * Warp_j;
  uint32_t __restrict__ a[WARP_ROW_TENSORS*2][4];
  uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS*2][4];
  // copying A & B tile from GlobalMemory to SharedMemory
  const half *ATileGlobalPTR = A + Tile_Start_M * K_Global + BatchID * AverageNumKBlock * TILE_K;
  const half *BTileGlobalPTR = B + Tile_Start_N * K_Global + BatchID * AverageNumKBlock * TILE_K;          // Address
for matrix B, taking SplitK into consideration CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_M, TilingConfig>
                                  (smem,
                                  ATileGlobalPTR,
                                  K_Global);
  CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>
                                  (smem + TilingConfig::TILE_M*TILE_K,
                                  BTileGlobalPTR,
                                  K_Global);
  // Initilazing C Matrix to Zeros
  float c[WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16];
  for(int i=0; i<WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS; i++)
    for(int j=0; j<REG_PER_C_TENSOR_16_16; j++)
      c[i][j] = 0.0f;
  //
  cp_async_wait_all();
  __syncthreads();
  // Go through the global K dimension by a fixed step at a time.
  // write buffer[1] first, read buffer[0] first
  #pragma unroll(1)
  for (int tile_id_k = 0; tile_id_k < NumIter; tile_id_k++)
  {
    // copying A & B tile from GlobalMemory to SharedMemory
    ATileGlobalPTR = A + Tile_Start_M * K_Global + BatchID * AverageNumKBlock * TILE_K + ((tile_id_k+1)*TILE_K);
    BTileGlobalPTR = B + Tile_Start_N * K_Global + BatchID * AverageNumKBlock * TILE_K + ((tile_id_k+1)*TILE_K);
    // double buffer
    half* __restrict__ smem_write_PTR = smem;
    half* __restrict__ smem_read_PTR = smem;
    smem_write_PTR = smem + ( (tile_id_k+1) % 2 )
                            * (TilingConfig::TILE_M*TILE_K + TILE_K*TilingConfig::TILE_N);
    smem_read_PTR  = smem + ( (tile_id_k) % 2 )
                            * (TilingConfig::TILE_M*TILE_K + TILE_K*TilingConfig::TILE_N);
    //
    bool GlobalCopy = (tile_id_k+1) < NumIter;
    //
    CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_M, TilingConfig>
                                    (smem_write_PTR,
                                    ATileGlobalPTR,
                                    K_Global,
                                    GlobalCopy);
    CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>
                                    (smem_write_PTR + TilingConfig::TILE_M*TILE_K,
                                    BTileGlobalPTR,
                                    K_Global,
                                    GlobalCopy);
    //
    PipelinedCoreComputations<TilingConfig>(c, a, b, smem_read_PTR, warp_start_row, warp_start_col);

    cp_async_wait_all();
    __syncthreads();
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Store the C fragments to shared memory.
  float (*smem_CFrag) [TilingConfig::TILE_M+PADDING_SHARED_MEM_FOR_C] =
        reinterpret_cast <float (*)[TilingConfig::TILE_M+PADDING_SHARED_MEM_FOR_C]> (smem);
  StoreToSharedMemoryFromRegister<TilingConfig>(smem_CFrag, c);
  __syncthreads();
  // Now that shared memory contains all the D tiles, stream them to global memory.
  half* BlockGlobalPTR = Reduction_Workspace + BatchID*(M_Global*N_Global) + Tile_Start_M + Tile_Start_N*M_Global;
  #pragma unroll
  for(int i=warpId; i<TilingConfig::TILE_N2; i+=TilingConfig::BLOCK_WARPS)    // i-th column
    #pragma unroll
    for(int j=threadIdx.x%WARP_SIZE; j<TilingConfig::TILE_M; j+=WARP_SIZE) // j-th row
      BlockGlobalPTR[j+i*M_Global] = __float2half_rn((*(smem_CFrag+i))[j]);
}
*/