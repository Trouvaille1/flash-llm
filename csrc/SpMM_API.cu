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

#include "./MatMulUtilities.cuh"
#include "./Reduction_Kernel.cuh"
#include "./SpMM_Kernel.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/**
 * Launch the Sparse Matrix Multiplication (SpMM) kernel with Split-K optimization
 * 启动具有Split-K优化的稀疏矩阵乘法(SpMM)内核
 * 
 * Template Parameters:
 * 模板参数:
 * @tparam TilingConfig    Configuration for GPU tile dimensions and memory layout
 *                        GPU tile维度和内存布局的配置
 * @tparam SparseKernelConfig  Configuration specific to sparse matrix operations
 *                            稀疏矩阵操作的特定配置
 * 
 * Parameters:
 * 参数:
 * @param stream           CUDA stream for asynchronous execution
 *                        用于异步执行的CUDA流
 * @param A               Dense input matrix A (M x K) in half precision
 *                        dense输入矩阵A (M x K)，半精度
 * @param Compressed_A    Compressed sparse representation of matrix A using uint4 format
 *                        使用uint4格式的矩阵A的压缩稀疏表示
 * @param TileOffsets     Offset indices for tile-based sparse matrix access
 *                        基于tile的稀疏矩阵访问的偏移索引
 * @param B               Dense input matrix B (K x N) in half precision
 *                        dense输入矩阵B (K x N)，半精度
 * @param Reduction_Workspace  Temporary workspace for Split-K reduction results
 *                            Split-K归约结果的临时工作空间
 * @param M_Global        Global M dimension of the matrix multiplication
 *                        矩阵乘法的全局M维度
 * @param N_Global        Global N dimension of the matrix multiplication
 *                        矩阵乘法的全局N维度
 * @param K_Global        Global K dimension of the matrix multiplication
 *                        矩阵乘法的全局K维度
 * @param Split_K         Number of K-dimension splits for improved parallelism
 *                        K维度分割数，用于提高并行性
 */
template<typename TilingConfig, typename SparseKernelConfig>
static void SpMM_SplitK_Kernel_Ex(cudaStream_t stream,
                                  const half*  A,
                                  const uint4* Compressed_A,
                                  const int*   TileOffsets,
                                  const half*  B,
                                  half*        Reduction_Workspace,
                                  const int    M_Global,
                                  const int    N_Global,
                                  const int    K_Global,
                                  int          Split_K)
{
    // Calculate required shared memory size for optimal performance
    // 计算最佳性能所需的共享内存大小
    //A矩阵Tile和B矩阵Tile*2（双缓冲）
    /*
    为什么取最大值?
    因为共享内存在不同计算阶段被复用：
    1. 数据加载阶段：需要第一部分大小存储A、Btile。
    2. 结果存储阶段：需要第二部分大小存储Ctile累加结果。
     */
    static int SHMEM_SZ = max((TilingConfig::TILE_M * TILE_K + TilingConfig::TILE_N * TILE_K) * sizeof(half) * 2,
                              (TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C) * TilingConfig::TILE_N * sizeof(float));
                              
    
    // Set maximum dynamic shared memory size for the kernel
    // 为内核设置最大动态共享内存大小
    cudaFuncSetAttribute(
        SpMM_Kernel<TilingConfig, SparseKernelConfig>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    // printf("Max shared memory size: %d B\n", SHMEM_SZ);
    
    
    // Calculate grid dimensions for kernel launch
    // 计算内核启动的网格维度
    int dimN =
        max(N_Global / TilingConfig::TILE_N, 1);  // max(N_Global/TilingConfig::TILE_N,1) used when N=8, TILE_N=16
                                                  // 当N=8, TILE_N=16时使用max(N_Global/TilingConfig::TILE_N,1)
    int  dimM = M_Global * Split_K / TilingConfig::TILE_M;
    dim3 GridDim(dimN, dimM, 1);  // Grid Size is increased due to SplitK for higher SM occupancy
                                  // 由于SplitK，网格大小增加以提高SM占用率
    
    // Configure thread block dimensions
    // 配置线程块维度(一维)
    dim3 BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1);
    
    // Launch the sparse matrix multiplication kernel with configured parameters
    // 使用配置的参数启动稀疏矩阵乘法内核
    SpMM_Kernel<TilingConfig, SparseKernelConfig><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
        A, Compressed_A, TileOffsets, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);
}

/*
half* Reduction_Workspace:  1. Requiring an extra memory space in device memory for un-reducted intermediate output
tensors
                            2. Reduction_Workspace_Size = max( Split_K * M_Global * N_Global ) * sizeof(fp16)
int Split_K:                Split K dimension into Split_K Parts
*/
cudaError_t SpMM_SplitK_API(cudaStream_t stream,
                            const half*  A,
                            const uint4* Compressed_A,//向量化访问。以uint4类型访问
                            const int*   TileOffsets,
                            const half*  B,
                            half*        C,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            half*        Reduction_Workspace,  // Identical workspace for all SpMM kernel launches
                            int          Split_K)
{
#ifdef DEBUG_MODE
    printf(
        "SpMM_API.cu->SpMM_SplitK_API():  M: %d, N: %d, K: %d, SplitK: %d \n", M_Global, N_Global, K_Global, Split_K);
    assert(K_Global % TILE_K == 0);
    assert(M_Global % 256 == 0);
#endif
    half* SpMM_SplitK_OutputPTR;
    if (Split_K == 1)
        SpMM_SplitK_OutputPTR = C;
    else
        SpMM_SplitK_OutputPTR = Reduction_Workspace;
    // Batched SpMM
    switch (N_Global) {
        case 8:
            SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 1, 1>, SparseKernelConfig<64>>(
                stream, A, Compressed_A, TileOffsets, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        case 16:
            SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 1>, SparseKernelConfig<64>>(
                stream, A, Compressed_A, TileOffsets, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        case 32:
            SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 2>, SparseKernelConfig<64>>(
                stream, A, Compressed_A, TileOffsets, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        case 64:
            // return SpMM_SplitK_Kernel_Ex< TilingConfig<4, 1, 4>, SparseKernelConfig<64> >
            SpMM_SplitK_Kernel_Ex<TilingConfig<2, 2, 2>, SparseKernelConfig<32>>(
                stream, A, Compressed_A, TileOffsets, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        case 128:
            SpMM_SplitK_Kernel_Ex<TilingConfig<2, 2, 4>, SparseKernelConfig<32>>(
                stream, A, Compressed_A, TileOffsets, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        default:
            if (N_Global % 128 == 0)
                SpMM_SplitK_Kernel_Ex<TilingConfig<2, 2, 4>, SparseKernelConfig<32>>(stream,
                                                                                     A,
                                                                                     Compressed_A,
                                                                                     TileOffsets,
                                                                                     B,
                                                                                     SpMM_SplitK_OutputPTR,
                                                                                     M_Global,
                                                                                     N_Global,
                                                                                     K_Global,
                                                                                     Split_K);
            else {
                printf("MM_Sparse_API Error: Unsupported N dimension %d!\n", N_Global);
                return cudaErrorUnknown;
            }
            break;
    }
    //
    cudaError_t Error = cudaGetLastError();
    if (Error != cudaSuccess)
        return Error;

    if (Split_K == 1)
        return Error;
    dim3 GridDim((M_Global * N_Global) / 256, 1, 1);
    dim3 BlockDim(WARP_SIZE, 1, 1);
    SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace, M_Global, N_Global, Split_K);
    return cudaGetLastError();
}

static int BankID_Minimum(std::vector<unsigned int> ItemsInBank[])
{
    int ID           = 0;
    int MinItemCount = ItemsInBank[0].size();
    for (int i = 1; i < 32; i++) {
        if (ItemsInBank[i].size() < MinItemCount) {
            ID           = i;
            MinItemCount = ItemsInBank[i].size();
        }
    }
    return ID;
}

static int BankID_Maximum(std::vector<unsigned int> ItemsInBank[])
{
    int ID           = 0;
    int MaxItemCount = ItemsInBank[0].size();
    for (int i = 1; i < 32; i++) {
        if (ItemsInBank[i].size() > MaxItemCount) {
            ID           = i;
            MaxItemCount = ItemsInBank[i].size();
        }
    }
    return ID;
}

/*
return: Number of Element in array TileOffsets
Note: TileOffsets[return-1] = NNZ / SparseKernelConfig::VECTOR_SIZE    (SparseKernelConfig::VECTOR_SIZE = 4)
返回：TileOffsets数组中的元素数量
注意：TileOffsets[return-1] = NNZ / SparseKernelConfig::VECTOR_SIZE    (SparseKernelConfig::VECTOR_SIZE = 4)
*/
// template<typename TilingConfig, typename SparseKernelConfig>
// 初始化稀疏矩阵A的API函数，将稀疏矩阵压缩为特定格式以供GPU kernel使用
// NOTE: 对应论文4.3节
__host__ int InitSparseMatrixA_API(half*      A_h,              // 输入：主机内存中的半精度浮点数矩阵A
                                   int        M,                // 输入：矩阵A的行数
                                   int        N,                // 输入：矩阵的列数（此函数中未使用）
                                   int        K,                // 输入：矩阵A的列数
                                   uint32_t** Compressed_A,     // 输出：CPU指针，指向压缩后的矩阵A数据（这里不是二维数组的意思。实际上是用一维数组模拟二维数组，并且为了支持修改外部调用者的指针，所以用二级指针）
                                   int**      TileOffsets)      // 输出：CPU指针，指向tile偏移量数组（可以支持修改外部调用者的指针，所以用二级指针）
{
    // Unified Sparse Fornat for different N, in our kernel, TILE_M=128 or 256
    // 为不同的N维度统一稀疏格式，在我们的kernel中，TILE_M=128或256
    const int TILE_M                       = 128;  // tile的行数
    const int VECTOR_SIZE                  = 4;    // 向量大小，用于对齐
    const int PADDING_SIZE_FOR_TILEOFFSETS = 2;    // TileOffsets数组的填充大小
#ifdef DEBUG_MODE
    printf("Weight Shuffle is Enabled\n");         // 权重重排已启用
#endif
    float ZERO_THRESHOLD = 0.0;                    // 零值阈值
    int   NumRow_offsets = M / TILE_M;             // 行方向的tile数量
    int   NumCol_offsets = K / TILE_K;             // 列方向的tile数量
    //
    // 计算原始矩阵中的非零元素数量
    int NNZ_Original = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            if (fabs(__half2float(A_h[i * K + j])) > ZERO_THRESHOLD)
                NNZ_Original++;
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ=%d, Pruning Ratio=%.2f\n",
           M,
           K,
           NNZ_Original,
           1.0f - static_cast<float>(NNZ_Original) / (M * K));
    // 矩阵A：M=%d K=%d，非零元素=%d，剪枝比例=%.2f
#endif
    //
    // 计算填充后的非零元素数量
    int  NNZ_AfterPadding   = 0;
    //用一维数组存储的二维数组。大小实际为(M/TILE_M) * (K/TILE_K)
    int* PaddingForEachTile = (int*)malloc(NumRow_offsets * NumCol_offsets * sizeof(int));  // 为每个tile分配填充数组
    if (!PaddingForEachTile) {
        printf("Error in InitSparseMatrixA line %d :malloc Error\n", __LINE__);
        // 在InitSparseMatrixA第%d行出错：内存分配错误
        exit(-1);
    }
    // 遍历每个tile计算需要的填充数量
    for (int i = 0; i < M / TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR = A_h + (i * TILE_M) * K + (j * TILE_K);  // 当前tile的指针
            int   TileNZCount    = 0;                                      // 当前tile的非零元素计数
            // 统计当前tile中的非零元素
            for (int m = 0; m < TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD)
                        TileNZCount++;
                }
            }
            // 计算为了对齐到VECTOR_SIZE需要的填充数量
            int NumPadding                           = (VECTOR_SIZE - (TileNZCount % VECTOR_SIZE)) % VECTOR_SIZE;
            PaddingForEachTile[i * (K / TILE_K) + j] = NumPadding;
            TileNZCount += NumPadding;
            NNZ_AfterPadding += TileNZCount;
        }
    }
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ_AfterPadding=%d, PruningRatio_AfterPadding=%.2f\n",
           M,
           K,
           NNZ_AfterPadding,
           1.0f - static_cast<float>(NNZ_AfterPadding) / (M * K));
    // 矩阵A：M=%d K=%d，填充后非零元素=%d，填充后剪枝比例=%.2f
#endif
    //
    // 分配压缩矩阵和tile偏移量数组的内存（对应论文中的NonZeros）
    *Compressed_A = (uint32_t*)malloc(NNZ_AfterPadding * sizeof(uint32_t));
    *TileOffsets  = (int*)malloc((NumRow_offsets * NumCol_offsets + PADDING_SIZE_FOR_TILEOFFSETS) * sizeof(int));
    if (*Compressed_A == NULL || *TileOffsets == NULL) {
        printf("InitSparseMatrixA: Error in malloc memory from host memory!\n");// InitSparseMatrixA：从主机内存分配内存时出错！
        exit(-1);
    }
    // Generating compressed format for A Matrix
    // 为矩阵A生成压缩格式
    assert(M % TILE_M == 0 && K % TILE_K == 0);    // 确保M和K能被tile大小整除
    int       TotalNZCount = 0;                     // 总非零元素计数
    uint32_t* Ptr_SubArray = *Compressed_A;        // 指向压缩数组的指针
    // NOTE: 对应论文中算法3.遍历每个tile进行压缩
    for (int i = 0; i < M / TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half*        CurrentTilePTR    = A_h + (i * TILE_M) * K + (j * TILE_K);  // 当前tile指针
            int          TileNZCount       = 0;                                      // 当前tile非零元素计数
            int          remainingPaddings = PaddingForEachTile[i * (K / TILE_K) + j];  // 剩余填充数量
            unsigned int Item              = 0;                                     // 用于存储压缩数据的变量
            // Processing each tile
            // 处理每个tile
            std::vector<unsigned int> ItemsInBank[32];  // 32个bank的数据项向量
            int                       ZeroPositionForBank[32];  // 每个bank的零位置记录
            for (int k = 0; k < 32; k++)
                ZeroPositionForBank[k] = -1;
            //
            // printf("Starting Processing Tile i:%d j:%d...\n", i, j);
            // 开始处理tile i:%d j:%d...
            // 遍历tile中的每个元素，提取非零元素，并记录每个bank中第一个零元素的位置
            for (int m = 0; m < TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    // Row permutation for bank-conflict-free shared memory layout
                    // 每一行分别做列置换以实现无bank冲突的共享内存布局
                    int      row            = m;
                    int      col            = n;
                    uint32_t mask           = (row % 8) << 3;          // 生成置换掩码
                    int      col_permutated = col ^ mask;              // 置换后的列索引
                    int      bank_smem      = (col_permutated / 2) % 32;  // 计算共享内存bank编号。两个half共享一个bank，所以除以2
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD) {
                        // 处理非零元素：将值和位置信息打包到32位整数中
                        half* half_ptr   = reinterpret_cast<half*>(&Item);      // 前16位存储值
                        *half_ptr        = value;
                        short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);  // 后16位存储位置
                        *short_ptr       = static_cast<short>(row * TILE_K + col_permutated);
                        ItemsInBank[bank_smem].push_back(Item);  // 添加到对应bank
                        //
                        TileNZCount++;
                    }
                    else {
                        // 记录每个bank中第一个零元素的位置，用于后续填充
                        if (ZeroPositionForBank[bank_smem] == -1)
                            ZeroPositionForBank[bank_smem] = row * TILE_K + col_permutated;
                    }
                }
            }
            //
            // printf("Starting Weight Padding...\n");
            // 开始权重填充...
            // 进行权重填充以满足向量对齐要求
            for (int k = 0; k < remainingPaddings; k++) {
                int BankID = BankID_Minimum(ItemsInBank);  // 找到元素最少的bank
                assert(BankID >= 0 && BankID < 32);
                int ZeroPosition = ZeroPositionForBank[BankID];  // 获取该bank的零位置
                assert(ZeroPosition != -1);
                //
                // 创建填充的零元素
                half* half_ptr   = reinterpret_cast<half*>(&Item);
                *half_ptr        = __float2half_rn(0.0f);  // 设置为零值
                short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                *short_ptr       = static_cast<short>(ZeroPosition);
                ItemsInBank[BankID].push_back(Item);  // 添加到最少的bank
                //
                TileNZCount++;
            }
            /*
            if(i==0 && j==0)
            {
              printf("For tile i:%d j:%d\n",i,j);
              for(int h=0; h<32; h++)
                printf("%ld ", ItemsInBank[h].size());
              printf("\n");
            }
            */
            //
            // printf("Starting Weight Shuffle...\n");
            // 开始权重重排...
            // 进行权重重排以优化内存访问模式
            //这里的32=WARP_SIZE.每个线程处理的数据量为uint4
            std::vector<unsigned int> MainPart[32];  // 主要部分：32个向量，对应warp中的32个线程
            std::vector<unsigned int> TailPart[32];  // 尾部部分：处理不足一个完整warp的剩余数据
            int                       TileVectorCount = TileNZCount / VECTOR_SIZE;  // 总向量数量（每4个元素一个向量）
            assert(TileNZCount % VECTOR_SIZE == 0);
            int Repeat_Vector   = TileVectorCount / WARP_SIZE;  // 完整warp轮次：32的倍数部分
            int Remained_Vector = TileVectorCount % WARP_SIZE;  // 剩余向量：不足32个的部分
            // Filing the TailPart
            // 填充尾部部分
            for (int v = 0; v < VECTOR_SIZE; v++) { //v no use
                for (int b = 0; b < Remained_Vector; b++) {
                    int BankID = BankID_Maximum(ItemsInBank);  // 找到元素最多的bank。负载均衡策略
                    Item       = ItemsInBank[BankID].back();   // 取出最后一个元素
                    ItemsInBank[BankID].pop_back();
                    TailPart[b].push_back(Item);  // 添加到尾部
                }
            }
            // Filing the MainPart
            // 填充主要部分
            // printf("Starting Filing the MainPart...\n");
            // 开始填充主要部分...
            for (int r = 0; r < Repeat_Vector; r++) {
                for (int v = 0; v < VECTOR_SIZE; v++) {
                    for (int b = 0; b < WARP_SIZE; b++) {
                        int BankID = BankID_Maximum(ItemsInBank);  // 找到元素最多的bank
                        Item       = ItemsInBank[BankID].back();   // 取出最后一个元素
                        ItemsInBank[BankID].pop_back();
                        MainPart[b].push_back(Item);  // 添加到主要部分
                    }
                }
            }
            // Writing to the Sub-Array
            // 写入子数组
            // printf("Starting Writing to the Sub-Array...\n");
            // 开始写入子数组...
            // 将主要部分的数据写入压缩数组
            for (int r = 0; r < Repeat_Vector; r++) {//完整warp轮次
                for (int v = 0; v < VECTOR_SIZE; v++) {//每个线程处理4个uint
                    for (int b = 0; b < 32; b++) { //1个warp中的32个线程
                        Item = MainPart[b].back();
                        MainPart[b].pop_back();
                        int V_Size                                     = VECTOR_SIZE;
                        Ptr_SubArray[r * V_Size * 32 + b * V_Size + v] = Item;
                    }
                }
            }
            Ptr_SubArray += Repeat_Vector * VECTOR_SIZE * WARP_SIZE;  // 移动指针
            // 将尾部部分的数据写入压缩数组（尾部部分只含一个不完整的warp，即一轮，所以省略了r的循环）
            for (int v = 0; v < VECTOR_SIZE; v++) {
                for (int b = 0; b < Remained_Vector; b++) {
                    Item = TailPart[b].back();
                    TailPart[b].pop_back();
                    Ptr_SubArray[b * VECTOR_SIZE + v] = Item;
                }
            }
            Ptr_SubArray += VECTOR_SIZE * Remained_Vector;  // 移动指针
            //
            TotalNZCount += TileNZCount;  // 累加总非零元素数
            (*TileOffsets)[i * K / TILE_K + j + 1] = TotalNZCount / VECTOR_SIZE;  // 设置tile偏移量
        }
    }
    //
    assert(TotalNZCount == NNZ_AfterPadding);  // 确保总计数正确
    (*TileOffsets)[0] = 0;  // 第一个偏移量为0
    (*TileOffsets)[(M / TILE_M) * (K / TILE_K) + 1] =
        TotalNZCount / VECTOR_SIZE;  // #define PADDING_SIZE_FOR_TILEOFFSETS 2  // (N+1 offsets) + 1 padding // adding
                                     // an empty tile at last
                                     // 在最后添加一个空tile
    //
    return (M / TILE_M) * (K / TILE_K) + 2;  // number of Elements in array TileOffsets
                                             // 返回TileOffsets数组中的元素数量
}

// A_h is host memory pointer, Compressed_A and TileOffsets are device memory pointers
__host__ int InitSparseMatrixA_API_NoReorder(half*      A_h,
                                             int        M,
                                             int        N,
                                             int        K,
                                             uint32_t** Compressed_A,  // CPU PTR
                                             int**      TileOffsets)        // CPU_PTR
{
    // Unified Sparse Fornat for different N, in our kernel, TILE_M=128 or 256
    const int TILE_M                       = 128;
    const int VECTOR_SIZE                  = 4;
    const int PADDING_SIZE_FOR_TILEOFFSETS = 2;
#ifdef DEBUG_MODE
    printf("Weight Shuffle is NOT Enabled\n");
#endif
    float ZERO_THRESHOLD = 0.0;
    int   NumRow_offsets = M / TILE_M;
    int   NumCol_offsets = K / TILE_K;
    //
    int NNZ_Original = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            if (fabs(__half2float(A_h[i * K + j])) > ZERO_THRESHOLD)
                NNZ_Original++;
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ=%d, Pruning Ratio=%.2f\n",
           M,
           K,
           NNZ_Original,
           1.0f - static_cast<float>(NNZ_Original) / (M * K));
#endif
    //
    int  NNZ_AfterPadding   = 0;
    int* PaddingForEachTile = (int*)malloc(NumRow_offsets * NumCol_offsets * sizeof(int));
    if (!PaddingForEachTile) {
        printf("Error in InitSparseMatrixA line %d :malloc Error\n", __LINE__);
        exit(-1);
    }
    for (int i = 0; i < M / TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR = A_h + (i * TILE_M) * K + (j * TILE_K);
            int   TileNZCount    = 0;
            for (int m = 0; m < TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD)
                        TileNZCount++;
                }
            }
            int NumPadding                           = (VECTOR_SIZE - (TileNZCount % VECTOR_SIZE)) % VECTOR_SIZE;
            PaddingForEachTile[i * (K / TILE_K) + j] = NumPadding;
            TileNZCount += NumPadding;
            NNZ_AfterPadding += TileNZCount;
        }
    }
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ_AfterPadding=%d, PruningRatio_AfterPadding=%.2f\n",
           M,
           K,
           NNZ_AfterPadding,
           1.0f - static_cast<float>(NNZ_AfterPadding) / (M * K));
#endif
    //
    *Compressed_A = (uint32_t*)malloc(NNZ_AfterPadding * sizeof(uint32_t));
    *TileOffsets  = (int*)malloc((NumRow_offsets * NumCol_offsets + PADDING_SIZE_FOR_TILEOFFSETS) * sizeof(int));
    if (*Compressed_A == NULL || *TileOffsets == NULL) {
        printf("InitSparseMatrixA: Error in malloc memory from host memory!\n");
        exit(-1);
    }
    // Generating compressed format for A Matrix
    assert(M % TILE_M == 0 && K % TILE_K == 0);
    int TotalNZCount = 0;
    for (int i = 0; i < M / TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR    = A_h + (i * TILE_M) * K + (j * TILE_K);
            int   TileNZCount       = 0;
            int   remainingPaddings = PaddingForEachTile[i * (K / TILE_K) + j];
            for (int m = 0; m < TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD) {
                        half* half_ptr   = reinterpret_cast<half*>(*Compressed_A + TotalNZCount);
                        *half_ptr        = value;
                        short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                        // Row permutation for bank-conflict-free shared memory layout
                        int      row            = m;
                        int      col            = n;
                        uint32_t mask           = (row % 8) << 3;
                        int      col_permutated = col ^ mask;
                        *short_ptr              = static_cast<short>(row * TILE_K + col_permutated);
                        //
                        TileNZCount++;
                        TotalNZCount++;
                    }
                    else {
                        if (remainingPaddings > 0) {
                            remainingPaddings--;
                            half* half_ptr   = reinterpret_cast<half*>(*Compressed_A + TotalNZCount);
                            *half_ptr        = value;  // zero
                            short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                            // Row permutation for bank-conflict-free shared memory layout
                            int      row            = m;
                            int      col            = n;
                            uint32_t mask           = (row % 8) << 3;
                            int      col_permutated = col ^ mask;
                            *short_ptr              = static_cast<short>(row * TILE_K + col_permutated);
                            //
                            TileNZCount++;
                            TotalNZCount++;
                        }
                    }
                }
            }
            //
            assert(TileNZCount % VECTOR_SIZE == 0);
            (*TileOffsets)[i * K / TILE_K + j + 1] = TotalNZCount / VECTOR_SIZE;
        }
    }
    assert(TotalNZCount == NNZ_AfterPadding);
    (*TileOffsets)[0] = 0;
    (*TileOffsets)[(M / TILE_M) * (K / TILE_K) + 1] =
        TotalNZCount / VECTOR_SIZE;  // #define PADDING_SIZE_FOR_TILEOFFSETS 2  // (N+1 offsets) + 1 padding // adding
                                     // an empty tile at last
    //

    //
    return (M / TILE_M) * (K / TILE_K) + 2;  // number of Elements in array TileOffsets
}

/*
input:    char* DenseMatrixFileName
          int   M
          int   N                   // N is used by void InitSparseMatrixA_API()
          int   K
          char* NZWeightsFileName
          char* TileOffsetsFileName
          char* OutputSizesFileName // NNZ -> NumOffsets
*/
extern "C" void GenSparseMatrixBinFile(char* DenseMatrixFileName,
                                       int   M,
                                       int   K,
                                       char* NZWeightsFileName,
                                       char* TileOffsetsFileName,
                                       char* OutputSizesFileName)
{
    std::vector<half> host_array(M * K);
    std::ifstream     in(DenseMatrixFileName, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        printf("file %s cannot be opened, loadDataArrayFromBin fails. \n", DenseMatrixFileName);
        exit(-1);
    }
    size_t loaded_data_size = sizeof(half) * M * K;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);
#ifdef DEBUG_MODE
    printf("Read %ld bytes from %s.\n", loaded_data_size, DenseMatrixFileName);
#endif
    in.read((char*)host_array.data(), loaded_data_size);
    size_t in_get_size = in.gcount();
    if (in_get_size != loaded_data_size) {
        printf("file %s only has %ld, but request %ld, loading DenseMatrix fails! \n",
               DenseMatrixFileName,
               in_get_size,
               loaded_data_size);
        exit(-1);
    }
    in.close();
    // Step 2: Dense to Sparse Transformation
    unsigned int* NZWeights_CPU   = nullptr;
    int*          TileOffsets_CPU = nullptr;
    int           NumOffsets      = InitSparseMatrixA_API(host_array.data(), M, 0, K, &NZWeights_CPU, &TileOffsets_CPU);
    int           NNZ             = TileOffsets_CPU[NumOffsets - 1] * 4;  // VectorSize = 4
    // Step 3: Write to FILE(OutputSizesFileName)
    //         Write to FILE(NZWeightsFileName), FILE(TileOffsetsFileName)
    std::ofstream out_SizesFile(OutputSizesFileName, std::ios::out | std::ios::binary);
    std::ofstream out_NZWeightsFile(NZWeightsFileName, std::ios::out | std::ios::binary);
    std::ofstream out_TileOffsetsFile(TileOffsetsFileName, std::ios::out | std::ios::binary);
    if (!out_SizesFile.is_open() || !out_NZWeightsFile.is_open() || !out_TileOffsetsFile.is_open()) {
        printf("GenSparseMatrixBinFile() ERROR: file %s, %s, or %s cannot be opened or creaetd. \n",
               OutputSizesFileName,
               NZWeightsFileName,
               TileOffsetsFileName);
        exit(-1);
    }
    //
    // out_SizesFile << NNZ << NumOffsets;
    out_SizesFile.write((char*)&NNZ, sizeof(int));
    out_SizesFile.write((char*)&NumOffsets, sizeof(int));
    out_SizesFile.close();
    out_NZWeightsFile.write((char*)NZWeights_CPU, sizeof(uint32_t) * NNZ);
    out_NZWeightsFile.close();
    out_TileOffsetsFile.write((char*)TileOffsets_CPU, sizeof(int) * NumOffsets);
    out_TileOffsetsFile.close();
}