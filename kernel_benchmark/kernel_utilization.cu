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

// #define USE_CUBLAS
#define USE_FLASH_LLM
// #define USE_SPUTNIK
// #define USE_CUSPARSE
// #define USE_SPARTA

#include "./spmm_test_utils.h"
#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <stdio.h>

#ifdef USE_FLASH_LLM
#include "SpMM_API.cuh"
#endif
//

// ITERATION wrongly used in SPMM

int main(int argc, char** argv)
{
    if (argc != 6) {
        printf("Wrong Inputs! Correct input format: ./spmm_test M K N Sparsity SplitK\n");
        return -1;
    }
    int M_GLOBAL                    = atoi(argv[1]);
    int K_GLOBAL                    = atoi(argv[2]);
    int N_GLOBAL                    = atoi(argv[3]);
    int MATRIX_A_PRUNING_PERCENTAGE = atoi(argv[4]);
    int SPLIT_K                     = atoi(argv[5]);
    //
    // printf("M: %d N: %d K: %d\n", M_GLOBAL, N_GLOBAL, K_GLOBAL);
    //
    cublasStatus_t cublas_status;
    // cusparseStatus_t  cusparse_status;
    // cudaError_t       cuda_error;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Host memory
    half* A_h            = NULL;  // row major
    half* B_h            = NULL;  // col major
    half* B_Transposed_h = NULL;  // row major
    // Device memory
    half* A            = NULL;
    half* B            = NULL;
    half* B_Transposed = NULL;
    //
    A_h            = (half*)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    B_h            = (half*)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    B_Transposed_h = (half*)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    if (A_h == NULL || B_h == NULL || B_Transposed_h == NULL) {
        printf("Error in CPU Malloc!\n");
        exit(-1);
    }
    cudaMalloc(reinterpret_cast<void**>(&A), sizeof(half) * M_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&B), sizeof(half) * N_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&B_Transposed), sizeof(half) * N_GLOBAL * K_GLOBAL);
    checkLastCudaError(__LINE__);
    if (A == NULL || B == NULL || B_Transposed == NULL) {
        printf("Error in cudaMalloc!\n");
        exit(-1);
    }
    //
    init_host_matrices(A_h, B_h, M_GLOBAL, K_GLOBAL, N_GLOBAL, MATRIX_A_PRUNING_PERCENTAGE);
    for (int i = 0; i < K_GLOBAL; i++)
        for (int j = 0; j < N_GLOBAL; j++)
            B_Transposed_h[i * N_GLOBAL + j] = B_h[i + j * K_GLOBAL];
    //
    // printf("Preparing dense data for GPU...\n");
    cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B_Transposed, B_Transposed_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    checkLastCudaError(__LINE__);

#ifdef USE_FLASH_LLM
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // SpMM_WithSplitK
    // printf("Preparing Compressed A matrix for GPU kernel: MM_Sparse_TC...\n");
    half* D_SpMM = NULL;
    cudaMalloc(reinterpret_cast<void**>(&D_SpMM), sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_SpMM == NULL) {
        printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemset(D_SpMM, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);
    uint32_t* NZWeights_CPU   = NULL;
    int*      TileOffsets_CPU = NULL;
    int       NumOffsets = InitSparseMatrixA_API(A_h, M_GLOBAL, N_GLOBAL, K_GLOBAL, &NZWeights_CPU, &TileOffsets_CPU);
    int       NNZ        = TileOffsets_CPU[NumOffsets - 1] * 4;  // VectorSize = 4
    // printf("NumOffsets: %d, NNZ: %d\n", NumOffsets, NNZ);
    //
    uint32_t* NZWeights_GPU   = NULL;
    int*      TileOffsets_GPU = NULL;
    cudaMalloc(&TileOffsets_GPU, sizeof(int) * NumOffsets);
    if (NNZ == 0)
        NNZ = 1;  // For 100% sparsity, NNZ = 0, malloc will return NULL
    cudaMalloc(&NZWeights_GPU, sizeof(uint32_t) * NNZ);
    if (TileOffsets_GPU == NULL || NZWeights_GPU == NULL) {
        printf("Error in malloc memory from device memory!\n");
        exit(-1);
    }
    cudaMemcpy(NZWeights_GPU, NZWeights_CPU, sizeof(uint32_t) * NNZ, cudaMemcpyHostToDevice);
    cudaMemcpy(TileOffsets_GPU, TileOffsets_CPU, sizeof(int) * NumOffsets, cudaMemcpyHostToDevice);
    ;
    free(TileOffsets_CPU);
    free(NZWeights_CPU);
    // printf("Done! Compressed A matrix for GPU kernel: MM_Sparse_TC.\n");
    //
    printf("Launching Flash-LLM...\n");
    int Split_K = SPLIT_K;
    // printf("Split_K = %d\n", Split_K);
    half* Reduction_Workspace = NULL;
    cudaMalloc(reinterpret_cast<void**>(&Reduction_Workspace), sizeof(half) * M_GLOBAL * N_GLOBAL * Split_K);
    if (Reduction_Workspace == NULL) {
        printf("Error in cudaMalloc\n");
        exit(-1);
    }
    //WARM_UP_ITERATION
    for (int i = 0; i < 5; i++)
        SpMM_SplitK_API(0,
                        A,
                        reinterpret_cast<uint4*>(NZWeights_GPU),
                        TileOffsets_GPU,
                        B,
                        D_SpMM,
                        M_GLOBAL,
                        N_GLOBAL,
                        K_GLOBAL,
                        Reduction_Workspace,
                        Split_K);
    cudaEventRecord(start);
    for (int i = 0; i < 5; i++)
        SpMM_SplitK_API(0,
                        A,
                        reinterpret_cast<uint4*>(NZWeights_GPU),
                        TileOffsets_GPU,
                        B,
                        D_SpMM,
                        M_GLOBAL,
                        N_GLOBAL,
                        K_GLOBAL,
                        Reduction_Workspace,
                        Split_K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    checkLastCudaError(__LINE__);
    //
    float milliseconds_SpMM = 0.0f;
    cudaEventElapsedTime(&milliseconds_SpMM, start, stop);
    milliseconds_SpMM = milliseconds_SpMM / BENCHMARK_ITERATION;
    float tflops_SpMM =
        static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_SpMM / 1000.))
        / 1e12;
    half* D_SpMM_h = NULL;  // col major
    D_SpMM_h       = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    cudaMemcpy(D_SpMM_h, D_SpMM, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);  // Col Major
    cudaFree(D_SpMM);
    cudaFree(NZWeights_GPU);
    cudaFree(TileOffsets_GPU);
    cudaFree(Reduction_Workspace);
#endif


    printf("******************************************Problem Size******************************************\n");
    printf("M: %d N: %d K: %d Pruning Rate: %d SplitK: %d\n",
           M_GLOBAL,
           N_GLOBAL,
           K_GLOBAL,
           MATRIX_A_PRUNING_PERCENTAGE,
           SPLIT_K);


    free(A_h);
    free(B_h);
    free(B_Transposed_h);
    cudaFree(A);
    cudaFree(B);
    cudaFree(B_Transposed);
    return 0;
}
