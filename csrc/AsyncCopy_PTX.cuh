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
// Extended from CUTLASS's source code

template<int SizeInBytes>
__device__ __forceinline__ void cp_async(half* smem_ptr, const half* global_ptr, bool pred_guard = true)
{
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported");
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),
                 "r"(smem_int_ptr),
                 "l"(global_ptr),
                 "n"(SizeInBytes));
}

// only used for kernel pipeline analysis
template<int SizeInBytes>
__device__ __forceinline__ void cp_async_test_only(half* smem_ptr, const half* global_ptr, bool pred_guard = true)
{
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported");
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global [%1], [%2], %3, 0;\n"
                 "}\n" ::"r"((int)pred_guard),
                 "r"(smem_int_ptr),
                 "l"(global_ptr),
                 "n"(SizeInBytes));
}

template<int SizeInBytes>
__device__ __forceinline__ void cp_async_ignore_src(half* smem_ptr, half* global_ptr)  // 异步复制函数，忽略源数据用于清零操作
{
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported");
    // 静态断言：只支持4、8、16字节的复制大小
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);  // 将通用指针转换为共享内存地址
    asm volatile("{ \n"  // 内联汇编开始
                 "  cp.async.cg.shared.global [%0], [%1], %2, 0;\n"  // PTX异步复制指令：从全局内存复制到共享内存
                 "}\n" ::"r"(smem_int_ptr),  // %0：共享内存地址（寄存器约束）
                 "l"(global_ptr),            // %1：全局内存地址（64位地址约束）
                 "n"(SizeInBytes));          // %2：复制字节数（编译时常量约束）
}

/// Establishes an ordering w.r.t previously issued cp.async instructions. Does not block.
/// 建立与之前发出的cp.async指令的顺序关系。不会阻塞。
__device__ __forceinline__ void cp_async_group_commit()  // 提交异步复制组，建立操作顺序
{
    asm volatile("cp.async.commit_group;\n" ::);  // PTX指令：提交当前异步复制组。将现有的操作打包为一个组
}

/// Blocks until all but <N> previous cp.async.commit_group operations have committed.
/// 阻塞等待，直到除了最新N个之外的所有cp.async.commit_group操作都已提交完成。
template<int N>
__device__ __forceinline__ void cp_async_wait_group()  // 等待异步复制组完成，N表示保留最新的N个组继续执行
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));  // PTX指令：等待异步复制组，N为编译时常量参数
}

/// Blocks until all previous cp.async.commit_group operations have committed.
// cp.async.wait_all is equivalent to :
// cp.async.commit_group;
// cp.async.wait_group 0;
__device__ __forceinline__ void cp_async_wait_all()
{
    asm volatile("cp.async.wait_all;\n" ::);
}