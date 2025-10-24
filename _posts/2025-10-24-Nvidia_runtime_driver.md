---
layout: post
title: CUDA Compilation and the Runtime vs. Driver API
subtitle: Ncidia GPU
gh-repo: daattali/beautiful-jekyll
gh-badge: [star, fork, follow]
categories: [Technology, GPU, CUDA]
tags: [NVIDIA, CUDA, GPU, Driver, Runtime]
comments: true
mathjax: true
author: Mingyuan Wang
---

When you write a CUDA program like the one below, it's easy to take the compilation and execution for granted. But under the hood, several layers of software - especially the **CUDA Runtime** and **NVIDIA driver** - work together to make GPU acceleration possible. Let's walk through what happens when you compile and run a simple CUDA program, and clarify the roles of the Runtime and Driver APIs.


File: `runtime_example.cu`
```cpp
// runtime_example.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void add_one(float* a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] += 1.0f;
    }
}

int main() {
    const int N = 256;
    const size_t size = N * sizeof(float);

    // Host data
    std::vector<float> h_a(N, 1.0f);

    // Device memory
    float* d_a;
    cudaMalloc(&d_a, size);
    cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);

    // Launch kernel
    add_one<<<1, 256>>>(d_a, N);

    // Copy back
    cudaMemcpy(h_a.data(), d_a, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);

    // Check result
    std::cout << "Result[0] = " << h_a[0] << std::endl;  // should be 2.0
    return 0;
}
```
This program:
- Allocates an array on the GPU
- Launches a kernel that adds 1.0 to each element
- copies the result back to the CPU

Now, let's see how it goes from source code to execution.


---
# Step1: Compilation with NVCC
The NVIDIA CUDA Compiler (NVCC) operates in two stages:
(Host means CPU and device means GPU)
1. Device Code compilation
- The `global` function `add_one` is compiled into PTX(Parallel Thread Execution) intermediate code, or directly into SASS(GPU machine code) for a specific GPU architecture(for example: sm80).By default, nvcc will embed both PTX (for backward compatibility) and SASS (for binary targeting the specific architecture) into the fatbin.

- This code is embedded into the final executable as a fat binary.

2. Host Code Compilation
- The  `main()` function and calls like `cudaMalloc`, `cudaMemcpy`, and the kernel launch syntax (`<<<>>>`) are host-side C++ code.
- `nvcc` translates the `<<<>>>` syntax into calls to the CUDA Runtime API(e.g. `cudaLaunchKernel`).
- The host code is then passed to a standard C++ compiler(like `g++` or `clang++`) for final compilation.

You typically compile with:

```Bash
nvcc -o runtime_example runtime_example.cu
```

# Step2: Execution - Where Runtime meets Driver?

When you run `./runtime_example`, the following happens:
- The CUDA Runtime library(libcudart.so in linux) is loaded.
- Runtime API calls(cudaMalloc, cudaMemcpy, etc.) are executed.
- Under the hood, the Runtime API calls the lower-level CUDA driver API (e.g., `cuMemAlloc`, `cuMemcpy`, `cuLaunchKernel`).
- The driver API communicates directly with the NVIDIA kernel driver (`nvidia.ko` on Linux) to:
    - Allocate GPU memory
    - Submit work to the GPU command queue
    - Synchronize execution
 
 #### 🔑Key Insight: The Runtime API is a high-level, easy-to-use wrapper around the driver API. The driver API is more verbose but offers finer control (e.g., explicit context management, module loading).

 In our example, we use the Runtime API—which is why the code is concise and readable. If we don't use the CUDA Runtime API, we must use lower-level CUDA driver API (`#include <cuda.h>`). This gives you more control—but requires significantly more boilerplate.
```cpp
//(...)
cuInit(0);
cuDeviceGet(&dev, 0);
cuCtxCreate(&ctx, 0, dev);
cuModuleLoad(&mod, "kernel.ptx");
cuModuleGetFunction(&func, mod, "add_one");
void* args[] = {&d_a, &N};
cuLaunchKernel(func, 1,1,1, 256,1,1, 0, 0, args, 0);
//(...)
```

The driver API requires you to handle:

- Explicit initialization: `cuInit()`
- Context management: `cuCtxCreate()` / `cuCtxDestroy()`
- Module loading: You must compile your kernel to PTX or CUBIN separately (e.g.,using nvcc -ptx add_one.cu) and load it via `cuModuleLoad()`
- Manual memory copies: `cuMemcpyHtoD`, `cuMemcpyDtoH`

In contrast, the Runtime API:

- Automatically initializes and manages context
- Embeds PTX/CUBIN into the executable at compile time
- Lets you call kernels with natural C++ syntax (`<<<>>>`)
- Handles argument marshaling for you.

So while the driver API offers maximum flexibility, the Runtime API is far more developer-friendly—which is why almost all CUDA applications and frameworks (like PyTorch) use it under the hood.

![Nvdia Software Stack](/assets/img/blog_20251024/blog_1.png){: .mx-auto.d-block :}

Above is the Nvidia Software Stack. Do you understand it now?😊In summary, the CUDA Runtime API abstracts away the complexity of GPU context and module management, making it ideal for most developers. The Driver API, while more verbose, is essential for system-level libraries that require fine-grained control. 
