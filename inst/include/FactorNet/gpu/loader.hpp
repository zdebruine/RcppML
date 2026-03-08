/**
 * @file gpu/loader.hpp
 * @brief Runtime GPU library loader via dlopen/dlsym
 *
 * Provides typed function-pointer resolution for gpu_bridge.cu symbols
 * that are already loaded into the process by R's dyn.load().
 *
 * Architecture:
 *   R gpu_available() → dyn.load("RcppML_gpu.so")   (loads symbols)
 *   C++ gpu::resolve("rcppml_gpu_detect")            (finds symbols via dlsym)
 *   C++ bridge callers call through function pointers
 *
 * This header uses POSIX dlsym(RTLD_DEFAULT, ...) which finds symbols
 * in any shared library loaded into the current process address space.
 * Since R loads the GPU .so before calling into C++, the symbols are
 * available without knowing the library path.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

// Only use dlsym when NOT compiled with nvcc (CPU-only build)
#ifndef FACTORNET_HAS_GPU

#include <dlfcn.h>
#include <cstdio>
#include <string>

namespace FactorNet {
namespace gpu {

/**
 * @brief Resolve a symbol from the GPU shared library at runtime.
 *
 * Uses dlsym(RTLD_DEFAULT, ...) to search all loaded shared libraries.
 * R must have already called dyn.load() on the GPU .so for this to work.
 *
 * @tparam FnPtr  Function pointer type (e.g., void(*)(int*, double*))
 * @param  name   Symbol name (e.g., "rcppml_gpu_detect")
 * @return        Function pointer, or nullptr if symbol not found
 */
template<typename FnPtr>
FnPtr resolve(const char* name) {
    void* sym = dlsym(RTLD_DEFAULT, name);
    return reinterpret_cast<FnPtr>(sym);
}

/**
 * @brief Check if the GPU bridge library is loaded (symbols available)
 *
 * Tests for the presence of rcppml_gpu_detect, which is always
 * present in gpu_bridge.cu.
 */
inline bool bridge_available() {
    static int cached = -1;  // -1 = not checked
    if (cached >= 0) return cached != 0;
    void* sym = dlsym(RTLD_DEFAULT, "rcppml_gpu_detect");
    cached = (sym != nullptr) ? 1 : 0;
    return cached != 0;
}

/**
 * @brief Detect GPUs via the bridge library (dlsym path)
 *
 * Calls rcppml_gpu_detect from the already-loaded GPU .so.
 *
 * @param[out] gpu_count  Number of GPUs found
 * @param[out] total_mem  Total GPU memory in bytes (sum of all GPUs)
 * @return     true if detection succeeded and GPUs were found
 */
inline bool detect_gpus_via_bridge(int& gpu_count, size_t& total_mem) {
    using detect_fn_t = void(*)(int*, double*, double*, int*, int*);
    auto fn = resolve<detect_fn_t>("rcppml_gpu_detect");
    if (!fn) return false;

    int num_gpus = 0;
    double total_mem_mb[8] = {0}, free_mem_mb[8] = {0};
    int max_gpus = 8, status = -1;

    fn(&num_gpus, total_mem_mb, free_mem_mb, &max_gpus, &status);

    if (status != 0 || num_gpus <= 0) return false;

    gpu_count = num_gpus;
    total_mem = 0;
    for (int i = 0; i < num_gpus; ++i) {
        total_mem += static_cast<size_t>(total_mem_mb[i] * 1024.0 * 1024.0);
    }
    return true;
}

}  // namespace gpu
}  // namespace FactorNet

#endif  // !FACTORNET_HAS_GPU
