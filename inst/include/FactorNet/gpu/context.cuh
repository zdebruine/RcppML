/**
 * @file context.cuh
 * @brief Backward-compatible GPU context header
 *
 * Redirects to gpu_types.cuh which contains the actual GPUContext definition.
 * Provides a 'handle' alias for 'cublas' for legacy NMF GPU code.
 */
#pragma once
#include <FactorNet/gpu/types.cuh>
