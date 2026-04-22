#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_WEBGPU_NAME "WebGPU"

// Needed for examples in ggml
GGML_BACKEND_API ggml_backend_t ggml_backend_webgpu_init(void);

// WebLLM browser readback helpers. These expose a request-based API so callers
// can poll GPU->CPU tensor readback progress without blocking in a single call.
GGML_BACKEND_API int32_t ggml_backend_webgpu_tensor_get_async_begin(
    const struct ggml_tensor * tensor,
    size_t offset,
    size_t size);
GGML_BACKEND_API int32_t ggml_backend_webgpu_tensor_get_async_poll(int32_t request_id);
GGML_BACKEND_API void ggml_backend_webgpu_tensor_get_async_finish(
    int32_t request_id,
    void * data,
    size_t size);
GGML_BACKEND_API void ggml_backend_webgpu_tensor_get_async_cancel(int32_t request_id);

// WebLLM graph profiling summary for the most recent graph compute.
GGML_BACKEND_API void ggml_backend_webgpu_set_graph_profiling_enabled(int32_t enabled);
GGML_BACKEND_API int32_t ggml_backend_webgpu_last_graph_profile_valid(void);
GGML_BACKEND_API int32_t ggml_backend_webgpu_last_graph_profile_breakdown_available(void);
GGML_BACKEND_API double ggml_backend_webgpu_last_graph_profile_total_ms(void);
GGML_BACKEND_API double ggml_backend_webgpu_last_graph_profile_matmul_ms(void);
GGML_BACKEND_API double ggml_backend_webgpu_last_graph_profile_attention_ms(void);
GGML_BACKEND_API double ggml_backend_webgpu_last_graph_profile_encode_overhead_ms(void);
GGML_BACKEND_API int32_t ggml_backend_webgpu_last_graph_profile_dispatch_count(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_webgpu_reg(void);

#ifdef  __cplusplus
}
#endif
