#ifndef TRITON_HIPBLAS_INSTANCE_H
#define TRITON_HIPBLAS_INSTANCE_H

#include "hipblas_types.h"
#include <dlfcn.h>
#include <stdexcept>
#include <string>

class HipBlasLtInstance {
  // Typedefs for hipblas functions
  typedef hipblasStatus_t (*hipblasLtCreate_t)(hipblasLtHandle_t *);
  typedef hipblasStatus_t (*hipblasLtDestroy_t)(hipblasLtHandle_t);
  typedef hipblasStatus_t (*hipblasLtMatmulDescCreate_t)(hipblasLtMatmulDesc_t *,
                                                       hipblasComputeType_t,
                                                       hipblasDataType_t);
  typedef hipblasStatus_t (*hipblasLtMatmulDescDestroy_t)(hipblasLtMatmulDesc_t);
  typedef hipblasStatus_t (*hipblasLtMatmulDescSetAttribute_t)(
      hipblasLtMatmulDesc_t, hipblasLtMatmulDescAttributes_t, const void *,
      size_t);
  typedef hipblasStatus_t (*hipblasLtMatrixLayoutCreate_t)(
      hipblasLtMatrixLayout_t *, hipblasDataType_t, uint64_t, uint64_t, int64_t);
  typedef hipblasStatus_t (*hipblasLtMatrixLayoutDestroy_t)(
      hipblasLtMatrixLayout_t);
  typedef hipblasStatus_t (*hipblasLtMatmulPreferenceCreate_t)(
      hipblasLtMatmulPreference_t *);
  typedef hipblasStatus_t (*hipblasLtMatmulPreferenceDestroy_t)(
      hipblasLtMatmulPreference_t);
  typedef hipblasStatus_t (*hipblasLtMatmulPreferenceSetAttribute_t)(
      hipblasLtMatmulPreference_t, hipblasLtMatmulPreferenceAttributes_t,
      const void *, size_t);
  typedef hipblasStatus_t (*hipblasLtMatmulAlgoGetHeuristic_t)(
      hipblasLtHandle_t, hipblasLtMatmulDesc_t, hipblasLtMatrixLayout_t,
      hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t,
      hipblasLtMatmulPreference_t, int, hipblasLtMatmulHeuristicResult_t *,
      int *);
  typedef hipblasStatus_t (*hipblasLtMatmul_t)(
      hipblasLtHandle_t, hipblasLtMatmulDesc_t, const void *, const void *,
      const hipblasLtMatrixLayout_t, const void *, const hipblasLtMatrixLayout_t,
      const void *, const void *, const hipblasLtMatrixLayout_t, void *,
      const hipblasLtMatrixLayout_t, const hipblasLtMatmulAlgo_t *, void *,
      size_t, cudaStream_t);

  static constexpr const char *name = "libhipblas.so"; // TODO: verify this is correct so name

  hipblasLtCreate_t hipblasLtCreate;
  hipblasLtDestroy_t hipblasLtDestroy;
  hipblasLtMatmulDescCreate_t hipblasLtMatmulDescCreate;
  hipblasLtMatmulDescDestroy_t hipblasLtMatmulDescDestroy;
  hipblasLtMatmulDescSetAttribute_t hipblasLtMatmulDescSetAttribute;
  hipblasLtMatrixLayoutCreate_t hipblasLtMatrixLayoutCreate;
  hipblasLtMatrixLayoutDestroy_t hipblasLtMatrixLayoutDestroy;
  hipblasLtMatmulPreferenceCreate_t hipblasLtMatmulPreferenceCreate;
  hipblasLtMatmulPreferenceDestroy_t hipblasLtMatmulPreferenceDestroy;
  hipblasLtMatmulPreferenceSetAttribute_t hipblasLtMatmulPreferenceSetAttribute;
  hipblasLtMatmulAlgoGetHeuristic_t hipblasLtMatmulAlgoGetHeuristic;
  hipblasLtMatmul_t hipblasLtMatmul;

  void *dylibHandle = nullptr;
  hipblasLtHandle_t ltHandle;

  void *workspace = nullptr;
  size_t workspaceSize = 0;

  hipblasLtMatmulPreference_t preference = NULL;

  void loadHipBlasDylib() {
    if (dylibHandle == nullptr) {
      // First reuse the existing handle
      dylibHandle = dlopen(name, RTLD_NOLOAD);
    }
    if (dylibHandle == nullptr) {
      // If not found, try to load it
      dylibHandle = dlopen(name, RTLD_LOCAL | RTLD_LAZY);
    }
    if (dylibHandle == nullptr) {
      throw std::runtime_error("Could not find `" + std::string(name) +
                               "`. Make sure it is in your "
                               "LD_LIBRARY_PATH.");
    }
    dlerror(); // Clear any existing error

    hipblasLtCreate = (hipblasLtCreate_t)dlsym(dylibHandle, "hipblasLtCreate");
    hipblasLtDestroy = (hipblasLtDestroy_t)dlsym(dylibHandle, "hipblasLtDestroy");
    cublasLtMatmulDescCreate = (cublasLtMatmulDescCreate_t)dlsym(
        dylibHandle, "hipblasLtMatmulDescCreate");
    hipblasLtMatmulDescDestroy = (hipblasLtMatmulDescDestroy_t)dlsym(
        dylibHandle, "hipblasLtMatmulDescDestroy");
    hipblasLtMatmulDescSetAttribute = (hipblasLtMatmulDescSetAttribute_t)dlsym(
        dylibHandle, "hipblasLtMatmulDescSetAttribute");
    hipblasLtMatrixLayoutCreate = (hipblasLtMatrixLayoutCreate_t)dlsym(
        dylibHandle, "hipblasLtMatrixLayoutCreate");
    hipblasLtMatrixLayoutDestroy = (hipblasLtMatrixLayoutDestroy_t)dlsym(
        dylibHandle, "hipblasLtMatrixLayoutDestroy");
    hipblasLtMatmulPreferenceCreate = (hipblasLtMatmulPreferenceCreate_t)dlsym(
        dylibHandle, "hipblasLtMatmulPreferenceCreate");
    hipblasLtMatmulPreferenceDestroy = (hipblasLtMatmulPreferenceDestroy_t)dlsym(
        dylibHandle, "hipblasLtMatmulPreferenceDestroy");
    hipblasLtMatmulPreferenceSetAttribute =
        (hipblasLtMatmulPreferenceSetAttribute_t)dlsym(
            dylibHandle, "hipblasLtMatmulPreferenceSetAttribute");
    hipblasLtMatmulAlgoGetHeuristic = (hipblasLtMatmulAlgoGetHeuristic_t)dlsym(
        dylibHandle, "hipblasLtMatmulAlgoGetHeuristic");
    hipblasLtMatmul = (hipblasLtMatmul_t)dlsym(dylibHandle, "hipblasLtMatmul");

    const char *dlsym_error = dlerror();
    if (dlsym_error) {
      throw std::runtime_error("Could not load symbol from `" +
                               std::string(name) +
                               "`: " + std::string(dlsym_error));
    }
  }

  void unloadHipBlasDylib() { dlclose(dylibHandle); }

  void successOrExit(hipblasStatus_t status) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("HIPBLAS Error: " + std::to_string(status) +
                               "\n");
    }
  }

  void gemm_impl(int m, int n, int k, uint64_t A, uint64_t B, uint64_t C,
                 uint64_t D, cudaDataType_t dtype, float alpha, float beta) {
    hipblasLtMatmulDesc_t matmulDesc = NULL;

    hipblasOperation_t transa = HIPBLAS_OP_T;
    hipblasOperation_t transb = HIPBLAS_OP_N;

    int8_t fastAccum = 1;

    hipblasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL,
                           Ddesc = NULL;

    int returnedResults = 0;
    hipblasLtMatmulHeuristicResult_t heuristicResult = {};

    // Select compute type. Use TF32 when inputs are FP32, otherwise default
    // FP32 accumulation.
    hipblasComputeType_t computeType = (dtype == CUDA_R_32F)
                                          ? CUBLAS_COMPUTE_32F_FAST_TF32
                                          : CUBLAS_COMPUTE_32F;
    successOrExit(
        hipblasLtMatmulDescCreate(&matmulDesc, computeType, CUDA_R_32F));
    successOrExit(hipblasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    successOrExit(hipblasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    if (dtype == CUDA_R_8F_E4M3) {
      successOrExit(hipblasLtMatmulDescSetAttribute(
          matmulDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccum,
          sizeof(fastAccum)));
    }

    auto c_dtype = dtype == CUDA_R_8F_E4M3 ? CUDA_R_16F : dtype;
    successOrExit(hipblasLtMatrixLayoutCreate(&Adesc, dtype, k, m, k));
    successOrExit(hipblasLtMatrixLayoutCreate(&Bdesc, dtype, k, n, k));
    successOrExit(hipblasLtMatrixLayoutCreate(&Cdesc, c_dtype, m, n, m));
    successOrExit(hipblasLtMatrixLayoutCreate(&Ddesc, dtype, m, n, m));

    successOrExit(hipblasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1,
        &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
      throw std::runtime_error(
          "No valid algorithm found by hipblasLtMatmulAlgoGetHeuristic");
    }

    successOrExit(hipblasLtMatmul(ltHandle, matmulDesc, &alpha, (void *)A, Adesc,
                                 (void *)B, Bdesc, &beta, (void *)C, Cdesc,
                                 (void *)D, Ddesc, &heuristicResult.algo,
                                 (void *)workspace, workspaceSize, 0));
    if (Ddesc)
      successOrExit(hipblasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc)
      successOrExit(hipblasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)
      successOrExit(hipblasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc)
      successOrExit(hipblasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc)
      successOrExit(hipblasLtMatmulDescDestroy(matmulDesc));
  }

public:
  HipBlasLtInstance(uint64_t workspace, size_t workspaceSize)
      : workspace((void *)workspace), workspaceSize(workspaceSize) {
    loadHipBlasDylib();
    hipblasLtCreate(&ltHandle);

    successOrExit(hipblasLtMatmulPreferenceCreate(&preference));
    successOrExit(hipblasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize,
        sizeof(workspaceSize)));
  }
  ~HipBlasLtInstance() {
    if (preference)
      successOrExit(hipblasLtMatmulPreferenceDestroy(preference));

    hipblasLtDestroy(ltHandle);
    unloadHipBlasDylib();
  }

  // TODO: Verify if this is sensical for AMD workflow
  // C = A * B
  // Matrix B needs to be transposed, while matrix A does not. The function
  // *will-not* transpose the matrices, so the caller is responsible for
  // ensuring that the matrices are in the correct format and have the correct
  // dimensions.
  void matmul(int m, int n, int k, uint64_t A, uint64_t B, uint64_t C,
              cudaDataType_t dtype) {
    // CUDA is column-major, while triton is row-major, therefore we need to
    // reverse the order of the matrices ( A * B = (B^T * A^T)^T ).
    gemm_impl(n, m, k, B, A, 0, C, dtype, 1.0f, 0.0f);
  }

  void gemm(int m, int n, int k, uint64_t A, uint64_t B, uint64_t C, uint64_t D,
            cudaDataType_t dtype, float alpha, float beta) {
    gemm_impl(n, m, k, B, A, C, D, dtype, alpha, beta);
  }
  
}
#endif // TRITON_HIPBLAS_INSTANCE_H
