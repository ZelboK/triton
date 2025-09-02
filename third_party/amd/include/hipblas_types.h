#ifndef TRITON_HIPBLAS_TYPES_H
#define TRITON_HIPBLAS_TYPES_H

#include <cstddef>
#include <cstdint>

// Forward declarations of hipBLAS types and functions.

/* HIPBLAS status type returns */
typedef enum {
  HIPBLAS_STATUS_SUCCESS = 0,
  HIPBLAS_STATUS_NOT_INITIALIZED = 1,
  HIPBLAS_STATUS_ALLOC_FAILED = 3,
  HIPBLAS_STATUS_INVALID_VALUE = 7,
  HIPBLAS_STATUS_ARCH_MISMATCH = 8,
  HIPBLAS_STATUS_MAPPING_ERROR = 11,
  HIPBLAS_STATUS_EXECUTION_FAILED = 13,
  HIPBLAS_STATUS_INTERNAL_ERROR = 14,
  HIPBLAS_STATUS_NOT_SUPPORTED = 15,
  HIPBLAS_STATUS_LICENSE_ERROR = 16
} hipblasStatus_t;

typedef enum {
  HIPBLAS_COMPUTE_16F = 64,          /* half - default */
  HIPBLAS_COMPUTE_16F_PEDANTIC = 65, /* half - pedantic */
  HIPBLAS_COMPUTE_32F = 68,          /* float - default */
  HIPBLAS_COMPUTE_32F_PEDANTIC = 69, /* float - pedantic */
  HIPBLAS_COMPUTE_32F_FAST_16F =
      74, /* float - fast, allows down-converting inputs to half or TF32 */
  HIPBLAS_COMPUTE_32F_FAST_16BF =
      75, /* float - fast, allows down-converting inputs to bfloat16 or TF32 */
  HIPBLAS_COMPUTE_32F_FAST_TF32 =
      77, /* float - fast, allows down-converting inputs to TF32 */
  HIPBLAS_COMPUTE_64F = 70,          /* double - default */
  HIPBLAS_COMPUTE_64F_PEDANTIC = 71, /* double - pedantic */
  HIPBLAS_COMPUTE_32I = 72,          /* signed 32-bit int - default */
  HIPBLAS_COMPUTE_32I_PEDANTIC = 73, /* signed 32-bit int - pedantic */
} hipblasComputeType_t;

typedef enum {
  HIPBLASLT_MATMUL_DESC_COMPUTE_TYPE = 0,
  HIPBLASLT_MATMUL_DESC_SCALE_TYPE = 1,
  HIPBLASLT_MATMUL_DESC_POINTER_MODE = 2,
  HIPBLASLT_MATMUL_DESC_TRANSA = 3,
  HIPBLASLT_MATMUL_DESC_TRANSB = 4,
  HIPBLASLT_MATMUL_DESC_TRANSC = 5,
  HIPBLASLT_MATMUL_DESC_FILL_MODE = 6,
  HIPBLASLT_MATMUL_DESC_EPILOGUE = 7,
  HIPBLASLT_MATMUL_DESC_BIAS_POINTER = 8,
  HIPBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE = 10,
  HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = 11,
  HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD = 12,
  HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE = 13,
  HIPBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE = 14,
  HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET = 15,
  HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER = 17,
  HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER = 18,
  HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER = 19,
  HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER = 20,
  HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER = 21,
  HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE = 22,
  HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER = 23,
  HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER = 24,
  HIPBLASLT_MATMUL_DESC_FAST_ACCUM = 25,
  HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = 26,
  HIPBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS = 27,
  HIPBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS = 28,
  HIPBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER = 29,
  HIPBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER = 30,
} hipblasLtMatmulDescAttributes_t;

typedef enum {
  HIPBLAS_OP_N = 0,
  HIPBLAS_OP_T = 1,
  HIPBLAS_OP_C = 2,
  HIPBLAS_OP_HERMITAN = 2, /* synonym if HIPBLAS_OP_C */
  HIPBLAS_OP_CONJG =
      3 /* conjugate, placeholder - not supported in the current release */
} hipblasOperation_t;

typedef enum {
  HIPBLASLT_MATMUL_PREF_SEARCH_MODE = 0,
  HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,
  HIPBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK = 3,
  HIPBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES = 5,
  HIPBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES = 6,
  HIPBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES = 7,
  HIPBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES = 8,
  HIPBLASLT_MATMUL_PREF_MAX_WAVES_COUNT = 9,
  HIPBLASLT_MATMUL_PREF_IMPL_MASK = 12,
} hipblasLtMatmulPreferenceAttributes_t;
typedef struct {
  uint64_t data[8];
} hipblasLtMatrixLayoutOpaque_t;
typedef hipblasLtMatrixLayoutOpaque_t *hipblasLtMatrixLayout_t;

typedef struct {
  uint64_t data[8];
} hipblasLtMatmulPreferenceOpaque_t;
typedef hipblasLtMatmulPreferenceOpaque_t *hipblasLtMatmulPreference_t;

typedef struct {
  uint64_t data[8];
} hipblasLtMatmulAlgo_t;

typedef struct {
  hipblasLtMatmulAlgo_t algo;
  size_t workspaceSize;
  hipblasStatus_t state;
  float wavesCount;
  int reserved[4];
} hipblasLtMatmulHeuristicResult_t;

typedef enum hipData_t {
  HIP_R_16F = 2,   /* real as a half */
  HIP_C_16F = 6,   /* complex as a pair of half numbers */
  HIP_R_16BF = 14, /* real as a nv_bfloat16 */
  HIP_C_16BF = 15, /* complex as a pair of nv_bfloat16 numbers */
  HIP_R_32F = 0,   /* real as a float */
  HIP_C_32F = 4,   /* complex as a pair of float numbers */
  HIP_R_64F = 1,   /* real as a double */
  HIP_C_64F = 5,   /* complex as a pair of double numbers */
  HIP_R_4I = 16,   /* real as a signed 4-bit int */
  HIP_C_4I = 17,   /* complex as a pair of signed 4-bit int numbers */
  HIP_R_4U = 18,   /* real as a unsigned 4-bit int */
  HIP_C_4U = 19,   /* complex as a pair of unsigned 4-bit int numbers */
  HIP_R_8I = 3,    /* real as a signed 8-bit int */
  HIP_C_8I = 7,    /* complex as a pair of signed 8-bit int numbers */
  HIP_R_8U = 8,    /* real as a unsigned 8-bit int */
  HIP_C_8U = 9,    /* complex as a pair of unsigned 8-bit int numbers */
  HIP_R_16I = 20,  /* real as a signed 16-bit int */
  HIP_C_16I = 21,  /* complex as a pair of signed 16-bit int numbers */
  HIP_R_16U = 22,  /* real as a unsigned 16-bit int */
  HIP_C_16U = 23,  /* complex as a pair of unsigned 16-bit int numbers */
  HIP_R_32I = 10,  /* real as a signed 32-bit int */
  HIP_C_32I = 11,  /* complex as a pair of signed 32-bit int numbers */
  HIP_R_32U = 12,  /* real as a unsigned 32-bit int */
  HIP_C_32U = 13,  /* complex as a pair of unsigned 32-bit int numbers */
  HIP_R_64I = 24,  /* real as a signed 64-bit int */
  HIP_C_64I = 25,  /* complex as a pair of signed 64-bit int numbers */
  HIP_R_64U = 26,  /* real as a unsigned 64-bit int */
  HIP_C_64U = 27,  /* complex as a pair of unsigned 64-bit int numbers */
  HIP_R_8F_E4M3_FNUZ = 28, /* real as a hip_fp8_e4m3_fnuz */
  HIP_R_8F_E5M2_FNUZ = 29, /* real as a hip_fp8_e5m2_fnuz */
} hipDataType;

struct hipblasContext;
typedef struct hipblasLtContext *hipblasLtHandle_t;
struct hipblasLtMatmulDescOpaque_t;
typedef hipblasLtMatmulDescOpaque_t *hipblasLtMatmulDesc_t;
struct hipStream_st;
typedef struct hipStream_st *hipStream_t;

#endif // TRITON_HIPBLAS_TYPES_H
