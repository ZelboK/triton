add_proton_library(ProtonProfiler
	Cupti/CuptiPCSampling.cpp
	Cupti/CuptiProfiler.cpp
	Rocm/RocprofilerProfiler.cpp
	Instrumentation/InstrumentationProfiler.cpp
	Instrumentation/CudaRuntime.cpp
	Instrumentation/HipRuntime.cpp
	Instrumentation/Metadata.cpp
)

# Standalone preload library for rocprofiler-sdk interception
# This library can be LD_PRELOADed to enable profiling
add_library(proton_preload SHARED Rocm/RocprofilerPreload.cpp)

target_include_directories(proton_preload
  SYSTEM PRIVATE
    "${ROCPROFILER_SDK_INCLUDE_DIR}"
    "${ROCPROFILER_SDK_INCLUDE_DIR}/rocprofiler-sdk"
)

target_include_directories(proton_preload
  PRIVATE
    "${PROTON_SRC_DIR}/include"
)

# No Python dependencies - this must be preloadable
set_target_properties(proton_preload PROPERTIES
  OUTPUT_NAME "proton_preload"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../python/triton/_C"
)

target_link_libraries(proton_preload PRIVATE dl)
