// Thin preload wrapper for rocprofiler-sdk integration
//
// This library can be LD_PRELOADed to enable rocprofiler-sdk interception.
// It exports rocprofiler_configure which rocprofiler-sdk will find at startup.
// When called, it stores the client info and later forwards to libproton.so.
//
// Usage: LD_PRELOAD=/path/to/libproton_preload.so python script.py

#include <dlfcn.h>
#include <iostream>
#include <mutex>
#include <string>

#include "rocprofiler-sdk/fwd.h"
#include "rocprofiler-sdk/registration.h"

namespace {

// Storage for rocprofiler client info - will be used when libproton loads
struct PreloadState {
  std::mutex mutex;
  bool configured = false;
  uint32_t version = 0;
  std::string runtimeVersion;
  uint32_t priority = 0;
  rocprofiler_client_id_t *clientId = nullptr;
  rocprofiler_tool_configure_result_t *result = nullptr;
};

PreloadState &getPreloadState() {
  static PreloadState state;
  return state;
}

// Forward declaration of the real configure function in libproton.so
using ConfigureFunc =
    rocprofiler_tool_configure_result_t *(*)(uint32_t, const char *, uint32_t,
                                             rocprofiler_client_id_t *);

// Try to find and call the real rocprofiler_configure from libproton.so
rocprofiler_tool_configure_result_t *
tryForwardToProton(uint32_t version, const char *runtimeVersion,
                   uint32_t priority, rocprofiler_client_id_t *clientId) {
  // Try to find libproton.so - it might already be loaded
  void *protonLib = dlopen("libproton.so", RTLD_NOLOAD | RTLD_LAZY);
  if (!protonLib) {
    // Try common paths
    const char *paths[] = {"python/triton/_C/libproton.so",
                           "../python/triton/_C/libproton.so", nullptr};
    for (const char **p = paths; *p; ++p) {
      protonLib = dlopen(*p, RTLD_LAZY | RTLD_GLOBAL);
      if (protonLib)
        break;
    }
  }

  if (protonLib) {
    // Find the real rocprofiler_configure (it's in proton:: namespace but has C
    // linkage)
    auto realConfigure = reinterpret_cast<ConfigureFunc>(
        dlsym(protonLib, "rocprofiler_configure"));
    if (realConfigure) {
      return realConfigure(version, runtimeVersion, priority, clientId);
    }
  }

  return nullptr;
}

// Dummy init/fini functions for when libproton isn't available yet
int dummyInit(rocprofiler_client_finalize_t, void *) {
  std::cerr << "[PROTON PRELOAD] Tool init called but libproton not loaded yet"
            << std::endl;
  return 0;
}

void dummyFini(void *) {}

} // namespace

// This is the symbol that rocprofiler-sdk looks for at startup
extern "C" __attribute__((visibility("default")))
rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t version, const char *runtimeVersion,
                      uint32_t priority, rocprofiler_client_id_t *clientId) {
  std::cerr << "[PROTON PRELOAD] rocprofiler_configure called (version="
            << version << ", priority=" << priority << ")" << std::endl;

  auto &state = getPreloadState();
  std::lock_guard<std::mutex> lock(state.mutex);

  // Store the client info for later
  state.version = version;
  state.runtimeVersion = runtimeVersion ? runtimeVersion : "";
  state.priority = priority;
  state.clientId = clientId;
  state.configured = true;

  if (clientId) {
    clientId->name = "ProtonPreload";
  }

  // Try to forward to the real libproton.so if it's already loaded
  auto result = tryForwardToProton(version, runtimeVersion, priority, clientId);
  if (result) {
    std::cerr << "[PROTON PRELOAD] Forwarded to libproton.so" << std::endl;
    state.result = result;
    return result;
  }

  // libproton not loaded yet - return a dummy config
  // The real configuration will happen when libproton loads
  std::cerr
      << "[PROTON PRELOAD] libproton.so not loaded yet, using dummy config"
      << std::endl;
  static rocprofiler_tool_configure_result_t dummyConfig{
      sizeof(rocprofiler_tool_configure_result_t), &dummyInit, &dummyFini,
      nullptr};
  state.result = &dummyConfig;
  return &dummyConfig;
}

// Function that libproton.so can call to check if preload happened
extern "C" __attribute__((visibility("default"))) int
proton_preload_was_configured(uint32_t *version, uint32_t *priority,
                              rocprofiler_client_id_t **clientId) {
  auto &state = getPreloadState();
  std::lock_guard<std::mutex> lock(state.mutex);

  if (!state.configured) {
    return 0;
  }

  if (version)
    *version = state.version;
  if (priority)
    *priority = state.priority;
  if (clientId)
    *clientId = state.clientId;

  return 1;
}
