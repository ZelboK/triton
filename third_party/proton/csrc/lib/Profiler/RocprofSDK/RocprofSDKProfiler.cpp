#include "Profiler/RocprofSDK/RocprofSDKProfiler.h"

#include "Context/Context.h"
#include "Data/Metric.h"
#include "Driver/GPU/HipApi.h"
#include "Driver/GPU/RocprofApi.h"
#include "Profiler/GPUProfiler.h"
#include "Runtime/HipRuntime.h"
#include "Utility/Env.h"
#include "Utility/Map.h"
#include "Utility/Singleton.h"

#include "hip/amd_detail/hip_prof_str.h"
#include "hip/hip_runtime_api.h"
#include "rocprofiler-sdk/agent.h"
#include "rocprofiler-sdk/buffer_tracing.h"
#include "rocprofiler-sdk/callback_tracing.h"
#include "rocprofiler-sdk/hip/api_args.h"
#include "rocprofiler-sdk/hip/runtime_api_id.h"
#include "rocprofiler-sdk/pc_sampling.h"
#include "rocprofiler-sdk/registration.h"
#include "roctracer/ext/prof_protocol.h"

#include <atomic>
#include <dlfcn.h>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace proton {

template <>
thread_local GPUProfiler<RocprofSDKProfiler>::ThreadState
    GPUProfiler<RocprofSDKProfiler>::threadState(
        RocprofSDKProfiler::instance());

namespace {

constexpr size_t BufferSize = 64 * 1024 * 1024;
constexpr const char *UnknownKernelName = "<unknown>";

// Thread-local bridge for PC sampling: when rocprofiler-sdk's HIP callback
// fires ENTER, it stores its correlation ID here. Then hipTracerPhaseEnter
// (which runs inside the HIP call) maps it to the extern scope so PC samples
// can look it up.
static thread_local uint64_t tls_sdkCorrId = 0;

// CLR-internal struct layout (not in public headers). Must match
// projects/clr/hipamd/src/hip_prof_api.h exactly.
struct HipApiTraceData {
  hip_api_data_t apiData;
  uint64_t phaseEnterTimestamp;
  uint64_t phaseData;
  void (*phaseEnter)(hip_api_id_t operationId, HipApiTraceData *data);
  void (*phaseExit)(hip_api_id_t operationId, HipApiTraceData *data);
};

using HipRegisterTracerCallbackFn = void (*)(int (*)(activity_domain_t,
                                                     uint32_t, void *));
constexpr uint32_t kOpIdDispatch = 0;

// ---- SDK runtime state (singleton, outlives any profiler instance) ----

struct PcSamplingAgentInfo {
  rocprofiler_agent_id_t agentId;
  rocprofiler_pc_sampling_method_t method;
  rocprofiler_pc_sampling_unit_t unit;
  uint64_t interval;
};

struct RocprofilerRuntimeState {
  std::mutex mutex;
  rocprofiler_context_id_t codeObjectContext{};
  rocprofiler_context_id_t profilingContext{};
  rocprofiler_buffer_id_t kernelBuffer{};
  rocprofiler_callback_thread_t callbackThread{};
  rocprofiler_client_finalize_t finalizeFunc = nullptr;
  rocprofiler_client_id_t *clientId{nullptr};
  bool configured{false};
  bool codeObjectStarted{false};
  bool profilingStarted{false};

  rocprofiler_context_id_t pcSamplingContext{};
  std::vector<rocprofiler_buffer_id_t> pcSamplingBuffers;
  rocprofiler_callback_thread_t pcSamplingThread{};
  std::vector<PcSamplingAgentInfo> pcSamplingAgents;
  bool pcSamplingConfigured{false};
  bool pcSamplingStarted{false};
  std::atomic<uint64_t> pcSampleCount{0};

  bool useHipTracer{false};
  bool hipTracerRegistered{false};

  // Time-based PC sampling correlation for late-attach mode.
  // Stores completed dispatch time ranges so PC samples can be matched
  // by timestamp when queue interception isn't available.
  struct DispatchTimeRange {
    uint64_t beginNs;
    uint64_t endNs;
    size_t externId;
  };
  std::mutex dispatchRangesMutex;
  std::vector<DispatchTimeRange> dispatchRanges;

  void addDispatchRange(uint64_t beginNs, uint64_t endNs, size_t externId) {
    std::lock_guard<std::mutex> lock(dispatchRangesMutex);
    dispatchRanges.push_back({beginNs, endNs, externId});
  }

  size_t findExternIdByTimestamp(uint64_t timestamp) const {
    std::lock_guard<std::mutex> lock(
        const_cast<std::mutex &>(dispatchRangesMutex));
    for (auto it = dispatchRanges.rbegin(); it != dispatchRanges.rend(); ++it) {
      if (timestamp >= it->beginNs && timestamp <= it->endNs)
        return it->externId;
    }
    return Scope::DummyScopeId;
  }
};

RocprofilerRuntimeState &getRuntimeState() {
  static RocprofilerRuntimeState state;
  return state;
}

// ---- PC Sampling helpers ----

constexpr size_t PcSamplingBufferSize = 16 * 1024 * 1024; // 16 MB
constexpr size_t PcSamplingWatermark = PcSamplingBufferSize - 1024;
constexpr uint64_t StochasticDefaultInterval = 131072; // 2^17 cycles

PCSamplingMetric::PCSamplingMetricKind mapStochasticReason(
    rocprofiler_pc_sampling_instruction_not_issued_reason_t reason,
    bool waveIssued) {
  using K = PCSamplingMetric::PCSamplingMetricKind;
  if (waveIssued)
    return K::StalledSelected;
  switch (reason) {
  case ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE:
    return K::StalledNoInstruction;
  case ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY:
    return K::StalledShortScoreboard;
  case ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT:
    return K::StalledLongScoreboard;
  case ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_BARRIER_WAIT:
    return K::StalledBarrier;
  case ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN:
    return K::StalledNotSelected;
  case ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL:
    return K::StalledDispatchStall;
  case ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT:
    return K::StalledWait;
  case ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_SLEEP_WAIT:
    return K::StalledSleeping;
  case ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_INTERNAL_INSTRUCTION:
  case ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NONE:
  default:
    return K::StalledMisc;
  }
}

void pcSamplingBufferCallback(rocprofiler_context_id_t context,
                              rocprofiler_buffer_id_t buffer,
                              rocprofiler_record_header_t **headers,
                              size_t numHeaders, void *userData,
                              uint64_t dropCount);

rocprofiler_status_t
pcSamplingAgentQueryCallback(rocprofiler_agent_version_t version,
                             const void **agents, size_t count,
                             void *userData) {
  auto *state = static_cast<RocprofilerRuntimeState *>(userData);
  if (version != ROCPROFILER_AGENT_INFO_VERSION_0)
    return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
  auto agentList = reinterpret_cast<const rocprofiler_agent_t *const *>(agents);
  for (size_t i = 0; i < count; ++i) {
    const auto *agent = agentList[i];
    if (agent->type != ROCPROFILER_AGENT_TYPE_GPU)
      continue;

    struct ConfigResult {
      bool found = false;
      rocprofiler_pc_sampling_configuration_t config{};
    } result;

    auto configCb = [](const rocprofiler_pc_sampling_configuration_t *configs,
                       size_t numConfigs, void *ud) {
      auto *res = static_cast<ConfigResult *>(ud);
      for (size_t j = 0; j < numConfigs; ++j) {
        if (configs[j].method == ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC) {
          res->config = configs[j];
          res->found = true;
          return ROCPROFILER_STATUS_SUCCESS;
        }
      }
      return ROCPROFILER_STATUS_SUCCESS;
    };

    auto status = rocprofiler::queryPcSamplingAgentConfigurations<false>(
        agent->id, configCb, &result);
    if (status != ROCPROFILER_STATUS_SUCCESS || !result.found)
      continue;

    auto interval = StochasticDefaultInterval;
    if (result.config.min_interval == result.config.max_interval)
      interval = result.config.min_interval;

    state->pcSamplingAgents.push_back(
        {agent->id, result.config.method, result.config.unit, interval});
  }
  return ROCPROFILER_STATUS_SUCCESS;
}

// ROCTx marker interception via libroctx64.so's callback registration.
// rocprofiler-sdk's own marker tracing requires its replacement roctx library
// to be loaded, which doesn't happen with late-start (force_configure). Instead
// we use the standard libroctx64.so's built-in callback mechanism.
constexpr uint32_t kRoctxPushA = 1;
constexpr uint32_t kRoctxPop = 2;

struct RoctxApiData {
  union {
    struct {
      const char *message;
    } roctxRangePushA;
    struct {
      const char *message;
    } roctxRangePop;
  } args;
};

using RoctxTracerCallbackFn = int (*)(uint32_t domain, uint32_t operationId,
                                      void *data);
using RoctxRegisterTracerCallbackFn = void (*)(RoctxTracerCallbackFn);

// registerRoctxCallback is defined after the Pimpl class (needs access to
// the static roctxCallback member).
void registerRoctxCallback(bool enable);

// ---- Agent (GPU) ID mapping ----

class AgentIdMapper : public Singleton<AgentIdMapper> {
public:
  AgentIdMapper() = default;

  void initialize() {
    std::call_once(initializeFlag, [this]() {
      rocprofiler::queryAvailableAgents<true>(
          ROCPROFILER_AGENT_INFO_VERSION_0, &AgentIdMapper::callback,
          sizeof(rocprofiler_agent_t), this);
    });
  }

  uint32_t map(uint64_t agentHandle) const {
    auto it = agentToDevice.find(agentHandle);
    if (it != agentToDevice.end())
      return it->second;
    return 0;
  }

private:
  static rocprofiler_status_t callback(rocprofiler_agent_version_t version,
                                       const void **agents, size_t count,
                                       void *userData) {
    auto *self = static_cast<AgentIdMapper *>(userData);
    if (version != ROCPROFILER_AGENT_INFO_VERSION_0)
      return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    auto agentList =
        reinterpret_cast<const rocprofiler_agent_t *const *>(agents);
    for (size_t i = 0; i < count; ++i) {
      const auto *agent = agentList[i];
      if (agent->type == ROCPROFILER_AGENT_TYPE_GPU) {
        self->agentToDevice[agent->id.handle] =
            static_cast<uint32_t>(agent->logical_node_type_id);
      }
    }
    return ROCPROFILER_STATUS_SUCCESS;
  }

  std::once_flag initializeFlag;
  std::unordered_map<uint64_t, uint32_t> agentToDevice;
};

// ---- Metric conversion ----

std::unique_ptr<Metric> convertDispatchToMetric(
    const rocprofiler_buffer_tracing_kernel_dispatch_record_t *record) {
  if (record->start_timestamp >= record->end_timestamp)
    return nullptr;
  auto deviceId = static_cast<uint64_t>(
      AgentIdMapper::instance().map(record->dispatch_info.agent_id.handle));
  return std::make_unique<KernelMetric>(
      static_cast<uint64_t>(record->start_timestamp),
      static_cast<uint64_t>(record->end_timestamp), 1, deviceId,
      static_cast<uint64_t>(DeviceType::HIP),
      static_cast<uint64_t>(record->dispatch_info.queue_id.handle));
}

// ---- Operation classification ----

bool isKernelLaunchOperation(rocprofiler_tracing_operation_t op) {
  switch (op) {
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchMultiKernelMultiDevice:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtModuleLaunchKernel:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipHccModuleLaunchKernel:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernelMultiDevice:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernel:
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernelMultiDevice:
    return true;
  default:
    return false;
  }
}

// ---- Kernel dispatch processing (matches main's GPUProfiler interface) ----

void processKernelRecord(
    RocprofSDKProfiler &profiler,
    RocprofSDKProfiler::CorrIdToExternIdMap &corrIdToExternId,
    RocprofSDKProfiler::ExternIdToStateMap &externIdToState,
    ThreadSafeMap<uint64_t, bool, std::unordered_map<uint64_t, bool>>
        &corrIdToIsHipGraph,
    std::map<Data *, std::pair<size_t, size_t>> &dataPhases,
    const std::string &kernelName,
    const rocprofiler_buffer_tracing_kernel_dispatch_record_t *record) {
  auto externId = Scope::DummyScopeId;
  bool hasCorrelation =
      corrIdToExternId.withRead(record->correlation_id.internal,
                                [&](const size_t &value) { externId = value; });

  if (!hasCorrelation)
    return;

  if (externId == Scope::DummyScopeId)
    return;

  bool isGraph = corrIdToIsHipGraph.contain(record->correlation_id.internal);
  auto &state = externIdToState[externId];

  if (!isGraph) {
    for (auto &[data, entry] : state.dataToEntry) {
      if (auto metric = convertDispatchToMetric(record)) {
        if (state.isMissingName) {
          auto childEntry =
              data->addOp(entry.phase, entry.id, {Context(kernelName)});
          childEntry.upsertMetric(std::move(metric));
          entry = childEntry;
        } else {
          entry.upsertMetric(std::move(metric));
        }
        detail::updateDataPhases(dataPhases, data, entry.phase);
      }
    }
  } else {
    for (auto &[data, entry] : state.dataToEntry) {
      if (auto metric = convertDispatchToMetric(record)) {
        auto childEntry =
            data->addOp(entry.phase, entry.id, {Context(kernelName)});
        childEntry.upsertMetric(std::move(metric));
        entry = childEntry;
        detail::updateDataPhases(dataPhases, data, entry.phase);
      }
    }
  }

  --state.numNodes;
  if (state.numNodes == 0) {
    auto &rtState = getRuntimeState();
    if (!rtState.pcSamplingStarted) {
      corrIdToExternId.erase(record->correlation_id.internal);
      corrIdToIsHipGraph.erase(record->correlation_id.internal);
      externIdToState.erase(externId);
    }
  }
}

} // namespace

// ---- Pimpl ----

struct RocprofSDKProfiler::RocprofSDKProfilerPimpl
    : public GPUProfiler<RocprofSDKProfiler>::GPUProfilerPimplInterface {
  RocprofSDKProfilerPimpl(RocprofSDKProfiler &profiler)
      : GPUProfiler<RocprofSDKProfiler>::GPUProfilerPimplInterface(profiler) {
    auto runtime = &HipRuntime::instance();
    profiler.metricBuffer =
        std::make_unique<MetricBuffer>(1024 * 1024 * 64, runtime);
  }
  virtual ~RocprofSDKProfilerPimpl() = default;

  void doStart() override;
  void doFlush() override;
  void doStop() override;

  static void hipRuntimeCallback(rocprofiler_callback_tracing_record_t record,
                                 rocprofiler_user_data_t *userData, void *arg);
  static void roctxCallback(uint32_t operationId, void *data);
  static void codeObjectCallback(rocprofiler_callback_tracing_record_t record,
                                 rocprofiler_user_data_t *userData, void *arg);
  static void kernelBufferCallback(rocprofiler_context_id_t context,
                                   rocprofiler_buffer_id_t buffer,
                                   rocprofiler_record_header_t **headers,
                                   size_t numHeaders, void *userData,
                                   uint64_t dropCount);
  static void pcSamplingBufferCallbackImpl(
      rocprofiler_context_id_t context, rocprofiler_buffer_id_t buffer,
      rocprofiler_record_header_t **headers, size_t numHeaders, void *userData,
      uint64_t dropCount);

  static void hipTracerPhaseEnterImpl(hip_api_id_t opId, HipApiTraceData *data);
  static void hipTracerPhaseExitImpl(hip_api_id_t opId, HipApiTraceData *data);
  static int hipTracerCallbackImpl(activity_domain_t domain,
                                   uint32_t operationId, void *data);

  using KernelNameMap =
      ThreadSafeMap<uint64_t, std::string,
                    std::unordered_map<uint64_t, std::string>>;

  std::string getKernelName(uint64_t kernelId) {
    std::string name;
    if (!kernelNames.withRead(kernelId,
                              [&](const std::string &v) { name = v; }))
      return UnknownKernelName;
    const std::string suffix = ".kd";
    if (name.size() > suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
      name.resize(name.size() - suffix.size());
    return name;
  }

  void setKernelName(uint64_t kernelId, const char *name) {
    if (name == nullptr)
      return;
    kernelNames[kernelId] = std::string(name);
  }

  ThreadSafeMap<uint64_t, bool, std::unordered_map<uint64_t, bool>>
      corrIdToIsHipGraph;

  ThreadSafeMap<hipGraphExec_t, hipGraph_t,
                std::unordered_map<hipGraphExec_t, hipGraph_t>>
      graphExecToGraph;

  ThreadSafeMap<hipGraph_t, uint32_t, std::unordered_map<hipGraph_t, uint32_t>>
      graphToNumInstances;

  ThreadSafeMap<hipStream_t, uint32_t,
                std::unordered_map<hipStream_t, uint32_t>>
      streamToCaptureCount;

  ThreadSafeMap<hipStream_t, bool, std::unordered_map<hipStream_t, bool>>
      streamToCapture;

  // Fast check: non-zero when any stream is being captured. Avoids acquiring
  // a shared_mutex on every kernel launch EXIT just to find an empty map.
  std::atomic<int> activeCaptureCount{0};

  KernelNameMap kernelNames;
};

// ---- HIP Runtime API callback (correlation tracking) ----

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::hipRuntimeCallback(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_user_data_t *userData, void *arg) {
  if (record.kind != ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API)
    return;

  auto operation =
      static_cast<rocprofiler_tracing_operation_t>(record.operation);
  bool isKernelOp = isKernelLaunchOperation(operation);
  auto &profiler = RocprofSDKProfiler::instance();
  auto *impl = static_cast<RocprofSDKProfiler::RocprofSDKProfilerPimpl *>(
      profiler.pImpl.get());
  auto *payload = static_cast<rocprofiler_callback_tracing_hip_api_data_t *>(
      record.payload);

  // When hipTracer is active, this callback only serves as a PC sampling
  // correlation bridge. Store rocprofiler-sdk's correlation ID at ENTER so
  // hipTracerPhaseEnter can map it to the extern scope.
  auto &rtState = getRuntimeState();
  if (rtState.useHipTracer) {
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER && isKernelOp)
      tls_sdkCorrId = record.correlation_id.internal;
    return;
  }

  if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
    if (!isKernelOp)
      return;
    threadState.enterOp(Scope(""));
    auto &dataToEntry = threadState.dataToEntry;
    size_t numInstances = 1;
    if (operation == ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch) {
      impl->corrIdToIsHipGraph[record.correlation_id.internal] = true;
      numInstances = std::numeric_limits<size_t>::max();
      bool foundGraph = false;
      auto graphExec = payload->args.hipGraphLaunch.graphExec;
      if (impl->graphExecToGraph.contain(graphExec)) {
        auto graph = impl->graphExecToGraph[graphExec];
        if (impl->graphToNumInstances.contain(graph)) {
          numInstances = impl->graphToNumInstances[graph];
          foundGraph = true;
        }
      }
      if (!foundGraph) {
        std::cerr
            << "[PROTON] Cannot find graph and it may cause a memory leak."
               "To avoid this problem, please start profiling before the "
               "graph is created."
            << std::endl;
      }
    }
    auto &scope = threadState.scopeStack.back();
    auto isMissingName = scope.name.empty();
    profiler.correlation.correlate(record.correlation_id.internal,
                                   scope.scopeId, numInstances, isMissingName,
                                   dataToEntry);
    return;
  }

  if (record.phase != ROCPROFILER_CALLBACK_PHASE_EXIT)
    return;

  switch (operation) {
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture: {
    auto stream = payload->args.hipStreamBeginCapture.stream;
    impl->streamToCaptureCount[stream] = 0;
    impl->streamToCapture[stream] = true;
    impl->activeCaptureCount.fetch_add(1, std::memory_order_release);
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture: {
    auto stream = payload->args.hipStreamEndCapture.stream;
    auto graph = *(payload->args.hipStreamEndCapture.pGraph);
    uint32_t captured = impl->streamToCaptureCount.contain(stream)
                            ? impl->streamToCaptureCount[stream]
                            : 0;
    impl->graphToNumInstances[graph] = captured;
    impl->streamToCapture.erase(stream);
    impl->streamToCaptureCount.erase(stream);
    impl->activeCaptureCount.fetch_sub(1, std::memory_order_release);
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithFlags: {
    auto graph = payload->args.hipGraphInstantiateWithFlags.graph;
    auto graphExec = *(payload->args.hipGraphInstantiateWithFlags.pGraphExec);
    impl->graphExecToGraph[graphExec] = graph;
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiate: {
    auto graph = payload->args.hipGraphInstantiate.graph;
    auto graphExec = *(payload->args.hipGraphInstantiate.pGraphExec);
    impl->graphExecToGraph[graphExec] = graph;
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecDestroy: {
    auto graphExec = payload->args.hipGraphExecDestroy.graphExec;
    impl->graphExecToGraph.erase(graphExec);
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphDestroy: {
    auto graph = payload->args.hipGraphDestroy.graph;
    impl->graphToNumInstances.erase(graph);
    break;
  }
  default:
    break;
  }

  // Count kernel launches during graph capture. The atomic fast-check avoids
  // acquiring the shared_mutex on streamToCapture for every kernel launch
  // when no capture is active (the overwhelmingly common case).
  if (isKernelOp &&
      impl->activeCaptureCount.load(std::memory_order_acquire) > 0) {
    hipStream_t stream = nullptr;
    switch (operation) {
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel:
      stream = payload->args.hipLaunchKernel.stream;
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel:
      stream = payload->args.hipExtLaunchKernel.stream;
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel:
      stream = payload->args.hipLaunchCooperativeKernel.stream;
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel:
      stream = payload->args.hipModuleLaunchKernel.stream;
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernel:
      stream = payload->args.hipModuleLaunchCooperativeKernel.stream;
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtModuleLaunchKernel:
      stream = payload->args.hipExtModuleLaunchKernel.stream;
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipHccModuleLaunchKernel:
      stream = payload->args.hipHccModuleLaunchKernel.stream;
      break;
    default:
      break;
    }
    if (stream && impl->streamToCapture.contain(stream))
      impl->streamToCaptureCount[stream]++;
  }

  if (isKernelOp) {
    threadState.exitOp();
    profiler.correlation.submit(record.correlation_id.internal);
  }
}

// ---- ROCTx marker callback via libroctx64.so ----

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::roctxCallback(
    uint32_t operationId, void *data) {
  auto *apiData = static_cast<RoctxApiData *>(data);
  if (operationId == kRoctxPushA) {
    threadState.enterScope(apiData->args.roctxRangePushA.message);
  } else if (operationId == kRoctxPop) {
    threadState.exitScope();
  }
}

namespace {
int roctxTracerCallback(uint32_t /*domain*/, uint32_t operationId, void *data) {
  RocprofSDKProfiler::RocprofSDKProfilerPimpl::roctxCallback(operationId, data);
  return 0;
}

void registerRoctxCallback(bool enable) {
  // libroctx64.so is typically loaded with RTLD_LOCAL (e.g. by PyTorch), so
  // dlsym(RTLD_DEFAULT, ...) won't find it. Use RTLD_NOLOAD to get a handle
  // to the already-loaded library.
  void *roctxLib = dlopen("libroctx64.so", RTLD_NOLOAD | RTLD_NOW);
  if (!roctxLib)
    return;
  auto *fn = reinterpret_cast<RoctxRegisterTracerCallbackFn>(
      dlsym(roctxLib, "roctxRegisterTracerCallback"));
  dlclose(roctxLib);
  if (!fn)
    return;
  fn(enable ? &roctxTracerCallback : nullptr);
}
} // namespace

// ---- HIP Tracer Callback (hipRegisterTracerCallback) ----
// Provides dispatch timing for ALL queues, including pre-existing ones that
// rocprofiler-sdk's buffer tracing cannot intercept in late-attach scenarios.

namespace {

std::atomic<uint64_t> hipTracerNextCorrId{1};

bool isHipApiKernelLaunch(uint32_t op) {
  switch (static_cast<hip_api_id_t>(op)) {
  case HIP_API_ID_hipLaunchKernel:
  case HIP_API_ID_hipExtLaunchKernel:
  case HIP_API_ID_hipModuleLaunchKernel:
  case HIP_API_ID_hipExtModuleLaunchKernel:
  case HIP_API_ID_hipHccModuleLaunchKernel:
  case HIP_API_ID_hipLaunchCooperativeKernel:
  case HIP_API_ID_hipModuleLaunchCooperativeKernel:
  case HIP_API_ID_hipExtLaunchMultiKernelMultiDevice:
  case HIP_API_ID_hipLaunchCooperativeKernelMultiDevice:
  case HIP_API_ID_hipModuleLaunchCooperativeKernelMultiDevice:
  case HIP_API_ID_hipGraphLaunch:
    return true;
  default:
    return false;
  }
}

void hipTracerPhaseEnterFwd(hip_api_id_t opId, HipApiTraceData *data) {
  RocprofSDKProfiler::RocprofSDKProfilerPimpl::hipTracerPhaseEnterImpl(opId,
                                                                       data);
}

void hipTracerPhaseExitFwd(hip_api_id_t opId, HipApiTraceData *data) {
  RocprofSDKProfiler::RocprofSDKProfilerPimpl::hipTracerPhaseExitImpl(opId,
                                                                      data);
}

int hipTracerCallbackFwd(activity_domain_t domain, uint32_t operationId,
                         void *data) {
  return RocprofSDKProfiler::RocprofSDKProfilerPimpl::hipTracerCallbackImpl(
      domain, operationId, data);
}

void registerHipTracerCallback(bool enable) {
  void *hipLib = dlopen("libamdhip64.so", RTLD_NOLOAD | RTLD_NOW);
  if (!hipLib)
    return;
  auto *fn = reinterpret_cast<HipRegisterTracerCallbackFn>(
      dlsym(hipLib, "hipRegisterTracerCallback"));
  dlclose(hipLib);
  if (!fn)
    return;
  fn(enable ? &hipTracerCallbackFwd : nullptr);
}

bool isHipTracerAvailable() {
  void *hipLib = dlopen("libamdhip64.so", RTLD_NOLOAD | RTLD_NOW);
  if (!hipLib)
    return false;
  auto *fn = dlsym(hipLib, "hipRegisterTracerCallback");
  dlclose(hipLib);
  return fn != nullptr;
}

} // namespace

// ---- HIP Tracer Callback static member implementations ----

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::hipTracerPhaseEnterImpl(
    hip_api_id_t opId, HipApiTraceData *data) {
  if (!isHipApiKernelLaunch(static_cast<uint32_t>(opId)))
    return;

  auto &profiler = RocprofSDKProfiler::instance();
  auto &ts = GPUProfiler<RocprofSDKProfiler>::threadState;
  ts.enterOp(Scope(""));
  auto &dataToEntry = ts.dataToEntry;
  auto &scope = ts.scopeStack.back();
  profiler.correlation.correlate(data->apiData.correlation_id, scope.scopeId,
                                 /*numNodes=*/1, scope.name.empty(),
                                 dataToEntry);

  // PC sampling bridge: map rocprofiler-sdk's correlation ID (stored earlier
  // by hipRuntimeCallback ENTER) to the same extern scope so PC samples can
  // find it. The SDK's ID space is separate from CLR's correlation IDs.
  auto &rtState = getRuntimeState();
  if (rtState.pcSamplingStarted && tls_sdkCorrId != 0) {
    profiler.correlation.corrIdToExternId.insert(tls_sdkCorrId, scope.scopeId);
    tls_sdkCorrId = 0;
  }
}

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::hipTracerPhaseExitImpl(
    hip_api_id_t opId, HipApiTraceData *data) {
  if (!isHipApiKernelLaunch(static_cast<uint32_t>(opId)))
    return;

  auto &profiler = RocprofSDKProfiler::instance();
  auto &ts = GPUProfiler<RocprofSDKProfiler>::threadState;
  ts.exitOp();
  profiler.correlation.submit(data->apiData.correlation_id);
}

int RocprofSDKProfiler::RocprofSDKProfilerPimpl::hipTracerCallbackImpl(
    activity_domain_t domain, uint32_t operationId, void *data) {
  if (domain == ACTIVITY_DOMAIN_HIP_OPS) {
    if (data == nullptr)
      return (operationId == kOpIdDispatch) ? 0 : -1;

    auto *record = static_cast<activity_record_t *>(data);
    if (record->begin_ns >= record->end_ns)
      return 0;

    auto &profiler = RocprofSDKProfiler::instance();
    auto &correlation = profiler.correlation;

    auto externId = Scope::DummyScopeId;
    bool found = correlation.corrIdToExternId.withRead(
        record->correlation_id, [&](const size_t &value) { externId = value; });
    if (!found || externId == Scope::DummyScopeId)
      return 0;

    std::string kernelName =
        record->kernel_name ? record->kernel_name : UnknownKernelName;
    const std::string suffix = ".kd";
    if (kernelName.size() > suffix.size() &&
        kernelName.compare(kernelName.size() - suffix.size(), suffix.size(),
                           suffix) == 0)
      kernelName.resize(kernelName.size() - suffix.size());

    auto deviceId = static_cast<uint64_t>(record->device_id);

    // Store time range for time-based PC sampling correlation in late-attach
    auto &rtState = getRuntimeState();
    if (rtState.pcSamplingStarted && record->begin_ns < record->end_ns)
      rtState.addDispatchRange(record->begin_ns, record->end_ns, externId);

    static thread_local std::map<Data *, std::pair<size_t, size_t>> dataPhases;
    dataPhases.clear();

    auto &state = correlation.externIdToState[externId];
    for (auto &[dataPtr, entry] : state.dataToEntry) {
      auto metric = std::make_unique<KernelMetric>(
          static_cast<uint64_t>(record->begin_ns),
          static_cast<uint64_t>(record->end_ns), /*count=*/1, deviceId,
          static_cast<uint64_t>(DeviceType::HIP), record->queue_id);
      if (state.isMissingName) {
        auto childEntry =
            dataPtr->addOp(entry.phase, entry.id, {Context(kernelName)});
        childEntry.upsertMetric(std::move(metric));
        entry = childEntry;
      } else {
        entry.upsertMetric(std::move(metric));
      }
      detail::updateDataPhases(dataPhases, dataPtr, entry.phase);
    }

    --state.numNodes;
    if (state.numNodes == 0) {
      auto &rtState = getRuntimeState();
      if (!rtState.pcSamplingStarted) {
        correlation.corrIdToExternId.erase(record->correlation_id);
        correlation.externIdToState.erase(externId);
      }
    }
    correlation.complete(record->correlation_id);

    static thread_local std::map<Data *, size_t> dataFlushedPhases;
    profiler.flushDataPhases(dataFlushedPhases, dataPhases,
                             profiler.pendingGraphPool.get());
    return 0;
  }

  if (domain == ACTIVITY_DOMAIN_HIP_API) {
    if (!isHipApiKernelLaunch(operationId))
      return -1;

    auto *trace = static_cast<HipApiTraceData *>(data);
    trace->apiData.correlation_id =
        hipTracerNextCorrId.fetch_add(1, std::memory_order_relaxed);
    trace->phaseEnter = &hipTracerPhaseEnterFwd;
    trace->phaseExit = &hipTracerPhaseExitFwd;
    return 0;
  }

  return -1;
}

// ---- Code object callback (kernel_id -> name mapping) ----

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::codeObjectCallback(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_user_data_t *userData, void *arg) {
  if (record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT ||
      record.operation !=
          ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER ||
      record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) {
    return;
  }
  auto *payload = static_cast<
      rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t *>(
      record.payload);
  auto &profiler = RocprofSDKProfiler::instance();
  auto *impl = static_cast<RocprofSDKProfiler::RocprofSDKProfilerPimpl *>(
      profiler.pImpl.get());
  impl->setKernelName(payload->kernel_id, payload->kernel_name);
}

// ---- Kernel dispatch buffer callback ----

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::kernelBufferCallback(
    rocprofiler_context_id_t context, rocprofiler_buffer_id_t buffer,
    rocprofiler_record_header_t **headers, size_t numHeaders, void *userData,
    uint64_t dropCount) {
  // When hipTracer is active, buffer tracing exists only for PC sampling
  // dispatch correlation. hipTracer handles timing; just consume the buffer.
  auto &rtState = getRuntimeState();
  if (rtState.useHipTracer)
    return;

  if (dropCount > 0) {
    std::cerr << "[PROTON] ROCProfiler-SDK dropped " << dropCount
              << " kernel dispatch records" << std::endl;
  }
  auto &profiler = RocprofSDKProfiler::instance();
  auto *impl = static_cast<RocprofSDKProfiler::RocprofSDKProfilerPimpl *>(
      profiler.pImpl.get());
  auto &correlation = profiler.correlation;

  static thread_local std::map<Data *, size_t> dataFlushedPhases;
  uint64_t maxCorrelationId = 0;
  std::map<Data *, std::pair<size_t, size_t>> dataPhases;

  for (size_t i = 0; i < numHeaders; ++i) {
    auto *header = headers[i];
    if (header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING ||
        header->kind != ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {
      continue;
    }
    auto *record =
        static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t *>(
            header->payload);
    maxCorrelationId =
        std::max(maxCorrelationId, record->correlation_id.internal);
    auto kernelName = impl->getKernelName(record->dispatch_info.kernel_id);
    processKernelRecord(profiler, correlation.corrIdToExternId,
                        correlation.externIdToState, impl->corrIdToIsHipGraph,
                        dataPhases, kernelName, record);
  }
  if (maxCorrelationId > 0) {
    correlation.complete(maxCorrelationId);
  }
  profiler.flushDataPhases(dataFlushedPhases, dataPhases,
                           profiler.pendingGraphPool.get());
}

// ---- PC sampling buffer callback ----

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::pcSamplingBufferCallbackImpl(
    rocprofiler_context_id_t /*context*/, rocprofiler_buffer_id_t /*buffer*/,
    rocprofiler_record_header_t **headers, size_t numHeaders,
    void * /*userData*/, uint64_t dropCount) {
  auto &state = getRuntimeState();
  auto &profiler = RocprofSDKProfiler::instance();
  auto &correlation = profiler.correlation;

  uint64_t sampleCount = 0;

  for (size_t i = 0; i < numHeaders; ++i) {
    auto *header = headers[i];
    if (!header)
      continue;
    if (header->category != ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING)
      continue;

    // Accept both stochastic and host-trap sample records
    bool isStochastic =
        (header->kind == ROCPROFILER_PC_SAMPLING_RECORD_STOCHASTIC_V0_SAMPLE);
    bool isHostTrap =
        (header->kind == ROCPROFILER_PC_SAMPLING_RECORD_HOST_TRAP_V0_SAMPLE);
    if (!isStochastic && !isHostTrap)
      continue;

    uint64_t corrIdInternal = 0;
    rocprofiler_pc_sampling_instruction_not_issued_reason_t reason{};
    bool waveIssued = false;

    if (isStochastic) {
      auto *rec = static_cast<rocprofiler_pc_sampling_record_stochastic_v0_t *>(
          header->payload);
      corrIdInternal = rec->correlation_id.internal;
      reason =
          static_cast<rocprofiler_pc_sampling_instruction_not_issued_reason_t>(
              rec->snapshot.reason_not_issued);
      waveIssued = rec->wave_issued != 0;
    } else {
      auto *rec = static_cast<rocprofiler_pc_sampling_record_host_trap_v0_t *>(
          header->payload);
      corrIdInternal = rec->correlation_id.internal;
      waveIssued = true;
    }
    auto stallKind = mapStochasticReason(reason, waveIssued);
    bool isStalled = !waveIssued;

    ++sampleCount;

    // Extract timestamp for time-based fallback correlation
    uint64_t sampleTimestamp = 0;
    if (isStochastic) {
      auto *recTs =
          static_cast<rocprofiler_pc_sampling_record_stochastic_v0_t *>(
              header->payload);
      sampleTimestamp = recTs->timestamp;
    }

    auto externId = Scope::DummyScopeId;
    if (corrIdInternal != 0) {
      // Primary path: correlate via rocprofiler-sdk correlation ID
      correlation.corrIdToExternId.withRead(
          corrIdInternal, [&](const size_t &val) { externId = val; });
    }
    if (externId == Scope::DummyScopeId && sampleTimestamp != 0) {
      // Fallback: time-based correlation (needed for late-attach where
      // queue interception doesn't produce dispatch marker packets)
      externId = state.findExternIdByTimestamp(sampleTimestamp);
    }
    if (externId == Scope::DummyScopeId)
      continue;

    uint64_t stalledSamples = isStalled ? 1 : 0;
    correlation.externIdToState.withRead(externId, [&](const auto &extState) {
      for (const auto &[data, entry] : extState.dataToEntry) {
        entry.upsertMetric(std::make_unique<PCSamplingMetric>(
            stallKind, /*samples=*/1, stalledSamples));
      }
    });
  }

  state.pcSampleCount.fetch_add(sampleCount, std::memory_order_relaxed);
}

namespace {
void pcSamplingBufferCallback(rocprofiler_context_id_t context,
                              rocprofiler_buffer_id_t buffer,
                              rocprofiler_record_header_t **headers,
                              size_t numHeaders, void *userData,
                              uint64_t dropCount) {
  RocprofSDKProfiler::RocprofSDKProfilerPimpl::pcSamplingBufferCallbackImpl(
      context, buffer, headers, numHeaders, userData, dropCount);
}
} // namespace

// ---- SDK tool init / fini (called by rocprofiler_force_configure) ----

namespace {

int protonToolInit(rocprofiler_client_finalize_t finiFunc, void *toolData) {
  auto *state = static_cast<RocprofilerRuntimeState *>(toolData);
  state->finalizeFunc = finiFunc;

  // Context 1: lightweight, always-active context for code object tracking.
  // Captures kernel_id -> name mappings as kernels are compiled.
  rocprofiler::createContext<true>(&state->codeObjectContext);

  const rocprofiler_tracing_operation_t codeObjectOps[] = {
      ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};
  rocprofiler::configureCallbackTracingService<true>(
      state->codeObjectContext, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
      codeObjectOps, 1,
      &RocprofSDKProfiler::RocprofSDKProfilerPimpl::codeObjectCallback,
      nullptr);

  int valid = 0;
  rocprofiler::contextIsValid<true>(state->codeObjectContext, &valid);
  if (valid == 0)
    return -1;

  // Start the code object context immediately so it captures kernel symbols
  // registered during API table re-propagation (force_configure path).
  rocprofiler::startContext<true>(state->codeObjectContext);
  state->codeObjectStarted = true;

  // Detect hipRegisterTracerCallback availability. When available, use it
  // instead of rocprofiler-sdk HIP callback/buffer tracing. This captures
  // dispatches on ALL queues including pre-existing ones.
  state->useHipTracer = isHipTracerAvailable();

  // Context 2: on-demand profiling context for HIP callback tracing and
  // kernel dispatch buffer tracing. Always configured regardless of hipTracer
  // mode -- in hipTracer mode, the context is only started when PC sampling
  // is requested (providing dispatch interception for PC sample correlation).
  rocprofiler::createContext<true>(&state->profilingContext);

  constexpr rocprofiler_tracing_operation_t kTracedHipOps[] = {
      ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchMultiKernelMultiDevice,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipExtModuleLaunchKernel,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipHccModuleLaunchKernel,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernelMultiDevice,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernel,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernelMultiDevice,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiate,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithFlags,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecDestroy,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphDestroy,
  };

  rocprofiler::configureCallbackTracingService<true>(
      state->profilingContext, ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
      kTracedHipOps, std::size(kTracedHipOps),
      &RocprofSDKProfiler::RocprofSDKProfilerPimpl::hipRuntimeCallback,
      nullptr);

  size_t watermark = BufferSize - (BufferSize / 8);
  rocprofiler::createBuffer<true>(
      state->profilingContext, BufferSize, watermark,
      ROCPROFILER_BUFFER_POLICY_LOSSLESS,
      &RocprofSDKProfiler::RocprofSDKProfilerPimpl::kernelBufferCallback,
      nullptr, &state->kernelBuffer);

  rocprofiler::configureBufferTracingService<true>(
      state->profilingContext, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
      nullptr, 0, state->kernelBuffer);

  rocprofiler::createCallbackThread<true>(&state->callbackThread);
  rocprofiler::assignCallbackThread<true>(state->kernelBuffer,
                                          state->callbackThread);

  valid = 0;
  rocprofiler::contextIsValid<true>(state->profilingContext, &valid);
  if (valid == 0)
    return -1;

  AgentIdMapper::instance().initialize();

  rocprofiler::queryAvailableAgents<true>(ROCPROFILER_AGENT_INFO_VERSION_0,
                                          &pcSamplingAgentQueryCallback,
                                          sizeof(rocprofiler_agent_t), state);

  if (!state->pcSamplingAgents.empty()) {
    rocprofiler::createContext<true>(&state->pcSamplingContext);

    rocprofiler::createCallbackThread<true>(&state->pcSamplingThread);

    for (auto &agentInfo : state->pcSamplingAgents) {
      rocprofiler_buffer_id_t bufId{};
      rocprofiler::createBuffer<true>(
          state->pcSamplingContext, PcSamplingBufferSize, PcSamplingWatermark,
          ROCPROFILER_BUFFER_POLICY_LOSSLESS, &pcSamplingBufferCallback,
          nullptr, &bufId);

      rocprofiler::assignCallbackThread<true>(bufId, state->pcSamplingThread);

      auto status = rocprofiler::configurePcSamplingService<false>(
          state->pcSamplingContext, agentInfo.agentId, agentInfo.method,
          agentInfo.unit, agentInfo.interval, bufId, 0);

      if (status == ROCPROFILER_STATUS_SUCCESS) {
        state->pcSamplingBuffers.push_back(bufId);
      }
    }

    if (!state->pcSamplingBuffers.empty()) {
      int pcValid = 0;
      rocprofiler::contextIsValid<true>(state->pcSamplingContext, &pcValid);
      state->pcSamplingConfigured = (pcValid != 0);
    }
  }

  state->configured = true;
  return 0;
}

void protonToolFini(void *toolData) {
  auto *state = static_cast<RocprofilerRuntimeState *>(toolData);
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->pcSamplingStarted) {
      rocprofiler::stopContext<false>(state->pcSamplingContext);
      state->pcSamplingStarted = false;
    }
    if (state->profilingStarted) {
      rocprofiler::stopContext<false>(state->profilingContext);
      state->profilingStarted = false;
    }
    if (state->codeObjectStarted) {
      rocprofiler::stopContext<false>(state->codeObjectContext);
      state->codeObjectStarted = false;
    }
  }
  for (auto bufId : state->pcSamplingBuffers)
    rocprofiler::flushBuffer<false>(bufId);
  rocprofiler::flushBuffer<false>(state->kernelBuffer);
  if (state->hipTracerRegistered) {
    registerHipTracerCallback(false);
    state->hipTracerRegistered = false;
  }
  if (state->finalizeFunc && state->clientId) {
    state->finalizeFunc(*state->clientId);
  }
}

rocprofiler_tool_configure_result_t *
protonConfigure(uint32_t version, const char *runtimeVersion, uint32_t priority,
                rocprofiler_client_id_t *id) {
  auto &state = getRuntimeState();
  id->name = "ProtonRocprofSDK";
  state.clientId = id;
  static rocprofiler_tool_configure_result_t config{
      sizeof(rocprofiler_tool_configure_result_t), &protonToolInit,
      &protonToolFini, static_cast<void *>(&state)};
  return &config;
}

} // namespace

// ---- Profiler lifecycle ----

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::doStart() {
  auto &state = getRuntimeState();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.useHipTracer) {
    if (!state.hipTracerRegistered) {
      registerHipTracerCallback(true);
      state.hipTracerRegistered = true;
    }
    // When PC sampling is requested, also start the profilingContext so
    // rocprofiler-sdk intercepts dispatches and assigns correlation IDs
    // that PC samples can reference.
    if (profiler.pcSamplingEnabled && state.pcSamplingConfigured &&
        !state.profilingStarted) {
      rocprofiler::startContext<true>(state.profilingContext);
      state.profilingStarted = true;
    }
  } else {
    if (!state.profilingStarted) {
      rocprofiler::startContext<true>(state.profilingContext);
      state.profilingStarted = true;
    }
  }
  if (profiler.pcSamplingEnabled && state.pcSamplingConfigured &&
      !state.pcSamplingStarted) {
    auto pcStatus = rocprofiler::startContext<false>(state.pcSamplingContext);
    state.pcSamplingStarted = (pcStatus == ROCPROFILER_STATUS_SUCCESS);
  }
  if (getBoolEnv("TRITON_ENABLE_NVTX", true))
    registerRoctxCallback(true);
}

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::doFlush() {
  auto &state = getRuntimeState();
  std::ignore = hip::deviceSynchronize<true>();
  if (state.useHipTracer) {
    profiler.correlation.flush(/*maxRetries=*/100, /*sleepUs=*/10, [&state]() {
      // If profilingContext is started (for PC sampling bridge), consume
      // its kernel buffer to prevent overflow.
      if (state.profilingStarted)
        rocprofiler::flushBuffer<true>(state.kernelBuffer);
    });
  } else {
    profiler.correlation.flush(
        /*maxRetries=*/100, /*sleepUs=*/10,
        [&state]() { rocprofiler::flushBuffer<true>(state.kernelBuffer); });
  }
  if (state.pcSamplingStarted) {
    for (auto bufId : state.pcSamplingBuffers)
      rocprofiler::flushBuffer<true>(bufId);
  }
}

void RocprofSDKProfiler::RocprofSDKProfilerPimpl::doStop() {
  registerRoctxCallback(false);
  auto &state = getRuntimeState();
  std::lock_guard<std::mutex> lock(state.mutex);
  bool wasPcSampling = state.pcSamplingStarted;
  if (state.pcSamplingStarted) {
    for (auto bufId : state.pcSamplingBuffers)
      rocprofiler::flushBuffer<true>(bufId);
    rocprofiler::stopContext<true>(state.pcSamplingContext);
    state.pcSamplingStarted = false;
    profiler.pcSamplingEnabled = false;
  }
  if (state.useHipTracer && state.hipTracerRegistered) {
    registerHipTracerCallback(false);
    state.hipTracerRegistered = false;
  }
  // profilingContext may be started in both hipTracer (for PC sampling bridge)
  // and non-hipTracer modes.
  if (state.profilingStarted) {
    rocprofiler::stopContext<true>(state.profilingContext);
    state.profilingStarted = false;
  }
  if (wasPcSampling) {
    profiler.correlation.corrIdToExternId.clear();
    profiler.correlation.externIdToState.clear();
    std::lock_guard<std::mutex> rangeLock(state.dispatchRangesMutex);
    state.dispatchRanges.clear();
  }
}

RocprofSDKProfiler::RocprofSDKProfiler() {
  pImpl = std::make_unique<RocprofSDKProfilerPimpl>(*this);
  auto &state = getRuntimeState();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!state.configured) {
    rocprofiler::forceConfigure<true>(&protonConfigure);
  }
  if (!state.codeObjectStarted) {
    rocprofiler::startContext<true>(state.codeObjectContext);
    state.codeObjectStarted = true;
  }
}

RocprofSDKProfiler::~RocprofSDKProfiler() = default;

void RocprofSDKProfiler::doSetMode(
    const std::vector<std::string> &modeAndOptions) {
  auto mode = modeAndOptions.empty() ? std::string() : modeAndOptions[0];
  if (proton::toLower(mode) == "pcsampling") {
    auto &state = getRuntimeState();
    if (!state.pcSamplingConfigured) {
      std::cerr << "[PROTON PC_SAMPLING] WARNING: PC sampling requested but "
                   "no GPU agents support it on this system"
                << std::endl;
    }
    pcSamplingEnabled = true;
  } else if (proton::toLower(mode) == "periodic_flushing") {
    detail::setPeriodicFlushingMode(periodicFlushingEnabled,
                                    periodicFlushingFormat, modeAndOptions,
                                    "RocprofSDKProfiler");
  } else if (!mode.empty()) {
    throw std::invalid_argument(
        "[PROTON] RocprofSDKProfiler: unsupported mode: " + mode);
  }
}

} // namespace proton
