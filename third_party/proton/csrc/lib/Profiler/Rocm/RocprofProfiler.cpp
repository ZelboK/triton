#include "Profiler/Rocm/RocprofProfiler.h"

#include "Context/Context.h"
#include "Data/Metric.h"
#include "Driver/GPU/HipApi.h"
#include "Driver/GPU/RocprofApi.h"
#include "Profiler/GPUProfiler.h"
#include "Utility/Env.h"
#include "Utility/Map.h"
#include "Utility/Singleton.h"

#include "hip/hip_runtime_api.h"
#include "rocprofiler-sdk/agent.h"
#include "rocprofiler-sdk/buffer_tracing.h"
#include "rocprofiler-sdk/callback_tracing.h"
#include "rocprofiler-sdk/hip/api_args.h"
#include "rocprofiler-sdk/hip/runtime_api_id.h"
#include "rocprofiler-sdk/marker/api_args.h"
#include "rocprofiler-sdk/marker/api_id.h"
#include "rocprofiler-sdk/registration.h"

#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

namespace proton {

template <>
thread_local GPUProfiler<RocprofProfiler>::ThreadState
    GPUProfiler<RocprofProfiler>::threadState(RocprofProfiler::instance());

template <>
thread_local std::deque<size_t>
    GPUProfiler<RocprofProfiler>::Correlation::externIdQueue{};

namespace {

constexpr size_t BufferSize = 64 * 1024 * 1024;
constexpr const char *UnknownKernelName = "<unknown>";

struct RocprofilerRuntimeState {
  std::mutex mutex;
  rocprofiler_context_id_t context{};
  rocprofiler_buffer_id_t kernelBuffer{};
  rocprofiler_buffer_id_t pcSamplingBuffer{};
  rocprofiler_callback_thread_t callbackThread{};
  rocprofiler_callback_thread_t pcSamplingCallbackThread{};
  rocprofiler_client_finalize_t finalizeFunc = nullptr;
  rocprofiler_client_id_t *clientId{nullptr};
  bool configured{false};
  bool markerCallbacksEnabled{false};
  bool started{false};
  bool pcSamplingEnabled{false};
};

RocprofilerRuntimeState &getRuntimeState() {
  static RocprofilerRuntimeState state;
  return state;
}

std::once_flag configureOnce;

void ensureRocprofilerConfigured();

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

  const std::vector<rocprofiler_agent_id_t> &getGpuAgentIds() const {
    return gpuAgentIds;
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
        self->gpuAgentIds.push_back(agent->id);
      }
    }
    return ROCPROFILER_STATUS_SUCCESS;
  }

  std::once_flag initializeFlag;
  std::unordered_map<uint64_t, uint32_t> agentToDevice;
  std::vector<rocprofiler_agent_id_t> gpuAgentIds;
};

std::shared_ptr<Metric> convertDispatchToMetric(
    const rocprofiler_buffer_tracing_kernel_dispatch_record_t *record) {
  if (record->start_timestamp >= record->end_timestamp)
    return nullptr;
  auto deviceId = static_cast<uint64_t>(
      AgentIdMapper::instance().map(record->dispatch_info.agent_id.handle));
  return std::make_shared<KernelMetric>(
      static_cast<uint64_t>(record->start_timestamp),
      static_cast<uint64_t>(record->end_timestamp), 1, deviceId,
      static_cast<uint64_t>(DeviceType::HIP),
      static_cast<uint64_t>(record->dispatch_info.queue_id.handle));
}

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

void ensureRocprofilerConfigured() {
  std::call_once(configureOnce, []() {
    // Check if already configured (e.g., via ROCP_TOOL_LIBRARIES env var)
    auto &state = getRuntimeState();
    if (state.configured) {
      return;
    }

    int status = 0;
    rocprofiler::isInitialized<true>(&status);
    if (status > 0) {
      // rocprofiler is initialized - check if this is our PC sampling setup
      // When ROCP_TOOL_LIBRARIES loads libproton.so, it gets a separate static
      // state from the Python-loaded version. Check if PC sampling was enabled
      // via environment variable - if so, we configured it ourselves.
      if (state.configured) {
        return;
      }
      bool pcSamplingEnabled = getBoolEnv("PROTON_PC_SAMPLING", false);
      if (pcSamplingEnabled) {
        // PC sampling was configured via ROCP_TOOL_LIBRARIES - mark this
        // instance as configured and continue
        state.configured = true;
        std::cerr << "[PROTON] Using PC sampling configuration from "
                     "ROCP_TOOL_LIBRARIES"
                  << std::endl;
        return;
      }
      throw std::runtime_error(
          "[PROTON] ROCProfiler-SDK is already configured by another tool");
    }
    auto ret = rocprofiler::forceConfigure<true>(&rocprofiler_configure);
    if (ret != ROCPROFILER_STATUS_SUCCESS) {
      throw std::runtime_error(
          "[PROTON] Failed to configure ROCProfiler-SDK runtime");
    }
  });

  if (!getRuntimeState().configured) {
    throw std::runtime_error(
        "[PROTON] ROCProfiler-SDK runtime is not initialized");
  }
}

} // namespace

struct RocprofProfiler::RocprofProfilerPimpl
    : public GPUProfiler<RocprofProfiler>::GPUProfilerPimplInterface {
  RocprofProfilerPimpl(RocprofProfiler &profiler)
      : GPUProfiler<RocprofProfiler>::GPUProfilerPimplInterface(profiler) {}
  virtual ~RocprofProfilerPimpl() = default;

  void doStart() override {
    ensureRocprofilerConfigured();
    auto &state = getRuntimeState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.started) {
      // When PC sampling is configured via ROCP_TOOL_LIBRARIES, the context
      // is managed by the other library instance. Check if we're in that mode
      // by seeing if our context handle is invalid (0).
      if (state.context.handle != 0) {
        rocprofiler::startContext<true>(state.context);
      }
      // Even if we didn't start a context, mark as started to prevent retries
      state.started = true;
    }
  }

  void doFlush() override {
    ensureRocprofilerConfigured();
    auto &state = getRuntimeState();
    std::ignore = hip::deviceSynchronize<true>();

    // When PC sampling is configured via ROCP_TOOL_LIBRARIES, this instance
    // doesn't have valid buffers - the other instance handles flushing
    if (state.context.handle == 0) {
      // No-op for the Python instance when using ROCP_TOOL_LIBRARIES
      return;
    }

    if (state.pcSamplingEnabled) {
      profiler.correlation.flush(
          /*maxRetries=*/100, /*sleepMs=*/10,
          /*flushFn=*/
          [&state]() {
            rocprofiler::flushBuffer<true>(state.pcSamplingBuffer);
          });
    } else {
      profiler.correlation.flush(
          /*maxRetries=*/100, /*sleepMs=*/10,
          /*flushFn=*/
          [&state]() { rocprofiler::flushBuffer<true>(state.kernelBuffer); });
    }
  }

  void doStop() override {
    auto &state = getRuntimeState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.started) {
      // Only stop context if we have a valid handle (not the Python instance
      // when using ROCP_TOOL_LIBRARIES)
      if (state.context.handle != 0) {
        rocprofiler::stopContext<true>(state.context);
      }
      state.started = false;
    }
  }

  static void hipRuntimeCallback(rocprofiler_callback_tracing_record_t record,
                                 rocprofiler_user_data_t *userData, void *arg);
  static void markerCallback(rocprofiler_callback_tracing_record_t record,
                             rocprofiler_user_data_t *userData, void *arg);
  static void codeObjectCallback(rocprofiler_callback_tracing_record_t record,
                                 rocprofiler_user_data_t *userData, void *arg);
  static void kernelBufferCallback(rocprofiler_context_id_t context,
                                   rocprofiler_buffer_id_t buffer,
                                   rocprofiler_record_header_t **headers,
                                   size_t numHeaders, void *userData,
                                   uint64_t dropCount);
  static void pcSamplingBufferCallback(rocprofiler_context_id_t context,
                                       rocprofiler_buffer_id_t buffer,
                                       rocprofiler_record_header_t **headers,
                                       size_t numHeaders, void *userData,
                                       uint64_t dropCount);

  using KernelNameMap =
      ThreadSafeMap<uint64_t, std::string,
                    std::unordered_map<uint64_t, std::string>>;

  std::string getKernelName(uint64_t kernelId) {
    if (kernelNames.contain(kernelId)) {
      std::string name = kernelNames[kernelId];
      // Strip ".kd" suffix (AMD kernel descriptor) for consistency
      const std::string suffix = ".kd";
      if (name.size() > suffix.size() &&
          name.compare(name.size() - suffix.size(), suffix.size(), suffix) ==
              0) {
        name = name.substr(0, name.size() - suffix.size());
      }
      return name;
    }
    return UnknownKernelName;
  }

  void setKernelName(uint64_t kernelId, const char *name) {
    if (name == nullptr)
      return;
    kernelNames[kernelId] = std::string(name);
  }

  ThreadSafeMap<uint64_t, bool, std::unordered_map<uint64_t, bool>>
      CorrIdToIsHipGraph;

  ThreadSafeMap<hipGraphExec_t, hipGraph_t,
                std::unordered_map<hipGraphExec_t, hipGraph_t>>
      GraphExecToGraph;

  ThreadSafeMap<hipGraph_t, uint32_t, std::unordered_map<hipGraph_t, uint32_t>>
      GraphToNumInstances;

  ThreadSafeMap<hipStream_t, uint32_t,
                std::unordered_map<hipStream_t, uint32_t>>
      StreamToCaptureCount;

  ThreadSafeMap<hipStream_t, bool, std::unordered_map<hipStream_t, bool>>
      StreamToCapture;

  KernelNameMap kernelNames;

private:
  static void processKernelRecord(
      RocprofProfiler &profiler, RocprofProfilerPimpl &impl,
      const rocprofiler_buffer_tracing_kernel_dispatch_record_t *record);
};

namespace {} // namespace

void RocprofProfiler::RocprofProfilerPimpl::hipRuntimeCallback(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_user_data_t *userData, void *arg) {
  if (record.kind != ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API)
    return;

  auto operation =
      static_cast<rocprofiler_tracing_operation_t>(record.operation);
  bool isKernelOp = isKernelLaunchOperation(operation);
  auto &profiler = RocprofProfiler::instance();
  auto *impl = static_cast<RocprofProfiler::RocprofProfilerPimpl *>(
      profiler.pImpl.get());
  auto *payload = static_cast<rocprofiler_callback_tracing_hip_api_data_t *>(
      record.payload);

  if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
    if (!isKernelOp)
      return;
    threadState.enterOp();
    size_t numInstances = 1;
    if (operation == ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch) {
      impl->CorrIdToIsHipGraph[record.correlation_id.internal] = true;
      numInstances = std::numeric_limits<size_t>::max();
      bool foundGraph = false;
      auto graphExec = payload->args.hipGraphLaunch.graphExec;
      if (impl->GraphExecToGraph.contain(graphExec)) {
        auto graph = impl->GraphExecToGraph[graphExec];
        if (impl->GraphToNumInstances.contain(graph)) {
          numInstances = impl->GraphToNumInstances[graph];
          foundGraph = true;
        }
      }
      if (!foundGraph) {
        std::cerr
            << "[PROTON] Unable to determine hipGraph kernel count. Start "
               "profiling before creating graphs to avoid leaks."
            << std::endl;
      }
    }
    profiler.correlation.correlate(record.correlation_id.internal,
                                   numInstances);
    return;
  }

  if (record.phase != ROCPROFILER_CALLBACK_PHASE_EXIT)
    return;

  switch (operation) {
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture: {
    auto stream = payload->args.hipStreamBeginCapture.stream;
    impl->StreamToCapture[stream] = true;
    impl->StreamToCaptureCount[stream] = 0;
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture: {
    auto stream = payload->args.hipStreamEndCapture.stream;
    auto graph = *(payload->args.hipStreamEndCapture.pGraph);
    uint32_t captured = impl->StreamToCaptureCount.contain(stream)
                            ? impl->StreamToCaptureCount[stream]
                            : 0;
    impl->GraphToNumInstances[graph] = captured;
    impl->StreamToCapture.erase(stream);
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel: {
    auto stream = payload->args.hipLaunchKernel.stream;
    if (impl->StreamToCapture.contain(stream))
      impl->StreamToCaptureCount[stream]++;
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel: {
    auto stream = payload->args.hipExtLaunchKernel.stream;
    if (impl->StreamToCapture.contain(stream))
      impl->StreamToCaptureCount[stream]++;
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel: {
    auto stream = payload->args.hipLaunchCooperativeKernel.stream;
    if (impl->StreamToCapture.contain(stream))
      impl->StreamToCaptureCount[stream]++;
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel: {
    auto stream = payload->args.hipModuleLaunchKernel.stream;
    if (impl->StreamToCapture.contain(stream))
      impl->StreamToCaptureCount[stream]++;
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernel: {
    auto stream = payload->args.hipModuleLaunchCooperativeKernel.stream;
    if (impl->StreamToCapture.contain(stream))
      impl->StreamToCaptureCount[stream]++;
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithFlags: {
    auto graph = payload->args.hipGraphInstantiateWithFlags.graph;
    auto graphExec = *(payload->args.hipGraphInstantiateWithFlags.pGraphExec);
    impl->GraphExecToGraph[graphExec] = graph;
    break;
  }
  case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiate: {
    auto graph = payload->args.hipGraphInstantiate.graph;
    auto graphExec = *(payload->args.hipGraphInstantiate.pGraphExec);
    impl->GraphExecToGraph[graphExec] = graph;
    break;
  }
  default:
    break;
  }

  if (isKernelOp) {
    threadState.exitOp();
    profiler.correlation.submit(record.correlation_id.internal);
  }
}

void RocprofProfiler::RocprofProfilerPimpl::markerCallback(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_user_data_t *userData, void *arg) {
  if (record.kind != ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API)
    return;
  auto *payload = static_cast<rocprofiler_callback_tracing_marker_api_data_t *>(
      record.payload);
  auto op = static_cast<rocprofiler_tracing_operation_t>(record.operation);
  if (op == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA &&
      record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
    threadState.enterScope(payload->args.roctxRangePushA.message);
  } else if (op == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop &&
             record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) {
    threadState.exitScope();
  }
}

void RocprofProfiler::RocprofProfilerPimpl::codeObjectCallback(
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
  auto &profiler = RocprofProfiler::instance();
  auto *impl = static_cast<RocprofProfiler::RocprofProfilerPimpl *>(
      profiler.pImpl.get());
  impl->setKernelName(payload->kernel_id, payload->kernel_name);
}

void RocprofProfiler::RocprofProfilerPimpl::kernelBufferCallback(
    rocprofiler_context_id_t context, rocprofiler_buffer_id_t buffer,
    rocprofiler_record_header_t **headers, size_t numHeaders, void *userData,
    uint64_t dropCount) {
  if (dropCount > 0) {
    std::cerr << "[PROTON] ROCProfiler-SDK dropped " << dropCount
              << " kernel dispatch records" << std::endl;
  }
  auto &profiler = RocprofProfiler::instance();
  auto *impl = static_cast<RocprofProfiler::RocprofProfilerPimpl *>(
      profiler.pImpl.get());
  uint64_t maxCorrelationId = 0;
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
    RocprofProfiler::RocprofProfilerPimpl::processKernelRecord(profiler, *impl,
                                                               record);
  }
  if (maxCorrelationId > 0) {
    profiler.correlation.complete(maxCorrelationId);
  }
}

void RocprofProfiler::RocprofProfilerPimpl::processKernelRecord(
    RocprofProfiler &profiler, RocprofProfilerPimpl &impl,
    const rocprofiler_buffer_tracing_kernel_dispatch_record_t *record) {
  auto metric = convertDispatchToMetric(record);
  if (!metric)
    return;

  auto &correlation = profiler.correlation;
  auto hasCorrelation =
      correlation.corrIdToExternId.contain(record->correlation_id.internal);
  auto externId =
      hasCorrelation
          ? correlation.corrIdToExternId[record->correlation_id.internal].first
          : Scope::DummyScopeId;
  auto isAPI = correlation.apiExternIds.contain(externId);
  bool isGraph =
      impl.CorrIdToIsHipGraph.contain(record->correlation_id.internal);

  auto dataSet = profiler.getDataSet();
  if (externId == Scope::DummyScopeId)
    isAPI = false;

  auto kernelName = impl.getKernelName(record->dispatch_info.kernel_id);

  if (!isGraph) {
    for (auto *data : dataSet) {
      auto scopeId = externId;
      if (isAPI) {
        scopeId = data->addOp(externId, kernelName);
      }
      data->addMetric(scopeId, metric);
    }
  } else {
    for (auto *data : dataSet) {
      auto childId = data->addOp(externId, kernelName);
      data->addMetric(childId, metric);
    }
  }

  if (hasCorrelation) {
    auto &[parentId, remaining] =
        correlation.corrIdToExternId[record->correlation_id.internal];
    if (remaining > 1) {
      correlation.corrIdToExternId[record->correlation_id.internal].second =
          remaining - 1;
    } else {
      correlation.corrIdToExternId.erase(record->correlation_id.internal);
    }
  } else {
    correlation.apiExternIds.erase(externId);
  }

  if (isGraph) {
    impl.CorrIdToIsHipGraph.erase(record->correlation_id.internal);
  }
}

void RocprofProfiler::RocprofProfilerPimpl::pcSamplingBufferCallback(
    rocprofiler_context_id_t context, rocprofiler_buffer_id_t buffer,
    rocprofiler_record_header_t **headers, size_t numHeaders, void *userData,
    uint64_t dropCount) {

  static size_t totalSamplesReceived = 0;
  totalSamplesReceived += numHeaders;

  std::cerr << "[PROTON PC_SAMPLING] Callback invoked: " << numHeaders
            << " samples (total so far: " << totalSamplesReceived << ")";
  if (dropCount > 0) {
    std::cerr << ", DROPPED: " << dropCount;
  }
  std::cerr << std::endl;

  auto &profiler = RocprofProfiler::instance();
  auto *impl = static_cast<RocprofProfiler::RocprofProfilerPimpl *>(
      profiler.pImpl.get());
  auto dataSet = profiler.getDataSet();

  size_t stochasticSamples = 0;
  size_t stalledSamples = 0;

  for (size_t i = 0; i < numHeaders; ++i) {
    auto *header = headers[i];
    if (header->category != ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING) {
      std::cerr << "[PROTON PC_SAMPLING] Warning: Unexpected category "
                << header->category << std::endl;
      continue;
    }

    if (header->kind == ROCPROFILER_PC_SAMPLING_RECORD_STOCHASTIC_V0_SAMPLE) {
      stochasticSamples++;
      auto *sample =
          static_cast<rocprofiler_pc_sampling_record_stochastic_v0_t *>(
              header->payload);

      // Determine stall reason
      RocprofPCSamplingMetric::RocprofPCSamplingMetricKind stallKind =
          RocprofPCSamplingMetric::NotIssuedNone;
      bool isStalled = !sample->wave_issued;

      if (isStalled) {
        auto reason = static_cast<
            rocprofiler_pc_sampling_instruction_not_issued_reason_t>(
            sample->snapshot.reason_not_issued);
        switch (reason) {
        case ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NONE:
          stallKind = RocprofPCSamplingMetric::NotIssuedNone;
          break;
        case ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE:
          stallKind = RocprofPCSamplingMetric::NotIssuedNoInstruction;
          break;
        case ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY:
          stallKind = RocprofPCSamplingMetric::NotIssuedAluDependency;
          break;
        case ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT:
          stallKind = RocprofPCSamplingMetric::NotIssuedWaitcnt;
          break;
        case ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_INTERNAL_INSTRUCTION:
          stallKind = RocprofPCSamplingMetric::NotIssuedInternal;
          break;
        case ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_BARRIER_WAIT:
          stallKind = RocprofPCSamplingMetric::NotIssuedBarrier;
          break;
        case ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN:
          stallKind = RocprofPCSamplingMetric::NotIssuedArbiterNotWin;
          break;
        case ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL:
          stallKind = RocprofPCSamplingMetric::NotIssuedArbiterExStall;
          break;
        case ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT:
          stallKind = RocprofPCSamplingMetric::NotIssuedOtherWait;
          break;
        case ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_SLEEP_WAIT:
          stallKind = RocprofPCSamplingMetric::NotIssuedSleep;
          break;
        default:
          stallKind = RocprofPCSamplingMetric::NotIssuedNone;
          break;
        }
      }

      if (isStalled) {
        stalledSamples++;
      }

      // Debug output for first few samples
      if (stochasticSamples <= 5) {
        std::cerr << "[PROTON PC_SAMPLING] Sample #" << stochasticSamples
                  << ": "
                  << "code_obj_id=" << sample->pc.code_object_id << " offset=0x"
                  << std::hex << sample->pc.code_object_offset << std::dec
                  << " corr_id=" << sample->correlation_id.internal
                  << " wave_issued=" << (int)sample->wave_issued
                  << " timestamp=" << sample->timestamp;
        if (isStalled) {
          std::cerr << " STALLED(reason=" << sample->snapshot.reason_not_issued
                    << ")";
        }
        std::cerr << std::endl;
      }

      auto metric = std::make_shared<RocprofPCSamplingMetric>(
          stallKind, 1, isStalled ? 1 : 0);

      // Try to find kernel name from correlation
      // For now, use a generic scope - correlation with kernels requires more
      // work
      auto &correlation = profiler.correlation;
      auto hasCorrelation =
          correlation.corrIdToExternId.contain(sample->correlation_id.internal);
      auto externId =
          hasCorrelation
              ? correlation.corrIdToExternId[sample->correlation_id.internal]
                    .first
              : Scope::DummyScopeId;

      // Debug correlation lookup
      if (stochasticSamples <= 5) {
        std::cerr << "[PROTON PC_SAMPLING]   Correlation lookup: "
                  << "hasCorrelation=" << hasCorrelation
                  << " externId=" << externId << std::endl;
      }

      // Add metric to data
      for (auto *data : dataSet) {
        data->addMetric(externId, metric);
      }
    }
    // Only stochastic sampling is supported - ignore other record types
  }

  std::cerr << "[PROTON PC_SAMPLING] Processed " << stochasticSamples
            << " stochastic samples (" << stalledSamples << " stalled)"
            << std::endl;
}

RocprofProfiler::RocprofProfiler() {
  pImpl = std::make_unique<RocprofProfilerPimpl>(*this);
  ensureRocprofilerConfigured();
}

RocprofProfiler::~RocprofProfiler() = default;

void RocprofProfiler::doSetMode(
    const std::vector<std::string> &modeAndOptions) {
  auto mode = modeAndOptions.empty() ? std::string() : modeAndOptions[0];
  if (!mode.empty()) {
    throw std::invalid_argument("[PROTON] RocprofProfiler: unsupported mode: " +
                                mode);
  }
}

extern "C" {

rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t version, const char *runtimeVersion,
                      uint32_t priority, rocprofiler_client_id_t *id);

} // extern "C"

namespace {

// Helper to configure stochastic PC sampling for a single agent
bool configurePCSamplingForAgent(RocprofilerRuntimeState *state,
                                 rocprofiler_agent_id_t agentId) {
  struct ConfigData {
    bool hasStochastic = false;
    rocprofiler_pc_sampling_unit_t stochasticUnit{};
    uint64_t stochasticMinInterval = 0;
    uint64_t stochasticMaxInterval = 0;
  };
  ConfigData configData;

  auto configCallback =
      [](const rocprofiler_pc_sampling_configuration_t *configs,
         size_t numConfigs, void *userData) {
        auto *data = static_cast<ConfigData *>(userData);
        for (size_t i = 0; i < numConfigs; ++i) {
          if (configs[i].method == ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC) {
            data->hasStochastic = true;
            data->stochasticUnit = configs[i].unit;
            data->stochasticMinInterval = configs[i].min_interval;
            data->stochasticMaxInterval = configs[i].max_interval;
            break; // Found stochastic, that's all we need
          }
        }
        return ROCPROFILER_STATUS_SUCCESS;
      };

  auto queryStatus = rocprofiler::queryPCSamplingAgentConfigurations<false>(
      agentId, configCallback, &configData);

  if (queryStatus != ROCPROFILER_STATUS_SUCCESS) {
    std::cerr << "[PROTON] Failed to query PC sampling configurations for agent"
              << std::endl;
    return false;
  }

  if (!configData.hasStochastic) {
    std::cerr << "[PROTON] Stochastic PC sampling not available for agent. "
              << "Ensure ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1 is set."
              << std::endl;
    return false;
  }

  // this should be configurable probably
  uint64_t interval =
      std::max(configData.stochasticMinInterval, uint64_t(65536));
  // Clamp to max if needed
  if (configData.stochasticMaxInterval > 0) {
    interval = std::min(interval, configData.stochasticMaxInterval);
  }

  auto configStatus = rocprofiler::configurePCSamplingService<false>(
      state->context, agentId, ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC,
      configData.stochasticUnit, interval, state->pcSamplingBuffer, 0);

  if (configStatus != ROCPROFILER_STATUS_SUCCESS) {
    std::cerr << "[PROTON] Failed to configure stochastic PC sampling: "
              << configStatus << std::endl;
    return false;
  }

  std::cerr << "[PROTON] ✓ Configured stochastic PC sampling for agent "
            << agentId.handle << " with interval " << interval << " cycles"
            << std::endl;
  return true;
}

int proton_tool_init(rocprofiler_client_finalize_t finiFunc, void *toolData) {
  auto *state = static_cast<RocprofilerRuntimeState *>(toolData);
  state->finalizeFunc = finiFunc;

  rocprofiler::createContext<true>(&state->context);

  bool enableMarkers = getBoolEnv("TRITON_ENABLE_NVTX", true);
  state->markerCallbacksEnabled = enableMarkers;

  bool enablePCSampling =
      getBoolEnv("PROTON_PC_SAMPLING", true); // change later
  state->pcSamplingEnabled = enablePCSampling;

  if (enablePCSampling) {
    std::cerr << "[PROTON] PC sampling mode enabled via PROTON_PC_SAMPLING=1"
              << std::endl;
    // Check if beta flag is set
    bool betaEnabled =
        getBoolEnv("ROCPROFILER_PC_SAMPLING_BETA_ENABLED", false);
    if (!betaEnabled) {
      std::cerr << "[PROTON] WARNING: ROCPROFILER_PC_SAMPLING_BETA_ENABLED is "
                   "not set. "
                << "PC sampling may not work." << std::endl;
    }
  }

  rocprofiler::configureCallbackTracingService<true>(
      state->context, ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API, nullptr, 0,
      &RocprofProfiler::RocprofProfilerPimpl::hipRuntimeCallback, nullptr);

  if (enableMarkers) {
    rocprofiler::configureCallbackTracingService<true>(
        state->context, ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API, nullptr,
        0, &RocprofProfiler::RocprofProfilerPimpl::markerCallback, nullptr);
  }

  // Code object tracing (needed for kernel names AND PC sampling symbol
  // mapping)
  const rocprofiler_tracing_operation_t codeObjectOps[] = {
      ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};
  rocprofiler::configureCallbackTracingService<true>(
      state->context, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, codeObjectOps,
      1, &RocprofProfiler::RocprofProfilerPimpl::codeObjectCallback, nullptr);

  size_t watermark = BufferSize - (BufferSize / 8);

  // Initialize agent mapper early - needed for PC sampling configuration
  AgentIdMapper::instance().initialize();

  if (enablePCSampling) {
    // Create PC sampling buffer
    rocprofiler::createBuffer<true>(
        state->context, BufferSize, watermark,
        ROCPROFILER_BUFFER_POLICY_LOSSLESS,
        &RocprofProfiler::RocprofProfilerPimpl::pcSamplingBufferCallback,
        nullptr, &state->pcSamplingBuffer);

    std::cerr << "[PROTON] Configuring stochastic PC sampling for "
              << AgentIdMapper::instance().getGpuAgentIds().size()
              << " GPU agent(s)..." << std::endl;

    bool anyConfigured = false;
    for (const auto &agentId : AgentIdMapper::instance().getGpuAgentIds()) {
      std::cerr << "[PROTON]   Attempting agent " << agentId.handle << "..."
                << std::endl;
      if (configurePCSamplingForAgent(state, agentId)) {
        anyConfigured = true;
      }
    }

    if (!anyConfigured) {
      std::cerr << "[PROTON] ERROR: PC sampling requested but no agents "
                   "support it. Falling back to kernel dispatch tracing."
                << std::endl;
      state->pcSamplingEnabled = false;
    } else {
      std::cerr << "[PROTON] ✓ PC sampling configured successfully"
                << std::endl;
      rocprofiler::createCallbackThread<true>(&state->pcSamplingCallbackThread);
      rocprofiler::assignCallbackThread<true>(state->pcSamplingBuffer,
                                              state->pcSamplingCallbackThread);
    }
  }

  if (!state->pcSamplingEnabled) {
    rocprofiler::createBuffer<true>(
        state->context, BufferSize, watermark,
        ROCPROFILER_BUFFER_POLICY_LOSSLESS,
        &RocprofProfiler::RocprofProfilerPimpl::kernelBufferCallback, nullptr,
        &state->kernelBuffer);

    rocprofiler::configureBufferTracingService<true>(
        state->context, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0,
        state->kernelBuffer);

    rocprofiler::createCallbackThread<true>(&state->callbackThread);
    rocprofiler::assignCallbackThread<true>(state->kernelBuffer,
                                            state->callbackThread);
  }

  int valid = 0;
  rocprofiler::contextIsValid<true>(state->context, &valid);
  if (valid == 0) {
    std::cerr << "[PROTON] ERROR: Context is invalid" << std::endl;
    return -1;
  }

  rocprofiler::startContext<true>(state->context);
  state->started = true;
  state->configured = true;

  std::cerr << "[PROTON] ✓ Context started, profiling active" << std::endl;
  return 0;
}

void proton_tool_fini(void *toolData) {
  std::cerr << "[PROTON] proton_tool_fini called" << std::endl;
  auto *state = static_cast<RocprofilerRuntimeState *>(toolData);
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->started) {
      std::cerr << "[PROTON] Stopping context..." << std::endl;
      rocprofiler::stopContext<false>(state->context);
      state->started = false;
    }
  }
  if (state->pcSamplingEnabled) {
    std::cerr << "[PROTON] Flushing PC sampling buffer..." << std::endl;
    rocprofiler::flushBuffer<false>(state->pcSamplingBuffer);
  } else {
    std::cerr << "[PROTON] Flushing kernel buffer..." << std::endl;
    rocprofiler::flushBuffer<false>(state->kernelBuffer);
  }
  std::cerr << "[PROTON] proton_tool_fini complete" << std::endl;
  if (state->finalizeFunc && state->clientId) {
    state->finalizeFunc(*state->clientId);
  }
}

} // namespace

// Must be exported for rocprofiler-register to find via ROCP_TOOL_LIBRARIES
extern "C" __attribute__((visibility("default")))
rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t version, const char *runtimeVersion,
                      uint32_t priority, rocprofiler_client_id_t *id) {
  auto &state = getRuntimeState();
  id->name = "ProtonRocprofiler";
  state.clientId = id;
  static rocprofiler_tool_configure_result_t config{
      sizeof(rocprofiler_tool_configure_result_t), &proton_tool_init,
      &proton_tool_fini, static_cast<void *>(&state)};
  (void)version;
  (void)runtimeVersion;
  (void)priority;
  return &config;
}

} // namespace proton
