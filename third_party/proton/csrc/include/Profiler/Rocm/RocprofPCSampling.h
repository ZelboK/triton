#ifndef PROTON_PROFILER_ROC_PC_SAMPLING_H_
#define PROTON_PROFILER_ROC_PC_SAMPLING_H_

#include "Utility/Singleton.h"
#include "rocprofiler-sdk/pc_sampling.h" // validate later
#include <rocprofiler-sdk/fwd.h>

namespace proton {

struct AgentPCSamplingConfig {
  rocprofiler_agent_id_t agent_id;
  rocprofiler_pc_sampling_method_t method;
  uint64_t interval;
};

class RocprofPCSampling : public Singleton<RocprofPCSampling> {
public:
  RocprofPCSampling();
  ~RocprofPCSampling();

  void queryAgentPCSamplingConfigurations(rocprofiler_context_id_t ctx);
  static void pcSamplingCallback();

  void processStochasticSample(
      const rocprofiler_pc_sampling_record_stochastic_v0_t *);
};

} // namespace proton

#endif // PROTON_PROFILER_ROC_PC_SAMPLING_H_
