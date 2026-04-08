# ruff: noqa
from .scope import scope, cpu_timed_scope, enter_scope, exit_scope
from .state import state, enter_state, exit_state, metadata_state
from .profile import (
    start,
    activate,
    deactivate,
    finalize,
    profile,
    DEFAULT_PROFILE_NAME,
)
from . import context, specs, mode, data


# Eagerly configure rocprofiler-sdk on AMD systems.  When the firmware
# dispatch ring is available, queue interception is not needed and
# force_configure can happen at any time (late-attach is supported).
# When the firmware ring is NOT available, force_configure must run before
# the first HSA queue is created so the SDK can intercept hsa_queue_create.
def _eager_rocprofiler_init():
    try:
        from triton.backends import backends
        if "amd" not in backends:
            return
        from triton._C.libproton import proton as _libproton
        _libproton.ensure_rocprofiler_configured()
    except Exception:
        pass


_eager_rocprofiler_init()
del _eager_rocprofiler_init
