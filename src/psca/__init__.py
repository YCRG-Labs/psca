
try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

from .config import ANES_ITEMS, DIMENSIONS, PROFILES
from .sampler import (
    generate_saltelli_specifications,
    generate_specifications,
)
from .prompts import build_prompt

_LAZY = {
    "run_batch":              ("psca.runner",   "run_batch"),
    "save_results":           ("psca.runner",   "save_results"),
    "anes_benchmark":         ("psca.analysis", "anes_benchmark"),
    "bootstrap_ci":           ("psca.analysis", "bootstrap_ci"),
    "compute_partisan_gaps":  ("psca.analysis", "compute_partisan_gaps"),
    "derive_coverage_threshold": ("psca.analysis", "derive_coverage_threshold"),
    "fisher_rz_dimension_test": ("psca.analysis", "fisher_rz_dimension_test"),
    "flipped_spec_analysis":  ("psca.analysis", "flipped_spec_analysis"),
    "load_results":           ("psca.analysis", "load_results"),
    "permutation_inference":  ("psca.analysis", "permutation_inference"),
    "profile_jackknife":      ("psca.analysis", "profile_jackknife"),
    "run_analysis":           ("psca.analysis", "run_analysis"),
    "sobol_analysis":         ("psca.analysis", "sobol_analysis"),
    "specification_curve":    ("psca.analysis", "specification_curve"),
    "variance_decomposition": ("psca.analysis", "variance_decomposition"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module_path, attr = _LAZY[name]
        return getattr(importlib.import_module(module_path), attr)
    raise AttributeError(f"module 'psca' has no attribute {name!r}")

__all__ = [
    "ANES_ITEMS",
    "DIMENSIONS",
    "PROFILES",
    "__version__",
    "anes_benchmark",
    "bootstrap_ci",
    "build_prompt",
    "compute_partisan_gaps",
    "derive_coverage_threshold",
    "fisher_rz_dimension_test",
    "flipped_spec_analysis",
    "generate_saltelli_specifications",
    "generate_specifications",
    "load_results",
    "permutation_inference",
    "profile_jackknife",
    "run_analysis",
    "run_batch",
    "save_results",
    "sobol_analysis",
    "specification_curve",
    "variance_decomposition",
]
