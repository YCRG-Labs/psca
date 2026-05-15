
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
from .runner import run_batch, save_results
from .analysis import (
    anes_benchmark,
    bootstrap_ci,
    compute_partisan_gaps,
    derive_coverage_threshold,
    fisher_rz_dimension_test,
    flipped_spec_analysis,
    load_results,
    permutation_inference,
    profile_jackknife,
    run_analysis,
    sobol_analysis,
    specification_curve,
    variance_decomposition,
)

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
