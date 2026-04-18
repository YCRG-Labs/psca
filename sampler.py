import numpy as np
from scipy.stats.qmc import LatinHypercube
from SALib.sample import saltelli as saltelli_sampler
from config import DIMENSIONS, DIMENSION_ORDER


def _discretize(continuous_samples, dim_order, dim_levels):
    specifications = []
    for i, row in enumerate(continuous_samples):
        spec = {"spec_id": i}
        for j, dim_name in enumerate(dim_order):
            levels = dim_levels[dim_name]
            idx = int(np.clip(np.floor(row[j]), 0, len(levels) - 1))
            spec[dim_name] = levels[idx]
        specifications.append(spec)
    return specifications


def generate_specifications(n_samples, seed=42, only_models=None, spec_id_offset=0):
    """Generate LHS specifications.

    only_models: optional list of model identifiers to restrict the model
                 dimension to. Used by the v2 incremental runs (e.g. add
                 Claude Haiku to the existing 3-item dataset without
                 re-querying the other systems).
    spec_id_offset: integer added to spec_id so that runs concatenated
                    across multiple invocations do not collide.
    """
    n_dims = len(DIMENSION_ORDER)
    sampler = LatinHypercube(d=n_dims, seed=seed)
    lhs_samples = sampler.random(n=n_samples)

    specifications = []
    for i, sample in enumerate(lhs_samples):
        spec = {"spec_id": i + spec_id_offset}
        for j, dim_name in enumerate(DIMENSION_ORDER):
            if dim_name == "model" and only_models is not None:
                levels = list(only_models)
            else:
                levels = DIMENSIONS[dim_name]
            idx = min(int(sample[j] * len(levels)), len(levels) - 1)
            spec[dim_name] = levels[idx]
        specifications.append(spec)

    return specifications


def get_saltelli_problem():
    return {
        "num_vars": len(DIMENSION_ORDER),
        "names": list(DIMENSION_ORDER),
        "bounds": [[0, len(DIMENSIONS[d])] for d in DIMENSION_ORDER],
    }


def generate_saltelli_specifications(n_base=512, calc_second_order=False, seed=42):
    problem = get_saltelli_problem()
    np.random.seed(seed)
    param_values = saltelli_sampler.sample(
        problem, n_base,
        calc_second_order=calc_second_order,
    )
    total = len(param_values)
    specs = _discretize(param_values, DIMENSION_ORDER, DIMENSIONS)
    return specs, problem, total
