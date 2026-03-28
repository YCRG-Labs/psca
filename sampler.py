import numpy as np
from scipy.stats.qmc import LatinHypercube
from config import DIMENSIONS, DIMENSION_ORDER


def generate_specifications(n_samples, seed=42):
    n_dims = len(DIMENSION_ORDER)
    sampler = LatinHypercube(d=n_dims, seed=seed)
    lhs_samples = sampler.random(n=n_samples)

    specifications = []
    for i, sample in enumerate(lhs_samples):
        spec = {"spec_id": i}
        for j, dim_name in enumerate(DIMENSION_ORDER):
            levels = DIMENSIONS[dim_name]
            idx = min(int(sample[j] * len(levels)), len(levels) - 1)
            spec[dim_name] = levels[idx]
        specifications.append(spec)

    return specifications
