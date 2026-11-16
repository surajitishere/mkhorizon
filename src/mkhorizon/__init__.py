"""mkhorizon: how far back can a discrete trait be reconstructed?"""

from .diagnose import (
    expected_information,
    horizon,
    scale_profile,
    signal_test,
    verdict,
)
from .fit import FitResult, fit_best, fit_mk, saturated_loglik
from .likelihood import Engine, entropy_bits, tip_likelihoods
from .model import MkModel, build_q, signal_half_life, stationary_distribution
from .tree import Tree, parse_newick, read_newick

__version__ = "0.1.0"

__all__ = [
    "Engine",
    "FitResult",
    "MkModel",
    "Tree",
    "build_q",
    "entropy_bits",
    "expected_information",
    "fit_best",
    "fit_mk",
    "horizon",
    "parse_newick",
    "read_newick",
    "saturated_loglik",
    "scale_profile",
    "signal_half_life",
    "signal_test",
    "stationary_distribution",
    "tip_likelihoods",
    "verdict",
]
