#/Users/leelagopinath/Desktop/020725_AutoKMeans/src/clustering/distance_metrics/__init__.py

from .numeric import Euclidean, Manhattan, Minkowski, Chebyshev, Cosine, Mahalanobis, Pearson, BrayCurtis, Canberra
from .binary import Hamming, Jaccard, Tanimoto, Dice, SimpleMatching, Levenshtein, DamerauLevenshtein, JaroWinkler, Ochiai
from .distribution import EMD, Hellinger, KLDivergence, JensenShannon, Bhattacharyya
from .time_series import DTW, FastDTW, ERP, SBD, Frechet
from .mixed_type import Gower, HEOM, KPrototypes, HVDM
from .graph_structural import GraphEditDistance, SpectralDistance
from .universal import NCD

METRIC_REGISTRY = {
    # Numeric
    "euclidean": Euclidean,
    "manhattan": Manhattan,
    "minkowski": Minkowski,
    "chebyshev": Chebyshev,
    "cosine": Cosine,
    "mahalanobis": Mahalanobis,
    "pearson": Pearson,
    "bray-Curtis": BrayCurtis,
    "canberra": Canberra,
    
    # Binary
    "hamming": Hamming,
    "jaccard": Jaccard,
    "tanimoto": Tanimoto,
    "dice": Dice,
    "simple_matching": SimpleMatching,
    "levenshtein": Levenshtein,
    "damerau_levenshtein": DamerauLevenshtein,
    "jaro_winkler": JaroWinkler,
    "ochiai": Ochiai,

    # Distribution
    "emd": EMD,
    "hellinger": Hellinger,
    "kl_divergence": KLDivergence,
    "jensen_shannon": JensenShannon,
    "bhattacharyya": Bhattacharyya,

    # Time Series
    "dtw": DTW,
    "fast_dtw": FastDTW,
    "erp": ERP,
    "sbd": SBD,
    "fréchet": Frechet,

    # Mixed-Type
    "gower": Gower,
    "heom": HEOM,
    "k_prototypes": KPrototypes,
    "hvdm": HVDM,

    # Graph
    "ged": GraphEditDistance,
    "spectral_distance": SpectralDistance,

    # Universal
    "ncd": NCD
}

def get_metric_by_name(name, X=None):
    normalized = name.strip().lower().replace(' ', '_')
    if normalized not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {normalized}")

    metric_class = METRIC_REGISTRY[normalized]
    
    # Handle metrics that don't require initialization data
    if normalized in ["dtw", "fast_dtw"]:
        return metric_class(window=10)  # Default parameters
    
    # Handle metrics that require data initialization
    if X is not None:
        try:
            return metric_class(X)
        except TypeError:
            return metric_class()   # Fallback to no-arg constructor
    return metric_class()