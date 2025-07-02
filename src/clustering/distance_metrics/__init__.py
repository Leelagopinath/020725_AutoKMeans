from .numeric import Euclidean, Manhattan, Minkowski, Chebyshev, Cosine, Mahalanobis, Pearson, BrayCurtis, Canberra
from .binary import Hamming, Jaccard, Tanimoto, Dice, SimpleMatching, Levenshtein, DamerauLevenshtein, JaroWinkler, Ochiai
from .distribution import EMD, Hellinger, KLDivergence, JensenShannon, Bhattacharyya
from .time_series import DTW, FastDTW, ERP, SBD, Frechet
from .mixed_type import Gower, HEOM, KPrototypes, HVDM
from .graph_structural import GraphEditDistance, SpectralDistance
from .universal import NCD

METRIC_REGISTRY = {
    # Numeric
    "Euclidean": Euclidean,
    "Manhattan": Manhattan,
    "Minkowski": Minkowski,
    "Chebyshev": Chebyshev,
    "Cosine": Cosine,
    "Mahalanobis": Mahalanobis,
    "Pearson": Pearson,
    "Bray-Curtis": BrayCurtis,
    "Canberra": Canberra,
    
    # Binary
    "Hamming": Hamming,
    "Jaccard": Jaccard,
    "Tanimoto": Tanimoto,
    "Dice": Dice,
    "Simple Matching": SimpleMatching,
    "Levenshtein": Levenshtein,
    "Damerau-Levenshtein": DamerauLevenshtein,
    "Jaro-Winkler": JaroWinkler,
    "Ochiai": Ochiai,
    
    # Distribution
    "EMD": EMD,
    "Hellinger": Hellinger,
    "KL Divergence": KLDivergence,
    "Jensen-Shannon": JensenShannon,
    "Bhattacharyya": Bhattacharyya,
    
    # Time Series
    "DTW": DTW,
    "FastDTW": FastDTW,
    "ERP": ERP,
    "SBD": SBD,
    "Fréchet": Frechet,
    
    # Mixed-Type
    "Gower": Gower,
    "HEOM": HEOM,
    "K-Prototypes": KPrototypes,
    "HVDM": HVDM,
    
    # Graph
    "GED": GraphEditDistance,
    "Spectral Distance": SpectralDistance,
    
    # Universal
    "NCD": NCD
}

def get_metric_by_name(name, **kwargs):
    if name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {name}")
    return METRIC_REGISTRY[name](**kwargs)