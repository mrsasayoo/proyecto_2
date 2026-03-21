"""
Re-exports de los 4 algoritmos de routing del ablation study.

Uso:
    from routers import (
        LinearGatingHead, train_linear_router,
        train_gmm_router, train_nb_router, train_knn_router,
    )
"""

from .linear import LinearGatingHead, train_linear_router
from .gmm import train_gmm_router
from .naive_bayes import train_nb_router
from .knn import train_knn_router

__all__ = [
    "LinearGatingHead",
    "train_linear_router",
    "train_gmm_router",
    "train_nb_router",
    "train_knn_router",
]
