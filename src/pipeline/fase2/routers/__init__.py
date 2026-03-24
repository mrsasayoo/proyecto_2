"""
Paquete routers — Algoritmos de routing del ablation study.

Re-exporta los cinco símbolos públicos para que ablation_runner.py
pueda importarlos desde un único punto.
"""

from linear import LinearGatingHead, train_linear_router
from gmm import train_gmm_router
from naive_bayes import train_nb_router
from knn import train_knn_router

__all__ = [
    "LinearGatingHead",
    "train_linear_router",
    "train_gmm_router",
    "train_nb_router",
    "train_knn_router",
]
