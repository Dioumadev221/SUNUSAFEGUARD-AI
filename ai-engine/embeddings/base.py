"""Interface commune des générateurs d'embeddings et similarité cosinus.

Tout embedder (Ollama, hachage local) expose `name`, `dim` et `embed(texts)` ;
le moteur de recherche ne dépend que de ce contrat.
"""
import math
from typing import Protocol


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)
