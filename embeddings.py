"""Load the specified embedding model on demand; generate normalized movie vectors."""
from functools import lru_cache
from pathlib import Path
from config import MODEL_CACHE, MODEL_NAME

class ModelLoadError(RuntimeError):
    pass

@lru_cache(maxsize=1)
def load_embedding_model(cache_folder: Path = MODEL_CACHE):
    try:
        from sentence_transformers import SentenceTransformer
        cache_folder = Path(cache_folder)
        cache_folder.mkdir(parents=True, exist_ok=True)
        return SentenceTransformer(MODEL_NAME, cache_folder=str(cache_folder), device="cpu", trust_remote_code=False)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise ModelLoadError("Could not load all-MiniLM-L6-v2. Check installed dependencies, internet access for the first download, and model-cache permissions.") from error


EMBEDDING_DIMENSION = 384


def embed_movies(movies, model=None) -> list[list[float]]:
    """Encode validated movie text and reject malformed model output."""
    import numpy as np
    from dataset import format_movie
    if not movies:
        raise ModelLoadError("Cannot embed an empty movie collection.")
    if model is None:
        model = load_embedding_model()
    try:
        vectors = np.asarray(model.encode(
            [format_movie(movie) for movie in movies],
            batch_size=32, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ), dtype=float)
        if vectors.shape != (len(movies), EMBEDDING_DIMENSION):
            raise ValueError("Unexpected embedding dimensions")
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise ValueError("Invalid or unnormalized vectors")
        return vectors.tolist()
    except (ValueError, RuntimeError, TypeError) as error:
        raise ModelLoadError("Could not generate valid normalized 384-dimensional movie vectors.") from error
