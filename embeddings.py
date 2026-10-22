"""Load the specified embedding model on demand; vector generation comes next."""
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
