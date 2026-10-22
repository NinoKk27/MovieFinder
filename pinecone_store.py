"""Pinecone document-index creation and idempotent movie uploads."""
from dataclasses import asdict
import hashlib
import json
import math
from pinecone import Pinecone
from config import MODEL_NAME
from embeddings import EMBEDDING_DIMENSION

VECTOR_FIELD = "embedding"

class IndexingError(RuntimeError):
    pass


def movie_id(movie) -> str:
    identity = json.dumps([" ".join(movie.title.split()).casefold(), movie.year], ensure_ascii=False)
    return "movie-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def build_documents(movies, vectors):
    if not movies or len(movies) != len(vectors):
        raise IndexingError("Movie and vector counts must match and be nonempty.")
    documents = []
    seen = set()
    for movie, vector in zip(movies, vectors):
        identifier = movie_id(movie)
        if identifier in seen:
            raise IndexingError("Duplicate movie IDs in upload.")
        seen.add(identifier)
        if len(vector) != EMBEDDING_DIMENSION or not all(math.isfinite(value) for value in vector):
            raise IndexingError("Every movie requires a finite 384-dimensional vector.")
        if not math.isclose(sum(value * value for value in vector), 1.0, abs_tol=1e-4):
            raise IndexingError("Every movie vector must be normalized.")
        documents.append({"_id": identifier, VECTOR_FIELD: vector, **asdict(movie), "embedding_model": MODEL_NAME})
    return documents


def prepare_index(client, settings, *, create_if_missing=True):
    """Create only when missing; never replace an incompatible existing index."""
    if not client.indexes.exists(settings.index_name):
        if not create_if_missing:
            raise IndexingError("Movie index does not exist. Run main.py --index before searching.")
        client.indexes.create(
            name=settings.index_name,
            schema={"fields": {VECTOR_FIELD: {"type": "dense_vector", "dimension": EMBEDDING_DIMENSION, "metric": "cosine"}}},
            deployment={"deployment_type": "managed", "cloud": settings.cloud, "region": settings.region},
            timeout=60,
        )
    description = client.indexes.describe(settings.index_name)
    schema = getattr(description, "schema", None)
    fields = getattr(schema, "fields", {}) if schema is not None else {}
    vector_field = fields.get(VECTOR_FIELD)
    if (set(fields) != {VECTOR_FIELD} or getattr(vector_field, "type", None) != "dense_vector"
            or getattr(vector_field, "dimension", None) != EMBEDDING_DIMENSION
            or getattr(vector_field, "metric", None) != "cosine"):
        raise IndexingError("Existing index is incompatible. Use a separate index with only a 384-dimensional cosine embedding field; the existing index was not changed.")
    if not description.status.ready:
        raise IndexingError("The index is not ready yet. Retry indexing shortly.")
    return description.host


def upload_movies(settings, movies, vectors, client_factory=Pinecone) -> int:
    """Upsert batches with deterministic IDs; repeating a batch replaces its records."""
    documents = build_documents(movies, vectors)
    with client_factory(api_key=settings.api_key) as client:
        host = prepare_index(client, settings)
        with client.index(host=host) as index:
            accepted = 0
            for start in range(0, len(documents), 50):
                batch = documents[start:start + 50]
                response = index.documents.upsert(namespace=settings.namespace, documents=batch, timeout=30)
                if response.upserted_count != len(batch):
                    raise IndexingError("Pinecone did not accept the full batch. Retry with the same IDs.")
                accepted += response.upserted_count
            return accepted
