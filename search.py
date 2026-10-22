"""Semantic movie search and result presentation."""
from dataclasses import dataclass
import math
from pinecone import DenseVectorQuery
from config import MODEL_NAME
from embeddings import embed_query
from pinecone_store import VECTOR_FIELD

DEFAULT_THRESHOLD = 0.35
RESULT_FIELDS = ["title", "director", "genre", "year", "description"]

@dataclass(frozen=True)
class MovieMatch:
    title: str
    director: str
    genre: str
    year: int
    description: str
    score: float


def validate_threshold(threshold: float) -> None:
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("Similarity threshold must be a finite number between -1 and 1.")


def search_movies(index, namespace: str, query: str, *, threshold=DEFAULT_THRESHOLD, model=None) -> list[MovieMatch]:
    validate_threshold(threshold)
    vector = embed_query(query, model)
    response = index.documents.search(
        namespace=namespace, top_k=3,
        score_by=[DenseVectorQuery(field=VECTOR_FIELD, values=vector)],
        include_fields=RESULT_FIELDS,
        filter={"embedding_model": {"$eq": MODEL_NAME}},
        timeout=30,
    )
    matches = []
    for document in response.matches:
        score = document.score
        if score is None or not math.isfinite(score) or score < threshold:
            continue
        matches.append(MovieMatch(**{name: getattr(document, name) for name in RESULT_FIELDS}, score=score))
    return sorted(matches, key=lambda match: match.score, reverse=True)[:3]


def format_results(matches: list[MovieMatch]) -> str:
    if not matches:
        return "No matching movies found. Try a different plot, genre, or theme."
    return "\n\n".join(
        f"{number}. {movie.title} ({movie.year})\n"
        f"Director: {movie.director}\nGenre: {movie.genre}\n"
        f"{movie.description}\nSimilarity: {movie.score:.3f}"
        for number, movie in enumerate(matches, 1)
    )


def run_search(index, namespace: str, *, threshold=DEFAULT_THRESHOLD) -> None:
    print("Welcome to MovieFinder! Describe a movie you'd like to watch. Type exit to leave.")
    while True:
        try:
            query = input("Search: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        if query.casefold() == "exit":
            print("Goodbye!")
            return
        if not query:
            print("Describe the kind of movie you want to find.")
            continue
        print(format_results(search_movies(index, namespace, query, threshold=threshold)))
