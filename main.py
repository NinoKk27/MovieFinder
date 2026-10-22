"""MovieFinder preparation, indexing, and interactive search commands."""
import argparse
import sys
from config import ConfigurationError, load_settings
from dataset import DatasetError, format_movie, load_movies
from embeddings import ModelLoadError, load_embedding_model, embed_movies
from pinecone_store import IndexingError, upload_movies, prepare_index
from pinecone import Pinecone
from search import DEFAULT_THRESHOLD, run_search, validate_threshold
from errors import PINECONE_ERRORS, pinecone_message

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prepare MovieFinder's dataset and embedding model.")
    parser.add_argument("--check-config", action="store_true", help="Validate Pinecone settings locally; no connection is made.")
    parser.add_argument("--load-model", action="store_true", help="Download/cache and load the embedding model.")
    parser.add_argument("--embed", action="store_true", help="Generate all movie vectors locally without Pinecone.")
    parser.add_argument("--index", action="store_true", help="Create/validate the Pinecone index and upload movie vectors.")
    parser.add_argument("--search", action="store_true", help="Search an existing indexed movie collection interactively.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Minimum cosine similarity for search (default: 0.35, evaluated locally).")
    args = parser.parse_args(argv)
    if args.search and (args.index or args.embed or args.load_model):
        parser.error("Run --search separately from --index, --embed, or --load-model.")
    try:
        validate_threshold(args.threshold)
        if args.search:
            settings = load_settings()
            with Pinecone(api_key=settings.api_key) as client:
                host = prepare_index(client, settings, create_if_missing=False)
                with client.index(host=host) as index:
                    run_search(index, settings.namespace, threshold=args.threshold)
            return 0
        if args.check_config or args.index:
            settings = load_settings()
            print(f"Configuration valid for index: {settings.index_name}")
        movies = load_movies()
        print(f"Validated {len(movies)} movies.")
        print("First searchable record:\n" + format_movie(movies[0]))
        if args.load_model:
            print("Loading all-MiniLM-L6-v2...", flush=True)
            model = load_embedding_model()
            print(f"Model ready. Vector dimension: {model.get_embedding_dimension()}")
        if args.embed or args.index:
            print("Generating movie embeddings...", flush=True)
            vectors = embed_movies(movies)
            print(f"Generated {len(vectors)} normalized vectors with {len(vectors[0])} dimensions.")
            if args.index:
                count = upload_movies(settings, movies, vectors)
                print(f"Pinecone accepted {count} movies in namespace {settings.namespace}. Search visibility may be delayed.")
        print("Preparation complete. Use --search to search an indexed collection.")
        return 0
    except (ConfigurationError, DatasetError, ModelLoadError, IndexingError, ValueError) as error:
        print(f"Setup error: {error}", file=sys.stderr)
        return 1
    except PINECONE_ERRORS as error:
        print(pinecone_message(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return 0

if __name__ == "__main__":
    raise SystemExit(main())
