"""MovieFinder preparation commands. Indexing and search are later milestones."""
import argparse
import sys
from config import ConfigurationError, load_settings
from dataset import DatasetError, format_movie, load_movies
from embeddings import ModelLoadError, load_embedding_model, embed_movies
from pinecone_store import IndexingError, upload_movies
from pinecone.exceptions import PineconeException

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prepare MovieFinder's dataset and embedding model.")
    parser.add_argument("--check-config", action="store_true", help="Validate Pinecone settings locally; no connection is made.")
    parser.add_argument("--load-model", action="store_true", help="Download/cache and load the embedding model.")
    parser.add_argument("--embed", action="store_true", help="Generate all movie vectors locally without Pinecone.")
    parser.add_argument("--index", action="store_true", help="Create/validate the Pinecone index and upload movie vectors.")
    args = parser.parse_args(argv)
    try:
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
        print("Preparation complete. Search is not implemented yet.")
        return 0
    except (ConfigurationError, DatasetError, ModelLoadError, IndexingError) as error:
        print(f"Setup error: {error}", file=sys.stderr)
        return 1
    except PineconeException:
        print("Pinecone request failed. Check your key, permissions, cloud/region, quota, and connection. You can retry safely using the same movie IDs.", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
