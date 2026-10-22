"""MovieFinder preparation commands. Indexing and search are later milestones."""
import argparse
import sys
from config import ConfigurationError, load_settings
from dataset import DatasetError, format_movie, load_movies
from embeddings import ModelLoadError, load_embedding_model

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prepare MovieFinder's dataset and embedding model.")
    parser.add_argument("--check-config", action="store_true", help="Validate Pinecone settings locally; no connection is made.")
    parser.add_argument("--load-model", action="store_true", help="Download/cache and load the embedding model.")
    args = parser.parse_args(argv)
    try:
        if args.check_config:
            settings = load_settings()
            print(f"Configuration valid for index: {settings.index_name}")
        movies = load_movies()
        print(f"Validated {len(movies)} movies.")
        print("First searchable record:\n" + format_movie(movies[0]))
        if args.load_model:
            print("Loading all-MiniLM-L6-v2...", flush=True)
            model = load_embedding_model()
            print(f"Model ready. Vector dimension: {model.get_embedding_dimension()}")
        print("Preparation complete. Indexing and search are not implemented yet.")
        return 0
    except (ConfigurationError, DatasetError, ModelLoadError) as error:
        print(f"Setup error: {error}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
