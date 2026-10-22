"""Project configuration; loading settings does not contact Pinecone."""
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from dotenv import dotenv_values

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE = PROJECT_DIR / ".cache" / "models"
DATASET_PATH = PROJECT_DIR / "movies.json"

class ConfigurationError(ValueError):
    pass

@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    index_name: str = "movie-finder"

    cloud: str = "aws"
    region: str = "us-east-1"
    namespace: str = "movies"

    def __post_init__(self):
        key = self.api_key.strip()
        name = self.index_name.strip()
        if not key or key == "replace_with_your_key":
            raise ConfigurationError("Set PINECONE_API_KEY in .env to your Pinecone API key.")
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,43}[a-z0-9])?", name):
            raise ConfigurationError("PINECONE_INDEX_NAME must be 1-45 lowercase letters, digits, or hyphens, starting and ending with a letter or digit.")
        if self.cloud not in ("aws", "gcp", "azure"):
            raise ConfigurationError("PINECONE_CLOUD must be aws, gcp, or azure.")
        if not self.region.strip() or not self.namespace.strip():
            raise ConfigurationError("PINECONE_REGION and PINECONE_NAMESPACE must not be blank.")
        object.__setattr__(self, "region", self.region.strip())
        object.__setattr__(self, "namespace", self.namespace.strip())
        object.__setattr__(self, "api_key", key)
        object.__setattr__(self, "index_name", name)

def load_settings(env_file: Path = PROJECT_DIR / ".env") -> Settings:
    values = dotenv_values(env_file, interpolate=False)
    def setting(name, default):
        return os.environ.get(name, values.get(name, default)) or ""
    return Settings(
        api_key=os.environ.get("PINECONE_API_KEY", values.get("PINECONE_API_KEY")) or "",
        cloud=setting("PINECONE_CLOUD", "aws"),
        region=setting("PINECONE_REGION", "us-east-1"),
        namespace=setting("PINECONE_NAMESPACE", "movies"),
        index_name=os.environ.get("PINECONE_INDEX_NAME", values.get("PINECONE_INDEX_NAME", "movie-finder")) or "",
    )
