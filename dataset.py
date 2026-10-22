"""Read, validate, and format movie records independently of external services."""
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from config import DATASET_PATH

class DatasetError(ValueError):
    pass

@dataclass(frozen=True)
class Movie:
    title: str
    director: str
    genre: str
    year: int
    description: str

def validate_movies(records: object, minimum_count: int = 20) -> list[Movie]:
    if not isinstance(records, list):
        raise DatasetError("The movie dataset must be a JSON array.")
    movies = []
    seen = set()
    for number, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise DatasetError(f"Movie {number} must be an object.")
        fields = {}
        for name in ("title", "director", "genre", "description"):
            value = record.get(name)
            if not isinstance(value, str) or not value.strip():
                raise DatasetError(f"Movie {number}: {name} must be nonempty text.")
            fields[name] = " ".join(value.split())
        year = record.get("year")
        if type(year) is not int or not 1888 <= year <= date.today().year:
            raise DatasetError(f"Movie {number}: year must be an integer from 1888 to the current year.")
        identity = (fields["title"].casefold(), year)
        if identity in seen:
            raise DatasetError(f"Movie {number}: duplicate title and release year.")
        seen.add(identity)
        movies.append(Movie(**fields, year=year))
    if len(movies) < minimum_count:
        raise DatasetError(f"At least {minimum_count} movies are required; found {len(movies)}.")
    return movies

def load_movies(path: Path = DATASET_PATH) -> list[Movie]:
    try:
        records = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetError("Cannot read the movie dataset. Check the path, UTF-8 encoding, and JSON syntax.") from error
    return validate_movies(records)

def format_movie(movie: Movie) -> str:
    return (f"Title: {movie.title}\nDirector: {movie.director}\n"
            f"Genre: {movie.genre}\nYear: {movie.year}\nDescription: {movie.description}")
