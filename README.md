# MovieFinder

Python console application for semantic movie search. This milestone implements
steps 1-10: configuration, a validated dataset, searchable text, and model loading.
Vector generation, Pinecone connections/indexing, and search are later milestones.

## Setup

Use Python 3.11 or newer and select `.venv` as the interpreter in PyCharm.
The existing virtual environment already contains the required libraries.
For a fresh environment, run these commands in the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Top-level dependencies are pinned to the versions verified in this environment.

Validate the dataset and preview its first searchable record without an API key:

```powershell
.\.venv\Scripts\python.exe main.py
```

Download/cache and load the actual embedding model on CPU:

```powershell
.\.venv\Scripts\python.exe main.py --load-model
```

The first load requires internet access and can take time. Model files are cached
under `.cache/models/` inside the project and excluded from Git. The loader reuses
the model within a process. This command does not generate movie vectors or contact
Pinecone. It reports the loaded model's vector dimension for the next milestone.

## Pinecone configuration

If `.env` does not exist, copy `.env.example` to `.env` and edit it locally:

```text
PINECONE_API_KEY=your_actual_key
PINECONE_INDEX_NAME=movie-finder
```

Then validate these settings without connecting to Pinecone:

```powershell
.\.venv\Scripts\python.exe main.py --check-config
```

The key must be nonempty and cannot be the template placeholder. The index name
must contain 1-45 lowercase letters, digits, or hyphens, and start and end with a
letter or digit. Missing index names default to `movie-finder`; explicit blank
values are rejected. Environment variables override the project `.env` file.
Keys are omitted from the settings representation. Never commit `.env`.
Cloud/region settings will be introduced when index creation is implemented.

## Modules

- `config.py`: paths, fixed model name, immutable Pinecone settings, validation.
- `dataset.py`: movie data structure, JSON loading, validation, text formatting.
- `embeddings.py`: lazy loading of `sentence-transformers/all-MiniLM-L6-v2`.
- `main.py`: preparation commands and user-facing error messages.
- `movies.json`: 20 movies across science fiction, fantasy, animation, comedy,
  romance, crime, horror, and drama, with original short descriptions.
- `tests/test_setup.py`: offline validation and command tests.

Pinecone access and search will be separate modules when those steps begin.

## Dataset and formatting

Each record has a nonempty title, director, genre, and description, plus an integer
release year between 1888 and the current year. Boolean and string years are rejected.
The dataset must contain at least 20 valid records. Duplicate title/year pairs are
rejected after case and whitespace normalization; different-year remakes are allowed.
Malformed JSON, missing files, and invalid records produce readable errors.

Searchable text uses fixed labels and order: Title, Director, Genre, Year, Description.
The same formatter will be used for movie indexing. Example:

```text
Title: The Matrix
Director: Lana Wachowski and Lilly Wachowski
Genre: Science fiction / Action
Year: 1999
Description: A computer programmer discovers that everyday reality is a simulation and joins rebels fighting the machines that control humanity.
```

## Tests and requirements status

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

The tests use no network, model download, or real Pinecone key. They check dataset
shape, fields, years, duplicates, file errors, configuration precedence, secret
redaction, default startup, and cached model loading/error handling.

The original README is the working requirements source. A separate programming
assignment was not supplied, so its exact submission rules remain unconfirmed.

| Requirement | Status |
| --- | --- |
| Load at least 20 movies with five required fields | Implemented |
| Load all-MiniLM-L6-v2 | Implemented |
| Generate movie embeddings | Next milestone |
| Store vectors in Pinecone using stable IDs | Next milestone |
| Embed queries and retrieve top three movies | Pending |
| Tune similarity threshold and report no matches | Pending |
| Console search and exit | Pending |

Similarity threshold selection, indexing duplicate prevention, and a real search
transcript will be documented after those features are implemented. No Pinecone
index or external records are created by the current commands.

References: [SentenceTransformer constructor](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)
and [Pinecone index naming](https://docs.pinecone.io/reference/api/2025-10/control-plane/configure_index).

## Git

Commit source, tests, dataset, README, requirements, and the environment template.
Keep `.env`, `.venv`, `.idea`, and `.cache` out of Git. Connect this project to its
GitHub repository before submission if required by the assignment.

Verified for this milestone: all 8 offline tests passed, dependency checks passed, and the actual model loaded successfully with 384-dimensional output. Pinecone credentials and connectivity have not been tested.
