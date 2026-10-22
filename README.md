# MovieFinder

Python console application for semantic movie search. This milestone implements
steps 1-20: configuration, a validated dataset, model loading, vector generation,
Pinecone indexing, and interactive semantic search. Final tuning and review remain.

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
Cloud, region, and namespace default to `aws`, `us-east-1`, and `movies`; override them with PINECONE_CLOUD, PINECONE_REGION, and PINECONE_NAMESPACE. Confirm that your chosen region is supported by your account.

## Modules

- `config.py`: paths, fixed model name, immutable Pinecone settings, validation.
- `dataset.py`: movie data structure, JSON loading, validation, text formatting.
- `embeddings.py`: lazy loading of `sentence-transformers/all-MiniLM-L6-v2`.
- `main.py`: preparation commands and user-facing error messages.
- `movies.json`: 20 movies across science fiction, fantasy, animation, comedy,
  romance, crime, horror, and drama, with original short descriptions.
- `tests/test_setup.py`: offline validation and command tests.

`pinecone_store.py` owns compatible index creation, stable IDs, and batched uploads. `search.py` owns semantic querying, threshold filtering, and result display.

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
| Generate movie embeddings | Implemented; real embeddings verified |
| Store vectors in Pinecone using stable IDs | Implemented; live connection/upload pending API key |
| Embed queries and retrieve top three movies | Implemented; live Pinecone retrieval pending key |
| Tune similarity threshold and report no matches | Filtering implemented; threshold calibration pending |
| Console search and exit | Implemented |

Similarity threshold selection, indexing duplicate prevention, and a real search
transcript will be documented after those features are implemented. The `--index` command can create a Pinecone index and upload records; the default command and `--embed` remain local.

References: [SentenceTransformer constructor](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)
and [Pinecone index naming](https://docs.pinecone.io/reference/api/2025-10/control-plane/configure_index).

## Git

Commit source, tests, dataset, README, requirements, and the environment template.
Keep `.env`, `.venv`, `.idea`, and `.cache` out of Git. Connect this project to its
GitHub repository before submission if required by the assignment.

Verified through step 15: all 14 offline tests passed and the actual model generated 20 finite, normalized 384-dimensional movie vectors. Live Pinecone connection and upload are blocked by the missing API key.


## Generate and index movies (steps 11-15)

```powershell
# Generate vectors locally, without an API key or upload:
.\.venv\Scripts\python.exe main.py --embed
# Validate settings, embed movies, create/validate an index, then upload:
.\.venv\Scripts\python.exe main.py --index
```

The encoder uses the same formatted movie metadata defined above and normalizes
vectors for cosine similarity. Shape, finite values, and vector norms are checked
before upload. Vectors are generated in memory rather than saved to a separate file.

The installed Pinecone v10 SDK creates document indexes. The application declares
one dense `embedding` field (384 dimensions, cosine) and uploads each vector with
all five movie fields and the embedding model name through `index.documents.upsert`.
Index creation waits at most 60 seconds. Existing indexes must be ready and have
exactly the expected schema; incompatible indexes are left unchanged. The default
managed deployment uses AWS us-east-1. Client construction alone does not authenticate;
index existence/description requests provide the connection check.

Each ID is a SHA-256 digest of the normalized title and year. Reordering the dataset
or updating a description preserves its ID; rerunning indexing overwrites the same
records in the same namespace. Changing a title or year creates a new identity.
Removing a movie from the local dataset does not delete its old remote record.
Uploads use batches of at most 50 records with a 30-second request timeout and check
accepted counts. Retrying after a partial failure is safe for unchanged IDs. Pinecone
applies writes asynchronously, so accepted records may not immediately appear in search.

The repeated-upload test uses a simulated service; it is not evidence of a successful
live upload. Add your Pinecone key locally and run `--index` to complete that check.
No cloud index or records have been created during this implementation.

API reference: [Pinecone SDK quickstart](https://sdk.pinecone.io/python/getting-started/quickstart.html).


## Search movies (steps 16-20)

After configuring the key and running `--index`, start an interactive search:

```powershell
.\.venv\Scripts\python.exe main.py --search
# Optional threshold override:
.\.venv\Scripts\python.exe main.py --search --threshold 0.40
```

Describe a plot or theme, for example `an astronaut stranded on Mars trying to survive`.
Queries use the same model and normalization as the movie vectors. The query is plain
natural-language text, without the metadata labels used for movie records.
Search requests three candidates from the configured namespace using cosine similarity
on the `embedding` field, restricted to records tagged with the same embedding model.

Candidates below the threshold are removed. Remaining results show title, year,
director, genre, description, and similarity, ordered by score. Similarity is not a
probability or a guarantee of relevance. If none qualify, the app reports no matching
movies and asks for a different description. Fewer than three results may be shown.

The default threshold is **0.35**, a provisional starting value, not a calibrated
cutoff. Relevant, ambiguous, and unrelated query evaluation belongs to the final
milestone. `--threshold` accepts finite values between -1 and 1, inclusive.
Type `exit` to leave; blank input repeats the prompt. Search never creates an index
or uploads records. If the index is missing, run `--index` first. Search does not
require the local dataset file; it reads indexed metadata from Pinecone.

Verification: all 21 offline tests passed, including top-three request parameters,
threshold boundaries, empty results, formatting, and read-only index access.
Live Pinecone retrieval remains blocked by the missing API key. A full live search
transcript and final error-handling review remain for the last milestone.
