# MovieFinder

Python console application for semantic movie search. It embeds a 20-movie dataset
using `sentence-transformers/all-MiniLM-L6-v2`, stores vectors in Pinecone, and
retrieves up to three movies for a natural-language query.

## Setup

Use Python 3.11 or newer. In PyCharm, select the project `.venv` interpreter.
For a fresh environment, run these commands from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `.env` does not already exist, copy `.env.example` to `.env` and edit it locally:

```text
PINECONE_API_KEY=your_actual_key
PINECONE_INDEX_NAME=movie-finder
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_NAMESPACE=movies
```

Keep the key out of Git and chat. Existing environment variables override `.env`.
The key must not be empty or the template placeholder. Index names must contain
1-45 lowercase letters, digits, or hyphens and start/end with a letter or digit.
Cloud must be aws, gcp, or azure; region and namespace must be nonempty. Confirm
that your account supports the selected region. Defaults match the template.

## Commands

```powershell
# Validate and preview the local dataset; no key required:
.\.venv\Scripts\python.exe main.py
# Validate settings without contacting Pinecone:
.\.venv\Scripts\python.exe main.py --check-config
# Download/cache the model; no Pinecone key required:
.\.venv\Scripts\python.exe main.py --load-model
# Generate real normalized movie vectors locally:
.\.venv\Scripts\python.exe main.py --embed
# Create/validate an index and upload movie vectors:
.\.venv\Scripts\python.exe main.py --index
# Search the existing indexed collection:
.\.venv\Scripts\python.exe main.py --search
# Optional similarity threshold override:
.\.venv\Scripts\python.exe main.py --search --threshold 0.40
```

The first model load needs internet access. Subsequent loads reuse files under
`.cache/models/`, which is ignored by Git. Indexing creates cloud resources and
upserts records. Search only reads the existing index. Pinecone may take time to
make accepted writes visible; an immediate search may not see all new records.

Describe a plot, genre, or theme. Results show title, year, director, genre,
description, and similarity. Similarity is not a probability. Blank input repeats
the prompt; `exit`, EOF, or Ctrl+C ends the session. Search remains open after an
expected service failure so you can retry or exit.

## Structure and design

- `config.py`: immutable settings, validation, paths, and fixed embedding model.
- `dataset.py`: movie records, JSON validation, and searchable text formatting.
- `embeddings.py`: lazy cached model loading and normalized embeddings.
- `pinecone_store.py`: index compatibility checks, stable IDs, batched uploads.
- `search.py`: query embedding, top-three retrieval, filtering, console results.
- `errors.py`: readable Pinecone diagnostics without raw response bodies.
- `main.py`: command handling, orchestration, and client cleanup.
- `tests/`: network-free tests with simulated service responses.
- `evaluation/`: reproducible real-model evaluation and its recorded results.

The 20 movie records contain title, director, genre, integer release year, and
original short descriptions. Validation rejects missing/empty fields, invalid
JSON, noninteger or out-of-range years, fewer than 20 records, and duplicate
normalized title/year pairs. Different-year remakes are allowed.

Movie text uses labeled fields in this order: Title, Director, Genre, Year,
Description. Queries use plain natural-language text. Both use the same model
and normalized 384-dimensional vectors. Dimensions, finite values, and norms
are checked. Model files are cached; vectors are generated in memory.

Pinecone v10 document indexes use a single dense `embedding` field with cosine
similarity. An existing index must be ready and have exactly that schema. The app
never replaces an incompatible index. Creation waits at most 60 seconds; upload
and search requests use 30-second timeouts. Clients close when commands finish.
Search requests three candidates from the configured namespace and filters on the
stored embedding-model name. Candidates below the threshold are removed, so fewer
than three movies may appear. No qualifying candidates produces a no-match message.

## Duplicate prevention

Movie IDs are SHA-256 digests of normalized title and year. Reordering the dataset
or editing descriptions preserves IDs. Repeated upserts overwrite the same records
in the same namespace. Changing a title or year creates a new identity. Removing a
local movie does not delete its old remote record. Batches contain at most 50 records;
accepted counts are checked, and retrying unchanged IDs after partial failure is safe.

The repeated-upload test inserts 20 records, edits a description, uploads again,
and verifies that 20 records remain with the edited description. This test uses a
simulated service; live duplicate prevention still needs verification with your key.

## Similarity threshold and evaluation

The default **0.35** was retained after local real-model evaluation on the current
20-movie dataset. Run the evaluation again with:

```powershell
.\.venv\Scripts\python.exe evaluation\evaluate_search.py
```

`evaluation/results.json` records the top three cosine scores for each query.
These are exact local vector comparisons, not live Pinecone search results.

- 12 specific movie queries: the expected movie ranked first in all 12 and passed
  the threshold. Expected-match scores ranged from 0.406 to 0.671.
- 6 unrelated queries: all were rejected. Their highest scores ranged from 0.047
  to 0.128.
- 3 ambiguous queries: two produced results. `An exciting adventure` had a best
  score of 0.277 and was rejected; `A romantic story` kept one result at 0.385.

A 0.35 cutoff separates the tested specific and unrelated queries while accepting
some broad requests. It can miss relevant movies for vague prompts, and results
for `A funny movie about friendship` include films that are not comedies. Semantic
search does not enforce genre constraints. This is a small, hand-written development
set, not an independent benchmark or proof of general accuracy. Retune after changing
the dataset or model. CLI overrides must be finite values between -1 and 1.

## Example results

Observed using the real model and local cosine comparisons:

```text
Query: An astronaut must grow food to survive alone on Mars
1. The Martian (2015)
Director: Ridley Scott
Genre: Science fiction / Adventure
An astronaut stranded on Mars uses engineering and botany to survive while colleagues on Earth organize a rescue.
Similarity: 0.488

Query: How do I reset my wireless router password?
No matching movies found. Try a different plot, genre, or theme.
```

This is an example based on recorded local evaluation, not a live Pinecone console
transcript. A live transcript remains pending credentials.

## Errors and troubleshooting

Missing settings stop startup with a readable message. Dataset errors identify
invalid records or file problems. Model-loading errors explain dependencies,
first-download connectivity, and cache permissions. Pinecone authentication,
permission, missing-resource, quota/rate-limit, timeout, connection, and server
failures receive readable diagnostics without raw provider bodies or credentials.
Malformed indexed metadata asks you to refresh records with `--index`.

If no matches appear, try a more specific plot description and verify that indexing
finished in the same namespace. Lowering the threshold may increase irrelevant
results. Missing indexes require `--index`; search does not create an empty index.

## Verification and remaining work

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

All 27 offline tests and dependency checks passed at final review. Real movie and
query embeddings were verified. Tests cover validation, input/exit behavior,
request parameters, threshold boundaries, failure recovery, incompatible indexes,
metadata formatting, and repeat uploads.

| Working requirement | Status |
| --- | --- |
| At least 20 movies and required metadata | Implemented and validated |
| SentenceTransformers all-MiniLM-L6-v2 embeddings | Verified with the real model |
| Pinecone indexing and stable IDs | Implemented; simulated tests passed |
| Natural-language top-three search | Implemented; simulated service tests passed |
| Similarity filtering and no-match handling | Implemented and evaluated locally |
| Empty input, exit, and failure handling | Implemented and tested |
| Setup, examples, and design documentation | Complete for the implemented behavior |
| Live Pinecone upload/search/repeated indexing | Pending API key |
| Original programming-assignment compliance | Pending original assignment |

The project README supplied the working requirements; the original assignment was
not available. Final review still found no configured Pinecone key. Before submission,
set it locally, run `--index` twice, verify the same 20 IDs remain, and run live searches.
Compare the result with the original assignment's exact submission requirements.

## Git and references

`.env`, `.venv`, `.idea`, `.cache`, and Python caches are ignored. No `.env` is tracked.
Review and commit source, dataset, tests, evaluation files, requirements, README,
and `.env.example`. Connect the project to GitHub if the assignment requires it.

API references: [SentenceTransformer](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)
and [Pinecone SDK quickstart](https://sdk.pinecone.io/python/getting-started/quickstart.html).
