# Movie Finder

Python console application for semantic movie search.

## Setup

1. Open this folder as a project in PyCharm.
2. Install Python 3.11 or newer and create a project virtual environment (`.venv`).
3. In the project terminal, run `python -m pip install -r requirements.txt`.
4. Copy `.env.example` to `.env` and add your Pinecone API key.
5. Run `python main.py`, or run `main.py` from PyCharm.

`main.py` is a placeholder; the assignment solution has not been implemented yet.
`movies.json` is an empty starter dataset. Populate it with at least 20 movies,
each containing title, director, genre, year, and a short description.

## Implementation checklist

- Load the movie dataset from `movies.json`.
- Embed movie metadata with SentenceTransformers `all-MiniLM-L6-v2`.
- Store embeddings in Pinecone using stable IDs to avoid duplicate records.
- Embed natural language queries and retrieve the top three relevant movies.
- Apply a similarity threshold and report when no matching movies are found.
- Quit on `exit`.

## Similarity threshold

TODO: Choose a threshold, test it with relevant and irrelevant queries, and explain why.

## Example conversation

TODO: Add an actual example after implementation.

## Design decisions

TODO: Explain metadata formatting, indexing, duplicate prevention, and search.

## Git

This folder has its own Git repository. Make meaningful commits while developing.
Connect it to a GitHub repository before submission. Keep `.env` out of Git.
