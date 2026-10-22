"""Reproduce local relevance checks using real embeddings, without Pinecone."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import load_movies
from embeddings import embed_movies, embed_query, load_embedding_model

CASES = [
    ('relevant', 'An astronaut must grow food to survive alone on Mars', 'The Martian'),
    ('relevant', 'People enter dreams to steal secrets and plant ideas', 'Inception'),
    ('relevant', 'A linguist tries to communicate with visitors from another planet', 'Arrival'),
    ('relevant', 'A father fish searches the ocean for his missing son', 'Finding Nemo'),
    ('relevant', 'A rat wants to become a chef in Paris', 'Ratatouille'),
    ('relevant', 'A weatherman lives the same day over and over', 'Groundhog Day'),
    ('relevant', 'A shark threatens swimmers in a seaside town', 'Jaws'),
    ('relevant', 'A crime boss and his son struggle for control of their family business', 'The Godfather'),
    ('relevant', 'A young girl enters a magical world of spirits', 'Spirited Away'),
    ('relevant', 'A hobbit carries a dangerous ring on a quest', 'The Lord of the Rings: The Fellowship of the Ring'),
    ('relevant', 'Prison friendship and hope over many years', 'The Shawshank Redemption'),
    ('relevant', 'Cloned dinosaurs escape at a theme park', 'Jurassic Park'),
    ('irrelevant', 'How do I reset my wireless router password?', None),
    ('irrelevant', 'A recipe for sourdough bread', None),
    ('irrelevant', 'How much income tax do I owe?', None),
    ('irrelevant', 'The latest football match score', None),
    ('irrelevant', 'Fix a leaking bathroom tap', None),
    ('irrelevant', 'Calculate the derivative of a polynomial', None),
    ('ambiguous', 'A funny movie about friendship', None),
    ('ambiguous', 'An exciting adventure', None),
    ('ambiguous', 'A romantic story', None),
]

def main():
    movies = load_movies()
    model = load_embedding_model()
    vectors = np.asarray(embed_movies(movies, model))
    results = []
    for kind, query, expected in CASES:
        scores = vectors @ np.asarray(embed_query(query, model))
        order = np.argsort(-scores)[:3]
        results.append({'kind': kind, 'query': query, 'expected': expected,
                        'top_three': [{'title': movies[i].title, 'score': round(float(scores[i]), 6)} for i in order]})
    path = Path(__file__).with_name('results.json')
    path.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    for result in results:
        print(result['kind'], result['query'], result['top_three'])

if __name__ == '__main__':
    main()
