import unittest
from unittest.mock import Mock, MagicMock
from types import SimpleNamespace as NS
from dataclasses import replace
import numpy as np
from config import Settings
from dataset import load_movies
from embeddings import embed_movies, ModelLoadError
from pinecone_store import movie_id, build_documents, prepare_index, upload_movies, IndexingError


def description(dimension=384, metric='cosine', ready=True):
    return NS(schema=NS(fields={'embedding': NS(type='dense_vector', dimension=dimension, metric=metric)}), status=NS(ready=ready), host='test-host')


class IndexingTests(unittest.TestCase):
    def setUp(self):
        self.movies = load_movies()
        self.vectors = [[1.0] + [0.0] * 383 for _ in self.movies]
        self.settings = Settings('test-key')

    def test_embedding_model_output(self):
        model = Mock()
        model.encode.return_value = np.asarray(self.vectors)
        self.assertEqual(embed_movies(self.movies, model), self.vectors)
        self.assertTrue(model.encode.call_args.kwargs['normalize_embeddings'])
        for bad in (np.zeros((20,384)), np.ones((20,3)), np.full((20,384), np.nan)):
            model.encode.return_value = bad
            with self.assertRaises(ModelLoadError):
                embed_movies(self.movies, model)

    def test_stable_ids_and_metadata(self):
        movie = self.movies[0]
        self.assertEqual(movie_id(movie), movie_id(replace(movie, title=' THE   MATRIX ', description='Updated description')))
        self.assertNotEqual(movie_id(movie), movie_id(replace(movie, year=2000)))
        documents = build_documents(self.movies, self.vectors)
        self.assertEqual(len({doc['_id'] for doc in documents}), 20)
        self.assertEqual(documents[0]['title'], 'The Matrix')
        self.assertIn('embedding_model', documents[0])
        self.assertEqual(documents[0]['year'], 1999)

    def test_malformed_uploads_rejected(self):
        for movies, vectors in ((self.movies, []), ([], []), ([self.movies[0]]*2, self.vectors[:2]), (self.movies[:1], [[0.0]*384])):
            with self.assertRaises(IndexingError):
                build_documents(movies, vectors)

    def test_create_missing_index(self):
        client = Mock()
        client.indexes.exists.return_value = False
        client.indexes.describe.return_value = description()
        self.assertEqual(prepare_index(client, self.settings), 'test-host')
        request = client.indexes.create.call_args.kwargs
        self.assertEqual(request['schema']['fields']['embedding']['dimension'],384)
        self.assertEqual(request['schema']['fields']['embedding']['metric'],'cosine')
        self.assertEqual(request['deployment']['region'], 'us-east-1')
        self.assertEqual(request['timeout'], 60)

    def test_existing_index_compatibility_and_readiness(self):
        client = Mock()
        client.indexes.exists.return_value = True
        for desc in (description(768), description(metric='dotproduct'), description(ready=False), NS(schema=None)):
            client.indexes.describe.return_value = desc
            with self.assertRaises(IndexingError):
                prepare_index(client, self.settings)
        client.indexes.create.assert_not_called()
        client.indexes.delete.assert_not_called()

    def test_repeated_upserts_replace_same_records_and_close_clients(self):
        client, index = MagicMock(), MagicMock()
        client.__enter__.return_value = client
        client.index.return_value = index
        index.__enter__.return_value = index
        client.indexes.exists.return_value = True
        client.indexes.describe.return_value = description()
        records = {}
        def upsert(**kwargs):
            self.assertEqual(kwargs['namespace'], 'movies')
            for doc in kwargs['documents']:
                records[doc['_id']] = doc
            return NS(upserted_count=len(kwargs['documents']))
        index.documents.upsert.side_effect = upsert
        factory = Mock(return_value=client)
        self.assertEqual(upload_movies(self.settings,self.movies,self.vectors,factory),20)
        updated = [replace(self.movies[0], description='New description')] + self.movies[1:]
        self.assertEqual(upload_movies(self.settings,updated,self.vectors,factory),20)
        self.assertEqual(len(records),20)
        self.assertEqual(records[movie_id(updated[0])]['description'], 'New description')
        self.assertEqual(client.__exit__.call_count,2)
        self.assertEqual(index.__exit__.call_count,2)
        index.documents.upsert.side_effect = None
        index.documents.upsert.return_value = NS(upserted_count=0)
        with self.assertRaises(IndexingError):
            upload_movies(self.settings,self.movies,self.vectors,factory)

if __name__ == '__main__':
    unittest.main()
