import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
from dataclasses import asdict

from config import ConfigurationError, Settings, load_settings
from dataset import DatasetError, load_movies, validate_movies, format_movie
from embeddings import load_embedding_model, ModelLoadError
import main


class SetupTests(unittest.TestCase):
    def test_real_dataset_and_format(self):
        movies = load_movies()
        self.assertEqual(len(movies), 20)
        text = format_movie(movies[0])
        for field in ('Title:', 'Director:', 'Genre:', 'Year:', 'Description:'):
            self.assertIn(field, text)
        self.assertIn('The Matrix', text)

    def test_bad_records(self):
        base = asdict(load_movies()[0])
        variants = [None, [], {}, {**base, 'year': True}, {**base, 'year': '1999'}, {**base, 'year': 1000}, {**base, 'year': 9999}]
        for field in ('title', 'director', 'genre', 'description'):
            variants += [{**base, field: '  '}, {**base, field: 12}]
        for record in variants:
            with self.subTest(record=record), self.assertRaises(DatasetError):
                validate_movies([record], minimum_count=1)
        for data in ({}, [], [base]):
            with self.assertRaises(DatasetError):
                validate_movies(data)

    def test_duplicates_and_remakes(self):
        base = asdict(load_movies()[0])
        with self.assertRaises(DatasetError):
            validate_movies([base, {**base, 'title': ' THE   MATRIX '}], minimum_count=2)
        self.assertEqual(len(validate_movies([base, {**base, 'year': 2000}], minimum_count=2)), 2)

    def test_file_errors(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / 'movies.json'
            with self.assertRaises(DatasetError):
                load_movies(path)
            path.write_text('{broken json')
            with self.assertRaises(DatasetError):
                load_movies(path)

    def test_settings_validation(self):
        for key in ('', ' ', 'replace_with_your_key'):
            with self.assertRaises(ConfigurationError):
                Settings(key)
        for name in ('', 'Uppercase', '-bad', 'bad-', 'bad_name', 'x' * 46):
            with self.assertRaises(ConfigurationError):
                Settings('fake-key', name)
        settings = Settings(' secret-key ', ' movie-finder ')
        self.assertEqual(settings.index_name, 'movie-finder')
        self.assertNotIn('secret-key', repr(settings))
        self.assertEqual(Settings('key', 'a').index_name, 'a')

    def test_environment_precedence(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / '.env'
            path.write_text('PINECONE_API_KEY=file-key\nPINECONE_INDEX_NAME=file-index\n')
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(load_settings(path).api_key, 'file-key')
            with patch.dict(os.environ, {'PINECONE_API_KEY': 'env-key', 'PINECONE_INDEX_NAME': 'env-index'}, clear=True):
                self.assertEqual(load_settings(path), Settings('env-key', 'env-index'))
            with patch.dict(os.environ, {'PINECONE_API_KEY': ''}, clear=True), self.assertRaises(ConfigurationError):
                load_settings(path)

    def test_default_cli_needs_no_key_or_model(self):
        with patch.object(main, 'load_embedding_model') as model, patch.object(main, 'load_settings') as settings, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main.main([]), 0)
        model.assert_not_called()
        settings.assert_not_called()
        self.assertIn('Validated 20 movies', output.getvalue())

    def test_model_loader_cached_and_errors_readable(self):
        load_embedding_model.cache_clear()
        constructor = Mock(return_value=object())
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder, patch.dict('sys.modules', {'sentence_transformers': Mock(SentenceTransformer=constructor)}):
            first = load_embedding_model(Path(folder))
            self.assertIs(load_embedding_model(Path(folder)), first)
            constructor.assert_called_once()
            self.assertEqual(constructor.call_args.kwargs['device'], 'cpu')
            load_embedding_model.cache_clear()
            constructor.side_effect = OSError('internal detail')
            with self.assertRaises(ModelLoadError):
                load_embedding_model(Path(folder))
        load_embedding_model.cache_clear()


if __name__ == '__main__':
    unittest.main()
