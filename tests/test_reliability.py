import contextlib
import io
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace as NS
from pinecone import ApiError, PineconeConnectionError, PineconeTimeoutError
from errors import pinecone_message
from search import run_search, search_movies
import main

class ReliabilityTests(unittest.TestCase):
    def test_service_errors_are_readable_and_redacted(self):
        cases=[(ApiError('SECRET',401),'API key'),(ApiError('SECRET',403),'permissions'),(ApiError('SECRET',404),'resource'),(ApiError('SECRET',429),'quota'),(ApiError('SECRET',400),'schema'),(ApiError('SECRET',500),'unavailable'),(PineconeConnectionError('SECRET'),'connect'),(PineconeTimeoutError('SECRET'),'timed out')]
        for error, expected in cases:
            with self.subTest(expected=expected):
                text=pinecone_message(error)
                self.assertIn(expected,text)
                self.assertNotIn('SECRET',text)

    def test_search_recovers_after_failure(self):
        with patch('builtins.input',side_effect=['first','second','exit']),patch('search.search_movies',side_effect=[PineconeTimeoutError('SECRET'),[]]) as search,contextlib.redirect_stdout(io.StringIO()) as output:
            run_search(Mock(),'movies')
        self.assertEqual(search.call_count,2)
        self.assertIn('timed out',output.getvalue())
        self.assertIn('No matching movies',output.getvalue())
        self.assertNotIn('SECRET',output.getvalue())

    def test_eof_and_interrupt_at_prompt(self):
        for error in (EOFError,KeyboardInterrupt):
            with patch('builtins.input',side_effect=error),patch('search.search_movies') as search,contextlib.redirect_stdout(io.StringIO()) as output:
                run_search(Mock(),'movies')
            search.assert_not_called()
            self.assertIn('Goodbye',output.getvalue())

    def test_interrupt_during_search(self):
        with patch('builtins.input',return_value='query'),patch('search.search_movies',side_effect=KeyboardInterrupt),contextlib.redirect_stdout(io.StringIO()) as output:
            run_search(Mock(),'movies')
        self.assertIn('Goodbye',output.getvalue())

    def test_malformed_remote_metadata(self):
        index=Mock()
        index.documents.search.return_value=NS(matches=[NS(score=.9,title='Missing fields')])
        with patch('search.embed_query',return_value=[1.0]+[0.0]*383),self.assertRaisesRegex(ValueError,'metadata'):
            search_movies(index,'movies','query')

    def test_cli_handles_new_sdk_errors(self):
        with patch.object(main,'load_settings'),patch.object(main,'Pinecone',side_effect=PineconeConnectionError('SECRET')),contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(main.main(['--search']),1)
        self.assertIn('connect',output.getvalue())
        self.assertNotIn('SECRET',output.getvalue())

if __name__ == '__main__':
    unittest.main()
