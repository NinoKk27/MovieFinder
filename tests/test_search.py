import contextlib
import io
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import numpy as np
from config import Settings
from embeddings import embed_query
from pinecone_store import prepare_index, IndexingError
from search import search_movies, format_results, run_search


def document(score, title='Example'):
    return NS(title=title, director='Director', year=2000, genre='Drama', description='A sample plot.', score=score)


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.index = Mock()
        self.model = Mock()
        self.model.encode.return_value = np.array([[1.0]+[0.0]*383])

    def test_query_embedding_uses_normalized_text(self):
        vector = embed_query('  lost   in space ', self.model)
        self.assertEqual(len(vector),384)
        self.assertEqual(self.model.encode.call_args.args[0], ['lost in space'])
        self.assertTrue(self.model.encode.call_args.kwargs['normalize_embeddings'])

    def test_top_three_request_and_threshold_boundary(self):
        self.index.documents.search.return_value = NS(matches=[document(.34,'Weak'), document(.35,'Boundary'),document(.8,'Strong')])
        results = search_movies(self.index,'movies','a story',model=self.model)
        self.assertEqual([m.title for m in results], ['Strong','Boundary'])
        args = self.index.documents.search.call_args.kwargs
        self.assertEqual(args['top_k'],3)
        self.assertEqual(args['namespace'],'movies')
        self.assertEqual(args['score_by'][0].field,'embedding')
        self.assertEqual(len(args['score_by'][0].values),384)
        self.assertIn('description', args['include_fields'])
        self.assertIn('embedding_model',args['filter'])

    def test_no_matches_and_invalid_scores(self):
        for matches in ([],[document(.1)],[document(float('nan'))],[document(None)]):
            self.index.documents.search.return_value = NS(matches=matches)
            self.assertEqual(search_movies(self.index,'movies','query',model=self.model),[])
        self.assertIn('No matching movies',format_results([]))

    def test_display_fields(self):
        self.index.documents.search.return_value = NS(matches=[document(.75)])
        text = format_results(search_movies(self.index,'movies','query',model=self.model))
        for value in ('Example (2000)', 'Director: Director', 'Genre: Drama','A sample plot.','0.750'):
            self.assertIn(value,text)

    def test_invalid_query_or_threshold_never_calls_service(self):
        for threshold in (float('nan'),float('inf'),1.1,-1.1):
            with self.assertRaises(ValueError):
                search_movies(self.index,'movies','query',threshold=threshold,model=self.model)
        with self.assertRaises(ValueError):
            search_movies(self.index,'movies','  ',model=self.model)
        self.index.documents.search.assert_not_called()

    def test_search_missing_index_does_not_create(self):
        client=Mock()
        client.indexes.exists.return_value=False
        with self.assertRaises(IndexingError):
            prepare_index(client,Settings('fake-key'),create_if_missing=False)
        client.indexes.create.assert_not_called()

    def test_console_repeats_and_exits(self):
        with patch('builtins.input',side_effect=[' ', 'space adventure', 'EXIT']), patch('search.search_movies',return_value=[]) as search,contextlib.redirect_stdout(io.StringIO()) as output:
            run_search(self.index,'movies')
        search.assert_called_once_with(self.index,'movies','space adventure',threshold=.35)
        self.assertIn('No matching movies',output.getvalue())
        self.assertIn('Goodbye',output.getvalue())

if __name__ == '__main__':
    unittest.main()
