import unittest

import faiss
import numpy as np

from src.tabr_training import CONTEXT_SIZE, NeighborSearch


class TabRTrainingTests(unittest.TestCase):
    def test_training_retrieval_excludes_query_even_with_duplicate_keys(self):
        faiss.omp_set_num_threads(2)
        for size in (64, 4200):
            with self.subTest(size=size):
                keys = np.random.default_rng(42).normal(size=(size, 8)).astype(np.float32)
                keys[1] = keys[0]
                search = NeighborSearch(keys)
                neighbors = search.search(keys, exclude_self=True).numpy()
                self.assertEqual(neighbors.shape, (size, CONTEXT_SIZE))
                self.assertFalse(np.equal(neighbors, np.arange(size)[:, None]).any())
                self.assertTrue(((neighbors >= 0) & (neighbors < size)).all())
                self.assertIn(1, neighbors[0])
                self.assertIn(0, neighbors[1])
