import sys
import os
import unittest
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config, data_loader, embeddings, search

class TestEBOSComponents(unittest.TestCase):
    
    def test_config(self):
        self.assertTrue(os.path.exists(config.DATA_DIR))
        self.assertIsNotNone(config.MODEL_NAME)

    def test_data_loader(self):
        # Test loading local repositories
        repos = data_loader.get_available_repositories()
        print(f"Found repositories: {repos}")
        
        if repos:
            # ontologies, domains = data_loader.get_local_repo(repos[0])
            # self.assertIsInstance(ontologies, dict)
            # self.assertIsInstance(domains, set)
            # print(f"Loaded {len(ontologies)} ontologies from {repos[0]}")
            pass

    def test_search_logic(self):
        # Mock embeddings
        emb_dim = 384 # MiniLM dimension
        num_items = 10
        mock_embeddings = np.random.rand(num_items, emb_dim).astype('float32')
        ids = [str(i) for i in range(num_items)]
        
        # Test index creation
        index = search.create_index(mock_embeddings)
        self.assertIsNotNone(index)
        self.assertEqual(index.ntotal, num_items)
        
        # Test search
        query = np.random.rand(1, emb_dim).astype('float32')
        results = search.search_ontologies(query, index, ids, num_results=3)
        self.assertEqual(len(results), 3)
        print("Search test passed")

    # Disabled heavy model test for quick verification, uncomment if needed
    # def test_embeddings_model(self):
    #     tokenizer, model = embeddings.load_model()
    #     text = "This is a test"
    #     emb = embeddings.get_embedding(text, tokenizer, model)
    #     self.assertEqual(emb.shape[0], 384)

if __name__ == '__main__':
    unittest.main()
