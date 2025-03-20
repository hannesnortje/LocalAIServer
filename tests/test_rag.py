import unittest
import json
import tempfile
import shutil
import os
import time
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from local_ai_server.config import QDRANT_PATH, MODELS_DIR  # Add MODELS_DIR import here
from local_ai_server.vector_store import VectorStore, get_vector_store
from local_ai_server.rag import RAG
from local_ai_server.model_manager import model_manager

class TestRAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create temporary directory for vector storage"""
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="rag_test_"))
        
        # Set permissions
        os.chmod(cls.temp_dir, 0o777)
        
        # Save original path
        cls.original_path = QDRANT_PATH
        
        # Override the path before any imports that might use it
        import local_ai_server.config as config
        config.QDRANT_PATH = cls.temp_dir
        
        # Import and create Flask test client
        from local_ai_server.server import app
        cls.flask_app = app
        cls.flask_app.config['TESTING'] = True
        cls.client = cls.flask_app.test_client()

        # Initialize vector store
        cls.vector_store = get_vector_store(storage_path=cls.temp_dir)
        
        # Add test documents
        cls.test_docs = [
            "Artificial intelligence (AI) is intelligence demonstrated by machines.",
            "Machine learning is a subset of AI focused on data and algorithms.",
            "Natural language processing (NLP) allows computers to understand text.",
            "Vector embeddings represent words or phrases as numerical vectors.",
            "Retrieval-Augmented Generation combines retrieval with language generation."
        ]
        
        cls.test_metadata = [
            {"source": "wiki", "category": "ai"},
            {"source": "textbook", "category": "ml"},
            {"source": "blog", "category": "nlp"},
            {"source": "paper", "category": "embeddings"},
            {"source": "documentation", "category": "rag"}
        ]
        
        cls.doc_ids = cls.vector_store.add_texts(cls.test_docs, cls.test_metadata)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory"""
        # Restore original path
        import local_ai_server.config as config
        config.QDRANT_PATH = cls.original_path
        
        # Clean up vector store
        if hasattr(cls, 'vector_store'):
            del cls.vector_store
        
        # Wait before cleanup
        time.sleep(0.5)
        
        # Clean up temporary directory
        if hasattr(cls, 'temp_dir') and cls.temp_dir.exists():
            try:
                # Force permissions for cleanup
                for root, dirs, files in os.walk(cls.temp_dir):
                    for d in dirs:
                        try:
                            os.chmod(os.path.join(root, d), 0o777)
                        except:
                            pass
                    for f in files:
                        try:
                            os.chmod(os.path.join(root, f), 0o666)
                        except:
                            pass
                
                # Remove directory
                shutil.rmtree(cls.temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Cleanup error (can be ignored): {e}")

    def test_document_retrieval(self):
        """Test that documents can be retrieved based on a query"""
        # Search for AI-related content
        results = self.vector_store.similarity_search(
            query="What is artificial intelligence?",
            k=2
        )
        
        # Should find relevant documents
        self.assertEqual(len(results), 2)
        self.assertTrue(any("intelligence" in r["text"].lower() for r in results))
        
        # Search with filter
        results = self.vector_store.similarity_search(
            query="machine learning",
            filter={"category": "ml"},
            k=1
        )
        
        # Should find the document with machine learning and correct category
        self.assertEqual(len(results), 1)
        self.assertTrue("machine learning" in results[0]["text"].lower())
        self.assertEqual(results[0]["metadata"]["category"], "ml")

    def test_document_formatting(self):
        """Test that retrieved documents are formatted correctly"""
        # Get some documents
        results = self.vector_store.similarity_search(
            query="vector embeddings", 
            k=2
        )
        
        # Format them
        formatted = RAG.format_retrieved_documents(results)
        
        # Should contain document numbers and sources
        self.assertTrue("Document 1" in formatted)
        self.assertTrue("Document 2" in formatted)
        self.assertTrue("Source:" in formatted)
        
        # Empty case
        empty_formatted = RAG.format_retrieved_documents([])
        self.assertEqual(empty_formatted, "No relevant documents found.")

    @unittest.skipIf(not Path(MODELS_DIR).exists() or len(list(Path(MODELS_DIR).iterdir())) == 0, 
                    "Skipping test that requires a model")
    def test_rag_generation_direct(self):
        """Test direct RAG generation with a model (requires installed models)"""
        # Check for installed models
        import os
        from local_ai_server.config import MODELS_DIR
        
        # List models
        available_models = []
        if Path(MODELS_DIR).exists():
            available_models = [f.name for f in Path(MODELS_DIR).iterdir() if f.is_file()]
        
        if not available_models:
            self.skipTest("No models available for testing")
            
        # Use the first available model
        model_name = available_models[0]
        
        # Generate RAG response
        try:
            response = RAG.generate_rag_response(
                query="What is machine learning?",
                model_name=model_name,
                search_params={"limit": 2},
                generation_params={"max_tokens": 50, "temperature": 0.1}
            )
            
            # Check response structure
            self.assertIn("answer", response)
            self.assertIn("retrieved_documents", response)
            self.assertIn("metadata", response)
            self.assertEqual(len(response["retrieved_documents"]), 2)
            
            # The answer should not be empty
            self.assertTrue(len(response["answer"]) > 0)
            
        except RuntimeError as e:
            if "No model loaded" in str(e):
                self.skipTest("Model could not be loaded")

    def test_rag_api_endpoint(self):
        """Test the RAG API endpoint"""
        # This test mocks the model response
        import unittest.mock
        from local_ai_server.rag import RAG
        
        # Mock the generate_rag_response method
        original_generate = RAG.generate_rag_response
        mock_response = {
            "answer": "Machine learning is a subset of AI focused on data and algorithms.",
            "model": "test-model",
            "retrieved_documents": [
                {"text": "Machine learning is a subset of AI focused on data and algorithms.", 
                "metadata": {"source": "textbook", "category": "ml"}, "score": 0.95}
            ],
            "metadata": {"query": "What is machine learning?", "document_count": 1, "response_time": 0.1}
        }
        
        def mock_generate_rag_response(*args, **kwargs):
            return mock_response
        
        try:
            # Apply the mock
            RAG.generate_rag_response = mock_generate_rag_response
            
            # Test the API endpoint
            response = self.client.post('/api/rag', json={
                "query": "What is machine learning?",
                "model": "test-model",
                "search_params": {"limit": 2},
                "temperature": 0.7
            })
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            
            # Check response format
            self.assertIn("answer", data)
            self.assertIn("retrieved_documents", data)
            self.assertIn("metadata", data)
            self.assertEqual(data["answer"], mock_response["answer"])
            
        finally:
            # Restore the original method
            RAG.generate_rag_response = original_generate

if __name__ == '__main__':
    unittest.main()
