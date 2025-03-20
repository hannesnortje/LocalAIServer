import unittest
import tempfile
import shutil
import os
import time
from pathlib import Path
import sys
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from local_ai_server.config import QDRANT_PATH
# Import the factory function instead of direct class
from local_ai_server.history_manager import get_response_history

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)

class TestHistoryManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create temporary directory for history storage"""
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="history_test_"))
        
        # Set permissions
        os.chmod(cls.temp_dir, 0o777)
        
        # Override config
        import local_ai_server.config as config
        cls.original_path = QDRANT_PATH
        config.QDRANT_PATH = cls.temp_dir
        
        # Explicitly enable response history for tests
        cls.original_enabled = config.ENABLE_RESPONSE_HISTORY
        config.ENABLE_RESPONSE_HISTORY = True
        
        print(f"Test using directory: {cls.temp_dir}")
        print(f"History enabled: {config.ENABLE_RESPONSE_HISTORY}")

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory"""
        # Restore original settings
        import local_ai_server.config as config
        config.QDRANT_PATH = cls.original_path
        config.ENABLE_RESPONSE_HISTORY = cls.original_enabled
        
        # Wait before cleanup
        time.sleep(0.5)
        
        # Clean up temporary directory
        if hasattr(cls, 'temp_dir') and cls.temp_dir.exists():
            try:
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
                
                shutil.rmtree(cls.temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Cleanup error (can be ignored): {e}")

    def setUp(self):
        """Initialize history manager for each test"""
        # Create a test-specific instance with custom storage
        import local_ai_server.config as config
        print(f"Test setUp - History enabled: {config.ENABLE_RESPONSE_HISTORY}")
        
        self.history_manager = get_response_history(storage_path=self.temp_dir)
        
        # Verify initialization
        self.history_manager._initialize()  # Force initialization
        self.assertTrue(self.history_manager.initialized)
        self.assertTrue(self.history_manager.enabled)
        
        # Force collection re-creation to ensure clean state
        try:
            self.history_manager.delete_all_history()
        except Exception as e:
            print(f"Error in setUp: {e}")
    
    def test_save_and_retrieve(self):
        """Test saving and retrieving response history"""
        # Save test responses
        query1 = "What is artificial intelligence?"
        response1 = "AI is intelligence demonstrated by machines."
        metadata1 = {"model": "test-model", "timestamp": time.time()}
        
        query2 = "How does machine learning work?"
        response2 = "ML uses algorithms to learn from data."
        metadata2 = {"model": "test-model", "timestamp": time.time()}
        
        # Save responses
        id1 = self.history_manager.save_response(query1, response1, metadata1)
        id2 = self.history_manager.save_response(query2, response2, metadata2)
        
        # Verify we got IDs back
        self.assertIsNotNone(id1)
        self.assertIsNotNone(id2)
        
        # Retrieve similar responses
        results = self.history_manager.find_similar_responses(
            query="Tell me about AI",
            limit=5
        )
        
        # Should find at least one result
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(any(r['query'] == query1 for r in results))
        
        # Test with filter
        filtered_results = self.history_manager.find_similar_responses(
            query="artificial intelligence",
            filter_params={"model": "test-model"}
        )
        
        self.assertTrue(len(filtered_results) > 0)
        for r in filtered_results:
            self.assertEqual(r['metadata']['model'], 'test-model')
    
    def test_cleanup(self):
        """Test cleaning up old entries"""
        # Add some responses with different timestamps
        now = time.time()
        
        # Current response
        self.history_manager.save_response(
            "Current query", 
            "Current response",
            {"timestamp": now, "test": True}
        )
        
        # Old response (31 days ago)
        self.history_manager.save_response(
            "Old query", 
            "Old response",
            {"timestamp": now - (31 * 24 * 60 * 60), "test": True}
        )
        
        # Count before cleanup
        all_responses = self.history_manager.find_similar_responses(
            query="query",
            min_score=0.5,
            limit=10
        )
        self.assertEqual(len(all_responses), 2)
        
        # Clean up entries older than 30 days
        count = self.history_manager.clean_old_entries(days=30)
        self.assertEqual(count, 1)  # Should remove 1 entry
        
        # Verify only current response remains
        remaining = self.history_manager.find_similar_responses(
            query="query",
            min_score=0.5,
            limit=10
        )
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]['response'], "Current response")
    
    def test_delete_all(self):
        """Test deleting all history"""
        # Add some test responses
        self.history_manager.save_response("Query 1", "Response 1")
        self.history_manager.save_response("Query 2", "Response 2")
        
        # Verify they exist
        results = self.history_manager.find_similar_responses(query="query", min_score=0.5)
        self.assertTrue(len(results) > 0)
        
        # Delete all history
        success = self.history_manager.delete_all_history()
        self.assertTrue(success)
        
        # Verify nothing remains
        results_after = self.history_manager.find_similar_responses(query="query", min_score=0.5)
        self.assertEqual(len(results_after), 0)

if __name__ == '__main__':
    unittest.main()
