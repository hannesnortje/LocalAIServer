import unittest
import json
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from local_ai_server.server import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health_check(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')

    def test_list_models(self):
        response = self.app.get('/v1/models')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIsInstance(data['data'], list)

    def test_available_models(self):
        response = self.app.get('/api/available-models')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, dict)
        # Check for required model fields
        for model_id, model_info in data.items():
            self.assertIn('name', model_info)
            self.assertIn('url', model_info)
            self.assertIn('type', model_info)

if __name__ == '__main__':
    unittest.main()
