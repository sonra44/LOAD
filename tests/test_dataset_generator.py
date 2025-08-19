import unittest
import os
from tools.generate_dataset import generate_dataset

class TestDatasetGenerator(unittest.TestCase):
    def test_generate_dataset(self):
        path = "test_dataset.jsonl"
        generate_dataset(path, size=10)
        self.assertTrue(os.path.exists(path))
        
        # Проверим содержимое
        with open(path, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 10)
        
        # Удалим тестовый файл
        os.remove(path)

if __name__ == "__main__":
    unittest.main()
