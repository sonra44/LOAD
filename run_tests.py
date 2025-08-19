import unittest
import sys
import os

# Добавляем пути
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ne_qiki'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))

def run_all_tests():
    """Запуск всех тестов"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Добавляем тесты из разных директорий
    suite.addTests(loader.discover('tests', pattern='test_*.py'))
    
    # Создаем runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Запускаем тесты
    result = runner.run(suite)
    
    # Возвращаем код выхода
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
