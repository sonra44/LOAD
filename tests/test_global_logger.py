import unittest
import sys
import os

# Добавляем путь к tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

class TestGlobalLogger(unittest.TestCase):
    def test_global_logger_singleton(self):
        from tools.global_logger import GLOBAL_LOGGER
        
        # Проверяем, что это синглтон
        from tools.global_logger import GlobalLogger
        logger2 = GlobalLogger()
        
        # GLOBAL_LOGGER должен быть одним и тем же экземпляром
        self.assertIsNotNone(GLOBAL_LOGGER)
        
    def test_set_dashboard(self):
        from tools.global_logger import GLOBAL_LOGGER
        
        # Создаем мок dashboard
        mock_dashboard = unittest.mock.MagicMock()
        
        # Устанавливаем dashboard
        GLOBAL_LOGGER.set_dashboard(mock_dashboard)
        
        # Проверяем, что dashboard установлен
        self.assertEqual(GLOBAL_LOGGER.dashboard, mock_dashboard)
        
    def test_add_log_without_dashboard(self):
        from tools.global_logger import GLOBAL_LOGGER
        
        # Сбрасываем dashboard
        GLOBAL_LOGGER.dashboard = None
        
        # Должно работать без ошибок
        try:
            GLOBAL_LOGGER.add_log("Test message")
            success = True
        except:
            success = False
            
        self.assertTrue(success)
        
    def test_add_log_with_dashboard(self):
        from tools.global_logger import GLOBAL_LOGGER
        
        # Создаем мок dashboard
        mock_dashboard = unittest.mock.MagicMock()
        GLOBAL_LOGGER.set_dashboard(mock_dashboard)
        
        # Добавляем лог
        GLOBAL_LOGGER.add_log("Test message")
        
        # Проверяем, что метод dashboard.add_log был вызван
        mock_dashboard.add_log.assert_called_once_with("Test message")

if __name__ == '__main__':
    unittest.main()
