import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import json

# Добавляем путь к tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

class TestTerminalDashboard(unittest.TestCase):
    def setUp(self):
        # Мокаем зависимости prompt_toolkit
        self.mock_application = MagicMock()
        self.mock_layout = MagicMock()
        self.mock_bindings = MagicMock()
        
        # Мокаем requests
        self.mock_requests = MagicMock()
        
    @patch('tools.terminal_dashboard.requests')
    def test_query_prometheus_success(self, mock_requests):
        # Подготовка мока
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'status': 'success',
            'data': {
                'result': [{'value': [None, '125.5']}]
            }
        }
        mock_requests.get.return_value = mock_response
        
        # Импортируем после мока
        from tools.terminal_dashboard import TerminalDashboard
        dashboard = TerminalDashboard()
        
        # Тест
        result = dashboard.query_prometheus('test_query')
        self.assertEqual(result, 125.5)
        
    @patch('tools.terminal_dashboard.requests')
    def test_query_prometheus_failure(self, mock_requests):
        # Мокаем ошибку
        mock_requests.get.side_effect = Exception("Connection failed")
        
        from tools.terminal_dashboard import TerminalDashboard
        dashboard = TerminalDashboard()
        
        # Тест
        result = dashboard.query_prometheus('test_query')
        self.assertEqual(result, 0.0)
        
    def test_add_log(self):
        from tools.terminal_dashboard import TerminalDashboard
        dashboard = TerminalDashboard()
        
        # Добавляем логи
        dashboard.add_log("Test log 1")
        dashboard.add_log("Test log 2")
        
        # Проверяем
        self.assertIn("Test log 1", dashboard.logs_text.text)
        self.assertIn("Test log 2", dashboard.logs_text.text)
        
    def test_log_limit(self):
        from tools.terminal_dashboard import TerminalDashboard
        dashboard = TerminalDashboard()
        
        # Добавляем больше 50 логов
        for i in range(60):
            dashboard.add_log(f"Log {i}")
            
        # Проверяем, что осталось только 50
        lines = dashboard.logs_text.text.strip().split('\n')
        self.assertEqual(len(lines), 50)
        self.assertIn("Log 59", dashboard.logs_text.text)  # Последний
        self.assertNotIn("Log 0", dashboard.logs_text.text)  # Первый удалён
        
    @patch('tools.terminal_dashboard.threading')
    def test_auto_refresh_starts_thread(self, mock_threading):
        from tools.terminal_dashboard import TerminalDashboard
        dashboard = TerminalDashboard()
        
        # Мокаем метод update_metrics
        dashboard.update_metrics = MagicMock()
        
        # Вызываем auto_refresh
        dashboard.auto_refresh()
        
        # Проверяем, что thread был создан
        mock_threading.Thread.assert_called()
        
    def test_metrics_text_format(self):
        from tools.terminal_dashboard import TerminalDashboard
        dashboard = TerminalDashboard()
        
        # Мокаем query_prometheus
        dashboard.query_prometheus = MagicMock(return_value=125.5)
        
        # Обновляем метрики
        dashboard.update_metrics()
        
        # Проверяем формат
        text = dashboard.metrics_text.text
        self.assertIn("Inference Rate", text)
        self.assertIn("125.50", text)
        self.assertIn("QIKI Neural Engine Metrics", text)

if __name__ == '__main__':
    unittest.main()
