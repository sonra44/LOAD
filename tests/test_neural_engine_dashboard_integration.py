import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Добавляем пути
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ne_qiki'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

class TestNeuralEngineDashboardIntegration(unittest.TestCase):
    def setUp(self):
        # Мокаем torch и модель
        self.mock_torch = MagicMock()
        self.mock_model = MagicMock()
        
    @patch('ne_qiki.core.neural_engine_impl.torch')
    @patch('ne_qiki.core.neural_engine_impl.NE_v1')
    def test_neural_engine_logs_to_dashboard(self, mock_ne_v1, mock_torch):
        # Настройка моков
        mock_ne_v1.return_value = self.mock_model
        mock_torch.tensor.return_value = MagicMock()
        mock_torch.ones.return_value = MagicMock()
        
        # Мокаем результат модели
        mock_logits = MagicMock()
        mock_logits.size.return_value = [1, 6]
        self.mock_model.return_value = (mock_logits, MagicMock(), MagicMock())
        
        # Мокаем topk
        mock_topk_result = MagicMock()
        mock_topk_result.indices = MagicMock()
        mock_topk_result.indices.size.return_value = [1, 3]
        mock_topk_result.indices.__getitem__.return_value = MagicMock()
        mock_topk_result.indices.__getitem__.return_value.item.return_value = 0
        mock_topk_result.values = MagicMock()
        mock_topk_result.values.__getitem__.return_value = MagicMock()
        mock_topk_result.values.__getitem__.return_value.item.return_value = 0.8
        
        mock_torch.topk.return_value = mock_topk_result
        
        # Мокаем softmax
        mock_torch.softmax.return_value = mock_topk_result.values
        
        # Импортируем после мока
        from ne_qiki.core.neural_engine_impl import NeuralEngineV1
        from ne_qiki.shared.models import Proposal, ActuatorCommand
        from dataclasses import dataclass
        
        @dataclass
        class MockBiosStatus:
            ok: bool = True
            temperature: float = 50.0
            power_draw: float = 50.0
            utilization: float = 50.0
        
        @dataclass
        class MockContext:
            bios_status: MockBiosStatus
            fsm_state: str = "ACTIVE"
            sensor_data: dict = None
        
        # Конфигурация
        config = {
            "window": 16,
            "in_dim": 32,
            "num_classes": 6,
            "param_dim": 4,
            "topk": 3,
            "min_confidence": 0.55,
            "time_budget_ms": 8,
            "calibration": {"temperature": 1.2},
            "action_catalog": {
                "actions": [
                    {"name": "HOLD_POSITION", "params": {}},
                ]
            }
        }
        
        # Создаем engine
        engine = NeuralEngineV1(config)
        
        # Создаем контекст
        context = MockContext(bios_status=MockBiosStatus())
        
        # Мокаем глобальный логгер
        with patch('ne_qiki.core.neural_engine_impl.GLOBAL_LOGGER') as mock_logger:
            # Вызываем generate_proposals
            proposals = engine.generate_proposals(context)
            
            # Проверяем, что логгер был вызван
            mock_logger.add_log.assert_called()

if __name__ == '__main__':
    unittest.main()
