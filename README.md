# 📘 QIKI Neural Engine - Полная документация

## 📋 Описание

QIKI Neural Engine - это компактный, безопасный и CPU-оптимизированный нейросетевой движок для интеграции в агентную систему QIKI. Реализует генерацию "предложений" на основе анализа контекста агента и интегрируется в цикл принятия решений через интерфейс `INeuralEngine`.

## 🗂️ Структура проекта

```
ne_qiki/
├── core/                          # Ядро движка
│   ├── interfaces.py              # Интерфейс INeuralEngine
│   ├── feature_extractor.py       # Векторизация контекста
│   ├── neural_engine_impl.py      # Реализация NeuralEngineV1
│   ├── proposal_evaluator.py      # Оценка и фильтрация предложений
│   ├── safety.py                  # Безопасность и анти-флаппинг
│   ├── calibration.py             # Температурная калибровка
│   ├── metrics.py                 # Prometheus-метрики
│   ├── nats_logger.py             # NATS-логгер
│   └── __init__.py
├── models/                        # Нейросетевые модели
│   ├── ne_v1.py                   # GRU+MLP модель
│   └── __init__.py
├── shared/                        # Общие модели данных
│   ├── models.py                  # ActuatorCommand, Proposal
│   └── __init__.py
├── configs/                       # Конфигурации
│   └── config.example.yaml        # Конфигурация
├── schemas/                       # JSON-схемы
│   ├── agent_context.schema.json  # Схема контекста
│   ├── action_catalog.schema.json # Схема действий
│   └── safety.schema.yaml         # Схема безопасности
├── api/                           # API
│   └── health_check.py            # Health-check API
├── benchmark/                     # Бенчмарки
│   └── onnx_benchmark.py          # Бенчмарк ONNX
├── datasets/                      # Работа с датасетами
│   └── jsonl_dataset.py           # Загрузчик JSONL
├── examples/                      # Примеры использования
│   └── mock_inference.py          # Пример инференса
├── tests/                         # Юнит-тесты
│   ├── test_ne_v1_contract.py     # Контракт NeuralEngine
│   ├── test_feature_extractor.py  # Тест векторизатора
│   ├── test_safety_shield.py      # Тест безопасности
│   ├── test_proposal_evaluator.py # Тест оценщика
│   ├── test_nats_logger.py        # Тест NATS-логгера
│   ├── test_dataset_generator.py  # Тест генератора датасета
│   ├── test_terminal_dashboard.py # Тест терминального dashboard
│   ├── test_neural_engine_dashboard_integration.py # Интеграционный тест
│   ├── test_global_logger.py      # Тест глобального логгера
│   └── __init__.py
├── tools/                         # Инструменты
│   ├── terminal_dashboard.py      # Терминальный dashboard
│   ├── global_logger.py           # Глобальный логгер
│   ├── generate_dataset.py        # Генератор датасета
│   └── __init__.py
├── monitoring/                    # Мониторинг
│   ├── prometheus.yml             # Конфиг Prometheus
│   └── dashboard.json             # Grafana дашборд
├── train_bc.py                    # Обучение и экспорт ONNX
├── run_tests.py                   # Единый тест-раннер
├── Dockerfile                     # Docker-образ
├── docker-compose.yml             # Docker Compose
├── requirements.txt               # Зависимости
└── .github/
    └── workflows/
        └── ci.yml                 # CI Pipeline
```

## 🚀 Быстрый старт

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Запуск всех компонентов

```bash
docker-compose up -d
```

### Пример использования

```bash
python examples/mock_inference.py
```

### Запуск тестов

```bash
python run_tests.py
```

### Терминальный dashboard

```bash
python tools/terminal_dashboard.py
```

## 🧪 Архитектура

### Основные компоненты

| Компонент | Назначение |
|----------|------------|
| `INeuralEngine` | Интерфейс для интеграции в QIKI |
| `NeuralEngineV1` | Реализация: векторизация → модель → предложения |
| `NE_v1` | GRU(2×64) + MLP головы |
| `FeatureExtractor` | Преобразует `AgentContext` в тензор |
| `SafetyShield` | Проверяет безопасность и предотвращает флаппинг |
| `ProposalEvaluator` | Фильтрует и сортирует предложения |

### Поток данных

```
AgentContext → FeatureExtractor → NE_v1 (GRU+MLP) → Calibration → Proposals → 
SafetyShield → ProposalEvaluator → Decision
```

## 📦 Конфигурация

Файл: `configs/config.example.yaml`

```yaml
window: 16
in_dim: 32
num_classes: 6
param_dim: 4
topk: 3
min_confidence: 0.55
time_budget_ms: 8
calibration:
  temperature: 1.2
action_catalog:
  actions:
    - name: HOLD_POSITION
      params:
        duration: [0.1, 5.0]
    - name: COOLING_BOOST
      params:
        intensity: [0.1, 1.0]
    - name: THROTTLE_DOWN
      params:
        level: [0.0, 1.0]
    - name: THROTTLE_UP
      params:
        level: [0.0, 1.0]
    - name: ROTATE_LEFT
      params:
        angle: [0.0, 3.14]
    - name: ROTATE_RIGHT
      params:
        angle: [0.0, 3.14]
safety:
  fsm_invariants:
    - ERROR_STATE
  bios_invariants:
    - bios_ok
```

## 🔐 Безопасность

### Реализованные механизмы

- **FSM-инварианты**: нельзя действовать при `ERROR_STATE`
- **BIOS-инварианты**: проверка аппаратных параметров
- **Анти-флаппинг**: защита от частых переключений действий
- **Маскирование действий**: модель не может предложить запрещённое действие
- **Валидация параметров**: проверка диапазонов параметров действий

### SafetyShield

Проверяет:
- Состояние FSM
- Статус BIOS
- Корректность параметров действий
- Частоту повторения действий (анти-флаппинг)

## 📊 Мониторинг

### Prometheus-метрики

| Метрика | Описание |
|--------|----------|
| `ne_inference_total` | Количество вызовов инференса |
| `ne_inference_duration_seconds` | Латентность инференса |
| `ne_active_proposals` | Количество активных предложений |
| `ne_avg_confidence` | Средняя уверенность предложений |
| `ne_safety_blocks_total` | Количество блокировок SafetyShield |
| `ne_degradation_to_rule` | Количество деградаций в RuleEngine |

### Health-check API

```bash
curl http://localhost:5000/health
```

### Терминальный dashboard

Интерактивный терминальный интерфейс с реалтайм метриками и логами:

```bash
python tools/terminal_dashboard.py
```

Горячие клавиши:
- `q` - Выход
- `Ctrl+C` - Аварийный выход

## 📡 Логирование

### NATS-логгер

Все предложения логируются в топик `qiki.neural.proposals`

### Терминальное логирование

Реалтайм вывод логов в терминальном dashboard

## 🧠 Обучение модели

### Скрипт обучения

```bash
python train_bc.py
```

Включает:
- Обучение методом Behavior Cloning
- Экспорт в ONNX
- INT8-квантизация
- Бенчмарк производительности

### Генератор датасета

```bash
python tools/generate_dataset.py
```

Генерирует датасет в формате JSONL для обучения

## 🧪 Тестирование

### Запуск всех тестов

```bash
python run_tests.py
```

### Покрытие тестами

| Компонент | Тест |
|----------|------|
| `NeuralEngine` | `test_ne_v1_contract.py` |
| `FeatureExtractor` | `test_feature_extractor.py` |
| `SafetyShield` | `test_safety_shield.py` |
| `ProposalEvaluator` | `test_proposal_evaluator.py` |
| `NATSLogger` | `test_nats_logger.py` |
| `DatasetGenerator` | `test_dataset_generator.py` |
| `TerminalDashboard` | `test_terminal_dashboard.py` |

## 🐳 Docker

### Docker Compose

Запуск всей системы:

```bash
docker-compose up -d
```

Включает:
- QIKI Neural Engine
- Prometheus
- Grafana
- NATS (опционально)

### Отдельный запуск

```bash
docker build -t qiki-ne .
docker run qiki-ne
```

## 🧾 CI/CD

### GitHub Actions

Файл: `.github/workflows/ci.yml`

Автоматический запуск:
- Тестов при каждом пуше
- Проверки контрактов
- Валидации тайм-бюджета

## 🧩 Интеграция с QIKI

### Шаги интеграции

1. Реализуйте `IDataProvider` для передачи `AgentContext`
2. Подключите `NeuralEngineV1` как `INeuralEngine`
3. Настройте конфигурацию в `configs/config.yaml`
4. Запустите систему через Docker Compose

### Контракт INeuralEngine

```python
class INeuralEngine(ABC):
    def generate_proposals(self, context: Any) -> List[Proposal]:
        pass
```

## 📈 План развития

### Уровень 3 (долгосрочно)

- Transformer-модель вместо GRU
- RL-обучение (PPO, Offline RL)
- ONNX Server как отдельный микросервис
- Human-in-the-loop обучение
- Асинхронный инференс
- Multi-agent координация

## 📄 Лицензия

MIT

## 📞 Контакты

- Версия: 2.0
- Дата: 2025-04-05
