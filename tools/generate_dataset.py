import json
import random
import numpy as np

def generate_sample():
    """Генерирует один пример контекста и действия"""
    fsm_states = ["BOOTING", "IDLE", "ACTIVE", "ERROR_STATE"]
    actions = list(range(6))
    
    # Контекст
    features = np.random.randn(32).tolist()
    
    # Метки
    label = random.choice(actions)
    priority = random.random()
    params = np.random.randn(4).tolist()
    mask = [True, True, True, True, False, False]
    
    return {
        "features": features,
        "label": label,
        "priority": priority,
        "params": params,
        "action_mask": mask
    }

def generate_dataset(path="dataset.jsonl", size=1000):
    with open(path, 'w') as f:
        for _ in range(size):
            sample = generate_sample()
            f.write(json.dumps(sample) + '\n')
    print(f"Dataset saved to {path}")

if __name__ == "__main__":
    generate_dataset()
