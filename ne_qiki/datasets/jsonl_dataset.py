import json
import torch
from torch.utils.data import Dataset

class JSONLDataset(Dataset):
    def __init__(self, path: str):
        self.data = []
        with open(path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.tensor(item['features'], dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.long)
        priority = torch.tensor(item['priority'], dtype=torch.float32)
        params = torch.tensor(item['params'], dtype=torch.float32)
        mask = torch.tensor(item['action_mask'], dtype=torch.bool)
        return features, label, priority, params, mask
