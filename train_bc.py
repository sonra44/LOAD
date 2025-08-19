import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ne_qiki.models.ne_v1 import NE_v1
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# Пример датасета
class MockDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randn(16, 32)
        y_class = torch.randint(0, 6, (1,)).long()
        y_priority = torch.rand(1)
        y_params = torch.rand(4)
        mask = torch.ones(6).bool()
        return x, y_class, y_priority, y_params, mask

def train_model():
    model = NE_v1(32, 64, 6, 4)
    dataset = MockDataset(1000)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(5):
        for x, y_cls, y_prio, y_param, mask in loader:
            logits, prio, param = model(x, mask)
            loss = criterion_cls(logits, y_cls) + \
                   criterion_reg(prio, y_prio) + \
                   criterion_reg(param, y_param)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "ne_v1.pt")
    print("Model saved as ne_v1.pt")

    # Экспорт в ONNX
    model.eval()
    dummy_input = torch.randn(1, 16, 32)
    dummy_mask = torch.ones(1, 6).bool()
    torch.onnx.export(
        model,
        (dummy_input, dummy_mask),
        "ne_v1.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input", "mask"],
        output_names=["logits", "priority", "params"],
        dynamic_axes={
            "input": {0: "batch", 1: "time"},
            "mask": {0: "batch"},
            "logits": {0: "batch"},
            "priority": {0: "batch"},
            "params": {0: "batch"}
        }
    )
    print("ONNX model exported as ne_v1.onnx")

    # Квантизация
    quantize_dynamic("ne_v1.onnx", "ne_v1_int8.onnx", weight_type=QuantType.QUInt8)
    print("INT8 quantized model saved as ne_v1_int8.onnx")

if __name__ == "__main__":
    train_model()
