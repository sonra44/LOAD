import time
import torch
import onnxruntime as ort

def benchmark_onnx(model_path: str, iterations: int = 100):
    session = ort.InferenceSession(model_path)
    dummy_input = torch.randn(1, 16, 32).numpy()
    dummy_mask = torch.ones(1, 6).bool().numpy()

    # Warmup
    for _ in range(10):
        session.run(None, {"input": dummy_input, "mask": dummy_mask})

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        session.run(None, {"input": dummy_input, "mask": dummy_mask})
    elapsed = time.time() - start

    avg_ms = (elapsed / iterations) * 1000
    print(f"ONNX Inference Avg Latency: {avg_ms:.2f} ms")
    return avg_ms

if __name__ == "__main__":
    benchmark_onnx("ne_v1_int8.onnx")
