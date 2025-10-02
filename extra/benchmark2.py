import time
import torch
import open_clip
import numpy as np

# Define the text inputs for benchmarking
text_inputs = ["Example text input 1", "Example text input 2", "Example text input 3"]


def benchmark_and_get_output(device_name):
    """
    Benchmark and get the model output on the given device.
    """
    # Set device
    device = torch.device(device_name)

    # Load the model and preprocessing
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-SO400M-14-SigLIP-384",
        pretrained="webli",
        precision="fp16",
        device=device,
    )

    # Tokenize the text inputs
    tokenizer = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-384")
    tokenized_text = tokenizer(text_inputs).to(device)

    # Move model to evaluation mode
    model.eval()

    # Warm-up run (to stabilize measurements)
    with torch.no_grad():
        model.encode_text(tokenized_text)

    # Measure performance
    start_time = time.time()
    outputs = None
    with torch.no_grad():
        for _ in range(1):  # Run multiple iterations for averaging
            text_features = model.encode_text(tokenized_text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            outputs = text_features.detach().cpu().numpy()
    end_time = time.time()

    avg_time = (end_time - start_time) / 10  # Average time per iteration
    return avg_time, outputs


# Dictionary to store results and outputs
results = {}
outputs = {}

# Backends to benchmark
backends = ["cpu", "cuda", "mps"]

# Benchmark each backend
for backend in backends:
    if backend == "cuda" and not torch.cuda.is_available():
        print(f"CUDA not available on this system. Skipping CUDA benchmark.")
        continue
    if backend == "mps" and not torch.backends.mps.is_available():
        print(f"MPS not available on this system. Skipping MPS benchmark.")
        continue

    print(f"Benchmarking on {backend}...")
    avg_time, output = benchmark_and_get_output(backend)
    results[backend] = avg_time
    outputs[backend] = output
    print(f"Average time per iteration on {backend}: {avg_time:.6f} seconds")

# Compare outputs using L2 norm (Euclidean distance)
print("\nOutput Comparison:")
backend_keys = list(outputs.keys())
for i in range(len(backend_keys)):
    for j in range(i + 1, len(backend_keys)):
        b1, b2 = backend_keys[i], backend_keys[j]
        diff = np.linalg.norm(
            outputs[b1] - outputs[b2], axis=1
        )  # L2 Norm for each input
        print(f"Difference between {b1.upper()} and {b2.upper()}: {diff}")

# Display final benchmark results
print("\nBenchmark Results:")
for backend, avg_time in results.items():
    print(f"{backend.upper()}: {avg_time:.6f} seconds per iteration")
