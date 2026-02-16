# Copyright (C) 2026 Ujjwal Sharma and Omar Shahbaz Khan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""Benchmarking script for testing CLIP model performance on different devices.

This script benchmarks the performance of the ViT-SO400M-14-SigLIP-384 model
on different compute devices (CPU, CUDA, MPS) to help optimize deployment
configuration for the Live Services Engine.
"""

import time
import torch
import open_clip

# Define the text inputs for benchmarking
text_inputs = ["Example text input 1", "Example text input 2", "Example text input 3"]


def benchmark_model(device_name):
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
        text_features = model.encode_text(tokenized_text)

    # Measure performance
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):  # Run multiple iterations for averaging
            text_features = model.encode_text(tokenized_text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.detach().cpu().numpy()
    end_time = time.time()

    avg_time = (end_time - start_time) / 10  # Average time per iteration
    return avg_time


# Benchmark for each backend
backends = ["cpu", "cuda", "mps"]
results = {}

for backend in backends:
    if backend == "cuda" and not torch.cuda.is_available():
        print(f"CUDA not available on this system. Skipping CUDA benchmark.")
        continue
    if backend == "mps" and not torch.backends.mps.is_available():
        print(f"MPS not available on this system. Skipping MPS benchmark.")
        continue

    print(f"Benchmarking on {backend}...")
    avg_time = benchmark_model(backend)
    results[backend] = avg_time
    print(f"Average time per iteration on {backend}: {avg_time:.6f} seconds")

# Display results
print("\nBenchmark Results:")
for backend, avg_time in results.items():
    print(f"{backend.upper()}: {avg_time:.6f} seconds per iteration")
