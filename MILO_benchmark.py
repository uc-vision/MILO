import torch
import argparse
import time
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms

from MILO.model import MILO, MILOLatentWithVAE
from tqdm import tqdm


def benchmark_model(model, input_data, device, args, model_name, dtype=torch.bfloat16, mode="max-autotune"):
    model.eval()

    print(f"Compiling {model_name}...")
    model = torch.compile(model, mode=mode)
    model.requires_grad_(False)
    model.to(dtype=dtype)

    def run_once(input):    
        with torch.autocast(device_type=device.type, dtype=dtype):
            for _ in range(args.warmup):
                image = torch.nn.Parameter(input, requires_grad=True)
                score = model(image, image.detach()).mean()
                score.backward()

    print(f"Warmup ({args.warmup} iterations)...")
    for i in tqdm(range(args.warmup)):
        run_once(input_data)

    print(f"Benchmarking {model_name} ({args.iterations} iterations)...")
    torch.cuda.synchronize() if device.type == "cuda" else None
    start_time = time.perf_counter()

    for i in tqdm(range(args.iterations)):
        run_once(input_data)

    torch.cuda.synchronize() if device.type == "cuda" else None
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time = total_time / args.iterations
    throughput = args.iterations / total_time

    print(f"\n{model_name} Results:")
    print(f"  Input shape: {input_data.shape}")
    print(f"  Total time: {total_time:.4f}s")
    print(f"  Average time per iteration: {avg_time*1000:.2f}ms")
    print(f"  Throughput: {throughput:.2f} iterations/s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MILO model performance")
    parser.add_argument("--image", type=str, default="images/ref.png", help="Path to image file")
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-mse", help="VAE model to use for milo-latent")
    parser.add_argument("--iterations", type=int, default=200, help="Number of iterations to benchmark")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup iterations")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("=" * 50)
    print("MILO Benchmark")
    print("=" * 50)
    model_milo = MILO().to(device)
    image = transforms.ToTensor()(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    benchmark_model(model_milo, image, device, args, "MILO")

    print("\n" + "=" * 50)
    print("MILO-Latent Benchmark")
    print("=" * 50)
    model_latent_with_vae = MILOLatentWithVAE(args.vae_model).to(device)
    image_tensor = transforms.ToTensor()(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    benchmark_model(model_latent_with_vae, image_tensor, device, args, "MILO-Latent")


if __name__ == "__main__":
    main()
