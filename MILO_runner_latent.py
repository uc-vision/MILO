import torch
import argparse
from PIL import Image
from torchvision import transforms

from MILO.model import MILOLatentWithVAE
from MILO.MILO_runner import run_visualization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, default="images/ref.png")
    parser.add_argument("--dist", type=str, default="images/dist.png")
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-mse", help="VAE model to use for encoding")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_tensor = transforms.ToTensor()(Image.open(args.ref).convert("RGB")).unsqueeze(0).to(device)
    dist_tensor = transforms.ToTensor()(Image.open(args.dist).convert("RGB")).unsqueeze(0).to(device)
    
    model = MILOLatentWithVAE(args.vae_model).to(device)
    run_visualization(model, dist_tensor, ref_tensor, device)


if __name__ == "__main__":
    main()
