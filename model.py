import torch
import math
from pathlib import Path

try:
    from diffusers import AutoencoderKL
except ImportError:
    AutoencoderKL = None


class ScalerNetwork(torch.nn.Module):
    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(ScalerNetwork, self).__init__()

        layers = [
            torch.nn.Conv2d(1, chn_mid, 1, stride=1, padding=0, bias=True),
        ]
        layers += [
            torch.nn.LeakyReLU(0.2, True),
        ]
        layers += [
            torch.nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),
        ]
        layers += [
            torch.nn.LeakyReLU(0.2, True),
        ]
        layers += [
            torch.nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),
        ]
        if use_sigmoid:
            layers += [
                torch.nn.Sigmoid(),
            ]
        self.model = torch.nn.Sequential(*layers)

    def forward(self, val):
        return self.model.forward(val)


class MaskFinder(torch.nn.Module):
    def __init__(self, input_channels, num_features=64):
        super(MaskFinder, self).__init__()

        self.netBasic = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1
            ),
        )

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputChannels):
        out_net = self.netBasic(inputChannels)
        out = self.sigmoid(out_net)
        return out


class MILO(torch.nn.Module):
    def __init__(self):
        super(MILO, self).__init__()

        self.mask_finder_1 = MaskFinder(7)
        self.mask_finder_1.requires_grad = False
        self.number_of_scales = 3

        self.scaler_network = ScalerNetwork()

        model_path = Path(__file__).parent / "weights" / "MILO.pth"
        self.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)

    def mask_generator(self, y, x):
        B, C, H, W = y.shape[0:4]

        refScale = [x]
        distScale = [y]

        for intLevel in range(self.number_of_scales):
            refScale.insert(
                0,
                torch.nn.functional.avg_pool2d(
                    input=refScale[0], kernel_size=2, stride=2, count_include_pad=False
                ),
            )
            distScale.insert(
                0,
                torch.nn.functional.avg_pool2d(
                    input=distScale[0], kernel_size=2, stride=2, count_include_pad=False
                ),
            )

        mask = refScale[0].new_zeros(
            [
                refScale[0].shape[0],
                1,
                int(math.floor(refScale[0].shape[2] / 2.0)),
                int(math.floor(refScale[0].shape[3] / 2.0)),
            ]
        )

        for intLevel in range(len(refScale)):
            maskUpsampled = torch.nn.functional.interpolate(
                input=mask, scale_factor=2, mode="bilinear", align_corners=True
            )

            if maskUpsampled.shape[2] != refScale[intLevel].shape[2]:
                maskUpsampled = torch.nn.functional.pad(
                    input=maskUpsampled, pad=[0, 0, 0, 1], mode="replicate"
                )
            if maskUpsampled.shape[3] != refScale[intLevel].shape[3]:
                maskUpsampled = torch.nn.functional.pad(
                    input=maskUpsampled, pad=[0, 1, 0, 0], mode="replicate"
                )

            mask = (
                self.mask_finder_1(
                    torch.cat(
                        [refScale[intLevel], distScale[intLevel], maskUpsampled], 1
                    )
                )
                + maskUpsampled
            )

        return mask

    def forward(self, y, x, as_loss=True, resize=True):
        mask = self.mask_generator(x, y)
        score = (mask * torch.abs(x - y)).mean()
        return score

    def MILO_map(self, y, x):
        C, H, W = x.shape[0:3]

        masks = self.mask_generator(x, y)

        return self.scaler_network(
            (masks * torch.abs(x - y)).mean([1], keepdim=True)
        ) - self.scaler_network(torch.tensor(0.0, device=masks.device).reshape(1, 1, 1, 1)), masks[0]


class MILOLatent(torch.nn.Module):
    def __init__(self):
        super(MILOLatent, self).__init__()

        self.mask_finder_1 = MaskFinder(9)
        self.mask_finder_1.requires_grad = False
        self.number_of_scales = 3

        self.scaler_network = ScalerNetwork()

        model_path = Path(__file__).parent / "weights" / "MILO_latent.pth"
        self.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)

    def mask_generator(self, y, x):
        B, C, H, W = y.shape[0:4]

        refScale = [x]
        distScale = [y]

        for intLevel in range(self.number_of_scales):
            refScale.insert(
                0,
                torch.nn.functional.avg_pool2d(
                    input=refScale[0], kernel_size=2, stride=2, count_include_pad=False
                ),
            )
            distScale.insert(
                0,
                torch.nn.functional.avg_pool2d(
                    input=distScale[0], kernel_size=2, stride=2, count_include_pad=False
                ),
            )

        mask = refScale[0].new_zeros(
            [
                refScale[0].shape[0],
                1,
                int(math.floor(refScale[0].shape[2] / 2.0)),
                int(math.floor(refScale[0].shape[3] / 2.0)),
            ]
        )

        for intLevel in range(len(refScale)):
            maskUpsampled = torch.nn.functional.interpolate(
                input=mask, scale_factor=2, mode="bilinear", align_corners=True
            )

            if maskUpsampled.shape[2] != refScale[intLevel].shape[2]:
                maskUpsampled = torch.nn.functional.pad(
                    input=maskUpsampled, pad=[0, 0, 0, 1], mode="replicate"
                )
            if maskUpsampled.shape[3] != refScale[intLevel].shape[3]:
                maskUpsampled = torch.nn.functional.pad(
                    input=maskUpsampled, pad=[0, 1, 0, 0], mode="replicate"
                )

            mask = (
                self.mask_finder_1(
                    torch.cat(
                        [refScale[intLevel], distScale[intLevel], maskUpsampled], 1
                    )
                )
                + maskUpsampled
            )

        return mask

    def forward(self, y, x, as_loss=True, resize=True):
        mask = self.mask_generator(x, y)
        score = (mask * torch.abs(x - y)).mean()
        return score

    def MILO_map(self, y, x):
        C, H, W = x.shape[0:3]

        masks = self.mask_generator(x, y)

        return self.scaler_network(
            (masks * torch.abs(x - y)).mean([1], keepdim=True)
        ) - self.scaler_network(torch.tensor(0.0, device=masks.device).reshape(1, 1, 1, 1)), masks[0]


class MILOLatentWithVAE(torch.nn.Module):
    VAE_SCALE = 0.18215

    def __init__(self, vae_model: str = "stabilityai/sd-vae-ft-mse"):
        super(MILOLatentWithVAE, self).__init__()
        
        if AutoencoderKL is None:
            raise ImportError("diffusers package is required for MILOLatentWithVAE")
        
        self.vae = AutoencoderKL.from_pretrained(vae_model)
        self.vae.requires_grad_(False)
        self.milo = MILOLatent()
    
    def encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Encode image tensor to latent space. Image should be in [0, 1] range."""
        image_normalized = image_tensor * 2 - 1
        latent = self.vae.encode(image_normalized).latent_dist.mean * self.VAE_SCALE
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image space."""
        with torch.no_grad():
            image = self.vae.decode(latent / self.VAE_SCALE).sample
        return image
    
    def forward(self, dist_image, ref_image):
        """Compute MILO score. Images should be in [0, 1] range."""
        ref_latent = self.encode(ref_image)
        dist_latent = self.encode(dist_image)
        return self.milo(dist_latent, ref_latent)
    
    def MILO_map(self, dist_image, ref_image):
        """Compute MILO error map and mask. Images should be in [0, 1] range."""
        ref_latent = self.encode(ref_image)
        dist_latent = self.encode(dist_image)
        return self.milo.MILO_map(dist_latent, ref_latent)
