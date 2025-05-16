import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from prettytable import PrettyTable
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

def count_parameters(model):
    table = PrettyTable(["Module", "Parameters"])
    total = 0
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        table.add_row([name, f"{params:,}"])
        total += params
    print(table)
    print(f"Total parameters: {total:,}")
    return total

class DirectionEncoder(nn.Module):
    """
    Simple patch-wise 2-D projection similar to a ViT patch embed.

    Args:
        in_channels (int): C from the StyleGAN feature map.
        hidden_size (int): Dimension that the UNet expects for cross-attention.
        patch_size (int): Size of square patches used to reduce sequence length.
    """
    def __init__(self, in_channels: int, hidden_size: int, patch_size: int = 32):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.norm = nn.LayerNorm(hidden_size)  # keeps things numerically stable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, 512, 512]  (Δ feature map)
        returns: [B, N, hidden_size]  where N = (512/patch)^2
        """
        b, c, h, w = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # [B, seq, hidden]
        x = self.norm(x)
        return x


class StyleGANFeatureCondUNet(nn.Module):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        dir_encoder: DirectionEncoder,
    ):
        super().__init__()
        self.unet = unet
        self.dir_encoder = dir_encoder

        assert (
            self.unet.config.cross_attention_dim == self.dir_encoder.norm.normalized_shape[0]
        ), "cross_attention_dim mismatch – set both to the same hidden size!"

    def forward(
        self,
        sample: torch.Tensor,       # [B, C, 64, 64] – StyleGAN feature map
        direction: torch.Tensor,    # [B, C, 64, 64] – Δ direction map
    ) -> torch.Tensor:
        # Encode direction → tokens
        encoder_hidden_states = self.dir_encoder(direction)  # [B, N, H]

        timesteps = torch.zeros(sample.shape[0], dtype=torch.long, device=sample.device)

        unet_out: UNet2DConditionOutput = self.unet(
            sample=sample,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
        )
        return unet_out.sample  # [B, C, 64, 64]

    
def build_stylegan_cond_unet(
    base_channels: int = 512,
    patch_size: int = 16, 
    hidden_size: int = 1024,
):
    
    unet_config = {
        "sample_size": 8,                 # spatial resolution - 64/2^n_blocks
        "in_channels": base_channels,      
        "out_channels": base_channels,
        "layers_per_block": 2, #change to 2
        "block_out_channels": [hidden_size, hidden_size * 2, hidden_size * 4],
        "cross_attention_dim": hidden_size, 
        "down_block_types": (
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        "up_block_types": (
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
        "norm_num_groups": 32,
        "mid_block_type": "UNetMidBlock2DCrossAttn",
    }
    unet = UNet2DConditionModel(**unet_config)

    dir_encoder = DirectionEncoder(
        in_channels=base_channels,
        hidden_size=hidden_size,
        patch_size=patch_size,
    )
    
    cond_net = StyleGANFeatureCondUNet(unet, dir_encoder)
    
    count_parameters(cond_net)
    count_parameters(unet)
    
    return cond_net
