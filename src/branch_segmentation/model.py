from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerModel


class MiTEncoder(nn.Module):
    """SegFormer MiT encoder that returns four multi-scale feature maps."""

    def __init__(self, model_name: str = "nvidia/mit-b2", pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.encoder = SegformerModel.from_pretrained(model_name)
        else:
            self.encoder = SegformerModel(SegformerConfig.from_pretrained(model_name))
        self.out_channels = list(self.encoder.config.hidden_sizes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = self.encoder(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )
        return tuple(outputs.hidden_states)


class HRStem(nn.Module):
    """Light high-resolution stem for preserving thin branch details."""

    def __init__(self, in_ch: int = 3, out_ch: int = 24):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StripConvBlock(nn.Module):
    """Asymmetric + local + dilated context block."""

    def __init__(self, in_ch: int, out_ch: int, strip_length: int = 15):
        super().__init__()
        pad = strip_length // 2
        self.h_conv = nn.Conv2d(
            in_ch, out_ch, (1, strip_length), padding=(0, pad), bias=False
        )
        self.v_conv = nn.Conv2d(
            in_ch, out_ch, (strip_length, 1), padding=(pad, 0), bias=False
        )
        self.local = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.dilated = nn.Conv2d(
            in_ch, out_ch, 3, padding=2, dilation=2, bias=False
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 4, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(
            torch.cat(
                [self.h_conv(x), self.v_conv(x), self.local(x), self.dilated(x)],
                dim=1,
            )
        )


class DecoderBlock(nn.Module):
    def __init__(self, enc_ch: int, prev_ch: int, out_ch: int, strip_length: int):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(enc_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        in_ch = out_ch + prev_ch if prev_ch > 0 else out_ch
        self.merge = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.strip = StripConvBlock(out_ch, out_ch, strip_length)

    def forward(
        self, enc_feat: torch.Tensor, prev_feat: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.reduce(enc_feat)
        if prev_feat is not None:
            prev_up = F.interpolate(
                prev_feat, size=x.shape[2:], mode="bilinear", align_corners=False
            )
            x = torch.cat([x, prev_up], dim=1)
        return self.strip(self.merge(x))


class SideOutputHead(nn.Module):
    def __init__(self, in_ch: int, num_spec_filters: int = 16):
        super().__init__()
        self.spec_conv = nn.Sequential(
            nn.Conv2d(in_ch, num_spec_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_spec_filters),
            nn.ReLU(inplace=True),
        )
        self.predict = nn.Conv2d(num_spec_filters, 1, 1)

    def forward(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        x = self.predict(self.spec_conv(x))
        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)


class FinalUpsample(nn.Module):
    """Bilinear upsample + convolutional refinement with HR feature fusion."""

    def __init__(self, in_ch: int, hr_ch: int = 24, mid_ch: int = 32):
        super().__init__()
        self.up_block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.fuse_hr = nn.Sequential(
            nn.Conv2d(mid_ch + hr_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.predict = nn.Conv2d(mid_ch, 1, 1)

    def forward(self, x: torch.Tensor, hr_features: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, size=hr_features.shape[2:], mode="bilinear", align_corners=False
        )
        x = self.up_block(x)
        x = self.fuse_hr(torch.cat([x, hr_features], dim=1))
        return self.predict(x)


class BranchHeatmapNet(nn.Module):
    def __init__(
        self,
        encoder_name: str = "nvidia/mit-b2",
        decoder_channels: tuple[int, int, int, int] = (256, 128, 64, 32),
        num_spec_filters: int = 16,
        hr_channels: int = 24,
        final_mid_channels: int = 32,
        pretrained_encoder: bool = True,
    ):
        super().__init__()
        self.hr_stem = HRStem(in_ch=3, out_ch=hr_channels)
        self.encoder = MiTEncoder(encoder_name, pretrained=pretrained_encoder)
        enc_ch = self.encoder.out_channels
        dc = tuple(decoder_channels)

        self.dec4 = DecoderBlock(enc_ch[3], 0, dc[0], strip_length=3)
        self.dec3 = DecoderBlock(enc_ch[2], dc[0], dc[1], strip_length=7)
        self.dec2 = DecoderBlock(enc_ch[1], dc[1], dc[2], strip_length=11)
        self.dec1 = DecoderBlock(enc_ch[0], dc[2], dc[3], strip_length=15)

        self.side4 = SideOutputHead(dc[0], num_spec_filters)
        self.side3 = SideOutputHead(dc[1], num_spec_filters)
        self.side2 = SideOutputHead(dc[2], num_spec_filters)
        self.final_up = FinalUpsample(dc[3], hr_ch=hr_channels, mid_ch=final_mid_channels)
        self.fuse = nn.Conv2d(4, 1, 1)

    def forward(
        self, x: torch.Tensor, return_side_outputs: bool | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h, w = x.shape[2:]
        hr_feat = self.hr_stem(x)
        f1, f2, f3, f4 = self.encoder(x)

        d4 = self.dec4(f4)
        d3 = self.dec3(f3, d4)
        d2 = self.dec2(f2, d3)
        d1 = self.dec1(f1, d2)

        s4 = self.side4(d4, (h, w))
        s3 = self.side3(d3, (h, w))
        s2 = self.side2(d2, (h, w))
        s1 = self.final_up(d1, hr_feat)
        fused = self.fuse(torch.cat([s4, s3, s2, s1], dim=1))

        if return_side_outputs is None:
            return_side_outputs = self.training
        if return_side_outputs:
            return fused, s1, s2, s3, s4
        return torch.sigmoid(fused)
