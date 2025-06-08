from typing import Literal, Optional

import torch
from architectures.supervised import SupervisedArchitecture
from ml4gw.nn.resnet.resnet_1d import NormLayer, ResNet1D


class MultiModalPsd(SupervisedArchitecture):
    """
    MultiModal embedding network that embeds time, frequency, and PSD data.

    We pass the data through their own ResNets defined by their layers
    and context dims, then concatenate the output embeddings.
    """
    def __init__(
        self,
        num_ifos: int,
        low_freq_classes: int,
        high_freq_classes: int,
        freq_classes: int,
        low_freq_layers: list[int],
        high_freq_layers: list[int],
        freq_layers: list[int],
        **kwargs
    ):
        super().__init__()

        self.low_resnet = ResNet1D(
            in_channels=num_ifos,
            layers=low_freq_layers,
            classes=low_freq_classes,
            kernel_size=3,
        )

        self.high_resnet = ResNet1d(
            in_channels=num_ifos,
            layers=high_freq_layers,
            classes=high_freq_classes,
            kernel_size=3,
        )

        self.freq_psd_resnet = ResNet1D(
            in_channels=int(num_ifos * 3),
            layers=freq_layers,
            classes=freq_classes,
            kernel_size=freq_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )

        self.classifier = torch.nn.Linear(low_freq_classes + high_freq_classes + freq_classes, 1)

    def forward(self, X, X_fft):
        low_out = self.low_resnet(X_low)
        high_out = self.high_resnet(X_high)
        freq_out = self.freq_resnet(X_fft)
        x = torch.cat([low_out, high_out, freq_out], dim=-1)
        return self.classifier(x)
