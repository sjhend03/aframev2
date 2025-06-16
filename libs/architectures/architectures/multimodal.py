from typing import Literal, Optional

import torch
from architectures.supervised import SupervisedArchitecture
from ml4gw.nn.resnet.resnet_1d import NormLayer, ResNet1D

class MultiModalPsd(SupervisedArchitecture):
    """
    MultiModal embedding network that embeds time, frequency, and PSD data.
    Each modality is processed with its own ResNet, then the outputs are concatenated and classified.
    """
    def __init__(
        self,
        num_ifos: int,
        time_classes: int,
        freq_classes: int,
        time_layers: list[int],
        freq_layers: list[int],
        time_kernel_size: int = 3,
        freq_kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
        **kwargs
    ):
        super().__init__()

        self.low_resnet = ResNet1D(
            in_channels=num_ifos,
            layers=time_layers,
            classes=time_classes,
            kernel_size=time_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )

        self.high_resnet = ResNet1D(
            in_channels=num_ifos,
            layers=time_layers,
            classes=time_classes,
            kernel_size=time_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
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

        self.classifier = torch.nn.Linear(2 * time_classes + freq_classes, 1)

    def forward(self, X_low, X_high, X_fft):
        low_out = self.low_resnet(X_low)
        high_out = self.high_resnet(X_high)
        freq_out = self.freq_psd_resnet(X_fft)
        x = torch.cat([low_out, high_out, freq_out], dim=-1)
        return self.classifier(x)

