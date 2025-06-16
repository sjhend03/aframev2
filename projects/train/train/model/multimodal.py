import torch
import torch.nn.functional as F
from train.model.supervised import SupervisedAframe
from architectures.supervised import SupervisedArchitecture

Tensor = torch.Tensor


class MultimodalSupervisedAframe(SupervisedAframe):
    def __init__(self, arch: SupervisedArchitecture, *args, **kwargs) -> None:
        super().__init__(arch, *args, **kwargs)

    def forward(self, X_low, X_high, X_fft):
        return self.model(X_low, X_high, X_fft)

    def train_step(self, batch: tuple[tuple[Tensor, Tensor, Tensor], Tensor]) -> Tensor:
        (X_low, X_high, X_fft), y = batch
        y_hat = self.forward(X_low, X_high, X_fft)
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def score(self, X):
        """
        Called during validation. X is expected to be a tuple: (X_low, X_high, X_fft)
        """
        X_low, X_high, X_fft = X
        return self.forward(X_low, X_high, X_fft)

    def validation_step(self, batch, _):
        shift, X_bg, X_fg, psds = batch

        X_low_bg, X_high_bg, X_fft_bg = X_bg
        X_low_fg, X_high_fg, X_fft_fg = X_fg

        y_bg = self.score((X_low_bg, X_high_bg, X_fft_bg))

        if X_low_fg.ndim == 3: # (num_views, batch, ...)
            num_views, batch_size, *_ = X_low_fg.shape
            X_low_fg = X_low_fg.view(num_views * batch_size, *X_low_fg.shape[2:])
            X_high_fg = X_high_fg.view(num_views * batch_size, *X_high_fg.shape[2:])
            X_fft_fg = X_fft_fg.view(num_views * batch_size, *X_fft_fg.shape[2:])

            y_fg = self.score((X_low_fg, X_high_fg, X_fft_fg))
            y_fg = y_fg.view(num_views, batch_size).mean(0)
        else:
            y_fg = self.score((X_low_fg, X_high_fg, X_fft_fg))

        self.metric.update(shift, y_bg, y_fg)

        self.log(
            "valid_auroc",
            self.metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
