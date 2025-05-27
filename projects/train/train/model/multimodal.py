import torch
from train.model.supervised import SupervisedAframe
from architectures.supervised import SupervisedArchitecture
import torch.nn.functional as F

Tensor = torch.Tensor


class MultimodalSupervisedAframe(SupervisedAframe):
    def __init__(self, arch: SupervisedArchitecture, *args, **kwargs) -> None:
        super().__init__(arch, *args, **kwargs)

    def forward(self, X, X_fft):
        return self.model(X, X_fft)

    def train_step(self, batch: tuple[tuple[Tensor, Tensor], Tensor]) -> Tensor:
        (X, X_fft), y = batch
        y_hat = self.forward(X, X_fft)
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def score(self, X):
        """
        Called during validation to get logits. X is expected to be a tuple (X, X_fft)
        """
        X, X_fft = X
        return self.forward(X, X_fft)

    def validation_step(self, batch, _):
        shift, X_bg, X_inj, psds = batch

        # === Prepare PSD-based features ===
        asds = psds**0.5 * 1e23
        asds = asds.float()

        # FFT + inverse ASD for background
        X_fft_bg = torch.fft.rfft(X_bg)
        num_freqs = X_fft_bg.shape[-1]
        if asds.shape[-1] != num_freqs:
            asds = F.interpolate(asds, size=(num_freqs,), mode="linear")
        inv_asds = 1 / asds
        X_fft_bg = torch.cat((X_fft_bg.real, X_fft_bg.imag, inv_asds), dim=1)

        y_bg = self.score((X_bg, X_fft_bg))

        # === Process foreground injections with multiple views ===
        num_views, batch_size, *signal_shape = X_inj.shape
        X_inj_flat = X_inj.view(num_views * batch_size, *signal_shape)

        X_fft_fg = torch.fft.rfft(X_inj_flat)
        inv_asds_fg = inv_asds.repeat(num_views, 1, 1)
        X_fft_fg = torch.cat((X_fft_fg.real, X_fft_fg.imag, inv_asds_fg), dim=1)

        y_fg = self.score((X_inj_flat, X_fft_fg))
        y_fg = y_fg.view(num_views, batch_size).mean(0)

        self.metric.update(shift, y_bg, y_fg)

        self.log(
            "valid_auroc",
            self.metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
