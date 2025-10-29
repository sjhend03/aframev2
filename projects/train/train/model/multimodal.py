import torch
from architectures.supervised import SupervisedArchitecture
from train.model.base import AframeBase
import torch.nn.functional as F

Tensor = torch.Tensor


class MultimodalAframe(AframeBase):
    def __init__(
        self,
        arch: SupervisedArchitecture,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(arch, *args, **kwargs)

    def forward(self, x_low: Tensor, x_high: Tensor, x_fft: Tensor) -> Tensor:
        return self.model(x_low, x_high, x_fft)

    def train_step(self, batch: tuple) -> Tensor:
        # Unpack depending on number of elements
        if len(batch) != 4:
            raise ValueError(
                f"Unexpected batch format in train_step: {len(batch)} elements"
            )

        X_low, X_high, X_fft, y = batch

        y_hat = self(X_low, X_high, X_fft).squeeze(-1)
        # Match shape of y_hat
        y = y.float().view_as(y_hat)
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def score(self, X):
        X_low, X_high, X_fft = X
        return self.model(X_low, X_high, X_fft).squeeze(-1)

    def validation_step(self, batch, batch_idx):
        try:
            shift, X_bg, X_inj = batch
        except ValueError:
            (
                shift,
                X_bg_low,
                X_bg_high,
                X_bg_fft,
                X_fg_low,
                X_fg_high,
                X_fg_fft,
                *_,
            ) = batch
            X_bg = (X_bg_low, X_bg_high, X_bg_fft)
            X_inj = (X_fg_low, X_fg_high, X_fg_fft)

        # Score background
        y_bg = self.score(X_bg)

        # Score injected signals
        x0 = X_inj[0]
        if x0.ndim >= 4:
            V, B = x0.shape[:2]
            x_flat = tuple(x.reshape(V * B, *x.shape[2:]) for x in X_inj)
            y_fg = self.score(x_flat).view(V, B).mean(0)
        else:
            y_fg = self.score(X_inj)

        shift_val = float(shift) if not isinstance(shift, float) else shift
        self.metric.update(shift_val, y_bg, y_fg)

        self.log(
            "valid_auroc",
            self.metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
