import torch
import numpy as np
from train.data.supervised.supervised import SupervisedAframeDataset
from utils.preprocessing import butter_bandpass_filter
import torch.nn.functional as F

class MultiModalSupervisedAframeDataset(SupervisedAframeDataset):
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        num_freqs = psds.shape[-1]
        freq_bins = np.linspace(0, self.hparams.sample_rate/2, num_freqs)
        # Perform a split into high/low freqs
        X_bg_low = self.whitener(X_bg, psds, lowpass=self.hparams.lowpass, highpass=None)
        X_bg_high = self.whitener(X_bg, psds, lowpass=None, highpass=self.hparams.highpass)
        X_fg_low = []
        X_fg_high = []
        for inj in X_inj:
            inj_low = self.whitener(inj, psds, lowpass=self.hparams.lowpass, highpass=None)
            inj_high = self.whitener(inj, psds, lowpass=None, highpass=self.hparams.highpass)
            X_fg_high.append(inj_high)
            X_fg_low.append(inj_low)
        X_fg_low = torch.stack(X_fg_low)
        X_fg_high = torch.stack(X_fg_high)
        # FFT and ASD for injs
        asds = psds**0.5 * 1e23
        asds = asds.float()

        X_bg_fft = torch.fft.rfft(X_bg)
        X_fg_fft = torch.fft.rfft(X_inj)
        num_freqs = X_bg_fft.shape[-1]
        if asds.shape[-1] != num_freqs:
            asds = F.interpolate(asds, size=(num_freqs,), mode="linear", align_corners=False)
        inv_asds = 1 / asds
        if X_fg_fft.real.ndim == 4:  # (num_views, batch, channels, freq_bins)
            num_views = X_fg_fft.real.shape[0]
            # inv_asds: [batch, channels, freq_bins] → [1, batch, channels, freq_bins] → [num_views, batch, channels, freq_bins]
            inv_asds_inj = inv_asds.unsqueeze(0).expand(num_views, -1, -1, -1)
        else:
            inv_asds_inj = inv_asds  # fallback for non-augmented case
        X_bg_fft = torch.cat((X_bg_fft.real, X_bg_fft.imag, inv_asds), dim=1)
        X_fg_fft = torch.cat((X_fg_fft.real, X_fg_fft.imag, inv_asds_inj), dim=2)
        # Return all 3 branches for both
        return (X_bg_low, X_bg_high, X_bg_fft), (X_fg_low, X_fg_high, X_fg_fft), psds

    def on_after_batch_transfer(self, batch, _):
        """
        This is a method inherited from the DataModule
        base class that gets called after data returned
        by a dataloader gets put on the local device,
        but before it gets passed to the LightningModule.
        Use this to do on-device augmentation/preprocessing.
        """
        if self.trainer.training:
            # if we're training, perform random augmentations
            # on input data and use it to impact labels
            (X, waveforms) = batch
            (X_low, X_high, X_fft), y = self.augment(X, waveforms)
            return (X_low, X_high, X_fft), y
        elif self.trainer.validating or self.trainer.sanity_checking:
            # If we're in validation mode but we're not validating
            # on the local device, the relevant tensors will be
            # empty, so just pass them through with a 0 shift to
            # indicate that this should be ignored
            [background, _, timeslide_idx], [signals] = batch

            # If we're validating, unfold the background
            # data into a batch of overlapping kernels now that
            # we're on the GPU so that we're not transferring as
            # much data from CPU to GPU. Once everything is
            # on-device, pre-inject signals into background.
            shift = self.timeslides[timeslide_idx].shift_size
            (X_bg_low, X_bg_high, X_bg_fft), (X_inj_low, X_inj_high, X_inj_fft), psds = self.build_val_batches(background, signals)
            # return everthing model and arch expect
            return (shift, (X_bg_low, X_bg_high, X_bg_fft), (X_inj_low, X_inj_high, X_inj_fft), psds)
        return batch

    def augment(self, X, waveforms):
        if X.ndim == 4 and X.shape[0] == 1:
            X = X.squeeze(0)

        X, y, psds = super().augment(X, waveforms)
        X_low = self.whitener(X, psds, lowpass=self.hparams.lowpass, highpass=None)
        X_high = self.whitener(X, psds, lowpass=None, highpass=self.hparams.highpass)

        # existing FFT pipeline
        asds = psds**0.5 * 1e23
        asds = asds.float()

        X_fft = torch.fft.rfft(X)
        num_freqs = X_fft.shape[-1]
        if asds.shape[-1] != num_freqs:
            asds = F.interpolate(asds, size=(num_freqs,), mode="linear", align_corners=False)
        inv_asds = 1 / asds
        X_fft = torch.cat((X_fft.real, X_fft.imag, inv_asds), dim=1)

        return (X_low, X_high, X_fft), y
