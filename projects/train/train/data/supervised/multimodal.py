import torch

from train.data.supervised.supervised import SupervisedAframeDataset
from utils.preprocessing import butter_bandpass_filter

class MultiModalSupervisedAframeDataset(SupervisedAframeDataset):
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg = self.whitener(X_bg, psds)
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            X_fg.append(inj)

        X_fg = torch.stack(X_fg)
        return X_bg, X_fg, psds

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
            [X], waveforms = batch
            batch = self.augment(X, waveforms)
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
            X_bg, X_fg, psds = self.build_val_batches(background, signals)
            batch = (shift, X_bg, X_fg, psds)
        return batch

    def augment(self, X, waveforms):
        X, y, psds = super().augment(X, waveforms)
        X = self.whitener(X, psds)

        X_low, X_high = self.split_frequency_components(X)

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

    def split_frequency_components(self, X, fs=4096):
        """
        Takes a batch of time-domain data and returns
        (low_freq, high_freq)
        """
        X_np = X.cpu().numpy() # (B, C, T)
        low = butter_bandpass_filter(X_np, None, 100, fs)
        high = butter_bandpass_filter(X_np, 100, None, fs)
        return (
            torch.tensor(low, dtype=X.dtype, device=X.device),
            torch.tensor(high, dtype=X.dtype, device=X.device)
        )
