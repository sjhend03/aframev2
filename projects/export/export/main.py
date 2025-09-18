import io
import logging
from typing import Optional

import h5py
import hermes.quiver as qv
import torch

from export.snapshotter import add_streaming_input_preprocessor
from utils.s3 import open_file


def scale_model(model, instances):
    """
    Scale the model to the number of instances per GPU desired
    at inference time
    """
    # TODO: should quiver handle this under the hood?
    try:
        model.config.scale_instance_group(instances)
    except ValueError:
        model.config.add_instance_group(count=instances)


def export(
    weights: str,
    repository_directory: str,
    batch_file: str,
    num_ifos: int,
    kernel_length: float,
    inference_sampling_rate: float,
    sample_rate: float,
    batch_size: int,
    fduration: float,
    psd_length: float,
    fftlength: Optional[float] = None,
    q: Optional[float] = None,
    highpass: Optional[float] = None,
    lowpass: Optional[float] = None,
    streams_per_gpu: int = 1,
    aframe_instances: Optional[int] = None,
    preproc_instances: Optional[int] = None,
    platform: qv.Platform = qv.Platform.TENSORRT,
    clean: bool = False,
    verbose: bool = False,
    **kwargs,
) -> None:
    """
    Export a aframe architecture to a model repository
    for streaming inference, including adding a model
    for caching input snapshot state on the server.

    Args:
        weights:
            File Like object or Path representing
            a set of trained weights that will be
            exported to a model_repository. Supports
            local and S3 paths.
        repository_directory:
            Directory to which to save the models and their
            configs
        batch_file:
            Path to file containing a batch of data from model
            training. This is used to determine the input size
            of the model. File structure is assumed to match
            the structure of the file written during training
        logdir:
            Directory to which logs will be written
        num_ifos:
            The number of interferometers contained along the
            channel dimension used to train aframe
        kernel_length:
            Length of segment in seconds that the network sees
        inference_sampling_rate:
            The rate at which kernels are sampled from the
            h(t) timeseries. This, along with the `sample_rate`,
            dictates the size of the update expected at the
            snapshotter model
        sample_rate:
            Rate at which the input kernel has been sampled, in Hz
        batch_size:
            Number of kernels per batch
        fduration:
            Length of the time-domain whitening filter in seconds
        psd_length:
            Length of background time in seconds to use for PSD
            calculation
        fftlength:
            Length of time in seconds to use to calculate the FFT
            during whitening
        highpass:
            Frequency to use for a highpass filter
        lowpass:
            Frequency to use for a lowpass filter
        streams_per_gpu:
            The number of snapshot states to host per GPU during
            inference
        aframe_instances:
            The number of concurrent execution instances of the
            aframe architecture to host per GPU during inference
        platform:
            The backend framework platform used to host the
            aframe architecture on the inference service. Right
            now only `"onnxruntime_onnx"` is supported.
        clean:
            Whether to clear the repository directory before starting
            export
        verbose:
            If True, log at `DEBUG` verbosity, otherwise log at
            `INFO` verbosity.
        **kwargs:
            Key word arguments specific to the export platform
    """

    # load in the model graph
    logging.info("Initializing model graph")
#UNCOMMENT BELOW FOR ACTUAL RUN
    with open_file(weights, "rb") as f:
        graph = nn = torch.jit.load(f, map_location="cpu")
        print(graph)
        print(graph.forward)
        print(graph.forward.__doc__)


#UNCOMMENT ON ACTUAL RUN
    graph.eval()
    logging.info(f"Initialize:\n{nn}")
    # instantiate a model repository at the
    # indicated location. Split up the preprocessor
    # and the neural network (which we'll call aframe)
    # to export/scale them separately, and start by
    # seeing if either already exists in the model repo
    repo = qv.ModelRepository(repository_directory, clean)
    try:
        aframe = repo.models["aframe"]
    except KeyError:
        aframe = repo.add("aframe", platform=platform)

    # if we specified a number of instances we want per-gpu
    # for each model at inference time, scale them now
    if aframe_instances is not None:
        scale_model(aframe, aframe_instances)

    with open_file(batch_file, "rb") as f:
        batch = h5py.File(io.BytesIO(f.read()))
        print("BATCH GILE XFFT SHAPE")
        print(batch["X_fft"].shape)
        if "X" in batch.keys():
            size = batch["X"].shape[2:]
        else:
            size = None
        if "X_fft" in batch.keys():
            size_fft = batch["X_fft"].shape[-2:]
        else: 
            size_fft = None
        if "X_low" in batch.keys():
            size_low = batch["X_low"].shape[2:]
        else: 
            size_low = None
        if "X_high" in batch.keys():
            size_high = batch["X_high"].shape[2:]
        else: 
            size_high = None

    input_shape_dict = {}
    if size_low is not None:
        input_shape_low = (batch_size, num_ifos) + tuple(size_low)
        input_shape_dict["whitened_low"] = input_shape_low
    if size_high is not None:
        input_shape_high = (batch_size, num_ifos) + tuple(size_high)
        input_shape_dict["whitened_high"] = input_shape_high
    if size_fft is not None:
        input_shape_fft = (batch_size,) + tuple(size_fft)
        input_shape_dict["whitened_fft"] = input_shape_fft

    # the network will have some different keyword
    # arguments required for export depending on
    # the target inference platform
    # TODO: hardcoding these kwargs for now, but worth
    # thinking about a more robust way to handle this

    aframe.export_version(
        graph,
        input_shapes=input_shape_dict,
        output_names=["discriminator"],
        **kwargs,
    )

    ensemble_name = "aframe-stream"

    try:
        # first see if we have an existing
        # ensemble with the given name
        ensemble = repo.models[ensemble_name]
    except KeyError:
        ensemble = repo.add(ensemble_name, platform=qv.Platform.ENSEMBLE)
        # If fft isn't specified, calculate default value
        fftlength = fftlength or kernel_length + fduration
        whitened_low, whitened_high, whitened_fft = add_streaming_input_preprocessor(
            ensemble,
            aframe.inputs["whitened"],
            psd_length=psd_length,
            sample_rate=sample_rate,
            kernel_length=kernel_length,
            inference_sampling_rate=inference_sampling_rate,
            fduration=fduration,
            fftlength=fftlength,
            q=q,
            highpass=highpass,
            lowpass=lowpass,
            preproc_instances=preproc_instances,
            streams_per_gpu=streams_per_gpu,
        )
        ensemble.pipe(whitened_low, aframe.inputs["whitened_low"])
        ensemble.pipe(whitened_high, aframe.inputs["whitened_high"])
        ensemble.pipe(whitened_fft, aframe.inputs["whitened_fft"])

        ensemble.add_output(aframe.outputs["discriminator"])
        ensemble.export_version(None)
    else:
        # if there does already exist an ensemble by
        # the given name, make sure it has aframe
        # and the snapshotter as a part of its models
        if aframe not in ensemble.models:
            raise ValueError(
                "Ensemble model '{}' already in repository "
                "but doesn't include model 'aframe'".format(ensemble_name)
            )
    # TODO: checks for snapshotter and preprocessor
    # keep snapshot states around for a long time in case there are
    # unexpected bottlenecks which throttle update for a few seconds
    snapshotter = repo.models["snapshotter"]
    snapshotter.config.sequence_batching.max_sequence_idle_microseconds = int(
        6e10
    )
    snapshotter.config.write()
