import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        primary = self.streams[0]
        return bool(getattr(primary, "isatty", lambda: False)())

    def fileno(self):
        primary = self.streams[0]
        if hasattr(primary, "fileno"):
            return primary.fileno()
        raise OSError("Underlying stream has no file descriptor")

    @property
    def encoding(self):
        primary = self.streams[0]
        return getattr(primary, "encoding", "utf-8")

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


def setup_file_logging(log_file_path):
    log_path = Path(log_file_path)
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, log_handle)
    sys.stderr = TeeStream(sys.stderr, log_handle)
    print("Logging to:", log_path)
    return log_path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device=None, gpu_only=False):
    if requested_device:
        device = torch.device(requested_device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda" and gpu_only:
        raise RuntimeError("GPU-only mode is enabled but CUDA device was not requested/found.")

    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(0)
        arch = "sm_{}{}".format(cap[0], cap[1])
        supported_arches = set(torch.cuda.get_arch_list()) if hasattr(torch.cuda, "get_arch_list") else set()
        if supported_arches and arch not in supported_arches:
            print(
                "Warning: GPU arch {} not listed in this torch build ({}). "
                "Trying runtime CUDA probe...".format(arch, ", ".join(sorted(supported_arches)))
            )

        try:
            _ = torch.tensor([0.0], device=device)
        except Exception as err:
            if gpu_only:
                raise RuntimeError("CUDA is unavailable in this runtime: {}".format(err))
            print("CUDA is unavailable in this runtime, falling back to CPU: {}".format(err))
            device = torch.device("cpu")

    print("Device:", device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM: {:.1f} GB".format(props.total_memory / 1e9))
    return device


def make_grad_scaler(use_amp):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def amp_autocast(device, use_amp):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=use_amp)
    return torch.cuda.amp.autocast(enabled=use_amp)


def maybe_compile_model(model, disable_compile=False):
    if disable_compile:
        print("torch.compile disabled by flag")
        return model

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as err:
            print("torch.compile unavailable:", err)
    return model


def maybe_quantize_for_cpu(model, enable_quantization=False, device=None):
    if not enable_quantization:
        return model
    if device is not None and device.type != "cpu":
        print("Skipping quantization: enabled only for CPU runtime.")
        return model

    qmodel = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    print("Dynamic int8 quantization applied for CPU inference")
    return qmodel
