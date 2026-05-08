import importlib
import os
import sys
import torch

def read_dicom(img_path):
    import SimpleITK as sitk
    dicom_names = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(img_path)
    itk_img = sitk.ReadImage(dicom_names)
    return itk_img


def load_config(config_file):

    dirname = os.path.dirname(config_file)
    basename = os.path.basename(config_file)
    modulename, _ = os.path.splitext(basename)

    sys.path.append(dirname)
    lib = importlib.import_module(modulename)
    del sys.path[-1]

    return lib.config


def _normalize_device_ids(device_ids, device=None):
    if device_ids is None:
        if isinstance(device, torch.device) and device.type == "cuda" and device.index is not None:
            return [int(device.index)]
        if isinstance(device, str) and device.startswith("cuda"):
            if ":" in device:
                try:
                    return [int(device.split(":", 1)[1])]
                except ValueError:
                    return [0]
            return [0]
        return []

    if isinstance(device_ids, str):
        text = device_ids.strip()
        if not text:
            return []
        return [int(part.strip()) for part in text.split(",") if part.strip() != ""]

    if isinstance(device_ids, int):
        return [int(device_ids)]

    return [int(d) for d in device_ids]


def resolve_device(device="cuda:0", device_ids=None):
    device_ids = _normalize_device_ids(device_ids, device=device)

    if torch.cuda.is_available() and device_ids:
        available = torch.cuda.device_count()
        invalid = [idx for idx in device_ids if idx < 0 or idx >= available]
        if invalid:
            raise ValueError(
                f"Invalid CUDA device ids {invalid}; available ids are 0..{available - 1} "
                f"for the currently visible GPUs."
            )
        primary = torch.device(f"cuda:{device_ids[0]}")
        return primary, device_ids

    return torch.device("cpu"), []


def wrap_model_for_devices(model, device_ids):
    if torch.cuda.is_available() and len(device_ids) > 1:
        return torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    return model


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def warmup_cuda_linalg(device_ids):
    if not torch.cuda.is_available() or not device_ids:
        return

    for idx in device_ids:
        with torch.cuda.device(idx):
            device = torch.device(f"cuda:{idx}")
            a = torch.eye(4, device=device).unsqueeze(0)
            b = torch.eye(4, device=device).unsqueeze(0)
            torch.linalg.solve(a, b)
            torch.cuda.synchronize(device)
