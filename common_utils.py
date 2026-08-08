"""
common_utils.py
---------------
توابع مشترک قابل‌ایمپورت برای تمام نوت‌بوک‌های مدل پایه (ResNet, ViT, Swin, ConvNeXt, MobileNet, ProtoNet).

تغییرات نسبت به کدهای قبلی:
- ذخیره heatmap به صورت .npy (به‌جای .txt)
- توابع مستقل از متغیرهای سراسری نوت‌بوک
- پشتیبانی از معماری‌های مختلف برای Grad-CAM
"""

from __future__ import annotations

import copy
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
import matplotlib.pyplot as plt


import cv2
import numpy as np
import psutil
import torch
import torch.nn as nn
import yaml
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ============================================================
# 1. Seed و Config
# ============================================================
def set_seed(seed: int = 42) -> None:
    """Seed را برای reproducibility تنظیم می‌کند."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str | Path = "MVconfig.yaml") -> dict[str, Any]:
    """فایل YAML تنظیمات را می‌خواند."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("YAML config must be a dictionary.")
    return config


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. Transforms
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_test_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_cam_transform(img_size: int = 224) -> transforms.Compose:
    """Transform بدون Normalize برای Grad-CAM (نرمال‌سازی داخل خود pytorch_grad_cam مدیریت می‌شود)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


# ============================================================
# 3. Dataset و DataLoader
# ============================================================
def prepare_datasets(
    data_dir: str | Path,
    img_size: int = 224,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, Any]:
    """
    ImageFolder را لود کرده، به train/val/test تقسیم می‌کند و DataLoader می‌سازد.

    Returns
    -------
    dict با کلیدهای:
        full_dataset, train_dataset, val_dataset, test_dataset,
        train_loader, val_loader, test_loader,
        class_names, infected_class_idx (اگر نام کلاس داده شود باید جداگانه ست شود)
    """
    data_dir = Path(data_dir)
    full_dataset = datasets.ImageFolder(root=str(data_dir))
    class_names = full_dataset.classes

    total_size = len(full_dataset)
    test_size = int(test_ratio * total_size)
    val_size = int(val_ratio * total_size)
    train_size = total_size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # deepcopy تا transform مستقل داشته باشند
    train_dataset.dataset = copy.deepcopy(full_dataset)
    val_dataset.dataset = copy.deepcopy(full_dataset)
    test_dataset.dataset = copy.deepcopy(full_dataset)

    train_dataset.dataset.transform = get_train_transform(img_size)
    val_dataset.dataset.transform = get_test_transform(img_size)
    test_dataset.dataset.transform = get_test_transform(img_size)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return {
        "full_dataset": full_dataset,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_names": class_names,
    }


# ============================================================
# 4. مانیتورینگ منابع
# ============================================================
def get_ram_usage_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


# ============================================================
# 5. آموزش و ارزیابی (مدل‌های استاندارد)
# ============================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    num_batches = len(loader)

    print(f"[train_one_epoch] Started. Num batches: {num_batches}", flush=True)

    for batch_idx, (inputs, labels) in enumerate(loader, start=1):
        print(
            f"[train_one_epoch] Batch {batch_idx}/{num_batches} loaded. "
            f"inputs={tuple(inputs.shape)}, labels={tuple(labels.shape)}",
            flush=True
        )

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if torch.isnan(loss):
            raise RuntimeError(f"NaN loss detected at batch {batch_idx}")

        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += labels.size(0)

        batch_acc = torch.sum(preds == labels).item() / labels.size(0)

        print(
            f"[train_one_epoch] Batch {batch_idx}/{num_batches} done. "
            f"loss={loss.item():.4f}, batch_acc={batch_acc:.4f}",
            flush=True
        )

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = running_corrects / max(total, 1)

    print(
        f"[train_one_epoch] Finished. "
        f"epoch_loss={epoch_loss:.4f}, epoch_acc={epoch_acc:.4f}",
        flush=True
    )

    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list, list]:
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    all_labels: list = []
    all_preds: list = []

    num_batches = len(loader)

    print(f"[evaluate] Started. Num batches: {num_batches}", flush=True)

    for batch_idx, (inputs, labels) in enumerate(loader, start=1):
        print(
            f"[evaluate] Batch {batch_idx}/{num_batches} loaded.",
            flush=True
        )

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

        print(
            f"[evaluate] Batch {batch_idx}/{num_batches} done. "
            f"loss={loss.item():.4f}",
            flush=True
        )

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = running_corrects / max(total, 1)

    print(
        f"[evaluate] Finished. "
        f"loss={epoch_loss:.4f}, acc={epoch_acc:.4f}",
        flush=True
    )

    return epoch_loss, epoch_acc, all_labels, all_preds

# ============================================================
# 6. لاگ آموزش
# ============================================================
def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, MODEL_NAME):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - {MODEL_NAME}")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve - {MODEL_NAME}")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def collect_hyperparameters(
    seed: int,
    img_size: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    val_ratio: float,
    test_ratio: float,
    **extra: Any,
) -> dict[str, Any]:
    params = {
        "seed": seed,
        "img_size": img_size,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }
    params.update(extra)
    return params


def log_training_run(
    hyperparameters: dict[str, Any],
    train_accs: list[float],
    val_accs: list[float],
    train_losses: Optional[list[float]] = None,
    val_losses: Optional[list[float]] = None,
    log_path: str | Path = "training_log.jsonl",
    epoch_times: Optional[list[float]] = None,
    epoch_ram_mb: Optional[list[float]] = None,
    epoch_gpu_peak_mb: Optional[list[float]] = None,
) -> None:
    """نتایج هر ران را در فرمت JSONL ذخیره می‌کند."""
    if len(train_accs) != len(val_accs):
        raise ValueError("train_accs and val_accs must have the same length")

    epochs = []
    for i in range(len(train_accs)):
        epoch_data: dict[str, Any] = {
            "epoch": i + 1,
            "train_acc": float(train_accs[i]),
            "val_acc": float(val_accs[i]),
        }
        if train_losses is not None:
            epoch_data["train_loss"] = float(train_losses[i])
        if val_losses is not None:
            epoch_data["val_loss"] = float(val_losses[i])
        if epoch_times is not None:
            epoch_data["time_sec"] = float(epoch_times[i])
        if epoch_ram_mb is not None:
            epoch_data["ram_mb"] = float(epoch_ram_mb[i])
        if epoch_gpu_peak_mb is not None:
            epoch_data["gpu_peak_mb"] = float(epoch_gpu_peak_mb[i])
        epochs.append(epoch_data)

    record = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparameters": hyperparameters,
        "epochs": epochs,
    }

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Training log appended to: {log_path.resolve()}")


# ============================================================
# 7. Grad-CAM – کمکی‌های مشترک
# ============================================================
def load_image_for_cam(
    image_path: str | Path,
    img_size: int = 224,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, torch.Tensor]:
    """
    تصویر را برای Grad-CAM آماده می‌کند.

    Returns
    -------
    rgb_img_float : ndarray float32 در بازه [0, 1] با شکل (H, W, 3)
    input_tensor  : Tensor با شکل (1, 3, H, W)
    """
    pil_img = Image.open(image_path).convert("RGB")
    pil_img = pil_img.resize((img_size, img_size))
    rgb_img_float = np.array(pil_img).astype(np.float32) / 255.0
    input_tensor = get_cam_transform(img_size)(pil_img).unsqueeze(0).to(device)
    return rgb_img_float, input_tensor


def clear_output_images(output_dir: str | Path) -> None:
    """فایل‌های تصویر و heatmap قبلی را از پوشه خروجی پاک می‌کند."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    heatmap_exts = {".npy", ".txt"}
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix.lower() in (image_exts | heatmap_exts):
            path.unlink()


# ---------- reshape برای Transformerها ----------
def vit_reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
    """تبدیل خروجی attention ViT به نقشه فضایی برای Grad-CAM."""
    # tensor: (B, N, C)  →  حذف token کلاس و reshape
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)  # (B, C, H, W)
    return result


def swin_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """تبدیل خروجی Swin به نقشه فضایی."""
    # torchvision Swin: (B, H, W, C) یا (B, N, C)
    if tensor.ndim == 4:
        result = tensor.permute(0, 3, 1, 2)
    else:
        # fallback
        side = int(tensor.size(1) ** 0.5)
        result = tensor.reshape(tensor.size(0), side, side, tensor.size(2))
        result = result.permute(0, 3, 1, 2)
    return result


# ---------- تولید Grad-CAM برای معماری‌های مختلف ----------
def _run_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layers: list,
    target_class_idx: int,
    reshape_transform: Optional[Callable] = None,
) -> np.ndarray:
    targets = [ClassifierOutputTarget(target_class_idx)]
    cam_kwargs: dict[str, Any] = {"model": model, "target_layers": target_layers}
    if reshape_transform is not None:
        cam_kwargs["reshape_transform"] = reshape_transform

    with GradCAM(**cam_kwargs) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    return grayscale_cam


def generate_gradcam(
    model: nn.Module,
    image_path: str | Path,
    target_class_idx: int,
    device: torch.device,
    architecture: str = "resnet",
    img_size: int = 224,
) -> dict[str, Any]:
    """
    Grad-CAM را برای معماری‌های پشتیبانی‌شده تولید می‌کند.

    architecture: یکی از
        resnet | resnet18 | resnet50 |
        vit | vit_b16 |
        swin | swin_tiny |
        convnext | convnext_tiny |
        mobilenet | mobilenetv3
    """
    model.eval()
    rgb_img_float, input_tensor = load_image_for_cam(image_path, img_size, device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_conf = float(probs[pred_idx])

    arch = architecture.lower().replace("-", "_")
    reshape_transform = None

    if arch in ("resnet", "resnet18", "resnet50"):
        target_layers = [model.layer4[-1]]
    elif arch in ("vit", "vit_b16", "vitb16"):
        target_layers = [model.encoder.layers[-1].ln_1]
        reshape_transform = vit_reshape_transform
    elif arch in ("swin", "swin_tiny", "swintiny"):
        # لایه نرمال‌سازی آخرین بلاک
        target_layers = [model.features[-1][-1].norm1]
        reshape_transform = swin_reshape_transform
    elif arch in ("convnext", "convnext_tiny", "convnexttiny"):
        target_layers = [model.features[-1][-1]]
    elif arch in ("mobilenet", "mobilenetv3", "mobilenet_v3_large"):
        target_layers = [model.features[-1]]
    else:
        raise ValueError(
            f"Unsupported architecture for Grad-CAM: {architecture}. "
            "Supported: resnet, vit, swin, convnext, mobilenet"
        )

    grayscale_cam = _run_gradcam(
        model=model,
        input_tensor=input_tensor,
        target_layers=target_layers,
        target_class_idx=target_class_idx,
        reshape_transform=reshape_transform,
    )
    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

    return {
        "pred_idx": pred_idx,
        "pred_conf": pred_conf,
        "probs": probs,
        "grayscale_cam": grayscale_cam,
        "visualization": visualization,
        "original_image": rgb_img_float,
    }


def save_gradcams_for_predicted_infected(
    model: nn.Module,
    test_dataset,
    full_dataset,
    class_names: list[str],
    infected_class_idx: int,
    output_dir: str | Path,
    device: torch.device,
    architecture: str = "resnet",
    img_size: int = 224,
    clear_previous: bool = True,
) -> int:
    """
    Grad-CAM را فقط برای نمونه‌هایی که مدل آن‌ها را Infected پیش‌بینی کرده ذخیره می‌کند.

    خروجی برای هر نمونه:
        {name}.png          → تصویر ترکیبی (اصلی + heatmap)
        {name}_heatmap.npy  → آرایه float32 نقشه اهمیت (مقادیر تقریباً در [0, 1])

    Returns
    -------
    تعداد نمونه‌های ذخیره‌شده
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if clear_previous:
        clear_output_images(output_dir)

    model.eval()
    saved_count = 0
    test_indices = test_dataset.indices

    print(f"Saving predicted-infected Grad-CAMs (.png + .npy) to: {output_dir.resolve()}")

    for i, original_idx in enumerate(test_indices):
        image_path, true_label = full_dataset.samples[original_idx]

        try:
            result = generate_gradcam(
                model=model,
                image_path=image_path,
                target_class_idx=infected_class_idx,
                device=device,
                architecture=architecture,
                img_size=img_size,
            )
        except Exception as exc:
            print(f"[SKIP] Grad-CAM failed for {image_path}: {exc}")
            continue

        pred_idx = result["pred_idx"]
        pred_conf = result["pred_conf"]

        # فقط نمونه‌هایی که مدل آن‌ها را بیمار تشخیص داده
        if pred_idx != infected_class_idx:
            continue

        true_name = class_names[true_label]
        pred_name = class_names[pred_idx]
        base_name = Path(image_path).stem
        case_type = "TP" if true_label == infected_class_idx else "FP"

        common_name = (
            f"{base_name}"
            f"_case-{case_type}"
        )

        save_path_img = output_dir / f"{common_name}.png"
        save_path_npy = output_dir / f"{common_name}_heatmap.npy"

        # ذخیره heatmap به صورت .npy
        grayscale_cam = result["grayscale_cam"].astype(np.float32)
        np.save(str(save_path_npy), grayscale_cam)

        # تصویر ترکیبی
        original_rgb = (result["original_image"] * 255).clip(0, 255).astype(np.uint8)
        heatmap_rgb = result["visualization"]

        if original_rgb.shape[:2] != heatmap_rgb.shape[:2]:
            heatmap_rgb = cv2.resize(
                heatmap_rgb, (original_rgb.shape[1], original_rgb.shape[0])
            )

        combined_rgb = np.concatenate([original_rgb, heatmap_rgb], axis=1)

        info_bar_height = 140
        _, w, _ = combined_rgb.shape
        info_bar = np.ones((info_bar_height, w, 3), dtype=np.uint8) * 255

        text1 = f"True: {true_name}    Pred: {pred_name}"
        text2 = f"Case: {case_type}"
        text3 = f"File: {base_name}"
        text4 = f"Conf: {pred_conf:.3f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(info_bar, text1, (15, 30), font, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(info_bar, text2, (30, 60), font, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(info_bar, text3, (45, 90), font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(info_bar, text4, (60, 120), font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        final_rgb = np.concatenate([combined_rgb, info_bar], axis=0)
        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(str(save_path_img), final_bgr)

        if success:
            saved_count += 1
            print(f"[{saved_count}] Saved: {save_path_img.name} + {save_path_npy.name}")
        else:
            print(f"Failed to save image for: {common_name}")

    print(f"\nTotal predicted-infected samples saved: {saved_count}")
    print(f"Output folder: {output_dir.resolve()}")
    return saved_count


# ============================================================
# 8. بارگذاری امن وزن‌ها
# ============================================================
def load_model_weights(
    model: nn.Module,
    model_save_path: str | Path,
    device: torch.device,
) -> nn.Module:
    """بارگذاری وزن‌ها با سازگاری نسخه‌های مختلف PyTorch."""
    model_save_path = Path(model_save_path)
    if not model_save_path.exists():
        raise FileNotFoundError(f"Weights not found: {model_save_path}")

    try:
        state_dict = torch.load(
            model_save_path, map_location=device, weights_only=True
        )
    except TypeError:
        state_dict = torch.load(model_save_path, map_location=device)

    model.load_state_dict(state_dict)
    return model
