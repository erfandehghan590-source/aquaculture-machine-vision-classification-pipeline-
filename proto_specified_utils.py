import os

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# =====================================================================
# توابع اصلی ProtoNet
# =====================================================================
def compute_distances(embeddings, prototypes):
    """
    محاسبه فاصله مجذور اقلیدسی بین embeddingها و prototypeها.

    Parameters
    ----------
    embeddings : torch.Tensor
        Tensor با شکل [B, D]
        B: batch size
        D: embedding dimension

    prototypes : torch.Tensor
        Tensor با شکل [C, D]
        C: number of classes
        D: embedding dimension

    Returns
    -------
    distances : torch.Tensor
        Tensor با شکل [B, C]
        فاصله هر نمونه تا prototype هر کلاس
    """
    n = embeddings.size(0)
    c = prototypes.size(0)

    embeddings_expanded = embeddings.unsqueeze(1).expand(n, c, -1)
    prototypes_expanded = prototypes.unsqueeze(0).expand(n, c, -1)

    distances = torch.sum(
        (embeddings_expanded - prototypes_expanded) ** 2,
        dim=2
    )

    return distances


def compute_dataset_prototypes(
    model,
    loader,
    device,
    num_classes: int = 2
):
    """
    محاسبه prototypeهای ثابت برای کل train set.

    این تابع معمولاً بعد از هر epoch یا بعد از load کردن مدل استفاده می‌شود.

    Parameters
    ----------
    model : torch.nn.Module
        مدل ProtoNet که خروجی آن embedding است.

    loader : torch.utils.data.DataLoader
        DataLoader مربوط به train set.
        بهتر است shuffle=False باشد.

    device : torch.device
        cuda یا cpu.

    num_classes : int
        تعداد کلاس‌ها.

    Returns
    -------
    prototypes : torch.Tensor
        Tensor با شکل [num_classes, embedding_dim]
    """
    model.eval()

    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)

            embeddings = model(inputs)

            embeddings_list.append(embeddings.detach().cpu())
            labels_list.append(labels.detach().cpu())

    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    prototypes = torch.zeros(
        num_classes,
        embeddings.size(1)
    )

    for class_idx in range(num_classes):
        class_mask = labels == class_idx
        class_indices = class_mask.nonzero(as_tuple=True)[0]

        if len(class_indices) > 0:
            prototypes[class_idx] = embeddings[class_indices].mean(dim=0)
        else:
            # اگر به هر دلیل یک کلاس در train set وجود نداشت،
            # از میانگین کل embeddingها استفاده می‌کنیم تا prototype صفر یا NaN نشود.
            prototypes[class_idx] = embeddings.mean(dim=0)

    return prototypes.to(device)


def train_one_epoch_proto(
    model,
    loader,
    optimizer,
    device,
    num_classes: int = 2,
):
    """
    آموزش یک epoch برای ProtoNet با prototypeهای موقت در هر batch.

    Returns
    -------
    epoch_loss : float
    epoch_acc : float
    """
    model.train()

    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_corrects = 0
    total = 0

    num_batches = len(loader)

    print(
        f"[train_one_epoch_proto] Started. Num batches: {num_batches}",
        flush=True,
    )

    for batch_idx, (inputs, labels) in enumerate(loader, start=1):
        print(
            f"[train_one_epoch_proto] Batch {batch_idx}/{num_batches} loaded. "
            f"inputs={tuple(inputs.shape)}, labels={tuple(labels.shape)}",
            flush=True,
        )

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        embeddings = model(inputs)

        batch_prototypes = torch.zeros(
            num_classes,
            embeddings.size(1),
            device=device,
            dtype=embeddings.dtype,
        )

        for class_idx in range(num_classes):
            class_mask = labels == class_idx

            if class_mask.any():
                batch_prototypes[class_idx] = embeddings[class_mask].mean(dim=0)
            else:
                # اگر این batch نمونه‌ای از این کلاس نداشت، از میانگین embeddingهای batch استفاده می‌شود.
                batch_prototypes[class_idx] = embeddings.mean(dim=0)

                print(
                    f"[train_one_epoch_proto] Warning: batch {batch_idx} "
                    f"has no samples for class {class_idx}; "
                    f"using the batch mean as fallback prototype.",
                    flush=True,
                )

        distances = compute_distances(
            embeddings=embeddings,
            prototypes=batch_prototypes,
        )

        logits = -distances
        loss = criterion(logits, labels)

        if torch.isnan(loss):
            raise RuntimeError(
                f"[train_one_epoch_proto] NaN loss detected at batch {batch_idx}."
            )

        loss.backward()
        optimizer.step()

        _, preds = torch.max(logits, dim=1)

        batch_size = labels.size(0)
        batch_corrects = torch.sum(preds == labels).item()
        batch_acc = batch_corrects / batch_size

        running_loss += loss.item() * batch_size
        running_corrects += batch_corrects
        total += batch_size

        print(
            f"[train_one_epoch_proto] Batch {batch_idx}/{num_batches} done. "
            f"loss={loss.item():.4f}, batch_acc={batch_acc:.4f}",
            flush=True,
        )

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = running_corrects / max(total, 1)

    print(
        f"[train_one_epoch_proto] Finished. "
        f"epoch_loss={epoch_loss:.4f}, epoch_acc={epoch_acc:.4f}",
        flush=True,
    )

    return epoch_loss, epoch_acc

def evaluate_proto(
    model,
    loader,
    prototypes,
    device,
):
    """
    ارزیابی ProtoNet با prototypeهای ثابت.

    Returns
    -------
    epoch_loss : float
    epoch_acc : float
    all_labels : list
    all_preds : list
    """
    model.eval()

    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_corrects = 0
    total = 0

    all_labels = []
    all_preds = []

    prototypes = prototypes.to(device)
    num_batches = len(loader)

    print(
        f"[evaluate_proto] Started. Num batches: {num_batches}",
        flush=True,
    )

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader, start=1):
            print(
                f"[evaluate_proto] Batch {batch_idx}/{num_batches} loaded. "
                f"inputs={tuple(inputs.shape)}, labels={tuple(labels.shape)}",
                flush=True,
            )

            inputs = inputs.to(device)
            labels = labels.to(device)

            embeddings = model(inputs)

            distances = compute_distances(
                embeddings=embeddings,
                prototypes=prototypes,
            )

            logits = -distances
            loss = criterion(logits, labels)

            if torch.isnan(loss):
                raise RuntimeError(
                    f"[evaluate_proto] NaN loss detected at batch {batch_idx}."
                )

            _, preds = torch.max(logits, dim=1)

            batch_size = labels.size(0)
            batch_corrects = torch.sum(preds == labels).item()
            batch_acc = batch_corrects / batch_size

            running_loss += loss.item() * batch_size
            running_corrects += batch_corrects
            total += batch_size

            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())

            print(
                f"[evaluate_proto] Batch {batch_idx}/{num_batches} done. "
                f"loss={loss.item():.4f}, batch_acc={batch_acc:.4f}",
                flush=True,
            )

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = running_corrects / max(total, 1)

    print(
        f"[evaluate_proto] Finished. "
        f"loss={epoch_loss:.4f}, acc={epoch_acc:.4f}",
        flush=True,
    )

    return epoch_loss, epoch_acc, all_labels, all_preds


def predict_proto(
    model,
    inputs,
    prototypes,
    device
):
    """
    گرفتن prediction از ProtoNet برای یک batch ورودی.

    Parameters
    ----------
    model : torch.nn.Module
        مدل ProtoNet.

    inputs : torch.Tensor
        Tensor تصویرها با شکل [B, C, H, W]

    prototypes : torch.Tensor
        Prototypeها با شکل [num_classes, embedding_dim]

    device : torch.device
        cuda یا cpu.

    Returns
    -------
    logits : torch.Tensor
    probs : torch.Tensor
    preds : torch.Tensor
    """
    model.eval()

    inputs = inputs.to(device)
    prototypes = prototypes.to(device)

    with torch.no_grad():
        embeddings = model(inputs)

        distances = compute_distances(
            embeddings=embeddings,
            prototypes=prototypes
        )

        logits = -distances
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    return logits, probs, preds


# =====================================================================
# Grad-CAM مخصوص ProtoNet
# =====================================================================
class ProtoNetCAMWrapper(nn.Module):
    """
    Wrapper برای Grad-CAM در ProtoNet.

    چون خروجی اصلی ProtoNet فقط embedding است و logits مستقیم ندارد،
    این wrapper embedding را به فاصله از prototypeها تبدیل می‌کند.
    سپس negative distance به عنوان logits استفاده می‌شود.
    """
    def __init__(self, model, prototypes):
        super().__init__()

        self.model = model
        self.prototypes = prototypes

    def forward(self, x):
        embeddings = self.model(x)

        distances = compute_distances(
            embeddings=embeddings,
            prototypes=self.prototypes
        )

        logits = -distances

        return logits


class ProtoNetTarget:
    """
    Target سفارشی برای Grad-CAM.

    این کلاس مشخص می‌کند Grad-CAM نسبت به کدام کلاس محاسبه شود.
    """
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]

        return model_output[:, self.category]


def identity_reshape_transform(tensor):
    """
    reshape_transform پیش‌فرض برای مدل‌هایی که خروجی لایه هدف آن‌ها
    از قبل به شکل [B, C, H, W] است.

    برای ResNet و ConvNeXt معمولاً همین کافی است.
    """
    return tensor


def load_image_for_proto_cam(
    image_path,
    cam_transform,
    img_size=224,
    device="cpu"
):
    """
    آماده‌سازی تصویر برای Grad-CAM.

    خروجی شامل دو بخش است:
    1. تصویر خام resize شده برای overlay
    2. input_tensor نرمال‌شده برای ورود به مدل

    Parameters
    ----------
    image_path : str
        مسیر تصویر.

    cam_transform : torchvision.transforms.Compose
        transform مربوط به validation/test.
        باید شامل Resize/ToTensor/Normalize باشد.

    img_size : int
        اندازه ورودی مدل.

    device : torch.device or str
        cuda یا cpu.

    Returns
    -------
    rgb_img_float : np.ndarray
        تصویر RGB در بازه [0, 1] برای overlay.

    input_tensor : torch.Tensor
        Tensor ورودی مدل با شکل [1, C, H, W]
    """
    pil_img = Image.open(image_path).convert("RGB")
    pil_img = pil_img.resize((img_size, img_size))

    rgb_img_float = np.array(pil_img).astype(np.float32) / 255.0

    input_tensor = cam_transform(pil_img).unsqueeze(0).to(device)

    return rgb_img_float, input_tensor


def generate_gradcam_proto(
    model,
    prototypes,
    image_path,
    target_class_idx,
    target_layers,
    cam_transform,
    device,
    img_size=224,
    reshape_transform=None
):
    """
    تولید Grad-CAM برای مدل ProtoNet.

    این تابع برای backboneهای مختلف قابل استفاده است؛
    فقط باید target_layers و در صورت نیاز reshape_transform مناسب را پاس بدهی.

    مثال target_layers:
    - ResNet50:
        [model.backbone.layer4[-1]]

    - ConvNeXt-Tiny:
        [model.backbone.features[-1][-1]]

    - Swin-Tiny:
        [model.backbone.features[-1][-1]]
        همراه با reshape_transform مخصوص Swin

    Parameters
    ----------
    model : torch.nn.Module
        مدل ProtoNet.

    prototypes : torch.Tensor
        Prototypeها با شکل [num_classes, embedding_dim]

    image_path : str
        مسیر تصویر.

    target_class_idx : int
        کلاس هدف برای Grad-CAM.

    target_layers : list
        لیست لایه‌های هدف Grad-CAM.

    cam_transform : torchvision.transforms.Compose
        transform ورودی مدل برای Grad-CAM.

    device : torch.device
        cuda یا cpu.

    img_size : int
        اندازه ورودی مدل.

    reshape_transform : callable or None
        تابع reshape برای مدل‌هایی مثل Swin.
        برای ResNet و ConvNeXt می‌تواند None باشد.

    Returns
    -------
    result : dict
        شامل prediction، confidence، probabilities، heatmap و visualization.
    """
    model.eval()

    prototypes = prototypes.to(device)

    rgb_img_float, input_tensor = load_image_for_proto_cam(
        image_path=image_path,
        cam_transform=cam_transform,
        img_size=img_size,
        device=device
    )

    wrapped_model = ProtoNetCAMWrapper(
        model=model,
        prototypes=prototypes
    ).to(device)

    wrapped_model.eval()

    with torch.no_grad():
        logits = wrapped_model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_conf = float(probs[pred_idx])

    targets = [
        ProtoNetTarget(target_class_idx)
    ]

    cam_kwargs = {
        "model": wrapped_model,
        "target_layers": target_layers
    }

    if reshape_transform is not None:
        cam_kwargs["reshape_transform"] = reshape_transform

    with GradCAM(**cam_kwargs) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

    visualization = show_cam_on_image(
        rgb_img_float,
        grayscale_cam,
        use_rgb=True
    )

    return {
        "pred_idx": pred_idx,
        "pred_conf": pred_conf,
        "probs": probs,
        "grayscale_cam": grayscale_cam,
        "visualization": visualization,
        "original_image": rgb_img_float
    }


def save_gradcams_for_predicted_infected_proto(
    model,
    prototypes,
    test_dataset,
    full_dataset,
    class_names,
    infected_class_idx,
    output_dir,
    target_layers,
    cam_transform,
    device,
    img_size=224,
    reshape_transform=None,
    clear_previous=True,
    output_prefix="protonet"
):
    """
    ذخیره Grad-CAM برای نمونه‌هایی که مدل آن‌ها را infected پیش‌بینی کرده است.

    این تابع فقط مواردی را ذخیره می‌کند که:

        pred_idx == infected_class_idx

    بنابراین خروجی‌ها شامل TP و FP خواهند بود.

    Parameters
    ----------
    model : torch.nn.Module
        مدل ProtoNet.

    prototypes : torch.Tensor
        Prototypeهای مدل.

    test_dataset : torch.utils.data.Subset
        دیتاست validation/test.
        باید Subset باشد و attribute به نام indices داشته باشد.

    full_dataset : torchvision.datasets.ImageFolder
        دیتاست اصلی ImageFolder که samples دارد.

    class_names : list
        نام کلاس‌ها.

    infected_class_idx : int
        index کلاس infected.

    output_dir : str
        مسیر ذخیره خروجی‌ها.

    target_layers : list
        لایه‌های هدف Grad-CAM.

    cam_transform : torchvision.transforms.Compose
        transform ورودی Grad-CAM.

    device : torch.device
        cuda یا cpu.

    img_size : int
        اندازه ورودی مدل.

    reshape_transform : callable or None
        برای مدل‌هایی مثل Swin لازم است.

    clear_previous : bool
        اگر True باشد، فایل‌های قبلی داخل output_dir پاک می‌شوند.

    output_prefix : str
        پیشوند نام فایل‌ها برای تمایز مدل‌ها.
        مثال:
            protonet_resnet50
            protonet_convnext_tiny
            protonet_swin_tiny

    Returns
    -------
    saved_count : int
        تعداد فایل‌های ذخیره‌شده.
    """
    os.makedirs(output_dir, exist_ok=True)

    if clear_previous:
        removable_exts = (
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp",
            ".txt", ".npy"
        )

        for file_name in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file_name)

            if os.path.isfile(file_path) and file_name.lower().endswith(removable_exts):
                os.remove(file_path)

    model.eval()
    prototypes = prototypes.to(device)

    saved_count = 0

    if not hasattr(test_dataset, "indices"):
        raise AttributeError(
            "test_dataset must be a Subset and must have an .indices attribute."
        )

    test_indices = test_dataset.indices

    print(
        "Saving predicted-infected ProtoNet Grad-CAMs to:",
        os.path.abspath(output_dir)
    )

    for i, original_idx in enumerate(test_indices):
        image_path, true_label = full_dataset.samples[original_idx]

        result = generate_gradcam_proto(
            model=model,
            prototypes=prototypes,
            image_path=image_path,
            target_class_idx=infected_class_idx,
            target_layers=target_layers,
            cam_transform=cam_transform,
            device=device,
            img_size=img_size,
            reshape_transform=reshape_transform
        )

        pred_idx = result["pred_idx"]
        pred_conf = result["pred_conf"]

        # فقط نمونه‌هایی که مدل آن‌ها را infected پیش‌بینی کرده ذخیره می‌شوند.
        if pred_idx != infected_class_idx:
            continue

        true_name = class_names[true_label]
        pred_name = class_names[pred_idx]
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        case_type = "TP" if true_label == infected_class_idx else "FP"

        common_name = (
            f"{output_prefix}"
            "_"
            f"{base_name}"
            f"_case-{case_type}"
        )

        save_path_img = os.path.join(
            output_dir,
            f"{common_name}.png"
        )

        save_path_heatmap = os.path.join(
            output_dir,
            f"{common_name}_heatmap.npy"
        )

        grayscale_cam = result["grayscale_cam"]

        # ذخیره heatmap با فرمت npy برای حفظ دقت عددی
        np.save(save_path_heatmap, grayscale_cam)

        original_rgb = (
            result["original_image"] * 255
        ).clip(0, 255).astype(np.uint8)

        heatmap_rgb = result["visualization"]

        if original_rgb.shape[:2] != heatmap_rgb.shape[:2]:
            heatmap_rgb = cv2.resize(
                heatmap_rgb,
                (original_rgb.shape[1], original_rgb.shape[0])
            )

        combined_rgb = np.concatenate(
            [original_rgb, heatmap_rgb],
            axis=1
        )

        info_bar_height = 140
        _, w, _ = combined_rgb.shape

        info_bar = np.ones(
            (info_bar_height, w, 3),
            dtype=np.uint8
        ) * 255

        text1 = f"True: {true_name}    Pred: {pred_name}"
        text2 = f"Case: {case_type}"
        text3 = f"File: {base_name}"
        text4 = f"Conf: {pred_conf:.3f}"

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(
            info_bar,
            text1,
            (15, 30),
            font,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            info_bar,
            text2,
            (15, 60),
            font,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            info_bar,
            text3,
            (15, 90),
            font,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            info_bar,
            text4,
            (15, 120),
            font,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

        final_rgb = np.concatenate(
            [combined_rgb, info_bar],
            axis=0
        )

        final_bgr = cv2.cvtColor(
            final_rgb,
            cv2.COLOR_RGB2BGR
        )

        success = cv2.imwrite(
            save_path_img,
            final_bgr
        )

        if success:
            saved_count += 1
            print(f"[{saved_count}] Saved: {save_path_img}")
            print(f"    Heatmap: {save_path_heatmap}")
        else:
            print(f"Failed to save image for: {common_name}")

    print(f"\nTotal predicted-infected samples saved: {saved_count}")
    print(f"Output folder: {os.path.abspath(output_dir)}")

    return saved_count


# =====================================================================
# reshape_transformهای آماده برای بعضی backboneها
# =====================================================================
def swin_reshape_transform(tensor):
    """
    reshape_transform برای Swin Transformer.

    بعضی خروجی‌های Swin در torchvision به شکل زیر هستند:

        [B, H, W, C]

    اما Grad-CAM انتظار دارد tensor به شکل زیر باشد:

        [B, C, H, W]

    این تابع همان تبدیل را انجام می‌دهد.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor با شکل احتمالی [B, H, W, C]

    Returns
    -------
    tensor : torch.Tensor
        Tensor با شکل [B, C, H, W]
    """
    if len(tensor.shape) == 4:
        # [B, H, W, C] -> [B, C, H, W]
        return tensor.permute(0, 3, 1, 2)

    return tensor


def convnext_reshape_transform(tensor):
    """
    reshape_transform برای ConvNeXt.

    خروجی لایه‌های features در ConvNeXt معمولاً از قبل به شکل [B, C, H, W] است.
    """
    return tensor


def resnet_reshape_transform(tensor):
    """
    reshape_transform برای ResNet.

    خروجی layer4 در ResNet معمولاً از قبل به شکل [B, C, H, W] است.
    """
    return tensor
