"""
model_comparison_utils.py
-------------------------
ابزارهای مقایسه‌ی مدل‌های طبقه‌بندی و XAI برای پروژه SalmonScan.

قابلیت‌ها:
1) خواندن لاگ‌های JSONL آموزشی مدل‌ها
2) استخراج hyperparameterها و خلاصه‌ی عملکرد هر مدل
3) مقایسه‌ی Heatmapهای Grad-CAM با Soft Ground Truth پزشکی
4) ذخیره‌ی جدول مقایسه در CSV و Excel
5) تولید نمودارهای مقایسه‌ای
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim


# ============================================================
# 1. معیارهای مقایسه Heatmap
# ============================================================
def iou(gt: np.ndarray, pred: np.ndarray, threshold: float = 0.65) -> float:
    """IoU دودویی پس از threshold کردن نقشه‌ها."""
    gt_bin = gt >= threshold
    pred_bin = pred >= threshold

    intersection = np.logical_and(gt_bin, pred_bin).sum()
    union = np.logical_or(gt_bin, pred_bin).sum()

    # اگر هر دو نقشه کاملاً خالی باشند
    if union == 0:
        return 1.0

    return float(intersection / union)


def dice(gt: np.ndarray, pred: np.ndarray, threshold: float = 0.65) -> float:
    """Dice coefficient دودویی پس از threshold کردن نقشه‌ها."""
    gt_bin = gt >= threshold
    pred_bin = pred >= threshold

    intersection = np.logical_and(gt_bin, pred_bin).sum()
    denominator = gt_bin.sum() + pred_bin.sum()

    if denominator == 0:
        return 1.0

    return float((2.0 * intersection) / denominator)


def soft_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    """IoU پیوسته، بدون threshold."""
    intersection = np.minimum(gt, pred).sum()
    union = np.maximum(gt, pred).sum()

    return float(intersection / (union + 1e-8))


def soft_dice(gt: np.ndarray, pred: np.ndarray) -> float:
    """Dice پیوسته، بدون threshold."""
    intersection = (gt * pred).sum()
    denominator = gt.sum() + pred.sum()

    return float((2.0 * intersection) / (denominator + 1e-8))


# ============================================================
# 2. آماده‌سازی Heatmap
# ============================================================
def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """
    نرمال‌سازی heatmap در بازه [0, 1].

    همچنین اگر آرایه شکل غیرمنتظره‌ای داشته باشد،
    آن را به float32 تبدیل می‌کند.
    """
    heatmap = np.asarray(heatmap, dtype=np.float32)

    # برای حالت‌هایی مثل (1, H, W) یا (H, W, 1)
    heatmap = np.squeeze(heatmap)

    if heatmap.ndim != 2:
        raise ValueError(
            f"Heatmap must be 2D after squeeze. Got shape: {heatmap.shape}"
        )

    min_value = float(heatmap.min())
    max_value = float(heatmap.max())

    heatmap = heatmap - min_value

    if max_value - min_value > 0:
        heatmap = heatmap / (max_value - min_value)

    return heatmap.astype(np.float32)


def resize_heatmap_if_needed(
    heatmap: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """
    اگر اندازه‌ی heatmap با Ground Truth متفاوت باشد،
    با interpolation آن را resize می‌کند.

    نیازمند OpenCV نیست؛ از matplotlib/NumPy استفاده نمی‌کند،
    بلکه در صورت لزوم skimage transform را import می‌کند.
    """
    if heatmap.shape == target_shape:
        return heatmap

    from skimage.transform import resize

    resized = resize(
        heatmap,
        target_shape,
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    )

    return resized.astype(np.float32)


# ============================================================
# 3. تطبیق نام فایل‌های Ground Truth و Heatmap
# ============================================================
def extract_image_key(filename_or_path: str | Path) -> str:
    """
    استخراج کلید نام تصویر از فایل heatmap یا annotation.
    
    این تابع به دنبال الگوی 'salmon_dis_X' می‌گردد که در آن X یک یا چند رقم است.
    
    مثال‌ها
    -------
    salmon_dis_01.npy
        -> salmon_dis_01
    
    0007_salmon_dis_189_case-TP_true-InfectedFish_pred-InfectedFish_conf-0.972_heatmap.npy
        -> salmon_dis_189
        
    protonet_resnet50_salmon_dis_029_heatmap.npy
        -> salmon_dis_029
    """
    name = Path(filename_or_path).name  # استفاده از name به جای stem برای امنیت بیشتر روی نام فایل
    
    # جستجوی الگوی salmon_dis_ به همراه یک یا چند رقم (\d+) بدون حساسیت به حروف بزرگ و کوچک
    match = re.search(r"salmon_dis_\d+", name, flags=re.IGNORECASE)
    
    if match:
        return match.group(0).lower()  # خروجی همیشه به صورت حروف کوچک یکدست بازگردانده می‌شود
    
    # اگر الگو پیدا نشد، به عنوان رفتار زاپاس (fallback) stem فایل را برمی‌گردانیم
    return Path(filename_or_path).stem.strip().lower()



def build_heatmap_file_map(folder: str | Path) -> dict[str, Path]:
    """
    فایل‌های npy را از یک پوشه می‌خواند و به شکل:
    {image_key: file_path}
    برمی‌گرداند.
    """
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder.resolve()}")

    file_map: dict[str, Path] = {}

    for file_path in folder.glob("*.npy"):
        key = extract_image_key(file_path.name)

        # در حالت تکراری بودن کلید، آخرین فایل جایگزین می‌شود.
        # بهتر است چنین تکراری‌ای در خروجی نداشته باشی.
        file_map[key] = file_path

    return file_map


# ============================================================
# 4. ارزیابی XAI یک مدل
# ============================================================
def evaluate_heatmaps_against_ground_truth(
    pred_folder: str | Path,
    gt_folder: str | Path,
    threshold: float = 0.65,
    model_name: Optional[str] = None,
    save_per_image_csv: Optional[str | Path] = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Heatmapهای یک مدل را با Soft Ground Truthهای پزشکی مقایسه می‌کند.

    Parameters
    ----------
    pred_folder:
        پوشه‌ی heatmapهای پیش‌بینی‌شده‌ی یک مدل.

    gt_folder:
        پوشه‌ی annotation/Soft Ground Truth با فرمت .npy.

    threshold:
        آستانه‌ی معیارهای Binary IoU و Dice.

    model_name:
        نام مدل برای ثبت در خروجی.

    save_per_image_csv:
        در صورت تعیین، نتایج تک‌تصویری در فایل CSV ذخیره می‌شوند.

    Returns
    -------
    summary:
        دیکشنری شامل mean/std معیارها و تعداد فایل‌های مشترک.

    per_image_df:
        DataFrame شامل نتایج هر تصویر.
    """
    pred_folder = Path(pred_folder)
    gt_folder = Path(gt_folder)

    gt_map = build_heatmap_file_map(gt_folder)
    pred_map = build_heatmap_file_map(pred_folder)

    common_keys = sorted(set(gt_map) & set(pred_map))
    missing_gt = sorted(set(pred_map) - set(gt_map))
    missing_pred = sorted(set(gt_map) - set(pred_map))

    if not common_keys:
        raise RuntimeError(
            "No matching .npy files were found between GT and prediction folders.\n"
            f"GT folder: {gt_folder.resolve()}\n"
            f"Prediction folder: {pred_folder.resolve()}\n"
            "Check annotation filenames and extract_image_key()."
        )

    rows: list[dict[str, Any]] = []

    for key in common_keys:
        gt = np.load(gt_map[key])
        pred = np.load(pred_map[key])

        gt = normalize_heatmap(gt)
        pred = normalize_heatmap(pred)

        # در صورت تفاوت سایز annotation و heatmap
        pred = resize_heatmap_if_needed(pred, gt.shape)

        row = {
            "model": model_name,
            "image_key": key,
            "gt_file": gt_map[key].name,
            "pred_file": pred_map[key].name,
            "IoU": iou(gt, pred, threshold),
            "Dice": dice(gt, pred, threshold),
            "SoftIoU": soft_iou(gt, pred),
            "SoftDice": soft_dice(gt, pred),
            "SSIM": float(ssim(gt, pred, data_range=1.0)),
        }

        rows.append(row)

    per_image_df = pd.DataFrame(rows)

    metric_columns = ["IoU", "Dice", "SoftIoU", "SoftDice", "SSIM"]

    summary: dict[str, Any] = {
        "model": model_name,
        "heatmap_folder": str(pred_folder.resolve()),
        "gt_folder": str(gt_folder.resolve()),
        "threshold": threshold,
        "matched_heatmaps": len(common_keys),
        "prediction_heatmaps": len(pred_map),
        "ground_truth_heatmaps": len(gt_map),
        "predictions_without_gt": len(missing_gt),
        "ground_truth_without_prediction": len(missing_pred),
    }

    for metric in metric_columns:
        summary[f"{metric}_mean"] = float(per_image_df[metric].mean())
        summary[f"{metric}_std"] = float(per_image_df[metric].std(ddof=0))

    if save_per_image_csv is not None:
        save_per_image_csv = Path(save_per_image_csv)
        save_per_image_csv.parent.mkdir(parents=True, exist_ok=True)
        per_image_df.to_csv(save_per_image_csv, index=False, encoding="utf-8-sig")

    print("=" * 85)
    print(f"Model: {model_name or pred_folder.name}")
    print(f"Matched files: {len(common_keys)}")
    print(f"Predictions without GT: {len(missing_gt)}")
    print(f"GT files without prediction: {len(missing_pred)}")
    print("-" * 85)
    print(f"IoU       : {summary['IoU_mean']:.4f} ± {summary['IoU_std']:.4f}")
    print(f"Dice      : {summary['Dice_mean']:.4f} ± {summary['Dice_std']:.4f}")
    print(f"Soft IoU  : {summary['SoftIoU_mean']:.4f} ± {summary['SoftIoU_std']:.4f}")
    print(f"Soft Dice : {summary['SoftDice_mean']:.4f} ± {summary['SoftDice_std']:.4f}")
    print(f"SSIM      : {summary['SSIM_mean']:.4f} ± {summary['SSIM_std']:.4f}")
    print("=" * 85)

    return summary, per_image_df


# ============================================================
# 5. خواندن لاگ‌های آموزشی JSONL
# ============================================================
def read_latest_training_log(log_path: str | Path) -> dict[str, Any]:
    """
    آخرین Run ثبت‌شده را از فایل JSONL می‌خواند.

    هر خط JSONL نشان‌دهنده‌ی یک اجرای مستقل آموزش است.
    """
    log_path = Path(log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"Training log not found: {log_path.resolve()}")

    records = []

    with log_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {log_path.name}, line {line_number}: {exc}"
                ) from exc

    if not records:
        raise RuntimeError(f"No valid training records found in: {log_path.resolve()}")

    return records[-1]


def summarize_training_log(
    log_path: str | Path,
    model_name: Optional[str] = None,
) -> dict[str, Any]:
    """
    اطلاعات اصلی آخرین ران یک مدل را از فایل JSONL استخراج می‌کند.

    شامل:
    - نام مدل
    - Hyperparameters
    - بهترین epoch بر اساس Val Accuracy
    - آخرین epoch
    - میانگین زمان/RAM/GPU Peak
    """
    record = read_latest_training_log(log_path)

    hyperparameters = record.get("hyperparameters", {})
    epochs = record.get("epochs", [])

    if not epochs:
        raise RuntimeError(f"No epoch information in log: {log_path}")

    if model_name is None:
        model_name = hyperparameters.get("model", Path(log_path).stem)

    best_epoch_data = max(
        epochs,
        key=lambda item: item.get("val_acc", float("-inf")),
    )

    last_epoch_data = epochs[-1]

    summary: dict[str, Any] = {
        "model": model_name,
        "log_path": str(Path(log_path).resolve()),
        "datetime": record.get("datetime"),
        "num_logged_epochs": len(epochs),

        # نتایج بهترین epoch از نظر Validation Accuracy
        "best_epoch": best_epoch_data.get("epoch"),
        "best_train_loss": best_epoch_data.get("train_loss"),
        "best_train_acc": best_epoch_data.get("train_acc"),
        "best_val_loss": best_epoch_data.get("val_loss"),
        "best_val_acc": best_epoch_data.get("val_acc"),

        # نتایج آخرین epoch
        "last_epoch": last_epoch_data.get("epoch"),
        "last_train_loss": last_epoch_data.get("train_loss"),
        "last_train_acc": last_epoch_data.get("train_acc"),
        "last_val_loss": last_epoch_data.get("val_loss"),
        "last_val_acc": last_epoch_data.get("val_acc"),

        # میانگین منابع در همه epochها
        "mean_epoch_time_sec": _safe_mean(epochs, "time_sec"),
        "mean_ram_mb": _safe_mean(epochs, "ram_mb"),
        "max_ram_mb": _safe_max(epochs, "ram_mb"),
        "mean_gpu_peak_mb": _safe_mean(epochs, "gpu_peak_mb"),
        "max_gpu_peak_mb": _safe_max(epochs, "gpu_peak_mb"),
    }

    # Hyperparameterها را با prefix اضافه می‌کنیم تا در جدول مشخص باشند
    for key, value in hyperparameters.items():
        summary[f"hp_{key}"] = value

    return summary


def _safe_mean(epoch_data: list[dict[str, Any]], key: str) -> Optional[float]:
    values = [
        float(item[key])
        for item in epoch_data
        if item.get(key) is not None
    ]

    return float(np.mean(values)) if values else None


def _safe_max(epoch_data: list[dict[str, Any]], key: str) -> Optional[float]:
    values = [
        float(item[key])
        for item in epoch_data
        if item.get(key) is not None
    ]

    return float(np.max(values)) if values else None


def epochs_to_dataframe(
    log_path: str | Path,
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    همه‌ی اطلاعات epochهای آخرین Run یک مدل را به DataFrame تبدیل می‌کند.
    """
    record = read_latest_training_log(log_path)

    hyperparameters = record.get("hyperparameters", {})
    epochs = record.get("epochs", [])

    if model_name is None:
        model_name = hyperparameters.get("model", Path(log_path).stem)

    df = pd.DataFrame(epochs)
    df.insert(0, "model", model_name)
    df.insert(1, "run_datetime", record.get("datetime"))

    return df


# ============================================================
# 6. مقایسه‌ی جامع چند مدل
# ============================================================
def _empty_xai_summary(
    model_name: str,
    pred_folder: str | Path,
    gt_folder: Optional[str | Path],
    threshold: float,
) -> dict[str, Any]:
    """خلاصه‌ی XAI وقتی Annotation پزشکی در دسترس نیست."""
    pred_folder = Path(pred_folder)

    pred_count = 0
    if pred_folder.exists():
        pred_count = len(build_heatmap_file_map(pred_folder))

    return {
        "model": model_name,
        "heatmap_folder": str(pred_folder.resolve()),
        "gt_folder": str(Path(gt_folder).resolve()) if gt_folder is not None else None,
        "threshold": threshold,
        "matched_heatmaps": 0,
        "prediction_heatmaps": pred_count,
        "ground_truth_heatmaps": 0,
        "predictions_without_gt": pred_count,
        "ground_truth_without_prediction": 0,
        "IoU_mean": np.nan,
        "IoU_std": np.nan,
        "Dice_mean": np.nan,
        "Dice_std": np.nan,
        "SoftIoU_mean": np.nan,
        "SoftIoU_std": np.nan,
        "SoftDice_mean": np.nan,
        "SoftDice_std": np.nan,
        "SSIM_mean": np.nan,
        "SSIM_std": np.nan,
    }

def compare_models(
    model_configs: list[dict[str, Any]],
    gt_folder: Optional[str | Path] = None,
    threshold: float = 0.65,
    output_dir: str | Path = "model_comparison_results",
    save_excel: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    مقایسه‌ی جامع مدل‌ها با یا بدون Soft Ground Truth پزشکی.

    اگر gt_folder برابر None باشد یا مسیر آن وجود نداشته باشد،
    مقایسه‌ی آموزش، Accuracy، زمان و حافظه اجرا می‌شود؛
    اما معیارهای XAI با NaN ثبت می‌شوند.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_path = Path(gt_folder) if gt_folder is not None else None
    gt_available = (
        gt_path is not None
        and gt_path.exists()
        and gt_path.is_dir()
        and any(gt_path.glob("*.npy"))
    )

    if gt_available:
        print(
            f"Soft Ground Truth found: {gt_path.resolve()}",
            flush=True,
        )
    else:
        print(
            "[WARNING] Soft Ground Truth is not available. "
            "XAI metrics will be skipped.",
            flush=True,
        )

    model_summaries: list[dict[str, Any]] = []
    epoch_dfs: list[pd.DataFrame] = []
    heatmap_dfs: list[pd.DataFrame] = []

    for config in model_configs:
        required_keys = {"name", "log_path", "heatmap_folder"}
        missing = required_keys - set(config)

        if missing:
            raise ValueError(
                f"Model configuration is missing keys: {missing}. "
                f"Current config: {config}"
            )

        model_name = config["name"]

        print("\n" + "#" * 90)
        print(f"Processing model: {model_name}")
        print("#" * 90)

        training_summary = summarize_training_log(
            log_path=config["log_path"],
            model_name=model_name,
        )

        epochs_df = epochs_to_dataframe(
            log_path=config["log_path"],
            model_name=model_name,
        )

        if gt_available:
            per_image_csv = (
                output_dir / f"{_safe_filename(model_name)}_per_image_xai.csv"
            )

            xai_summary, per_image_df = evaluate_heatmaps_against_ground_truth(
                pred_folder=config["heatmap_folder"],
                gt_folder=gt_path,
                threshold=threshold,
                model_name=model_name,
                save_per_image_csv=per_image_csv,
            )
        else:
            xai_summary = _empty_xai_summary(
                model_name=model_name,
                pred_folder=config["heatmap_folder"],
                gt_folder=gt_folder,
                threshold=threshold,
            )

            per_image_df = pd.DataFrame(
                columns=[
                    "model",
                    "image_key",
                    "gt_file",
                    "pred_file",
                    "IoU",
                    "Dice",
                    "SoftIoU",
                    "SoftDice",
                    "SSIM",
                ]
            )

            print(
                f"[{model_name}] XAI evaluation skipped: "
                "medical annotations are unavailable.",
                flush=True,
            )

        full_summary = {**training_summary, **xai_summary}

        model_summaries.append(full_summary)
        epoch_dfs.append(epochs_df)
        heatmap_dfs.append(per_image_df)

    model_summary_df = pd.DataFrame(model_summaries)

    all_epochs_df = (
        pd.concat(epoch_dfs, ignore_index=True)
        if epoch_dfs
        else pd.DataFrame()
    )

    all_heatmaps_df = (
        pd.concat(heatmap_dfs, ignore_index=True)
        if heatmap_dfs
        else pd.DataFrame()
    )

    sort_columns = [
        column
        for column in ["best_val_acc", "SoftDice_mean", "SSIM_mean"]
        if (
            column in model_summary_df.columns
            and model_summary_df[column].notna().any()
        )
    ]

    if sort_columns:
        model_summary_df = model_summary_df.sort_values(
            by=sort_columns,
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    summary_csv = output_dir / "models_comparison_summary.csv"
    epochs_csv = output_dir / "models_all_epochs.csv"
    heatmaps_csv = output_dir / "models_xai_per_image.csv"

    model_summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    all_epochs_df.to_csv(epochs_csv, index=False, encoding="utf-8-sig")
    all_heatmaps_df.to_csv(heatmaps_csv, index=False, encoding="utf-8-sig")

    if save_excel:
        excel_path = output_dir / "models_comparison.xlsx"

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            model_summary_df.to_excel(
                writer,
                sheet_name="Model Summary",
                index=False,
            )

            all_epochs_df.to_excel(
                writer,
                sheet_name="All Epochs",
                index=False,
            )

            all_heatmaps_df.to_excel(
                writer,
                sheet_name="XAI Per Image",
                index=False,
            )

        print(f"\nExcel saved to: {excel_path.resolve()}")

    print("\n" + "=" * 90)
    print("Final model comparison summary")
    print("=" * 90)

    display_columns = [
        column
        for column in [
            "model",
            "best_epoch",
            "best_val_acc",
            "best_val_loss",
            "mean_epoch_time_sec",
            "mean_ram_mb",
            "max_gpu_peak_mb",
            "matched_heatmaps",
            "IoU_mean",
            "Dice_mean",
            "SoftIoU_mean",
            "SoftDice_mean",
            "SSIM_mean",
        ]
        if column in model_summary_df.columns
    ]

    print(model_summary_df[display_columns].to_string(index=False))

    print(f"\nCSV summary saved to: {summary_csv.resolve()}")
    print(f"CSV epochs saved to: {epochs_csv.resolve()}")
    print(f"CSV per-image XAI saved to: {heatmaps_csv.resolve()}")

    return model_summary_df, all_epochs_df, all_heatmaps_df


def _safe_filename(name: str) -> str:
    """تبدیل نام مدل به نام فایل امن."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")


# ============================================================
# 7. نمودارهای مقایسه‌ای
# ============================================================
def plot_model_comparison(
    model_summary_df: pd.DataFrame,
    output_dir: str | Path = "model_comparison_results",
) -> None:
    """
    نمودارهای مقایسه‌ی Accuracy، زمان، RAM/GPU و معیارهای XAI را ذخیره می‌کند.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = model_summary_df.copy()

    # --------------------------------------------------------
    # 1. Validation Accuracy
    # --------------------------------------------------------
    if "best_val_acc" in df.columns:
        plt.figure(figsize=(10, 5))

        plt.bar(df["model"], df["best_val_acc"], color="steelblue")

        plt.title("Best Validation Accuracy Comparison")
        plt.xlabel("Model")
        plt.ylabel("Best Validation Accuracy")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=25, ha="right")

        for index, value in enumerate(df["best_val_acc"]):
            if pd.notna(value):
                plt.text(index, value + 0.015, f"{value:.3f}", ha="center")

        plt.tight_layout()
        plt.savefig(
            output_dir / "comparison_best_val_accuracy.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # --------------------------------------------------------
    # 2. زمان متوسط هر Epoch
    # --------------------------------------------------------
    if "mean_epoch_time_sec" in df.columns:
        plt.figure(figsize=(10, 5))

        plt.bar(df["model"], df["mean_epoch_time_sec"], color="darkorange")

        plt.title("Average Epoch Time Comparison")
        plt.xlabel("Model")
        plt.ylabel("Average Epoch Time (seconds)")
        plt.xticks(rotation=25, ha="right")

        plt.tight_layout()
        plt.savefig(
            output_dir / "comparison_epoch_time.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # --------------------------------------------------------
    # 3. RAM و GPU Peak
    # --------------------------------------------------------
    resource_columns = [
        column for column in ["mean_ram_mb", "max_gpu_peak_mb"]
        if column in df.columns
    ]

    if resource_columns:
        ax = df.set_index("model")[resource_columns].plot(
            kind="bar",
            figsize=(11, 5),
        )

        ax.set_title("Memory Consumption Comparison")
        ax.set_xlabel("Model")
        ax.set_ylabel("Memory (MB)")
        ax.tick_params(axis="x", rotation=25)

        plt.tight_layout()
        plt.savefig(
            output_dir / "comparison_memory_usage.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # --------------------------------------------------------
    # 4. معیارهای XAI
    # --------------------------------------------------------
    xai_columns = [
        column for column in [
            "IoU_mean",
            "Dice_mean",
            "SoftIoU_mean",
            "SoftDice_mean",
            "SSIM_mean",
        ]
        if column in df.columns
    ]

    if xai_columns:
        ax = df.set_index("model")[xai_columns].plot(
            kind="bar",
            figsize=(13, 6),
        )

        ax.set_title("XAI Heatmap vs Medical Annotation Comparison")
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=25)

        plt.tight_layout()
        plt.savefig(
            output_dir / "comparison_xai_metrics.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    print(f"Comparison plots saved in: {output_dir.resolve()}")
