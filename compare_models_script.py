# ============================================================
# SalmonScan - Model Comparison Notebook Cell
# ============================================================

from pathlib import Path
from IPython.display import display
import yaml

from model_comparison_utils import (
    compare_models,
    plot_model_comparison,
)


def resolve_path(path_value, root_dir):
    """
    اگر path_value مسیر absolute باشد، همان را برمی‌گرداند.
    اگر relative باشد، آن را نسبت به root_dir کامل می‌کند.
    """
    if path_value is None:
        return None

    path_value = Path(path_value).expanduser()

    if path_value.is_absolute():
        return path_value.resolve()

    return (root_dir / path_value).resolve()


def load_model_config(config_path):
    """
    خواندن کانفیگ YAML و تبدیل مسیرهای relative به absolute.

    اولویت تعیین root:
    1. اگر root_dir داخل YAML باشد، از آن استفاده می‌شود.
    2. اگر root_dir نباشد، پوشه‌ای که config.yaml داخل آن است root در نظر گرفته می‌شود.
    """
    config_path = Path(config_path).resolve()

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    # اگر root_dir داخل YAML تعریف شده باشد، از آن استفاده کن؛
    # در غیر این صورت root برابر پوشه خود config می‌شود.
    if config.get("root_dir"):
        root_dir = Path(config["root_dir"]).expanduser().resolve()
    else:
        root_dir = config_path.parent

    config["root_dir"] = root_dir

    # مسیر GT و خروجی‌ها
    config["gt_folder"] = resolve_path(
        config.get("gt_folder", "ground_truth"),
        root_dir
    )

    config["output_dir"] = resolve_path(
        config.get("output_dir", "model_comparison_results"),
        root_dir
    )

    # تنظیمات عمومی
    config["threshold"] = float(config.get("threshold", 0.65))
    config["save_excel"] = bool(config.get("save_excel", True))

    if "models" not in config or not isinstance(config["models"], list):
        raise ValueError("Config must contain a 'models' list.")

    for model_cfg in config["models"]:
        if "name" not in model_cfg:
            raise ValueError("Each model config must have a 'name' field.")

        if "log_path" not in model_cfg:
            raise ValueError(f"Model '{model_cfg['name']}' has no 'log_path'.")

        if "heatmap_folder" not in model_cfg:
            raise ValueError(f"Model '{model_cfg['name']}' has no 'heatmap_folder'.")

        model_cfg["log_path"] = resolve_path(model_cfg["log_path"], root_dir)
        model_cfg["heatmap_folder"] = resolve_path(model_cfg["heatmap_folder"], root_dir)

    return config


# =========================
# Load Config
# =========================
CONFIG_PATH = "C:/Users/conceptD/mode-comparison-config.yaml"

config = load_model_config(CONFIG_PATH)

MODELS = config["models"]
GT_FOLDER = config["gt_folder"]
OUTPUT_DIR = config["output_dir"]
THRESHOLD = config["threshold"]
SAVE_EXCEL = config["save_excel"]

print("\nLoaded configuration:")
print("=" * 80)
print("Config path:", Path(CONFIG_PATH).resolve())
print("Root dir:", config["root_dir"])
print("GT folder:", GT_FOLDER)
print("Output dir:", OUTPUT_DIR)
print("Threshold:", THRESHOLD)
print("Save Excel:", SAVE_EXCEL)

print("\nLoaded models:")
print("=" * 80)

for model_cfg in MODELS:
    print(model_cfg["name"])
    print("  Log path      :", model_cfg["log_path"])
    print("  Heatmap folder:", model_cfg["heatmap_folder"])
    print()


# =========================
# Optional Path Checks
# =========================
print("\nChecking paths:")
print("=" * 80)

if not Path(GT_FOLDER).exists():
    print(f"Warning: GT_FOLDER does not exist: {GT_FOLDER}")
else:
    print(f"GT_FOLDER exists: {GT_FOLDER}")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
print(f"OUTPUT_DIR ready: {OUTPUT_DIR}")

for model_cfg in MODELS:
    log_path = Path(model_cfg["log_path"])
    heatmap_folder = Path(model_cfg["heatmap_folder"])

    if not log_path.exists():
        print(f"Warning: log_path not found for {model_cfg['name']}: {log_path}")

    if not heatmap_folder.exists():
        print(f"Warning: heatmap_folder not found for {model_cfg['name']}: {heatmap_folder}")


# =========================
# Compare Models
# =========================
summary_df, all_epochs_df, xai_per_image_df = compare_models(
    model_configs=MODELS,
    gt_folder=GT_FOLDER,
    threshold=THRESHOLD,
    output_dir=OUTPUT_DIR,
    save_excel=SAVE_EXCEL,
)


# =========================
# Plot Comparison
# =========================
plot_model_comparison(
    model_summary_df=summary_df,
    output_dir=OUTPUT_DIR,
)


# =========================
# Display Final Summary
# =========================
columns_to_show = [
    "model",
    "best_epoch",
    "best_train_loss",
    "best_train_acc",
    "best_val_loss",
    "best_val_acc",
    "mean_epoch_time_sec",
    "mean_ram_mb",
    "max_ram_mb",
    "mean_gpu_peak_mb",
    "max_gpu_peak_mb",
    "matched_heatmaps",
    "IoU_mean",
    "Dice_mean",
    "SoftIoU_mean",
    "SoftDice_mean",
    "SSIM_mean",
]

available_columns = [
    column
    for column in columns_to_show
    if column in summary_df.columns
]

print("\n" + "=" * 110)
print("FINAL MODEL COMPARISON")
print("=" * 110)

display(summary_df[available_columns])


# =========================
# Display All Epoch Results
# =========================
print("\n" + "=" * 110)
print("ALL EPOCH RESULTS")
print("=" * 110)

if not all_epochs_df.empty and {"model", "epoch"}.issubset(all_epochs_df.columns):
    display(all_epochs_df.sort_values(by=["model", "epoch"]))
else:
    display(all_epochs_df)


# =========================
# Display XAI Per-Image Results
# =========================
if xai_per_image_df is not None and not xai_per_image_df.empty:
    print("\n" + "=" * 110)
    print("XAI PER-IMAGE RESULTS")
    print("=" * 110)

    sort_cols = [
        col for col in ["model", "image_key"]
        if col in xai_per_image_df.columns
    ]

    if sort_cols:
        display(
            xai_per_image_df.sort_values(
                by=sort_cols
            ).head(20)
        )
    else:
        display(xai_per_image_df.head(20))
else:
    print("\nXAI comparison was skipped because Soft Ground Truth is not available.")


print("\nResults saved in:")
print(Path(OUTPUT_DIR).resolve())
