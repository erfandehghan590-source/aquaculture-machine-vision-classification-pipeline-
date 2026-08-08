# ============================================================
# SalmonScan - Model Comparison Notebook Cell
# بدون Soft Ground Truth پزشکی
# ============================================================

from pathlib import Path
from IPython.display import display

from model_comparison_utils import (
    compare_models,
    plot_model_comparison,
)

GT_FOLDER = r"C:\Users\conceptD\ground_truth"

OUTPUT_DIR = r"C:\Users\conceptD\model_comparison_results"

MODELS = [
    # ============================================================
    # مدل های detection
    # ============================================================
    {
        "name":"YOLOv8n-cls",
        "log_path":r"C:\Users\conceptD\Downloads\results.json",
        "heatmap_folder": None
    },
    # ============================================================
    # مدل‌های پایه CNN
    # ============================================================
    {
        "name": "MobileNetV3-Large",
        "log_path": r"C:\Users\conceptD\mobilenet_training_log_augmented.jsonl",
        "heatmap_folder": r"C:\Users\conceptD\gradcam_MOBILENET_augmented_infected_predictions",
    },
    {
        "name": "ResNet50",
        "log_path": r"C:\Users\conceptD\resnet50_training_log_augmented.jsonl",
        "heatmap_folder": r"C:\Users\conceptD\gradcam_RESNET50_augmented_infected_predictions",
    },
    {
        "name": "ResNet18",
        "log_path": r"C:\Users\conceptD\resnet18_training_log_augmented.jsonl",
        "heatmap_folder": r"C:\Users\conceptD\gradcam_RESNET18_augmented_infected_predictions",
    },
    

    # ============================================================
    # مدل‌های Transformer
    # ============================================================
    {
        "name": "ViT-B16",
        "log_path": r"C:\Users\conceptD\vit_training_log_augmented.jsonl",
        "heatmap_folder": r"C:\Users\conceptD\gradcam_VIT_augmented_infected_predictions",
    },
    {
        "name": "Swin-Tiny",
        "log_path": r"C:\Users\conceptD\swin_tiny_training_log_augmented.jsonl",
        "heatmap_folder": r"C:\Users\conceptD\gradcam_SWIN_TINY_augmented_infected_predictions",
    },

    # ============================================================
    # مدل‌های مدرن CNN
    # ============================================================
    {
        "name": "ConvNeXt-Tiny",
        "log_path": r"C:\Users\conceptD\convnext_tiny_training_log_augmented.jsonl",
        "heatmap_folder": r"C:\Users\conceptD\gradcam_CONVNEXT_TINY_augmented_infected_predictions",
    },

    # ============================================================
    # ProtoNetها با backboneهای متفاوت
    # ============================================================
    {
        "name": "ProtoNet-ResNet50",
        "log_path": r"C:\Users\conceptD\protonet_resnet50_training_log_augmented.jsonl",
        "heatmap_folder": r"C:\Users\conceptD\gradcam_PROTONET_RESNET50_augmented_infected_predictions",
    },
    {
        "name": "ProtoNet-ResNet18",
        "log_path": r"C:\Users\conceptD\protonet_resnet18_training_log_augmented.jsonl",
        "heatmap_folder": r"C:\Users\conceptD\gradcam_PROTONET_RESNET18_augmented_infected_predictions",
    },
    {
        "name": "ProtoNet-ConvNeXt-Tiny",
        "log_path": r"C:\Users\conceptD\protonet_convnext_tiny_training_log_augmented.jsonl",
        "heatmap_folder": r"C:\Users\conceptD\gradcam_PROTONET_CONVNEXT_TINY_augmented_infected_predictions",
    },
]

summary_df, all_epochs_df, xai_per_image_df = compare_models(
    model_configs=MODELS,
    gt_folder=GT_FOLDER,
    threshold=0.65,
    output_dir=OUTPUT_DIR,
    save_excel=True,
)

plot_model_comparison(
    model_summary_df=summary_df,
    output_dir=OUTPUT_DIR,
)

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

print("\n" + "=" * 110)
print("ALL EPOCH RESULTS")
print("=" * 110)

display(all_epochs_df.sort_values(by=["model", "epoch"]))

if not xai_per_image_df.empty:
    print("\n" + "=" * 110)
    print("XAI PER-IMAGE RESULTS")
    print("=" * 110)

    display(
        xai_per_image_df.sort_values(
            by=["model", "image_key"]
        ).head(20)
    )
else:
    print("\nXAI comparison was skipped because Soft Ground Truth is not available.")

print("\nResults saved in:")
print(Path(OUTPUT_DIR).resolve())
