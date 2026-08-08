import os
import copy
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common_utils import (
    set_seed,
    load_config,
    get_device,
    get_train_transform,
    get_test_transform,
    get_ram_usage_mb,
    train_one_epoch,
    evaluate,
    collect_hyperparameters,
    log_training_run,
    save_gradcams_for_predicted_infected,
    load_model_weights,
    plot_learning_curves,
)


# =====================================================================
# تنظیمات اجرا
# =====================================================================
MODE = "train"  # "train" یا "eval"
CONFIG_PATH = "MVconfig.yaml"

config = load_config(CONFIG_PATH)

DATA_ROOT = config["DATA_ROOT_SPLITED"]
DATA_SUBSET = config["DATA_SUBSET"]
IMG_SIZE = config["IMG_SIZE"]
BATCH_SIZE = config["BATCH_SIZE"]
NUM_EPOCHS = config["NUM_EPOCHS"]
LEARNING_RATE = float(config["LEARNING_RATE"])
VAL_RATIO = config["VAL_RATIO"]
TEST_RATIO = config["TEST_RATIO"]
NUM_CLASSES = config["NUM_CLASSES"]
INFECTED_CLASS_NAME = config["INFECTED_CLASS_NAME"]

MODEL_NAME = "MobileNetV3-Large"
MODEL_TAG = "mobilenet_v3_large"

# =====================================================================
# تنظیمات اولیه
# =====================================================================
SEED = 42
set_seed(SEED)

DEVICE = get_device()
print("Using device:", DEVICE)

PIN_MEMORY = DEVICE.type == "cuda"

# =====================================================================
# مسیرها
# =====================================================================
TRAIN_DATA_DIR = os.path.join(DATA_ROOT, "train")
VAL_DATA_DIR = os.path.join(DATA_ROOT, "val")

RUN_LOG_PATH = str(SCRIPT_DIR / f"mobilenet_training_log_{DATA_SUBSET.lower()}.jsonl")

GRADCAM_OUTPUT_DIR =  str(SCRIPT_DIR / f"gradcam_MOBILENET_{DATA_SUBSET.lower()}_infected_predictions")
MODEL_SAVE_PATH = str(SCRIPT_DIR / f"mobilenet_v3_large_{DATA_SUBSET.lower()}_salmonscan.pth")


# =====================================================================
# Transforms
# =====================================================================
train_transform = get_train_transform(IMG_SIZE)
val_transform = get_test_transform(IMG_SIZE)

# =====================================================================
# لود دیتاست
# =====================================================================
train_dataset = datasets.ImageFolder(
    root=TRAIN_DATA_DIR,
    transform=train_transform)

# دیتاست پایه validation
# این دیتاست را هم برای DataLoader و هم برای Grad-CAM استفاده می‌کنیم.
val_base_dataset = datasets.ImageFolder(
    root=VAL_DATA_DIR,
    transform=val_transform)

# برای سازگاری با تابع save_gradcams_for_predicted_infected در common_utils
# چون آن تابع انتظار دارد test_dataset.indices وجود داشته باشد.
val_dataset = Subset(
    val_base_dataset,
    list(range(len(val_base_dataset))))

class_names = train_dataset.classes

print("Classes:", class_names)
print("Train images:", len(train_dataset))
print("Val images:", len(val_dataset))
print("Total images:", len(train_dataset) + len(val_dataset))

if INFECTED_CLASS_NAME not in class_names:
    raise ValueError(f"Class '{INFECTED_CLASS_NAME}' not found in dataset classes: {class_names}")

INFECTED_CLASS_IDX = class_names.index(INFECTED_CLASS_NAME)
print("Infected class index:", INFECTED_CLASS_IDX)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=PIN_MEMORY)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=PIN_MEMORY)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# =====================================================================
# ساخت مدل MobileNetV3-Large
# =====================================================================
def build_model(num_classes: int = 2) -> nn.Module:
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model

model = build_model(NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =====================================================================
# تابع اصلی
# =====================================================================
def main():
    if MODE == "train":
        print("\n--- Running Mode: TRAIN ---")

        best_model_wts = copy.deepcopy(model.state_dict())
        best_val_acc = 0.0

        train_losses = []
        val_losses = []

        train_accs = []
        val_accs = []

        epoch_times = []
        epoch_ram_mb = []
        epoch_gpu_peak_mb = []

        for epoch in range(NUM_EPOCHS):
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(DEVICE)
                torch.cuda.synchronize()

            epoch_start = time.perf_counter()

            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=DEVICE
            )

            val_loss, val_acc, _, _ = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=DEVICE
            )

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
                gpu_peak_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
            else:
                gpu_peak_mb = 0.0

            epoch_time = time.perf_counter() - epoch_start
            ram_usage = get_ram_usage_mb()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            train_accs.append(train_acc)
            val_accs.append(val_acc)

            epoch_times.append(epoch_time)
            epoch_ram_mb.append(ram_usage)
            epoch_gpu_peak_mb.append(gpu_peak_mb)

            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Time: {epoch_time:.2f}s | "
                f"RAM: {ram_usage:.2f} MB | "
                f"GPU Peak: {gpu_peak_mb:.2f} MB"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_model_wts)

        torch.save(
            model.state_dict(),
            MODEL_SAVE_PATH
        )

        print(f"\nModel saved to: {MODEL_SAVE_PATH}")
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")

        plot_learning_curves(
            train_losses=train_losses,
            val_losses=val_losses,
            train_accs=train_accs,
            val_accs=val_accs,
            MODEL_NAME = MODEL_NAME
        )

        hyperparameters = collect_hyperparameters(
            seed=SEED,
            img_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            model="MobileNetV3-Large",
            data_subset=DATA_SUBSET,
            device=str(DEVICE),
            train_size=len(train_dataset),
            val_size=len(val_dataset)
        )

        log_training_run(
            hyperparameters=hyperparameters,
            train_accs=train_accs,
            val_accs=val_accs,
            train_losses=train_losses,
            val_losses=val_losses,
            log_path=RUN_LOG_PATH,
            epoch_times=epoch_times,
            epoch_ram_mb=epoch_ram_mb,
            epoch_gpu_peak_mb=epoch_gpu_peak_mb
        )

    elif MODE == "eval":
        print("\n--- Running Mode: EVAL (Load Weights) ---")

        if not os.path.exists(MODEL_SAVE_PATH):
            raise FileNotFoundError(
                f"No saved model weights found at: {MODEL_SAVE_PATH}. "
                f"Please train the model first."
            )

        load_model_weights(
            model=model,
            model_save_path=MODEL_SAVE_PATH,
            device=DEVICE
        )

        print(f"Weights successfully loaded from: {MODEL_SAVE_PATH}")

    else:
        raise ValueError("MODE must be either 'train' or 'eval'.")

    # =================================================================
    # بخش مشترک ارزیابی نهایی و XAI
    # =================================================================
    eval_loss, eval_acc, y_true, y_pred = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=DEVICE
    )

    print("\n===== Validation/Test Results =====")
    print(f"Loss: {eval_loss:.4f}")
    print(f"Accuracy: {eval_acc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {DATA_SUBSET} - MobileNetV3-Large")
    plt.show()

    # =================================================================
    # Grad-CAM
    # =================================================================
    save_gradcams_for_predicted_infected(
        model=model,
        test_dataset=val_dataset,
        full_dataset=val_base_dataset,
        class_names=class_names,
        infected_class_idx=INFECTED_CLASS_IDX,
        output_dir=GRADCAM_OUTPUT_DIR,
        device=DEVICE,
        architecture="mobilenet_v3_large",
        img_size=IMG_SIZE,
        clear_previous=True
    )


if __name__ == "__main__":
    main()
