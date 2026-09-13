"""
AI SMART WASTE MANAGEMENT & SEGREGATION SYSTEM
PHASE 5 - TRANSFER LEARNING
Model: MobileNetV3-Small
"""

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001

DATA_DIR = "data/processed"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "validation.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "mobilenetv3_transfer_best.pth"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "mobilenetv3_transfer_final.pth"
)

HISTORY_PATH = os.path.join(
    OUTPUT_DIR,
    "mobilenetv3_training_history.csv"
)

ACCURACY_GRAPH = os.path.join(
    OUTPUT_DIR,
    "mobilenetv3_accuracy.png"
)

LOSS_GRAPH = os.path.join(
    OUTPUT_DIR,
    "mobilenetv3_loss.png"
)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("PHASE 5 - TRANSFER LEARNING")
print("=" * 70)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available. Using CPU.")


# ============================================================
# 4. LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

print("\nDataset:")
print(f"Training samples   : {len(train_df)}")
print(f"Validation samples : {len(val_df)}")
print(f"Test samples       : {len(test_df)}")


# ============================================================
# 5. CLASS MAPPING
# ============================================================

with open(
    "models/class_mapping.json",
    "r",
    encoding="utf-8"
) as f:
    class_mapping = json.load(f)


# JSON mapping from previous phase:
# {"0": "battery", "1": "biological", ...}

class_to_idx = {
    class_name: int(idx)
    for idx, class_name in class_mapping.items()
}

idx_to_class = {
    int(idx): class_name
    for idx, class_name in class_mapping.items()
}

NUM_CLASSES = len(class_to_idx)

print(f"Number of classes : {NUM_CLASSES}")

print("\nClasses:")
for idx in sorted(idx_to_class):
    print(f"{idx}: {idx_to_class[idx]}")


# ============================================================
# 6. CONVERT CLASS TO NUMERIC LABEL
# ============================================================

train_df["label"] = train_df["class"].map(class_to_idx)
val_df["label"] = val_df["class"].map(class_to_idx)
test_df["label"] = test_df["class"].map(class_to_idx)

if train_df["label"].isna().any():
    raise ValueError("Unknown class found in training data.")

if val_df["label"].isna().any():
    raise ValueError("Unknown class found in validation data.")

if test_df["label"].isna().any():
    raise ValueError("Unknown class found in test data.")

train_df["label"] = train_df["label"].astype(int)
val_df["label"] = val_df["label"].astype(int)
test_df["label"] = test_df["label"].astype(int)


# ============================================================
# 7. TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 8. DATASET CLASS
# ============================================================

class WasteDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_path = row["image_path"]
        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# 9. CREATE DATASETS
# ============================================================

train_dataset = WasteDataset(
    train_df,
    train_transform
)

val_dataset = WasteDataset(
    val_df,
    val_test_transform
)

test_dataset = WasteDataset(
    test_df,
    val_test_transform
)


# ============================================================
# 10. CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 11. LOAD MOBILENETV3
# ============================================================

print("\nLoading MobileNetV3-Small...")

try:

    weights = models.MobileNet_V3_Small_Weights.DEFAULT

    model = models.mobilenet_v3_small(
        weights=weights
    )

    print("Pretrained ImageNet weights loaded.")

except Exception as e:

    print("\nWARNING: Could not load pretrained weights.")
    print("Reason:", e)

    print("Using MobileNetV3-Small without pretrained weights.")

    model = models.mobilenet_v3_small(
        weights=None
    )


# ============================================================
# 12. FREEZE BACKBONE
# ============================================================

for parameter in model.features.parameters():
    parameter.requires_grad = False


# ============================================================
# 13. REPLACE CLASSIFIER
# ============================================================

in_features = model.classifier[-1].in_features

model.classifier[-1] = nn.Linear(
    in_features,
    NUM_CLASSES
)

model = model.to(device)


# ============================================================
# 14. CLASS WEIGHTS
# ============================================================

weights_df = pd.read_csv(
    "data/processed/class_weights.csv"
)

class_weights = torch.tensor(
    weights_df["weight"].values,
    dtype=torch.float32
).to(device)


# ============================================================
# 15. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# 16. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 17. TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 18. VALIDATION FUNCTION
# ============================================================

def validate(model, loader):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 19. TRAIN MODEL
# ============================================================

history = []

best_val_accuracy = 0.0

print("\nStarting Transfer Learning...")
print("-" * 70)

for epoch in range(EPOCHS):

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader
    )

    val_loss, val_accuracy = validate(
        model,
        val_loader
    )

    history.append({

        "epoch": epoch + 1,

        "train_loss": train_loss,

        "train_accuracy": train_accuracy,

        "val_loss": val_loss,

        "val_accuracy": val_accuracy

    })

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}%"
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print(
            f"  ✓ Best model saved "
            f"({best_val_accuracy * 100:.2f}%)"
        )


# ============================================================
# 20. SAVE FINAL MODEL
# ============================================================

torch.save(
    model.state_dict(),
    FINAL_MODEL_PATH
)


# ============================================================
# 21. SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)


# ============================================================
# 22. LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ============================================================
# 23. TEST EVALUATION
# ============================================================

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# ============================================================
# 24. TEST METRICS
# ============================================================

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=[
        idx_to_class[i]
        for i in range(NUM_CLASSES)
    ],
    digits=4
)


# ============================================================
# 25. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TRANSFER LEARNING RESULTS")
print("=" * 70)

print(
    f"Best Validation Accuracy : "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Test Accuracy            : "
    f"{test_accuracy * 100:.2f}%"
)

print("\nClassification Report:")
print(report)


# ============================================================
# 26. TRAINING ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history_df["epoch"],
    history_df["train_accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    history_df["epoch"],
    history_df["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "MobileNetV3 Transfer Learning - Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    ACCURACY_GRAPH,
    dpi=300
)

plt.close()


# ============================================================
# 27. TRAINING LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history_df["epoch"],
    history_df["train_loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    history_df["epoch"],
    history_df["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "MobileNetV3 Transfer Learning - Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    LOSS_GRAPH,
    dpi=300
)

plt.close()


# ============================================================
# 28. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 5 COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print(BEST_MODEL_PATH)
print(FINAL_MODEL_PATH)
print(HISTORY_PATH)
print(ACCURACY_GRAPH)
print(LOSS_GRAPH)

print("\nBaseline CNN Test Accuracy : 74.36%")
print(
    f"MobileNetV3 Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

improvement = (
    test_accuracy * 100
) - 74.36

print(
    f"Improvement               : "
    f"{improvement:+.2f}%"
)

print("\nNext Phase: Prediction + Recommendation System")