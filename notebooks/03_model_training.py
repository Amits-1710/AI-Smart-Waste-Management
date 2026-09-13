# ============================================================
# AI SMART WASTE MANAGEMENT & SEGREGATION SYSTEM
# File: 03_model_training.py
# Purpose: Baseline CNN Model Training using PyTorch
# ============================================================

from pathlib import Path
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
OUTPUTS_PATH = PROJECT_ROOT / "outputs"

MODELS_PATH.mkdir(parents=True, exist_ok=True)
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 15

LEARNING_RATE = 0.001

RANDOM_SEED = 42


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 4. DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


# ============================================================
# 5. PROJECT INFORMATION
# ============================================================

print("=" * 70)
print("AI SMART WASTE MANAGEMENT & SEGREGATION SYSTEM")
print("BASELINE CNN MODEL TRAINING - PyTorch")
print("=" * 70)

print("\nPyTorch version:")
print(torch.__version__)

print("\nDevice:")
print(DEVICE)

if torch.cuda.is_available():

    print("GPU:")
    print(torch.cuda.get_device_name(0))

else:

    print("GPU not available - using CPU")


print("\nConfiguration:")
print(f"Image Size    : {IMAGE_SIZE} x {IMAGE_SIZE}")
print(f"Batch Size    : {BATCH_SIZE}")
print(f"Epochs        : {EPOCHS}")
print(f"Learning Rate : {LEARNING_RATE}")


# ============================================================
# 6. LOAD CSV FILES
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)


train_csv = PROCESSED_PATH / "train.csv"
validation_csv = PROCESSED_PATH / "validation.csv"
test_csv = PROCESSED_PATH / "test.csv"
class_weights_csv = PROCESSED_PATH / "class_weights.csv"


if not train_csv.exists():

    raise FileNotFoundError(
        f"Training CSV not found:\n{train_csv}"
    )


if not validation_csv.exists():

    raise FileNotFoundError(
        f"Validation CSV not found:\n{validation_csv}"
    )


if not test_csv.exists():

    raise FileNotFoundError(
        f"Test CSV not found:\n{test_csv}"
    )


if not class_weights_csv.exists():

    raise FileNotFoundError(
        f"Class weights CSV not found:\n{class_weights_csv}"
    )


train_df = pd.read_csv(train_csv)

validation_df = pd.read_csv(validation_csv)

test_df = pd.read_csv(test_csv)

class_weights_df = pd.read_csv(class_weights_csv)


print(f"\nTraining samples   : {len(train_df)}")

print(f"Validation samples : {len(validation_df)}")

print(f"Test samples       : {len(test_df)}")


# ============================================================
# 7. CLASS NAMES
# ============================================================

class_names = sorted(
    train_df["class"].unique().tolist()
)

NUM_CLASSES = len(class_names)


print("\nNumber of classes:")
print(NUM_CLASSES)

print("\nClasses:")

for index, class_name in enumerate(class_names):

    print(f"{index}: {class_name}")


# ============================================================
# 8. SAVE CLASS MAPPING
# ============================================================

class_mapping = {
    index: class_name
    for index, class_name in enumerate(class_names)
}


mapping_path = MODELS_PATH / "class_mapping.json"


with open(
    mapping_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        class_mapping,
        file,
        indent=4
    )


print("\nClass mapping saved:")
print(mapping_path)


# ============================================================
# 9. CUSTOM DATASET CLASS
# ============================================================

class WasteDataset(Dataset):

    def __init__(
        self,
        dataframe,
        class_names,
        transform=None
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.class_names = class_names

        self.class_to_index = {
            class_name: index
            for index, class_name
            in enumerate(class_names)
        }

        self.transform = transform


    def __len__(self):

        return len(self.dataframe)


    def __getitem__(self, index):

        image_path = self.dataframe.loc[
            index,
            "image_path"
        ]

        class_name = self.dataframe.loc[
            index,
            "class"
        ]

        # Load image
        image = Image.open(
            image_path
        ).convert("RGB")


        # Convert class name to integer
        label = self.class_to_index[
            class_name
        ]


        # Apply transformations
        if self.transform is not None:

            image = self.transform(image)


        return image, label


# ============================================================
# 10. IMAGE TRANSFORMATIONS
# ============================================================

print("\n" + "=" * 70)
print("CREATING IMAGE TRANSFORMS")
print("=" * 70)


# Training augmentation
train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=15
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# Validation transform
validation_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# Test transform
test_transform = validation_transform


# ============================================================
# 11. CREATE DATASETS
# ============================================================

train_dataset = WasteDataset(
    train_df,
    class_names,
    train_transform
)


validation_dataset = WasteDataset(
    validation_df,
    class_names,
    validation_transform
)


test_dataset = WasteDataset(
    test_df,
    class_names,
    test_transform
)


print("\nDatasets created successfully.")

print(f"Training dataset   : {len(train_dataset)}")

print(f"Validation dataset : {len(validation_dataset)}")

print(f"Test dataset       : {len(test_dataset)}")


# ============================================================
# 12. CREATE DATALOADERS
# ============================================================

print("\n" + "=" * 70)
print("CREATING DATALOADERS")
print("=" * 70)


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0
)


validation_loader = DataLoader(

    validation_dataset,

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


print("\nDataLoaders created successfully.")


# ============================================================
# 13. LOAD CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 70)
print("LOADING CLASS WEIGHTS")
print("=" * 70)


class_weights = np.zeros(
    NUM_CLASSES,
    dtype=np.float32
)


for _, row in class_weights_df.iterrows():

    class_name = row["class"]

    weight = float(row["weight"])

    class_index = class_names.index(
        class_name
    )

    class_weights[class_index] = weight


class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)


print("\nClass weights:")

for index, weight in enumerate(class_weights):

    print(
        f"{class_names[index]:15s} : {weight:.4f}"
    )


# ============================================================
# 14. BASELINE CNN MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING BASELINE CNN")
print("=" * 70)


class BaselineCNN(nn.Module):

    def __init__(
        self,
        num_classes
    ):

        super().__init__()


        # ----------------------------------------------------
        # Convolution Block 1
        # ----------------------------------------------------

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=2
        )


        # ----------------------------------------------------
        # Convolution Block 2
        # ----------------------------------------------------

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.pool2 = nn.MaxPool2d(
            kernel_size=2
        )


        # ----------------------------------------------------
        # Convolution Block 3
        # ----------------------------------------------------

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.pool3 = nn.MaxPool2d(
            kernel_size=2
        )


        # ----------------------------------------------------
        # Convolution Block 4
        # ----------------------------------------------------

        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            padding=1
        )

        self.pool4 = nn.MaxPool2d(
            kernel_size=2
        )


        # ----------------------------------------------------
        # Calculate flattened size
        # ----------------------------------------------------

        self.flatten_size = (
            256
            * (IMAGE_SIZE // 16)
            * (IMAGE_SIZE // 16)
        )


        # ----------------------------------------------------
        # Fully Connected Layer
        # ----------------------------------------------------

        self.fc1 = nn.Linear(
            self.flatten_size,
            256
        )


        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        self.dropout = nn.Dropout(
            p=0.5
        )


        # ----------------------------------------------------
        # Output Layer
        # ----------------------------------------------------

        self.fc2 = nn.Linear(
            256,
            num_classes
        )


    def forward(self, x):

        # Block 1
        x = torch.relu(
            self.conv1(x)
        )

        x = self.pool1(x)


        # Block 2
        x = torch.relu(
            self.conv2(x)
        )

        x = self.pool2(x)


        # Block 3
        x = torch.relu(
            self.conv3(x)
        )

        x = self.pool3(x)


        # Block 4
        x = torch.relu(
            self.conv4(x)
        )

        x = self.pool4(x)


        # Flatten
        x = x.view(
            x.size(0),
            -1
        )


        # Fully connected
        x = torch.relu(
            self.fc1(x)
        )


        # Dropout
        x = self.dropout(x)


        # Output
        x = self.fc2(x)


        return x


# ============================================================
# 15. CREATE MODEL
# ============================================================

model = BaselineCNN(
    NUM_CLASSES
).to(DEVICE)


print("\nModel created successfully.")

print(model)


# ============================================================
# 16. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor
)


# ============================================================
# 17. OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 18. TRAINING HISTORY
# ============================================================

history = {

    "train_loss": [],

    "train_accuracy": [],

    "validation_loss": [],

    "validation_accuracy": []
}


# ============================================================
# 19. TRAINING LOOP
# ============================================================

print("\n" + "=" * 70)
print("STARTING MODEL TRAINING")
print("=" * 70)


best_validation_accuracy = 0.0

best_model_path = (
    MODELS_PATH /
    "baseline_cnn_best.pth"
)


for epoch in range(EPOCHS):


    # ========================================================
    # TRAINING
    # ========================================================

    model.train()


    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)


        # Clear gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(images)


        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()


        # Statistics
        running_loss += (
            loss.item()
            * images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()


    validation_loss_total = 0.0

    validation_correct = 0

    validation_total = 0


    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            validation_loss_total += (
                loss.item()
                * images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            validation_total += (
                labels.size(0)
            )


            validation_correct += (
                predicted == labels
            ).sum().item()


    validation_loss = (
        validation_loss_total
        / validation_total
    )


    validation_accuracy = (
        validation_correct
        / validation_total
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["validation_loss"].append(
        validation_loss
    )

    history["validation_accuracy"].append(
        validation_accuracy
    )


    # ========================================================
    # DISPLAY EPOCH RESULTS
    # ========================================================

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{validation_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{validation_accuracy * 100:.2f}%"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = (
            validation_accuracy
        )


        torch.save(

            {
                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    class_names,

                "image_size":
                    IMAGE_SIZE,

                "validation_accuracy":
                    validation_accuracy
            },

            best_model_path
        )


        print(
            "✓ Best model saved."
        )


# ============================================================
# 20. SAVE FINAL MODEL
# ============================================================

final_model_path = (
    MODELS_PATH /
    "baseline_cnn_final.pth"
)


torch.save(

    {
        "model_state_dict":
            model.state_dict(),

        "class_names":
            class_names,

        "image_size":
            IMAGE_SIZE
    },

    final_model_path
)


print("\nFinal model saved:")
print(final_model_path)


# ============================================================
# 21. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)


history_path = (
    OUTPUTS_PATH /
    "baseline_training_history.csv"
)


history_df.to_csv(
    history_path,
    index=False
)


print("\nTraining history saved:")
print(history_path)


# ============================================================
# 22. TRAINING ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    history["train_accuracy"],
    label="Training Accuracy"
)


plt.plot(
    history["validation_accuracy"],
    label="Validation Accuracy"
)


plt.title(
    "Baseline CNN - Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.legend()


plt.grid(True)


accuracy_path = (
    OUTPUTS_PATH /
    "baseline_cnn_accuracy.png"
)


plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 23. TRAINING LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    history["train_loss"],
    label="Training Loss"
)


plt.plot(
    history["validation_loss"],
    label="Validation Loss"
)


plt.title(
    "Baseline CNN - Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.legend()


plt.grid(True)


loss_path = (
    OUTPUTS_PATH /
    "baseline_cnn_loss.png"
)


plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 24. LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)


checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


print(
    f"\nBest validation accuracy: "
    f"{checkpoint['validation_accuracy'] * 100:.2f}%"
)


# ============================================================
# 25. TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING ON TEST DATA")
print("=" * 70)


test_correct = 0

test_total = 0

test_loss_total = 0.0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        test_loss_total += (
            loss.item()
            * images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        test_total += labels.size(0)


        test_correct += (
            predicted == labels
        ).sum().item()


test_loss = (
    test_loss_total /
    test_total
)


test_accuracy = (
    test_correct /
    test_total
)


print(
    f"\nTest Loss: "
    f"{test_loss:.4f}"
)


print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# 26. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("BASELINE CNN TRAINING COMPLETED")
print("=" * 70)


print(
    f"\nTotal training images   : "
    f"{len(train_df)}"
)


print(
    f"Total validation images : "
    f"{len(validation_df)}"
)


print(
    f"Total test images       : "
    f"{len(test_df)}"
)


print(
    f"\nBest validation accuracy: "
    f"{best_validation_accuracy * 100:.2f}%"
)


print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


print("\nGenerated files:")

print(
    f"1. {best_model_path}"
)

print(
    f"2. {final_model_path}"
)

print(
    f"3. {mapping_path}"
)

print(
    f"4. {history_path}"
)

print(
    f"5. {accuracy_path}"
)

print(
    f"6. {loss_path}"
)


print("\n" + "=" * 70)
print("PHASE 3 COMPLETED SUCCESSFULLY!")
print("=" * 70)