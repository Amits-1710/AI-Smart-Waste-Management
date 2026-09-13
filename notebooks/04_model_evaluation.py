# ==============================================================
# AI SMART WASTE MANAGEMENT & SEGREGATION SYSTEM
# PHASE 4 - MODEL EVALUATION & ANALYSIS
# ==============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# ==============================================================
# 1. CONFIGURATION
# ==============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

TEST_CSV = os.path.join(
    DATA_DIR,
    "test.csv"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "baseline_cnn_best.pth"
)

CLASS_MAPPING_PATH = os.path.join(
    MODEL_DIR,
    "class_mapping.json"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

IMAGE_SIZE = 224
BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ==============================================================
# 2. HEADER
# ==============================================================

print("=" * 70)

print(
    "AI SMART WASTE MANAGEMENT & SEGREGATION SYSTEM"
)

print(
    "PHASE 4 - MODEL EVALUATION & ANALYSIS"
)

print("=" * 70)


print("\nDevice:")
print(DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    print(
        "GPU not available - using CPU"
    )


# ==============================================================
# 3. LOAD CLASS MAPPING
# ==============================================================

print("\n" + "=" * 70)

print(
    "LOADING CLASS MAPPING"
)

print("=" * 70)


with open(
    CLASS_MAPPING_PATH,
    "r"
) as f:

    class_mapping = json.load(f)


# JSON keys are strings
class_mapping = {
    int(k): v
    for k, v in class_mapping.items()
}


NUM_CLASSES = len(
    class_mapping
)


print(
    "\nNumber of classes:",
    NUM_CLASSES
)


for idx, class_name in class_mapping.items():

    print(
        f"{idx}: {class_name}"
    )


# Create reverse mapping
# class name -> numeric index

class_to_idx = {
    class_name: idx
    for idx, class_name
    in class_mapping.items()
}


# ==============================================================
# 4. LOAD TEST DATA
# ==============================================================

print("\n" + "=" * 70)

print(
    "LOADING TEST DATA"
)

print("=" * 70)


test_df = pd.read_csv(
    TEST_CSV
)


print(
    "\nTest samples:",
    len(test_df)
)


print(
    "\nTest CSV columns:"
)

print(
    test_df.columns.tolist()
)


# Check required columns

required_columns = [
    "image_path",
    "class"
]

for column in required_columns:

    if column not in test_df.columns:

        raise ValueError(
            f"Required column '{column}' "
            f"not found in test.csv"
        )


print(
    "\nTest class distribution:"
)


for class_name, count in (
    test_df["class"]
    .value_counts()
    .sort_index()
    .items()
):

    print(
        f"{class_name:15s}: {count}"
    )


# ==============================================================
# 5. CREATE NUMERIC LABELS
# ==============================================================

test_df["label"] = (
    test_df["class"]
    .map(class_to_idx)
)


# Check for unknown classes

if test_df["label"].isna().any():

    unknown_classes = (
        test_df.loc[
            test_df["label"].isna(),
            "class"
        ]
        .unique()
    )

    raise ValueError(
        "Unknown classes found in test.csv: "
        + str(unknown_classes)
    )


test_df["label"] = (
    test_df["label"]
    .astype(int)
)


print(
    "\nNumeric labels created successfully."
)


# ==============================================================
# 6. TEST DATASET
# ==============================================================

class WasteDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.dataframe = (
            dataframe
            .reset_index(drop=True)
        )

        self.transform = transform


    def __len__(self):

        return len(
            self.dataframe
        )


    def __getitem__(
        self,
        index
    ):

        image_path = (
            self.dataframe
            .iloc[index]["image_path"]
        )

        label = int(
            self.dataframe
            .iloc[index]["label"]
        )


        image = Image.open(
            image_path
        ).convert("RGB")


        if self.transform:

            image = self.transform(
                image
            )


        return image, label


# ==============================================================
# 7. IMAGE TRANSFORMS
# ==============================================================

print("\n" + "=" * 70)

print(
    "CREATING IMAGE TRANSFORMS"
)

print("=" * 70)


test_transform = transforms.Compose([

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


# ==============================================================
# 8. CREATE TEST DATASET
# ==============================================================

test_dataset = WasteDataset(

    test_df,

    transform=test_transform

)


print(
    "\nTest dataset created successfully."
)

print(
    "Test dataset:",
    len(test_dataset)
)


# ==============================================================
# 9. CREATE TEST DATALOADER
# ==============================================================

print("\n" + "=" * 70)

print(
    "CREATING TEST DATALOADER"
)

print("=" * 70)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)


print(
    "\nTest DataLoader created successfully."
)


# ==============================================================
# 10. DEFINE BASELINE CNN
# ==============================================================

print("\n" + "=" * 70)

print(
    "BUILDING BASELINE CNN"
)

print("=" * 70)


class BaselineCNN(nn.Module):

    def __init__(
        self,
        num_classes
    ):

        super().__init__()


        # Convolution block 1

        self.conv1 = nn.Conv2d(
            3,
            32,
            kernel_size=3,
            padding=1
        )

        self.pool1 = nn.MaxPool2d(
            2,
            2
        )


        # Convolution block 2

        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1
        )

        self.pool2 = nn.MaxPool2d(
            2,
            2
        )


        # Convolution block 3

        self.conv3 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            padding=1
        )

        self.pool3 = nn.MaxPool2d(
            2,
            2
        )


        # Convolution block 4

        self.conv4 = nn.Conv2d(
            128,
            256,
            kernel_size=3,
            padding=1
        )

        self.pool4 = nn.MaxPool2d(
            2,
            2
        )


        # Fully connected layer

        self.fc1 = nn.Linear(
            256 * 14 * 14,
            256
        )


        self.dropout = nn.Dropout(
            0.5
        )


        self.fc2 = nn.Linear(
            256,
            num_classes
        )


    def forward(
        self,
        x
    ):

        x = torch.relu(
            self.conv1(x)
        )

        x = self.pool1(x)


        x = torch.relu(
            self.conv2(x)
        )

        x = self.pool2(x)


        x = torch.relu(
            self.conv3(x)
        )

        x = self.pool3(x)


        x = torch.relu(
            self.conv4(x)
        )

        x = self.pool4(x)


        x = x.view(
            x.size(0),
            -1
        )


        x = torch.relu(
            self.fc1(x)
        )


        x = self.dropout(x)


        x = self.fc2(x)


        return x


# ==============================================================
# 11. LOAD TRAINED MODEL
# ==============================================================

print("\n" + "=" * 70)

print(
    "LOADING TRAINED MODEL"
)

print("=" * 70)


model = BaselineCNN(
    NUM_CLASSES
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# Handle different checkpoint formats

if (
    isinstance(
        checkpoint,
        dict
    )
    and
    "model_state_dict"
    in checkpoint
):

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(
    DEVICE
)


model.eval()


print(
    "\nModel loaded successfully."
)


print(
    "Model path:"
)

print(
    MODEL_PATH
)


# ==============================================================
# 12. GENERATE PREDICTIONS
# ==============================================================

print("\n" + "=" * 70)

print(
    "GENERATING PREDICTIONS"
)

print("=" * 70)


all_labels = []

all_predictions = []

all_probabilities = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        outputs = model(
            images
        )


        probabilities = (
            torch.softmax(
                outputs,
                dim=1
            )
        )


        predictions = (
            torch.argmax(
                probabilities,
                dim=1
            )
        )


        all_labels.extend(
            labels
            .cpu()
            .numpy()
        )


        all_predictions.extend(
            predictions
            .cpu()
            .numpy()
        )


        all_probabilities.extend(
            probabilities
            .cpu()
            .numpy()
        )


all_labels = np.array(
    all_labels
)

all_predictions = np.array(
    all_predictions
)

all_probabilities = np.array(
    all_probabilities
)


print(
    "\nPredictions generated successfully."
)


print(
    "Total predictions:",
    len(all_predictions)
)


# ==============================================================
# 13. OVERALL METRICS
# ==============================================================

print("\n" + "=" * 70)

print(
    "OVERALL MODEL PERFORMANCE"
)

print("=" * 70)


accuracy = accuracy_score(
    all_labels,
    all_predictions
)


precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


print(
    f"\nAccuracy  : "
    f"{accuracy * 100:.2f}%"
)


print(
    f"Precision : "
    f"{precision * 100:.2f}%"
)


print(
    f"Recall    : "
    f"{recall * 100:.2f}%"
)


print(
    f"F1 Score  : "
    f"{f1 * 100:.2f}%"
)


# ==============================================================
# 14. CLASSIFICATION REPORT
# ==============================================================

print("\n" + "=" * 70)

print(
    "CLASSIFICATION REPORT"
)

print("=" * 70)


class_names = [
    class_mapping[i]
    for i in range(
        NUM_CLASSES
    )
]


report = classification_report(

    all_labels,

    all_predictions,

    labels=list(
        range(NUM_CLASSES)
    ),

    target_names=class_names,

    zero_division=0

)


print("\n")

print(
    report
)


# Save classification report

report_dict = classification_report(

    all_labels,

    all_predictions,

    labels=list(
        range(NUM_CLASSES)
    ),

    target_names=class_names,

    output_dict=True,

    zero_division=0

)


report_df = (
    pd.DataFrame(
        report_dict
    )
    .transpose()
)


report_path = os.path.join(

    OUTPUT_DIR,

    "classification_report.csv"

)


report_df.to_csv(
    report_path
)


print(
    "Classification report saved:"
)

print(
    report_path
)


# ==============================================================
# 15. CONFUSION MATRIX
# ==============================================================

print("\n" + "=" * 70)

print(
    "CREATING CONFUSION MATRIX"
)

print("=" * 70)


cm = confusion_matrix(

    all_labels,

    all_predictions,

    labels=list(
        range(NUM_CLASSES)
    )

)


plt.figure(
    figsize=(12, 10)
)


plt.imshow(cm)


plt.title(
    "Baseline CNN - Confusion Matrix"
)


plt.colorbar()


plt.xticks(

    range(NUM_CLASSES),

    class_names,

    rotation=45,

    ha="right"

)


plt.yticks(

    range(NUM_CLASSES),

    class_names

)


plt.xlabel(
    "Predicted Class"
)


plt.ylabel(
    "Actual Class"
)


# Add values inside cells

for i in range(
    NUM_CLASSES
):

    for j in range(
        NUM_CLASSES
    ):

        plt.text(

            j,

            i,

            cm[i, j],

            ha="center",

            va="center"

        )


plt.tight_layout()


cm_path = os.path.join(

    OUTPUT_DIR,

    "confusion_matrix.png"

)


plt.savefig(

    cm_path,

    dpi=200,

    bbox_inches="tight"

)


plt.close()


print(
    "\nConfusion matrix saved:"
)

print(
    cm_path
)


# ==============================================================
# 16. CLASS-WISE F1 SCORE
# ==============================================================

print("\n" + "=" * 70)

print(
    "CREATING CLASS-WISE F1 SCORE"
)

print("=" * 70)


class_f1 = []


for class_name in class_names:

    class_f1.append(

        report_dict[
            class_name
        ][
            "f1-score"
        ]

    )


plt.figure(
    figsize=(12, 7)
)


bars = plt.bar(

    class_names,

    class_f1

)


plt.title(
    "Class-wise F1 Score"
)


plt.xlabel(
    "Waste Class"
)


plt.ylabel(
    "F1 Score"
)


plt.ylim(
    0,
    1
)


plt.xticks(

    rotation=45,

    ha="right"

)


for bar, value in zip(

    bars,

    class_f1

):

    plt.text(

        bar.get_x()
        + bar.get_width() / 2,

        value + 0.02,

        f"{value:.2f}",

        ha="center"

    )


plt.tight_layout()


f1_path = os.path.join(

    OUTPUT_DIR,

    "classwise_f1_score.png"

)


plt.savefig(

    f1_path,

    dpi=200,

    bbox_inches="tight"

)


plt.close()


print(
    "\nClass-wise F1 graph saved:"
)

print(
    f1_path
)


# ==============================================================
# 17. CONFIDENCE ANALYSIS
# ==============================================================

print("\n" + "=" * 70)

print(
    "CONFIDENCE ANALYSIS"
)

print("=" * 70)


max_confidence = np.max(

    all_probabilities,

    axis=1

)


average_confidence = (
    np.mean(
        max_confidence
    )
)


print(

    f"\nAverage prediction confidence: "

    f"{average_confidence * 100:.2f}%"

)


print(

    f"Minimum confidence: "

    f"{np.min(max_confidence) * 100:.2f}%"

)


print(

    f"Maximum confidence: "

    f"{np.max(max_confidence) * 100:.2f}%"

)


# ==============================================================
# 18. SAVE TEST PREDICTIONS
# ==============================================================

print("\n" + "=" * 70)

print(
    "SAVING TEST PREDICTIONS"
)

print("=" * 70)


predictions_df = test_df.copy()


predictions_df[
    "actual_class"
] = [

    class_mapping[
        int(x)
    ]

    for x in all_labels

]


predictions_df[
    "predicted_class"
] = [

    class_mapping[
        int(x)
    ]

    for x in all_predictions

]


predictions_df[
    "confidence"
] = (
    max_confidence
)


predictions_df[
    "correct"
] = (

    all_labels
    ==
    all_predictions

)


predictions_path = os.path.join(

    OUTPUT_DIR,

    "test_predictions.csv"

)


predictions_df.to_csv(

    predictions_path,

    index=False

)


print(
    "\nTest predictions saved:"
)

print(
    predictions_path
)


# ==============================================================
# 19. MISCLASSIFICATION ANALYSIS
# ==============================================================

print("\n" + "=" * 70)

print(
    "MISCLASSIFICATION ANALYSIS"
)

print("=" * 70)


incorrect = predictions_df[
    predictions_df[
        "correct"
    ] == False
]


correct = predictions_df[
    predictions_df[
        "correct"
    ] == True
]


print(

    f"\nCorrect predictions: "
    f"{len(correct)}"

)


print(

    f"Incorrect predictions: "
    f"{len(incorrect)}"

)


# Find common mistakes

mistakes = (

    incorrect

    .groupby(

        [
            "actual_class",
            "predicted_class"
        ]

    )

    .size()

    .reset_index(
        name="count"
    )

    .sort_values(

        "count",

        ascending=False

    )

)


print(
    "\nTop 15 misclassification pairs:"
)


if len(mistakes) > 0:

    print(

        mistakes
        .head(15)
        .to_string(
            index=False
        )

    )

else:

    print(
        "No misclassifications found."
    )


mistakes_path = os.path.join(

    OUTPUT_DIR,

    "misclassification_pairs.csv"

)


mistakes.to_csv(

    mistakes_path,

    index=False

)


print(
    "\nMisclassification analysis saved:"
)

print(
    mistakes_path
)


# ==============================================================
# 20. SAVE SUMMARY METRICS
# ==============================================================

summary_metrics = pd.DataFrame({

    "Metric": [

        "Accuracy",

        "Weighted Precision",

        "Weighted Recall",

        "Weighted F1 Score",

        "Average Confidence",

        "Correct Predictions",

        "Incorrect Predictions",

        "Total Test Samples"

    ],


    "Value": [

        accuracy,

        precision,

        recall,

        f1,

        average_confidence,

        len(correct),

        len(incorrect),

        len(test_df)

    ]

})


summary_path = os.path.join(

    OUTPUT_DIR,

    "evaluation_summary.csv"

)


summary_metrics.to_csv(

    summary_path,

    index=False

)


print(
    "\nEvaluation summary saved:"
)

print(
    summary_path
)


# ==============================================================
# 21. FINAL SUMMARY
# ==============================================================

print("\n" + "=" * 70)

print(
    "PHASE 4 EVALUATION COMPLETED"
)

print("=" * 70)


print(
    "\nFinal Results:"
)


print(

    f"Accuracy  : "
    f"{accuracy * 100:.2f}%"

)


print(

    f"Precision : "
    f"{precision * 100:.2f}%"

)


print(

    f"Recall    : "
    f"{recall * 100:.2f}%"

)


print(

    f"F1 Score  : "
    f"{f1 * 100:.2f}%"

)


print(

    f"Average Confidence : "
    f"{average_confidence * 100:.2f}%"

)


print(
    "\nGenerated files:"
)


print(
    "1.",
    report_path
)


print(
    "2.",
    cm_path
)


print(
    "3.",
    f1_path
)


print(
    "4.",
    predictions_path
)


print(
    "5.",
    mistakes_path
)


print(
    "6.",
    summary_path
)


print("\n" + "=" * 70)

print(
    "PHASE 4 COMPLETED SUCCESSFULLY!"
)

print("=" * 70)