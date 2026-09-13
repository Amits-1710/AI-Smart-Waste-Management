# ============================================================
# AI SMART WASTE MANAGEMENT & SEGREGATION SYSTEM
# File: 02_data_preprocessing.py
# Purpose: Dataset preprocessing and train/validation/test split
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Project root = AI-Smart-Waste-Management
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw dataset path
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "garbage-classification"

# Processed data path
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

# Create processed directory if it doesn't exist
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# 3. CHECK DATASET PATH
# ============================================================

print("=" * 70)
print("AI SMART WASTE MANAGEMENT & SEGREGATION SYSTEM")
print("DATA PREPROCESSING")
print("=" * 70)

print("\nChecking dataset path...")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found!\n"
        f"Expected location:\n{DATASET_PATH}"
    )

print(f"Dataset found at:\n{DATASET_PATH}")


# ============================================================
# 4. FIND CLASS FOLDERS
# ============================================================

classes = sorted([
    folder.name
    for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
])

print(f"\nNumber of classes: {len(classes)}")

print("\nClasses:")
for i, class_name in enumerate(classes, start=1):
    print(f"{i:2d}. {class_name}")


# ============================================================
# 5. SCAN DATASET
# ============================================================

print("\n" + "=" * 70)
print("SCANNING DATASET")
print("=" * 70)

image_paths = []
labels = []

for class_name in classes:

    class_folder = DATASET_PATH / class_name

    # IMPORTANT:
    # iterdir() scans only the files directly inside the class folder.
    # It does NOT scan nested subfolders.
    #
    # This prevents folders accidentally placed inside "trash"
    # from being counted as trash images.

    for image_path in class_folder.iterdir():

        if (
            image_path.is_file()
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ):

            image_paths.append(str(image_path.resolve()))
            labels.append(class_name)


# ============================================================
# 6. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame({
    "image_path": image_paths,
    "class": labels
})


# ============================================================
# 7. BASIC DATASET INFORMATION
# ============================================================

print(f"\nTotal images found: {len(df)}")

print(f"Number of classes: {df['class'].nunique()}")

print("\nClass distribution:")
print(df["class"].value_counts().sort_index())


# ============================================================
# 8. CHECK DUPLICATE IMAGE PATHS
# ============================================================

duplicate_paths = df["image_path"].duplicated().sum()

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

print(f"Duplicate image paths: {duplicate_paths}")

if duplicate_paths == 0:
    print("No duplicate image paths found.")
else:
    print("WARNING: Duplicate image paths found!")


# ============================================================
# 9. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CREATING DATASET SPLITS")
print("=" * 70)

# ------------------------------------------------------------
# First split:
# 70% Training
# 30% Temporary
# ------------------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["class"]
)


# ------------------------------------------------------------
# Second split:
# Temporary 30% -> 15% Validation + 15% Test
# ------------------------------------------------------------

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["class"]
)


# Reset indexes
train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ============================================================
# 10. PRINT SPLIT SIZES
# ============================================================

print("\nDataset split:")

print(f"Training images:   {len(train_df)}")
print(f"Validation images: {len(validation_df)}")
print(f"Test images:       {len(test_df)}")

print("\nExpected ratio:")
print("Training   = 70%")
print("Validation = 15%")
print("Test       = 15%")


# ============================================================
# 11. TRAINING DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING SET DISTRIBUTION")
print("=" * 70)

print(train_df["class"].value_counts().sort_index())


# ============================================================
# 12. VALIDATION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION SET DISTRIBUTION")
print("=" * 70)

print(validation_df["class"].value_counts().sort_index())


# ============================================================
# 13. TEST DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TEST SET DISTRIBUTION")
print("=" * 70)

print(test_df["class"].value_counts().sort_index())


# ============================================================
# 14. SAVE CSV FILES
# ============================================================

print("\n" + "=" * 70)
print("SAVING PROCESSED DATA")
print("=" * 70)

train_path = PROCESSED_PATH / "train.csv"
validation_path = PROCESSED_PATH / "validation.csv"
test_path = PROCESSED_PATH / "test.csv"

train_df.to_csv(train_path, index=False)
validation_df.to_csv(validation_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"\nTraining CSV saved at:")
print(train_path)

print(f"\nValidation CSV saved at:")
print(validation_path)

print(f"\nTest CSV saved at:")
print(test_path)


# ============================================================
# 15. CALCULATE CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING CLASS WEIGHTS")
print("=" * 70)

class_labels = np.array(sorted(train_df["class"].unique()))

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=class_labels,
    y=train_df["class"]
)

class_weights = dict(
    zip(class_labels, class_weights_array)
)


# ============================================================
# 16. DISPLAY CLASS WEIGHTS
# ============================================================

print("\nClass weights:")

for class_name, weight in class_weights.items():
    print(f"{class_name:15s} : {weight:.4f}")


# ============================================================
# 17. SAVE CLASS WEIGHTS
# ============================================================

class_weights_df = pd.DataFrame({
    "class": list(class_weights.keys()),
    "weight": list(class_weights.values())
})

class_weights_path = PROCESSED_PATH / "class_weights.csv"

class_weights_df.to_csv(
    class_weights_path,
    index=False
)

print(f"\nClass weights saved at:")
print(class_weights_path)


# ============================================================
# 18. DATA LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("DATA LEAKAGE CHECK")
print("=" * 70)

train_paths = set(train_df["image_path"])
validation_paths = set(validation_df["image_path"])
test_paths = set(test_df["image_path"])


train_validation_overlap = train_paths.intersection(
    validation_paths
)

train_test_overlap = train_paths.intersection(
    test_paths
)

validation_test_overlap = validation_paths.intersection(
    test_paths
)


print(
    f"\nTrain ∩ Validation: "
    f"{len(train_validation_overlap)}"
)

print(
    f"Train ∩ Test: "
    f"{len(train_test_overlap)}"
)

print(
    f"Validation ∩ Test: "
    f"{len(validation_test_overlap)}"
)


if (
    len(train_validation_overlap) == 0
    and len(train_test_overlap) == 0
    and len(validation_test_overlap) == 0
):

    print("\nNo data leakage detected!")

else:

    print("\nWARNING: Possible data leakage detected!")


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING SUMMARY")
print("=" * 70)

print(f"\nTotal images:       {len(df)}")
print(f"Number of classes:  {len(classes)}")
print(f"Training images:    {len(train_df)}")
print(f"Validation images:  {len(validation_df)}")
print(f"Test images:        {len(test_df)}")

print("\nFiles generated:")

print("1. train.csv")
print("2. validation.csv")
print("3. test.csv")
print("4. class_weights.csv")

print("\n" + "=" * 70)
print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 70)