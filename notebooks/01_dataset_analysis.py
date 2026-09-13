from pathlib import Path
from PIL import Image
import pandas as pd


# ==========================================
# 1. DATASET PATH
# ==========================================

DATASET_PATH = Path("./data/raw/garbage-classification")


# ==========================================
# 2. CHECK DATASET PATH
# ==========================================

if not DATASET_PATH.exists():
    print("❌ Dataset folder not found!")
    print("Expected path:", DATASET_PATH)
    exit()

print("✅ Dataset found!")
print("Path:", DATASET_PATH)


# ==========================================
# 3. FIND CLASSES
# ==========================================

classes = [
    folder.name
    for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
]

classes.sort()

print("\n================================")
print("WASTE CLASSES")
print("================================")

for i, class_name in enumerate(classes, start=1):
    print(f"{i}. {class_name}")


# ==========================================
# 4. COLLECT IMAGE INFORMATION
# ==========================================

records = []

valid_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

corrupted_images = []


for class_name in classes:

    class_path = DATASET_PATH / class_name

    for image_path in class_path.iterdir():

        if image_path.suffix.lower() not in valid_extensions:
            continue

        try:

            with Image.open(image_path) as img:

                width, height = img.size

                records.append({
                    "class": class_name,
                    "filename": image_path.name,
                    "width": width,
                    "height": height,
                    "path": str(image_path)
                })

        except Exception:

            corrupted_images.append(
                str(image_path)
            )


# ==========================================
# 5. CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(records)


# ==========================================
# 6. BASIC INFORMATION
# ==========================================

print("\n================================")
print("DATASET SUMMARY")
print("================================")

print("Total images:", len(df))
print("Total classes:", len(classes))
print("Corrupted images:", len(corrupted_images))


# ==========================================
# 7. CLASS DISTRIBUTION
# ==========================================

print("\n================================")
print("IMAGES PER CLASS")
print("================================")

class_counts = df["class"].value_counts()

print(class_counts)


# ==========================================
# 8. IMAGE DIMENSIONS
# ==========================================

print("\n================================")
print("IMAGE DIMENSIONS")
print("================================")

print(
    df[["width", "height"]].describe()
)


# ==========================================
# 9. UNIQUE IMAGE SIZES
# ==========================================

unique_sizes = (
    df[["width", "height"]]
    .drop_duplicates()
)

print(
    "\nUnique image dimensions:",
    len(unique_sizes)
)


# ==========================================
# 10. SAVE DATASET INFORMATION
# ==========================================

output_path = Path(
    "./data/dataset_metadata.csv"
)

df.to_csv(
    output_path,
    index=False
)

print(
    "\n✅ Dataset metadata saved to:",
    output_path
)


# ==========================================
# 11. CORRUPTED IMAGES
# ==========================================

if corrupted_images:

    print("\n⚠️ Corrupted images:")

    for image in corrupted_images:
        print(image)

else:

    print(
        "\n✅ No corrupted images detected."
    )


print("\n================================")
print("EDA part 1 COMPLETED")
print("================================")

# ==========================================
# EDA PART 2 — VISUALIZATION
# ==========================================

print("\nStarting EDA Part 2...")

# ==========================================
# EDA PART 2 — VISUALIZATION
# ==========================================

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

graph_folder = Path("outputs/graphs")
graph_folder.mkdir(parents=True, exist_ok=True)

print(f"Graph folder ready: {graph_folder}")


# ==========================================
# 1. CLASS DISTRIBUTION BAR CHART
# ==========================================

print("\nCreating class distribution graph...")

plt.figure(figsize=(12, 7))

class_counts.sort_values(ascending=False).plot(
    kind="bar"
)

plt.title("Waste Dataset - Images per Class")
plt.xlabel("Waste Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

class_distribution_path = graph_folder / "class_distribution.png"

plt.savefig(class_distribution_path, dpi=300)
plt.show()
plt.close()

print(f"Saved: {class_distribution_path}")


# ==========================================
# 2. IMAGE WIDTH DISTRIBUTION
# ==========================================

print("\nCreating image width distribution graph...")

plt.figure(figsize=(10, 6))

sns.histplot(
    df["width"],
    bins=30,
    kde=True
)

plt.title("Distribution of Image Widths")
plt.xlabel("Width (pixels)")
plt.ylabel("Number of Images")
plt.tight_layout()

width_path = graph_folder / "width_distribution.png"

plt.savefig(width_path, dpi=300)
plt.show()
plt.close()

print(f"Saved: {width_path}")


# ==========================================
# 3. IMAGE HEIGHT DISTRIBUTION
# ==========================================

print("\nCreating image height distribution graph...")

plt.figure(figsize=(10, 6))

sns.histplot(
    df["height"],
    bins=30,
    kde=True
)

plt.title("Distribution of Image Heights")
plt.xlabel("Height (pixels)")
plt.ylabel("Number of Images")
plt.tight_layout()

height_path = graph_folder / "height_distribution.png"

plt.savefig(height_path, dpi=300)
plt.show()
plt.close()

print(f"Saved: {height_path}")


# ==========================================
# 4. IMAGE WIDTH VS HEIGHT
# ==========================================

print("\nCreating width vs height graph...")

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=df,
    x="width",
    y="height",
    alpha=0.4
)

plt.title("Image Width vs Height")
plt.xlabel("Width (pixels)")
plt.ylabel("Height (pixels)")
plt.tight_layout()

dimension_path = graph_folder / "width_vs_height.png"

plt.savefig(dimension_path, dpi=300)
plt.show()
plt.close()

print(f"Saved: {dimension_path}")


# ==========================================
# 5. SAMPLE IMAGES FROM EACH CLASS
# ==========================================

# ==========================================
# 5. SAMPLE IMAGES FROM EACH CLASS
# ==========================================

print("\nCreating sample image visualization...")

from PIL import Image

classes = sorted(df["class"].unique())

fig, axes = plt.subplots(
    nrows=4,
    ncols=3,
    figsize=(15, 18)
)

axes = axes.flatten()

for i, class_name in enumerate(classes):

    # Current class folder
    class_folder = DATASET_PATH / class_name

    # Get image files
    image_files = []

    for extension in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_files.extend(class_folder.glob(extension))

    # Check if images exist
    if len(image_files) == 0:

        axes[i].text(
            0.5,
            0.5,
            "No image found",
            ha="center",
            va="center"
        )

        axes[i].set_title(class_name)
        axes[i].axis("off")

        continue

    # Select one image
    sample_image = image_files[0]

    try:

        image = Image.open(sample_image)

        axes[i].imshow(image)
        axes[i].set_title(
            class_name,
            fontsize=12
        )

        axes[i].axis("off")

    except Exception as e:

        axes[i].text(
            0.5,
            0.5,
            "Image could not be loaded",
            ha="center",
            va="center"
        )

        axes[i].set_title(class_name)
        axes[i].axis("off")


# Hide unused axes
for j in range(len(classes), len(axes)):
    axes[j].axis("off")


plt.suptitle(
    "Sample Images from Each Waste Class",
    fontsize=16
)

plt.tight_layout()

sample_images_path = graph_folder / "sample_images.png"

plt.savefig(
    sample_images_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

print(f"Saved: {sample_images_path}")
# ==========================================
# 6. DATASET IMBALANCE ANALYSIS
# ==========================================

print("\nDataset imbalance analysis...")

largest_class = class_counts.idxmax()
smallest_class = class_counts.idxmin()

largest_count = class_counts.max()
smallest_count = class_counts.min()

imbalance_ratio = largest_count / smallest_count

print(f"Largest class  : {largest_class}")
print(f"Images         : {largest_count}")

print(f"\nSmallest class : {smallest_class}")
print(f"Images         : {smallest_count}")

print(f"\nImbalance ratio: {imbalance_ratio:.2f}")


# ==========================================
# 7. TOP AND BOTTOM CLASSES
# ==========================================

print("\nTop 5 classes by image count:")

print(
    class_counts
    .sort_values(ascending=False)
    .head(5)
)


print("\nBottom 5 classes by image count:")

print(
    class_counts
    .sort_values(ascending=True)
    .head(5)
)


# ==========================================
# EDA PART 2 COMPLETED
# ==========================================

print("\n================================")
print("EDA PART 2 COMPLETED")
print("================================")

# ==========================================
# EDA PART 3 — ML READINESS ANALYSIS
# ==========================================

print("\n================================")
print("Starting EDA Part 3...")
print("================================")


# ==========================================
# 1. CLASS PERCENTAGE
# ==========================================

print("\nCalculating class percentages...")

class_percentage = (
    df["class"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nClass percentage distribution:")

print(class_percentage)


# ==========================================
# 2. ASPECT RATIO
# ==========================================

print("\nCalculating image aspect ratio...")

df["aspect_ratio"] = df["width"] / df["height"]

print("\nAspect ratio statistics:")

print(
    df["aspect_ratio"].describe()
)


# ==========================================
# 3. ASPECT RATIO VISUALIZATION
# ==========================================

print("\nCreating aspect ratio distribution graph...")

plt.figure(figsize=(10, 6))

sns.histplot(
    df["aspect_ratio"],
    bins=40,
    kde=True
)

plt.title("Distribution of Image Aspect Ratios")
plt.xlabel("Aspect Ratio (Width / Height)")
plt.ylabel("Number of Images")

plt.tight_layout()

aspect_ratio_path = (
    graph_folder / "aspect_ratio_distribution.png"
)

plt.savefig(
    aspect_ratio_path,
    dpi=300
)

plt.show()
plt.close()

print(f"Saved: {aspect_ratio_path}")


# ==========================================
# 4. VERY SMALL IMAGES
# ==========================================

print("\nChecking for very small images...")

small_images = df[
    (df["width"] < 100) |
    (df["height"] < 100)
]

print(
    f"Images smaller than 100 pixels in width "
    f"or height: {len(small_images)}"
)


# ==========================================
# 5. LARGE IMAGES
# ==========================================

print("\nChecking for large images...")

large_images = df[
    (df["width"] > 800) |
    (df["height"] > 800)
]

print(
    f"Images larger than 800 pixels in width "
    f"or height: {len(large_images)}"
)


# ==========================================
# 6. EXTREME ASPECT RATIO IMAGES
# ==========================================

print("\nChecking extreme aspect ratios...")

extreme_aspect_images = df[
    (df["aspect_ratio"] < 0.5) |
    (df["aspect_ratio"] > 2.0)
]

print(
    f"Images with extreme aspect ratio: "
    f"{len(extreme_aspect_images)}"
)


# ==========================================
# 7. RESOLUTION STATISTICS
# ==========================================

print("\nResolution statistics:")

print("\nWidth:")
print(df["width"].describe())

print("\nHeight:")
print(df["height"].describe())


# ==========================================
# 8. SAVE UPDATED METADATA
# ==========================================

print("\nSaving updated metadata...")

updated_metadata_path = Path(
    "data/dataset_metadata.csv"
)

df.to_csv(
    updated_metadata_path,
    index=False
)

print(
    f"Updated metadata saved: "
    f"{updated_metadata_path}"
)


# ==========================================
# 9. ML PREPROCESSING OBSERVATIONS
# ==========================================

print("\n================================")
print("ML PREPROCESSING OBSERVATIONS")
print("================================")

print("\n1. Images have different dimensions.")
print("   → Images should be resized to a fixed size.")

print("\n2. Dataset has class imbalance.")
print("   → Class weights or augmentation should be considered.")

print("\n3. Image aspect ratios vary.")
print("   → Resize strategy should preserve image content.")

print("\n4. No corrupted images were detected.")
print("   → Dataset is suitable for preprocessing.")

print("\n5. Data augmentation can improve model generalization.")
print("   → Rotation, flipping and zoom can be considered.")


# ==========================================
# EDA PART 3 COMPLETED
# ==========================================

print("\n================================")
print("EDA PART 3 COMPLETED")
print("================================")
