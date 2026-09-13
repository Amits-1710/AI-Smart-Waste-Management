"""
AI SMART WASTE MANAGEMENT & SEGREGATION SYSTEM

PHASE 6
Prediction + Waste Disposal Recommendation

Model:
MobileNetV3-Small Transfer Learning
"""

import os
import sys
import json

import torch
import torch.nn as nn

from torchvision import transforms, models
from PIL import Image


# ============================================================
# 1. CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

MODEL_PATH = "models/mobilenetv3_transfer_best.pth"

CLASS_MAPPING_PATH = "models/class_mapping.json"


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("AI SMART WASTE MANAGEMENT & SEGREGATION SYSTEM")
print("PHASE 6 - WASTE IMAGE PREDICTION")
print("=" * 70)

print(f"Device: {device}")


# ============================================================
# 3. LOAD CLASS MAPPING
# ============================================================

if not os.path.exists(CLASS_MAPPING_PATH):

    raise FileNotFoundError(
        f"Class mapping not found: {CLASS_MAPPING_PATH}"
    )


with open(
    CLASS_MAPPING_PATH,
    "r",
    encoding="utf-8"
) as f:

    class_mapping = json.load(f)


idx_to_class = {
    int(idx): class_name
    for idx, class_name in class_mapping.items()
}


NUM_CLASSES = len(idx_to_class)


print(f"Number of classes: {NUM_CLASSES}")


# ============================================================
# 4. LOAD MOBILENETV3 MODEL
# ============================================================

print("\nLoading MobileNetV3-Small...")

model = models.mobilenet_v3_small(
    weights=None
)


# Replace final classifier

in_features = model.classifier[-1].in_features

model.classifier[-1] = nn.Linear(
    in_features,
    NUM_CLASSES
)


# ============================================================
# 5. LOAD TRAINED WEIGHTS
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("✓ Trained model loaded successfully.")


# ============================================================
# 6. IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ============================================================
# 7. WASTE RECOMMENDATION DATABASE
# ============================================================

recommendations = {

    "battery": {

        "category": "Hazardous Waste",

        "bin": "Battery / E-waste Collection",

        "recommendation":
        "Do not mix batteries with regular household waste. "
        "Give batteries to an authorized battery or e-waste "
        "collection point."

    },

    "biological": {

        "category": "Organic Waste",

        "bin": "Wet / Organic Waste",

        "recommendation":
        "Place biological or biodegradable waste in the "
        "appropriate organic waste collection stream."

    },

    "brown-glass": {

        "category": "Recyclable Glass",

        "bin": "Glass Recycling",

        "recommendation":
        "Place brown glass in the appropriate glass recycling "
        "collection stream. Avoid mixing it with general waste."

    },

    "cardboard": {

        "category": "Recyclable Paper",

        "bin": "Paper / Cardboard Recycling",

        "recommendation":
        "Keep cardboard reasonably clean and dry and place it "
        "in the paper/cardboard recycling stream."

    },

    "clothes": {

        "category": "Textile Waste",

        "bin": "Textile Collection",

        "recommendation":
        "Donate reusable clothes where appropriate or use a "
        "textile collection/recycling facility."

    },

    "green-glass": {

        "category": "Recyclable Glass",

        "bin": "Glass Recycling",

        "recommendation":
        "Place green glass in the appropriate glass recycling "
        "collection stream."

    },

    "metal": {

        "category": "Recyclable Metal",

        "bin": "Metal Recycling",

        "recommendation":
        "Place clean metal items in the appropriate metal "
        "recycling collection stream."

    },

    "paper": {

        "category": "Recyclable Paper",

        "bin": "Paper Recycling",

        "recommendation":
        "Keep paper clean and dry and place it in the appropriate "
        "paper recycling collection stream."

    },

    "plastic": {

        "category": "Recyclable Plastic",

        "bin": "Plastic Recycling",

        "recommendation":
        "Place suitable clean plastic items in the appropriate "
        "plastic recycling collection stream."

    },

    "shoes": {

        "category": "Textile / Footwear Waste",

        "bin": "Textile or Footwear Collection",

        "recommendation":
        "Donate reusable footwear where appropriate or use a "
        "textile/footwear collection facility."

    },

    "trash": {

        "category": "General / Residual Waste",

        "bin": "General Waste",

        "recommendation":
        "Place residual waste in the appropriate general waste "
        "stream. Check local rules before disposal."

    },

    "white-glass": {

        "category": "Recyclable Glass",

        "bin": "Glass Recycling",

        "recommendation":
        "Place white/clear glass in the appropriate glass "
        "recycling collection stream."

    }

}


# ============================================================
# 8. PREDICTION FUNCTION
# ============================================================

def predict_waste(image_path):

    # Check image

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )


    # Open image

    image = Image.open(
        image_path
    ).convert("RGB")


    # Transform image

    image_tensor = transform(
        image
    )


    # Add batch dimension

    image_tensor = image_tensor.unsqueeze(
        0
    ).to(device)


    # Prediction

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        confidence, predicted_idx = torch.max(
            probabilities,
            dim=1
        )


    predicted_idx = predicted_idx.item()

    confidence = confidence.item()


    predicted_class = idx_to_class[
        predicted_idx
    ]


    # Recommendation

    info = recommendations.get(
        predicted_class,
        {

            "category": "Unknown",

            "bin": "Check local waste guidelines",

            "recommendation":
            "Please verify the disposal method "
            "with local waste-management guidelines."

        }
    )


    return {

        "class": predicted_class,

        "confidence": confidence,

        "category": info["category"],

        "bin": info["bin"],

        "recommendation": info["recommendation"]

    }


# ============================================================
# 9. DISPLAY RESULT
# ============================================================

def display_result(
    image_path,
    result
):

    print("\n")
    print("=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)

    print(
        f"Image       : {image_path}"
    )

    print(
        f"Prediction  : {result['class']}"
    )

    print(
        f"Confidence  : "
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        f"Waste Type  : {result['category']}"
    )

    print(
        f"Recommended : {result['bin']}"
    )

    print("\nDisposal Guidance:")

    print(
        result["recommendation"]
    )

    print("=" * 70)


# ============================================================
# 10. COMMAND-LINE INPUT
# ============================================================

if len(sys.argv) < 2:

    print("\nUsage:")

    print(
        "python notebooks/06_predict.py "
        "<image_path>"
    )

    print("\nExample:")

    print(
        'python notebooks/06_predict.py '
        '"data/test/sample.jpg"'
    )

    sys.exit(1)


image_path = sys.argv[1]


# ============================================================
# 11. RUN PREDICTION
# ============================================================

try:

    result = predict_waste(
        image_path
    )

    display_result(
        image_path,
        result
    )

except Exception as e:

    print("\nERROR:")
    print(e)