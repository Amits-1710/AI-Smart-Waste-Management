"""
AI SMART WASTE MANAGEMENT & SEGREGATION SYSTEM
PHASE 8.4 - STREAMLIT + RAG INTEGRATION
"""

import sys
import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# =========================================================
# PATH SETUP
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAG_PATH = PROJECT_ROOT / "rag"

if str(RAG_PATH) not in sys.path:
    sys.path.insert(0, str(RAG_PATH))


from rag.rag_assistant import ask_assistant


# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Smart Waste Management",
    page_icon="♻️",
    layout="wide"
)


# =========================================================
# MODEL CONFIGURATION
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mobilenetv3_transfer_best.pth"
)

CLASS_MAPPING_PATH = (
    PROJECT_ROOT
    / "models"
    / "class_mapping.json"
)


# =========================================================
# LOAD CLASS MAPPING
# =========================================================

with open(CLASS_MAPPING_PATH, "r") as f:
    class_mapping = json.load(f)


# ---------------------------------------------------------
# The project's class_mapping.json uses:
#
# {
#     "0": "battery",
#     "1": "biological",
#     ...
# }
#
# So we convert the JSON keys into integer indexes.
# ---------------------------------------------------------

idx_to_class = {
    int(index): class_name
    for index, class_name in class_mapping.items()
}


# =========================================================
# WASTE RECOMMENDATIONS
# =========================================================

RECOMMENDATIONS = {

    "battery": {
        "waste_type": "Hazardous Waste / Battery-E-waste",
        "recommendation": "Battery or appropriate e-waste collection",
        "guidance": (
            "Do not mix batteries with general household waste. "
            "Use an appropriate battery or e-waste collection facility."
        )
    },

    "biological": {
        "waste_type": "Organic Waste / Wet-Organic",
        "recommendation": "Organic waste collection / Composting",
        "guidance": (
            "Keep biological or organic waste separate from dry "
            "recyclable materials. Composting may be suitable "
            "where facilities exist."
        )
    },

    "brown-glass": {
        "waste_type": "Recyclable Glass",
        "recommendation": "Glass Recycling",
        "guidance": (
            "Separate glass from organic and incompatible waste "
            "and use an appropriate glass collection or recycling stream."
        )
    },

    "cardboard": {
        "waste_type": "Recyclable Paper / Cardboard",
        "recommendation": "Paper-Cardboard Recycling",
        "guidance": (
            "Keep cardboard clean and dry and place it in the "
            "appropriate paper or cardboard recycling stream."
        )
    },

    "clothes": {
        "waste_type": "Textile Waste",
        "recommendation": "Textile Collection / Reuse",
        "guidance": (
            "Reusable clothing can be donated where suitable. "
            "Damaged textiles may require specialized collection."
        )
    },

    "green-glass": {
        "waste_type": "Recyclable Glass",
        "recommendation": "Glass Recycling",
        "guidance": (
            "Separate glass from other incompatible waste and "
            "use an appropriate glass collection or recycling stream."
        )
    },

    "metal": {
        "waste_type": "Recyclable Metal",
        "recommendation": "Metal Recycling",
        "guidance": (
            "Separate suitable metal items from organic waste "
            "and direct them to an appropriate metal recycling stream."
        )
    },

    "paper": {
        "waste_type": "Recyclable Paper",
        "recommendation": "Paper Recycling",
        "guidance": (
            "Keep paper clean and dry and place it in the "
            "appropriate paper recycling stream."
        )
    },

    "plastic": {
        "waste_type": "Recyclable Plastic",
        "recommendation": "Plastic Recycling",
        "guidance": (
            "Suitable clean plastic items can be directed to "
            "an appropriate plastic or recyclable waste stream."
        )
    },

    "shoes": {
        "waste_type": "Textile / Footwear Waste",
        "recommendation": "Textile or Footwear Collection",
        "guidance": (
            "Reusable footwear can be donated where appropriate. "
            "Other footwear may require textile or specialized collection."
        )
    },

    "trash": {
        "waste_type": "General / Residual Waste",
        "recommendation": "General Waste",
        "guidance": (
            "Waste that cannot reasonably be reused, recycled, "
            "composted, or sent to a specialized collection stream "
            "may belong here."
        )
    },

    "white-glass": {
        "waste_type": "Recyclable Glass",
        "recommendation": "Glass Recycling",
        "guidance": (
            "Separate glass from organic and incompatible waste "
            "and use an appropriate glass recycling or collection stream."
        )
    }
}


# =========================================================
# IMAGE TRANSFORMATION
# =========================================================

IMAGE_TRANSFORM = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = models.mobilenet_v3_small(
        weights=None
    )

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(idx_to_class)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)

    model.eval()

    return model


model = load_model()


# =========================================================
# IMAGE PREDICTION
# =========================================================

def predict_image(image):

    image = image.convert("RGB")

    image_tensor = IMAGE_TRANSFORM(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = idx_to_class[
        predicted_index.item()
    ]

    confidence_value = confidence.item()

    return predicted_class, confidence_value


# =========================================================
# HEADER
# =========================================================

st.title(
    "♻️ AI Smart Waste Management & Segregation System"
)

st.write(
    "AI-powered waste classification, segregation guidance "
    "and sustainability assistance."
)

st.divider()


# =========================================================
# SECTION 1 — IMAGE CLASSIFICATION
# =========================================================

st.header("📷 Waste Image Classification")

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # IMAGE
    # -----------------------------------------------------

    with col1:

        st.image(
            image,
            caption="Uploaded Waste Image",
            use_container_width=True
        )

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    with col2:

        with st.spinner(
            "Analyzing image..."
        ):

            predicted_class, confidence = predict_image(
                image
            )

        confidence_percent = confidence * 100

        recommendation = RECOMMENDATIONS.get(
            predicted_class,
            {
                "waste_type": "Unknown",
                "recommendation": "Check local guidelines",
                "guidance": (
                    "The system could not provide a specific "
                    "disposal recommendation."
                )
            }
        )

        st.subheader("Prediction")

        st.write(
            f"**Waste Class:** {predicted_class}"
        )

        st.write(
            f"**Confidence:** {confidence_percent:.2f}%"
        )

        st.progress(
            confidence
        )

        st.write(
            f"**Waste Type:** "
            f"{recommendation['waste_type']}"
        )

        st.write(
            f"**Recommended Action:** "
            f"{recommendation['recommendation']}"
        )

        st.info(
            recommendation["guidance"]
        )

        # -------------------------------------------------
        # RESPONSIBLE AI CONFIDENCE WARNING
        # -------------------------------------------------

        if confidence_percent >= 80:

            st.success(
                "High-confidence prediction."
            )

        elif confidence_percent >= 60:

            st.warning(
                "Moderate-confidence prediction. "
                "Please verify the waste type if the item "
                "is difficult to identify."
            )

        else:

            st.warning(
                "Low-confidence prediction. "
                "The AI may be uncertain about this image. "
                "Please verify the item before disposal."
            )


# =========================================================
# SECTION 2 — RAG SUSTAINABILITY ASSISTANT
# =========================================================

st.divider()

st.header("🤖 Sustainability Assistant")

st.write(
    "Ask questions about waste segregation, recycling, "
    "waste disposal and sustainability."
)


question = st.text_input(
    "Ask your question:",
    placeholder=(
        "Example: How should I dispose plastic waste?"
    )
)


if st.button("Ask Assistant"):

    if question.strip() == "":

        st.warning(
            "Please enter a question first."
        )

    else:

        with st.spinner(
            "Searching the sustainability knowledge base..."
        ):

            response = ask_assistant(
                query=question,
                top_k=3
            )

        st.subheader("💡 Assistant Answer")

        st.write(
            response["answer"]
        )

        # -------------------------------------------------
        # RETRIEVED DOCUMENTS
        # -------------------------------------------------

        if response["results"]:

            with st.expander(
                "🔎 View Retrieved Information"
            ):

                for index, result in enumerate(
                    response["results"],
                    start=1
                ):

                    document = result["document"]

                    st.write(
                        f"**{index}. {document['title']}**"
                    )

                    st.caption(
                        f"Category: {document['category']} | "
                        f"Relevance Score: {result['score']}"
                    )

                    st.write(
                        document["content"]
                    )

                    st.divider()

        else:

            st.warning(
                "No relevant information was found "
                "in the current knowledge base."
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Smart Waste Management & Segregation System | "
    "Responsible AI: predictions include confidence "
    "information and disposal recommendations should be "
    "verified with applicable local waste-management guidelines."
)

st.divider()

st.subheader("🤖 Responsible AI")

st.markdown("""
### How this AI system works responsibly

- **Confidence Score:** Every prediction includes a confidence score.
- **Uncertainty Warning:** Low-confidence predictions are clearly flagged.
- **Human Verification:** Users should verify uncertain predictions before disposal.
- **Transparency:** The predicted class, confidence, waste type, and recommendation are shown.
- **Privacy:** Avoid uploading images containing unnecessary personal information.
- **Local Guidelines:** Waste-management rules can vary by location, so official local guidance should be followed.
- **AI Limitations:** The model may make mistakes with unusual objects, poor-quality images, lighting changes, or waste categories not well represented in the training data.

> **AI is a decision-support tool, not a replacement for human judgment or official waste-management guidance.**
""")