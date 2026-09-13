# ♻️ AI Smart Waste Management & Segregation System

An AI-powered waste classification and segregation assistance system
designed to promote sustainable waste management and responsible
consumption.

---

## 📌 Project Overview

The AI Smart Waste Management & Segregation System uses computer vision
and Retrieval-Augmented Generation (RAG) to identify waste from images
and provide appropriate waste-segregation and disposal guidance.

The system combines:

- Image Classification
- Transfer Learning
- Confidence-based prediction
- Waste disposal recommendations
- RAG-based sustainability guidance
- Responsible AI principles
- Streamlit web interface

### Overall Workflow

User Image
    ↓
Image Preprocessing
    ↓
MobileNetV3-Small
    ↓
Waste Classification
    ↓
Confidence Score
    ↓
Disposal Recommendation
    ↓
RAG Knowledge Base
    ↓
Waste Management Guidance

---

## 🎯 Problem Statement

Improper waste segregation is a major challenge in sustainable waste
management.

People often have difficulty identifying the correct waste category
and deciding how different waste items should be handled.

### How might we use AI to automatically identify different types of
waste and provide proper segregation guidance so that waste management
can become more sustainable?

---

## 🌍 Sustainable Development Goals

### Primary SDG

**SDG 12 — Responsible Consumption and Production**

The project supports responsible consumption and production by encouraging
waste segregation, recycling, reuse and appropriate disposal.

### Secondary SDG

**SDG 11 — Sustainable Cities and Communities**

Better waste segregation can contribute to cleaner and more sustainable
communities.

---

## 👥 Target Users

The system can assist:

- Students
- Educational institutions
- Offices
- Communities
- Households
- Waste-management awareness programs

---

# 🤖 AI Components

## 1. Computer Vision

The system uses an image-classification model to identify different
types of waste.

The final model uses:

**MobileNetV3-Small with ImageNet pretraining**

Transfer learning was used to adapt the pretrained model for waste
classification.

---

## 2. Waste Classification

The model classifies images into 12 waste categories:

1. Battery
2. Biological
3. Brown Glass
4. Cardboard
5. Clothes
6. Green Glass
7. Metal
8. Paper
9. Plastic
10. Shoes
11. Trash
12. White Glass

---

## 3. Confidence-based Prediction

The application displays the confidence score associated with each
prediction.

The system provides a warning when confidence is low.

Example:

"Low-confidence prediction. The AI may be uncertain about this image.
Please verify the item before disposal."

This helps prevent users from treating every AI prediction as certain.

---

# 📊 Dataset

The project uses the Kaggle Garbage Classification dataset containing
12 waste categories.

### Dataset Statistics

- Total images: **15,515**
- Number of classes: **12**
- Corrupted images found: **0**

### Dataset Split

| Dataset | Images |
|---|---:|
| Training | 10,860 |
| Validation | 2,327 |
| Testing | 2,328 |

The dataset contains class imbalance, with some categories having
significantly more images than others.

Class weights were therefore used during model training.

---

# 🔬 Exploratory Data Analysis

EDA was performed to understand the dataset before model training.

The analysis included:

- Class distribution
- Number of images per class
- Sample image visualization
- Image dimensions
- Aspect-ratio distribution
- Dataset imbalance
- Corrupted-image checking

Generated EDA outputs are stored in:

```text
outputs/graphs/