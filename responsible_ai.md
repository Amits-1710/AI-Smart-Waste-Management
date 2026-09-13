# Responsible AI — AI Smart Waste Management & Segregation System

## 1. Purpose

This project uses artificial intelligence to classify waste images
and provide waste-segregation guidance.

The system is designed to assist users in making better waste-management
decisions. It is not intended to replace human judgment or official
waste-management authorities.

---

## 2. Dataset Limitations

The image-classification model was trained using a 12-class garbage
classification dataset.

The dataset contains different numbers of images for different classes.
Therefore, the model may perform differently across waste categories.

The dataset contains the following classes:

- Battery
- Biological
- Brown Glass
- Cardboard
- Clothes
- Green Glass
- Metal
- Paper
- Plastic
- Shoes
- Trash
- White Glass

Because the training data may not represent every real-world waste item,
the model can make incorrect predictions on images that differ from
the training data.

---

## 3. Class Imbalance

The dataset is not perfectly balanced.

Some categories contain substantially more images than others.
Class weights were therefore used during model training to reduce the
effect of class imbalance.

Even with class weighting, performance may vary between classes.

---

## 4. Confidence and Uncertainty

The system displays the model's confidence score along with its prediction.

High-confidence predictions can still be incorrect.

Low-confidence predictions are explicitly shown to the user with a warning.

Example:

"Low-confidence prediction. The AI may be uncertain about this image.
Please verify the item before disposal."

This prevents users from treating every AI prediction as certain.

---

## 5. Human Verification

The AI system should be treated as a decision-support tool.

Users should verify uncertain predictions before disposing of waste,
especially for hazardous or special waste such as batteries.

Official local waste-management instructions should take priority over
the AI recommendation.

---

## 6. Hazardous Waste

Certain waste categories require special handling.

For example, batteries should not be mixed with ordinary household waste.
Users should follow appropriate battery or e-waste collection guidance.

The system provides general educational guidance and does not replace
official disposal instructions.

---

## 7. Privacy

The prototype is designed around waste-image classification.

Users should avoid uploading photographs containing unnecessary
personal or sensitive information.

If the system is deployed publicly, appropriate data-protection and
retention policies should be implemented.

---

## 8. Transparency

The system clearly communicates:

- Predicted waste class
- Confidence score
- Waste category
- Recommended action
- Uncertainty warning when confidence is low
- RAG information used for guidance

This helps users understand how the AI output should be interpreted.

---

## 9. RAG Responsibility

The RAG assistant retrieves information from a predefined waste-management
knowledge base.

The assistant provides educational guidance based on the available
information.

Waste-management rules may vary by location. Therefore, users should
follow applicable local regulations and official collection guidelines.

---

## 10. Model Limitations

The model may make mistakes because of:

- Poor image quality
- Unusual objects
- Different lighting conditions
- Background clutter
- Objects not represented in the training dataset
- Similar-looking waste categories
- Differences between training images and real-world images

Therefore, predictions should not be treated as guaranteed.

---

## 11. Fairness and Reliability

The system should be evaluated across all waste categories rather than
using only overall accuracy.

Class-wise precision, recall and F1-score should be monitored to identify
categories where the model performs poorly.

Future versions should include more diverse real-world images and
additional evaluation datasets.

---

## 12. Environmental Responsibility

The purpose of the system is to encourage:

- Waste segregation at source
- Recycling
- Reuse
- Appropriate disposal
- Reduction of contamination between waste streams

The system should support sustainable waste-management practices rather
than encouraging unnecessary disposal.

---

## 13. Responsible AI Principle

The project follows the principle:

"AI should assist people, not blindly make disposal decisions."

The final disposal decision should consider the actual item, local
waste-management rules, and appropriate human verification.