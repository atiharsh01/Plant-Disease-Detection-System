# Performance Report — PlantCare AI

## Executive summary

The deployed demonstration uses the public `mesabo/agri-plant-disease-resnet50` checkpoint. Its published model card reports:

- Architecture: ResNet50
- Classes: 38
- Input: 224×224 RGB
- Reported PlantVillage test accuracy: 95%+
- Reported CPU inference time: <100 ms

These figures are **upstream reported metrics** and are not presented as an independently reproduced benchmark by this project.

## Evaluation plan

For an independent evaluation, use a held-out PlantVillage test split and report:

| Metric | Value |
|---|---:|
| Accuracy | Run evaluation |
| Macro Precision | Run evaluation |
| Macro Recall | Run evaluation |
| Macro F1 | Run evaluation |
| Confusion Matrix | Generate |
| Average inference time | Measure |

## Recommended experiment

1. Download PlantVillage.
2. Keep leaf-group-aware train/test separation where available.
3. Train using `train.py` or fine-tune the ResNet50 checkpoint.
4. Evaluate on an untouched test set.
5. Save the confusion matrix and classification report.
6. Replace the placeholders in this report with measured values.

## Interpretation

A high benchmark score on PlantVillage should not be treated as proof of field-level diagnostic reliability. Controlled datasets can differ substantially from real-world images.
