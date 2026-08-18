# Project Documentation — PlantCare AI

## 1. Objective

Develop an AI-powered computer-vision application that classifies plant leaf images into healthy or disease categories and presents the prediction through an interactive dashboard.

## 2. Functional modules

### Image processing
Uploaded images are converted to RGB and normalized through the model's image processor.

### Computer vision
The system uses a ResNet50 convolutional neural network checkpoint specialized for plant disease classification.

### Image classification
The classifier produces a probability distribution over 38 PlantVillage classes.

### Recommendation layer
The dashboard maps the predicted class to human-readable disease context, symptoms, actions and prevention reminders.

### Prediction dashboard
The Streamlit UI presents:
- uploaded image
- predicted class
- confidence
- inference time
- top-5 alternatives
- disease information
- model snapshot

## 3. System flow

```text
Leaf Image
   ↓
RGB Conversion
   ↓
ResNet50 Image Processor
   ↓
CNN Feature Extraction
   ↓
38-Class Softmax
   ↓
Top-K Predictions
   ↓
Disease Information Layer
   ↓
Interactive Dashboard
```

## 4. Model strategy

The deployed demonstration uses a pretrained ResNet50 checkpoint trained for PlantVillage classes. The repository also contains a separate MobileNetV2 fine-tuning script for producing an independently trained artifact.

## 5. Limitations

PlantVillage contains many controlled-condition images, so performance on field photographs can differ. Real-world factors include background clutter, lighting, camera angle, multiple leaves, mixed symptoms and unseen diseases.

## 6. Future improvements

- Fine-tune on field-condition datasets
- Add Grad-CAM visual explanations
- Add crop selection
- Add disease severity estimation
- Add multilingual agricultural guidance
- Add model monitoring and feedback collection
