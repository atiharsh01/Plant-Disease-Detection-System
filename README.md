# 🌿 PlantCare AI — Plant Disease Detection System

An AI-powered computer-vision application that classifies plant leaf images using a ResNet50 image-classification model trained/fine-tuned for PlantVillage disease classes.

## ✨ Features

- Leaf image upload (JPG/PNG/WEBP)
- CNN-based ResNet50 inference
- 38 plant disease / healthy classes
- Top-5 predictions
- Confidence visualization
- Disease information and practical next steps
- Modern Streamlit dashboard
- Training pipeline for an independent MobileNetV2 fine-tuning experiment
- Performance-report template
- Sample-data instructions

## 🧠 Model

The demo uses `mesabo/agri-plant-disease-resnet50`, an Apache-2.0 model published on Hugging Face. Its model card describes a ResNet50 architecture, 38 PlantVillage classes, 224×224 RGB input and a reported 95%+ PlantVillage test accuracy.

**Important:** the 95%+ figure is the upstream model-card result, not a benchmark independently reproduced by this repository.

## 📊 Dataset

PlantVillage is an open-access dataset containing 54,306 leaf images spanning 14 crop species and 38 crop-disease/healthy classes.

See `data/README.md` for download and organization instructions.

## 🚀 Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first prediction downloads the model checkpoint from Hugging Face.

## 🏋️ Independent training

For an independently trained CNN experiment:

```bash
python train.py
```

Place PlantVillage class folders under `data/raw/`. The training script creates `model/plant_disease_model.keras` and `model/class_labels.json`.

For GPU training, Google Colab or another CUDA-enabled environment is recommended.

## 📁 Project structure

```text
Plant-Disease-Detection-System/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── PROJECT_DOCUMENTATION.md
├── PERFORMANCE_REPORT.md
├── model/
│   └── README.md
├── src/
│   ├── predictor.py
│   └── disease_info.py
├── data/
│   ├── README.md
│   └── sample/
└── tests/
```

## ⚠️ Responsible-use note

This application is an educational decision-support prototype. Leaf appearance can vary with lighting, camera quality, crop variety and real-world conditions. Predictions should be confirmed by an agricultural expert before treatment decisions.

## References

- PlantVillage Dataset: https://github.com/spMohanty/PlantVillage-Dataset
- Mohanty, Hughes & Salathé (2016), *Using Deep Learning for Image-Based Plant Disease Detection*
- Demo checkpoint: https://huggingface.co/mesabo/agri-plant-disease-resnet50
