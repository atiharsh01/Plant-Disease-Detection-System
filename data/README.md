# Dataset

## PlantVillage

The project uses the PlantVillage dataset as its reference dataset.

Public repository:
https://github.com/spMohanty/PlantVillage-Dataset

The dataset contains 54,306 leaf images across 14 crop species and 38 classes.

## Local training layout

For the TensorFlow training script, organize a local subset as:

```text
data/raw/
├── Apple___Apple_scab/
├── Apple___healthy/
├── Potato___Early_blight/
├── Potato___healthy/
├── Tomato___Early_blight/
├── Tomato___Late_blight/
├── Tomato___healthy/
└── ...
```

Do not commit the full dataset to GitHub. It is large and has its own dataset terms.

## Sample data

For a demo, download a few images from the public PlantVillage repository and place them in `data/sample/`.
