# Model Artifact

The repository intentionally does not duplicate the ~91 MB upstream checkpoint in GitHub.

## Demo model

The Streamlit app loads:

`mesabo/agri-plant-disease-resnet50`

from Hugging Face on first use.

## Independent trained model

Run:

```bash
python train.py
```

After training, the script creates:

```text
model/plant_disease_model.keras
model/class_labels.json
```

If you need a fully self-contained offline deployment, place your trained artifact in this folder and modify `src/predictor.py` to load the local checkpoint.
