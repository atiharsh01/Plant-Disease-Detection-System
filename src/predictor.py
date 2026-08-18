from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification


MODEL_ID = "mesabo/agri-plant-disease-resnet50"


class PlantDiseasePredictor:
    """Predict plant diseases using a pretrained ResNet50 model."""

    def __init__(self, model_id: str = MODEL_ID):
        self.model = AutoModelForImageClassification.from_pretrained(model_id)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        top_k: int = 5,
    ) -> dict:
        image = image.convert("RGB")
        inputs = self.transform(image).unsqueeze(0)

        outputs = self.model(pixel_values=inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

        k = min(top_k, probs.shape[-1])
        values, indices = torch.topk(probs, k=k)

        items = []

        for value, index in zip(values.tolist(), indices.tolist()):
            label = self.model.config.id2label.get(
                index,
                f"Class {index}",
            )

            items.append(
                {
                    "label": label,
                    "confidence": float(value),
                }
            )

        return {
            "label": items[0]["label"],
            "confidence": items[0]["confidence"],
            "top_k": items,
        }
