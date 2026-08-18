"""Human-readable disease metadata used by the dashboard."""

def _clean(label: str) -> tuple[str, str, str]:
    parts = label.replace("(", "").replace(")", "").split("___", 1)
    crop = parts[0].replace("_", " ")
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    display = f"{crop.title()} — {disease.replace('healthy', 'Healthy').title()}"
    return crop, disease, display


def get_disease_info(label: str) -> dict:
    crop, disease, display = _clean(label)
    healthy = "healthy" in disease.lower()

    if healthy:
        return {
            "crop": crop.title(),
            "display_name": display,
            "category": "Healthy",
            "healthy": True,
            "description": "The model found the healthy-leaf class most likely for this image.",
            "symptoms": ["No strong disease signal detected", "Leaf appears consistent with healthy training examples"],
            "actions": ["Continue normal crop monitoring", "Maintain appropriate irrigation and nutrition"],
            "prevention": ["Inspect plants regularly", "Remove unusual leaves early", "Keep foliage dry where appropriate"],
        }

    category = "Disease / stress signal"
    disease_lower = disease.lower()
    if "virus" in disease_lower or "mosaic" in disease_lower:
        category = "Viral disease"
    elif "bacterial" in disease_lower:
        category = "Bacterial disease"
    elif "mold" in disease_lower:
        category = "Fungal / mold disease"
    elif "blight" in disease_lower or "spot" in disease_lower or "rust" in disease_lower:
        category = "Fungal / leaf disease"

    return {
        "crop": crop.title(),
        "display_name": display,
        "category": category,
        "healthy": False,
        "description": f"The model classified the image as {disease.replace('_', ' ')}.",
        "symptoms": ["Discoloration or lesions may be present", "Pattern resembles examples in the training data"],
        "actions": ["Isolate visibly affected foliage when practical", "Check nearby plants for similar symptoms", "Confirm diagnosis before applying treatment"],
        "prevention": ["Improve airflow", "Avoid unnecessary leaf wetness", "Clean tools between plants", "Monitor new growth"],
    }
