from src.disease_info import get_disease_info


def test_healthy_metadata():
    info = get_disease_info("Tomato___healthy")
    assert info["healthy"] is True
    assert "Tomato" in info["display_name"]


def test_disease_metadata():
    info = get_disease_info("Tomato___Early_blight")
    assert info["healthy"] is False
    assert "Early" in info["display_name"]
