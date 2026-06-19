"""
Basic tests for the Wildfire Detection API.

Run with: pytest tests/
"""

import io

from fastapi.testclient import TestClient
from PIL import Image

from api.main import app

client = TestClient(app)


def make_fake_image_bytes() -> bytes:
    """Create a small in-memory dummy image for upload tests."""
    img = Image.new("RGB", (224, 224), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_rejects_non_image_file():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_accepts_valid_image_format():
    """
    Note: this will return 503 if the model file isn't present locally,
    which is expected in CI without the trained weights checked in.
    It still confirms the endpoint accepts a valid image and routes correctly.
    """
    image_bytes = make_fake_image_bytes()
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code in (200, 503)
