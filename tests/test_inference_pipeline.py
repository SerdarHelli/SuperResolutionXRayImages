import pytest
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
from io import BytesIO
import pydicom
from pydicom.dataset import Dataset, FileDataset
import tempfile
import os

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent ))

from src.app.main import app
from src.pipeline import InferencePipeline

# Initialize test client
client = TestClient(app)

@pytest.fixture
def pipeline_config():
    return {
        "model": {
            "weights": "weights/model.pth",
            "scale": 4,
            "device": "cpu"
        },
        "preprocessing": {
            "unsharping_mask": {
                "kernel_size": 7,
                "strength": 0.5
            }
        },
        "postprocessing": {
            "clahe": {
                "clipLimit": 2,
                "tileGridSize": [16, 16]
            }
        }
    }

@pytest.fixture
def pipeline(pipeline_config):
    return InferencePipeline(pipeline_config)

def create_dummy_dicom():
    """Create a dummy DICOM file for testing."""
    meta = Dataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    meta.MediaStorageSOPInstanceUID = "1.2.3"
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset("", {}, file_meta=meta, preamble=b"\x00" * 128)
    ds.PatientName = "Test"
    ds.PatientID = "12345"
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = 128
    ds.Columns = 128
    ds.BitsAllocated = 16
    ds.PixelData = (np.random.rand(128, 128) * 65535).astype(np.uint16).tobytes()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
    ds.save_as(temp_file.name)
    return temp_file.name

def test_is_dicom(pipeline):
    dicom_path = create_dummy_dicom()
    with open(dicom_path, "rb") as f:
        dicom_bytes = f.read()

    assert pipeline.is_dicom(dicom_path) is True
    assert pipeline.is_dicom(dicom_bytes) is True

    non_dicom_bytes = BytesIO()
    non_dicom_bytes.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 128)
    non_dicom_bytes.seek(0)
    assert pipeline.is_dicom(non_dicom_bytes.getvalue()) is False

    os.remove(dicom_path)

def test_preprocess_dicom(pipeline):
    dicom_path = create_dummy_dicom()
    with open(dicom_path, "rb") as f:
        dicom_bytes = f.read()

    processed_image = pipeline.preprocess(dicom_bytes, is_dicom=True)
    assert isinstance(processed_image, Image.Image)

    os.remove(dicom_path)

def test_preprocess_normal_image(pipeline):
    # Create a dummy image
    image = Image.new("RGB", (128, 128), color="red")
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    processed_image = pipeline.preprocess(image_bytes.getvalue(), is_dicom=False)
    assert isinstance(processed_image, Image.Image)

def test_infer(pipeline):
    # Create a dummy image
    image = Image.new("RGB", (128, 128), color="red")

    # Perform inference
    result = pipeline.infer(image)
    assert isinstance(result, np.ndarray)

def test_postprocess(pipeline):
    # Create a dummy array
    array = np.random.randint(0, 255, (128, 128), dtype=np.uint8)

    # Perform postprocessing
    result = pipeline.postprocess(array)
    assert isinstance(result, Image.Image)

def test_api_predict_normal_image():
    # Create a dummy image
    image = Image.new("RGB", (128, 128), color="red")
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        data={"apply_clahe_postprocess": False}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

def test_api_predict_dicom():
    dicom_path = create_dummy_dicom()
    with open(dicom_path, "rb") as f:
        dicom_bytes = f.read()

    response = client.post(
        "/predict",
        files={"file": ("test.dcm", BytesIO(dicom_bytes), "application/dicom")},
        data={"apply_clahe_postprocess": False}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    os.remove(dicom_path)
