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
    
    # Required Patient and Image Information
    ds.PatientName = "Test"
    ds.PatientID = "12345"
    ds.Modality = "CT"
    ds.StudyInstanceUID = "1.2.3.4.5.6.7.8.9.10"
    ds.SeriesInstanceUID = "1.2.3.4.5.6.7.8.9.11"
    ds.SOPInstanceUID = "1.2.3.4.5.6.7.8.9.12"
    ds.StudyDate = "20240101"
    ds.StudyTime = "120000"
    ds.Manufacturer = "TestManufacturer"

    # Required Image Data Information
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = 128
    ds.Columns = 128
    ds.BitsAllocated = 16
    ds.BitsStored = 16  # Add missing Bits Stored
    ds.HighBit = 15  # Highest bit set
    ds.PixelRepresentation = 0  # Unsigned integer
    ds.SamplesPerPixel = 1  # Single-channel (grayscale)
    ds.PixelData = (np.random.rand(128, 128) * 65535).astype(np.uint16).tobytes()

    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
    ds.save_as(temp_file.name)
    return temp_file.name

    
def test_is_dicom(pipeline):
    dicom_path = create_dummy_dicom()
    
    # Test with file path
    assert pipeline.is_dicom(dicom_path) is True
    
    # Test with BytesIO
    with open(dicom_path, "rb") as f:
        dicom_bytes = BytesIO(f.read())
    assert pipeline.is_dicom(dicom_bytes) is True

    # Test with invalid BytesIO (non-DICOM content)
    non_dicom_bytes = BytesIO()
    non_dicom_bytes.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 128)  # Write invalid header
    non_dicom_bytes.seek(0)
    assert pipeline.is_dicom(non_dicom_bytes) is False

    os.remove(dicom_path)


def test_is_dicom(pipeline):
    dicom_path = create_dummy_dicom()
    
    # Test with file path
    assert pipeline.is_dicom(dicom_path) is True, "DICOM file path should be recognized as DICOM"
    
    # Test with BytesIO
    with open(dicom_path, "rb") as f:
        dicom_bytes = BytesIO(f.read())
    assert pipeline.is_dicom(dicom_bytes) is True, "BytesIO DICOM content should be recognized as DICOM"



    # Test with invalid BytesIO (non-DICOM content)
    non_dicom_bytes = BytesIO()
    non_dicom_bytes.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 128)  # Write invalid header
    non_dicom_bytes.seek(0)
    assert pipeline.is_dicom(non_dicom_bytes) is False, "Non-DICOM BytesIO should not be recognized as DICOM"

    # Test with invalid raw bytes
    invalid_raw_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 128
    assert pipeline.is_dicom(invalid_raw_bytes) is False, "Invalid raw bytes should not be recognized as DICOM"

    os.remove(dicom_path)



def test_preprocess_normal_image(pipeline):
    # Create a dummy image
    image = Image.new("RGB", (128, 128), color="red")
    
    # Test with BytesIO
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    processed_image_bytes = pipeline.preprocess(image_bytes, is_dicom=False)
    assert isinstance(processed_image_bytes, Image.Image)
    
    # Test with file path
    temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
    image.save(temp_image_path)
    
    processed_image_path = pipeline.preprocess(temp_image_path, is_dicom=False)
    assert isinstance(processed_image_path, Image.Image)

    os.remove(temp_image_path)


def test_infer(pipeline):
    # Create a dummy image
    image = Image.new("RGB", (128, 128), color="red")

    # Perform inference
    result = pipeline.infer(image)
    assert isinstance(result, Image.Image)


def test_postprocess(pipeline):

    image = Image.new("RGB", (128, 128), color="red")
    result = pipeline.postprocess(image)

    assert isinstance(result, Image.Image)


def test_api_predict_normal_image():
    # Create a dummy image
    image = Image.new("RGB", (128, 128), color="red")
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    response = client.post(
        "/inference/predict",  # Adjusted to include the prefix
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        data={"apply_clahe_postprocess": "false"}  # Ensure proper boolean conversion
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"


def test_api_predict_dicom():
    dicom_path = create_dummy_dicom()
    
    # Use BytesIO for testing
    with open(dicom_path, "rb") as f:
        dicom_bytes = BytesIO(f.read())

    response = client.post(
        "/inference/predict",  # Adjusted to include the prefix
        files={"file": ("test.dcm", dicom_bytes, "application/dicom")},
        data={"apply_clahe_postprocess": "false"}  # Ensure proper boolean conversion
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"

    os.remove(dicom_path)

