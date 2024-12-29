# Super Resolution DICOM & Image Processing API

This project provides a FastAPI-based application for super-resolving medical DICOM images and standard images (PNG, JPEG). The API integrates preprocessing, inference using a RealESRGAN model, and optional postprocessing steps like CLAHE.

## Features

- **DICOM Support**: Detects and preprocesses DICOM images, handling VOI LUT and monochrome corrections.
- **Image Super-Resolution**: Utilizes a RealESRGAN model for enhancing image resolution.
- **Flexible Configuration**: Dynamic configuration for model parameters, preprocessing, and postprocessing.
- **Streaming API**: Efficient in-memory processing with FastAPI.
- **Robust Testing**: Comprehensive unit and integration tests for API endpoints and internal pipeline components.
- **Docker Support**: Seamless deployment using Docker with NVIDIA CUDA support.

## Project Structure

- `configs/`: Contains the YAML configuration file for model and processing parameters.
- `src/`: Source code directory.
  - `app/`: FastAPI application components.
    - `main.py`: Entry point for the API.
    - `pipeline.py`: Core inference pipeline.
    - `routes/`: API routes.
  - `network/`: Neural network model definitions.
  - `preprocess.py`: Preprocessing utilities for images and DICOM files.
- `weights/`: Pre-trained model weights.
- `tests/`: Test cases for pipeline and API functionality.
- `Dockerfile`: Docker configuration for containerized deployment.
- 
## Model Description

The super-resolution functionality is powered by the [RealESRGAN](https://arxiv.org/abs/2107.10833) model, fine-tuned specifically for enhancing dental X-ray images. Key steps in the model preparation include:

1. **Preprocessing Dataset**:
   - Downloaded datasets dental X-ray images.
   - Organized data into training and validation sets with appropriate folder structures.
   - Applied multiscale patch generation and meta-information extraction to prepare the data for training.

2. **Model Fine-Tuning**:
   - Used the RealESRGAN training pipeline for fine-tuning on the dental dataset.
   - Leveraged base pre-trained weights (`RealESRGAN_x4plus.pth`) as the starting point.
   - Applied advanced cropping strategies and resolution-specific adjustments for optimal results.

3. **Training Configuration**:
   - Fine-tuned using a custom parametres.
   - Utilized NVIDIA GPUs with CUDA acceleration for efficient training.
   - Periodically validated on unseen data to monitor improvements.

The resulting model significantly improves the clarity and resolution of dental X-rays while preserving diagnostic details.

## Configuration

The application uses a YAML configuration file (`configs/config.yaml`) for setting:

- Model weights path, scale, and device.
- Preprocessing parameters like unsharp mask kernel size and strength.
- Postprocessing parameters for CLAHE.

### Example Configuration

```yaml
model:
  weights: "weights/model.pth"
  scale: 4
  device: "cuda"

preprocessing:
  unsharping_mask:
    kernel_size: 7
    strength: 0.5

postprocessing:
  clahe:
    clipLimit: 2
    tileGridSize:
      - 16
      - 16
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/super-resolution-api.git
   cd super-resolution-api
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the FastAPI application:
   ```bash
   uvicorn src.app.main:app --host 0.0.0.0 --port 8080

   ```

## Docker Deployment

A `Dockerfile` is included for containerized deployment with NVIDIA CUDA support.

### NVIDIA Container Toolkit Before Build
To enable GPU usage in your Docker container (i.e., by using ```--gpus all```), you need to install the NVIDIA Container Toolkit before building. [Please follow the guide in the provided link for detailed instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Also, make sure that your drivers are installed and up to date.


### Build the Docker Image

```bash
docker build -t super-resolution-api .
```

### Run the Docker Container

```bash
docker run --gpus all -p 8080:8080 super-resolution-api uvicorn main:app --host 0.0.0.0 --port 8080
```

This will launch the API at `http://0.0.0.0:8080`.



## Usage

### API Endpoints

#### `POST /predict`

- **Description**: Processes an uploaded image (DICOM, PNG, JPEG) and returns the super-resolved image.
- **Parameters**:

  - `file`: The image file to process.
  - `apply_clahe_postprocess`: Boolean indicating if CLAHE should be applied after inference.
- **Response**: Returns the processed image as a PNG.

### Example Request

Using `curl`:
```bash
curl -X POST "http://0.0.0.0:8080/inference/predict" \
  -F "file=@test_image.dcm" \
  -F "apply_clahe_postprocess=true"
```
***An example request for python :***
```python 

import requests
from io import BytesIO

# URL of your FastAPI endpoint
url = "http://0.0.0.0:8080/inference/predict"

# Path to the DICOM file
dicom_file_path = "path/to/your/dicom_file.dcm"

# Read the DICOM file into bytes
with open(dicom_file_path, "rb") as f:
    dicom_bytes = f.read()

# Create the request payload
files = {
    "file": ("dicom_file.dcm", BytesIO(dicom_bytes), "application/dicom"),
}
data = {
    "apply_clahe_postprocess": "false"  # Set to "true" if CLAHE postprocessing is needed
}

# Send the POST request
response = requests.post(url, files=files, data=data)

# Check the response
if response.status_code == 200:
    print("Prediction successful!")
    # Save the returned PNG image
    with open("output.png", "wb") as f:
        f.write(response.content)
    print("Output saved to output.png")
else:
    print(f"Error: {response.status_code}, {response.text}")

```
---
#### `GET /health`

- **Description**: Checks the health of the API and the system.
- **Response**: Returns system and CUDA-related details.

### Example Health Check Request

Using `curl`:
```bash
curl http://0.0.0.0:8080/health
```

#### Example Response:

```json
{
  "status": "Healthy",
  "message": "API is running successfully.",
  "cuda": {
    "sys.version": "3.9.9 (default, Nov 16 2021, 06:04:44)\n[GCC 9.3.0]",
    "torch.__version__": "1.12.1",
    "torch.cuda.is_available()": true,
    "torch.version.cuda": "11.8",
    "torch.backends.cudnn.version()": 8200,
    "torch.backends.cudnn.enabled": true,
    "nvidia-smi": "... output from nvidia-smi ..."
  }
}
```
## Testing

Run tests with `pytest`:

- To run all tests:
  ```bash
  pytest
  ```


