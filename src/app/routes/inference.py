from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
from io import BytesIO
from ..config import load_config
from ...pipeline import InferencePipeline

# Define the router
router = APIRouter()

# Load configuration
config = load_config()
inference_pipeline = InferencePipeline(config)

@router.post("/predict")
async def process_image(
    file: UploadFile = File(...),
    apply_clahe_postprocess: bool = False
):
    """
    API endpoint to process and super-resolve an image.

    Args:
        file: Image file to process (PNG, JPEG, or DICOM).
        is_dicom: Boolean indicating if the input is a DICOM file.
        apply_clahe_postprocess: Boolean indicating if CLAHE should be applied post-processing.

    Returns:
        JSONResponse: Processed image or error message.
    """
    try:
        # Read the uploaded file
        contents = await file.read()


        # Perform super-resolution
        sr_image = inference_pipeline.run(BytesIO(contents),apply_clahe_postprocess = apply_clahe_postprocess)

        # Save or return result
        output_path = f"output_{file.filename}"
        sr_image.save(output_path)

        return {"message": "Super-resolution completed successfully.", "output_path": output_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
