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
    is_dicom: bool = False,
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

        # Convert to PIL Image or handle as DICOM
        if is_dicom:
            input_image = inference_pipeline.preprocess(BytesIO(contents), is_dicom=True)
        else:
            input_image = Image.open(BytesIO(contents)).convert("RGB")

        if input_image is None:
            raise HTTPException(status_code=400, detail="Failed to process the input image.")

        # Perform super-resolution
        sr_image = inference_pipeline.infer(input_image)

        if apply_clahe_postprocess:
            sr_image = inference_pipeline.postprocess(sr_image)

        # Save or return result
        output_path = f"output_{file.filename}"
        sr_image.save(output_path)

        return {"message": "Super-resolution completed successfully.", "output_path": output_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")
