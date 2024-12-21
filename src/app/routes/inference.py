from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import APIRouter, UploadFile, File
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
        apply_clahe_postprocess: Boolean indicating if CLAHE should be applied post-processing.

    Returns:
        StreamingResponse: Processed image or error message.
    """
    try:
        # Read the uploaded file into memory
        file_bytes = await file.read()

        # Perform super-resolution
        sr_image = inference_pipeline.run(BytesIO(file_bytes), apply_clahe_postprocess=apply_clahe_postprocess)

        # Save the result to a BytesIO stream
        output_stream = BytesIO()
        sr_image.save(output_stream, format="PNG")
        output_stream.seek(0)

        # Return the processed image as a streaming response
        return StreamingResponse(output_stream, media_type="image/png")

    except Exception as e:
        return JSONResponse(
            content={"error": f"Error during prediction: {str(e)}"},
            status_code=500
        )
