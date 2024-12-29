from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import APIRouter, UploadFile, File,status
from io import BytesIO
from src.app.config import load_config
from src.pipeline import InferencePipeline
from src.app.exceptions import InputError
# Define the router
router = APIRouter()

# Load configuration
config = load_config()
inference_pipeline = InferencePipeline(config)

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from io import BytesIO
from PIL import Image
import traceback

router = APIRouter()

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
        # Validate the file
        if not file.content_type in ["image/png", "image/jpeg", "application/dicom"]:
            raise HTTPException(
                status_code=415,
                detail="Unsupported file type. Supported types are PNG, JPEG, and DICOM."
            )

        # Validate apply_clahe_postprocess parameter
        if not isinstance(apply_clahe_postprocess, bool):
            raise HTTPException(
                status_code=400,
                detail="The 'apply_clahe_postprocess' parameter must be a boolean."
            )

        # Read the uploaded file into memory
        file_bytes = await file.read()

        # Perform inference with the pipeline
        sr_image = inference_pipeline.run(BytesIO(file_bytes), apply_clahe_postprocess=apply_clahe_postprocess)

        # Save the result to a BytesIO stream
        output_stream = BytesIO()
        sr_image.save(output_stream, format="PNG")
        output_stream.seek(0)

        # Return the processed image as a streaming response
        return StreamingResponse(output_stream, media_type="image/png")

    except HTTPException as e:
        return JSONResponse(
            content={"error": e.detail},
            status_code=e.status_code
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"Error during prediction: {str(e)}"},
            status_code=500
        )
