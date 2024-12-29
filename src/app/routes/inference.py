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

import os
from fastapi import HTTPException
from fastapi.responses import FileResponse

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
        FileResponse: Processed image file or error message.
    """
    try:

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

        # Save the processed image to a temporary file
        output_file_path = "output_highres.png"
        sr_image.save(output_file_path, format="PNG")

        # Return the file as a response
        return FileResponse(
            path=output_file_path,
            media_type="image/png",
            filename="processed_image.png"
        )

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during processing: {str(e)}"
        )
    finally:
        # Cleanup temporary file if it exists
        if os.path.exists("output_highres.png"):
            os.remove("output_highres.png")

