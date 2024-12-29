from fastapi import FastAPI,Request,status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.app.routes import inference
from src.app.exceptions import ModelLoadError, PreprocessingError, InferenceError,InputError, PostprocessingError
import torch
import os
import sys

app = FastAPI(title="Super Resolution Dental X-ray API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Include API routes
app.include_router(inference.router, prefix="/inference", tags=["Inference"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Super Resolution Dental X-ray API"}

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint to ensure the API and CUDA are running.

    Returns:
        dict: Status message indicating the API and CUDA availability.
    """
    def bash(command):
        return os.popen(command).read()

    # Check CUDA status

    # Construct response
    return {
        "status": "Healthy",
        "message": "API is running successfully.",
        "cuda": {
            "sys.version": sys.version,
            "torch.__version__": torch.__version__,
            "torch.cuda.is_available()": torch.cuda.is_available(),
            "torch.version.cuda": torch.version.cuda,
            "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
            "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
            "nvidia-smi": bash('nvidia-smi')
        }
    }

# Custom exception handlers
@app.exception_handler(ModelLoadError)
async def model_load_error_handler(request: Request, exc: ModelLoadError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "ModelLoadError", "message": exc.message},
    )

@app.exception_handler(PreprocessingError)
async def preprocessing_error_handler(request: Request, exc: PreprocessingError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "PreprocessingError", "message": exc.message},
    )

@app.exception_handler(PostprocessingError)
async def postprocessing_error_handler(request: Request, exc: PostprocessingError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "PostprocessingError", "message": exc.message},
    )

@app.exception_handler(InferenceError)
async def inference_error_handler(request: Request, exc: InferenceError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "InferenceError", "message": exc.message},
    )


@app.exception_handler(InputError)
async def input_load_error_handler(request: Request, exc: InputError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "InputError", "message": exc.message},
    )