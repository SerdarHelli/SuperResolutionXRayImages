from fastapi import APIRouter
import torch
import sys
import os
# Initialize router
router = APIRouter()

@router.get("/health", tags=["Health"])
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
