from fastapi import APIRouter
import torch

# Initialize router
router = APIRouter()

@router.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint to ensure the API and CUDA are running.

    Returns:
        dict: Status message indicating the API and CUDA availability.
    """
    # Check CUDA status
    cuda_available = torch.cuda.is_available()
    cuda_device = torch.cuda.get_device_name(0) if cuda_available else "No CUDA device found"

    # Construct response
    return {
        "status": "Healthy",
        "message": "API is running successfully.",
        "cuda": {
            "available": cuda_available,
            "device": cuda_device
        }
    }
