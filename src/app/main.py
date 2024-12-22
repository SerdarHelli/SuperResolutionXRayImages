from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.app.routes import inference, health
from src.app.exceptions import ModelLoadError, PreprocessingError, InferenceError,InputError, PostprocessingError


app = FastAPI(title="Super Resolution Dental X-ray API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Include API routes
app.include_router(inference.router, prefix="/inference", tags=["Inference"])
app.include_router(health.router, prefix="/health", tags=["Health"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Super Resolution Dental X-ray API"}

# Custom exception handlers
@app.exception_handler(ModelLoadError)
async def model_load_error_handler(request: Request, exc: ModelLoadError):
    return JSONResponse(
        status_code=500,
        content={"error": "ModelLoadError", "message": exc.message},
    )

@app.exception_handler(PreprocessingError)
async def preprocessing_error_handler(request: Request, exc: PreprocessingError):
    return JSONResponse(
        status_code=400,
        content={"error": "PreprocessingError", "message": exc.message},
    )

@app.exception_handler(PostprocessingError)
async def postprocessing_error_handler(request: Request, exc: PostprocessingError):
    return JSONResponse(
        status_code=400,
        content={"error": "PostprocessingError", "message": exc.message},
    )

@app.exception_handler(InferenceError)
async def inference_error_handler(request: Request, exc: InferenceError):
    return JSONResponse(
        status_code=500,
        content={"error": "InferenceError", "message": exc.message},
    )


@app.exception_handler(InputError)
async def input_load_error_handler(request: Request, exc: InputError):
    return JSONResponse(
        status_code=500,
        content={"error": "InputError", "message": exc.message},
    )