from fastapi import FastAPI
from src.app.routes import inference,health

app = FastAPI(title="Super Resolution Dental X-ray API", version="1.0.0")

# Include API routes
app.include_router(inference.router, prefix="/inference", tags=["Inference"])
app.include_router(health.router, prefix="/health", tags=["Health"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Super Resolution Dental X-ray API"}
