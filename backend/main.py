from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import shutil
from pathlib import Path
import base64
from io import BytesIO
import numpy as np
from PIL import Image

import config
from model_inference import get_model
from grad_cam import generate_gradcam
from mock_data import get_mock_analysis_result, get_sample_result


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup/shutdown"""
    # Startup
    print("Loading model...")
    get_model()
    print("Model loaded successfully!")
    yield
    # Shutdown (if needed)


app = FastAPI(
    title="CheXCA API",
    description="Chest X-ray AI Diagnosis with Explainable AI",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CheXCA API - Chest X-ray AI Diagnosis",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "diseases": config.DISEASE_CLASSES
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict diseases from chest X-ray image

    Args:
        file: Uploaded X-ray image

    Returns:
        predictions: Disease probabilities
        top_predictions: Top 5 most likely diseases
        co_occurrence: Disease co-occurrence matrix
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Save uploaded file
        upload_path = config.UPLOAD_DIR / file.filename
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Get model and run inference
        model = get_model()
        predictions = model.predict(str(upload_path))

        # Get top predictions
        top_predictions = model.get_top_predictions(predictions, top_k=5)

        # Calculate co-occurrence matrix
        co_occurrence = model.calculate_co_occurrence(predictions)

        return {
            "predictions": predictions,
            "top_predictions": [
                {"disease": disease, "probability": prob}
                for disease, prob in top_predictions
            ],
            "co_occurrence": co_occurrence.tolist(),
            "disease_classes": config.DISEASE_CLASSES,
            "image_path": str(upload_path)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/explain")
async def explain(file: UploadFile = File(...)):
    """
    Generate explainable AI visualization (Grad-CAM)

    Args:
        file: Uploaded X-ray image

    Returns:
        heatmap: Base64 encoded heatmap overlay image
        predictions: Disease probabilities
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Save uploaded file
        upload_path = config.UPLOAD_DIR / file.filename
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Get model and preprocess image
        model_instance = get_model()
        image_tensor, original_image = model_instance.preprocess_image(str(upload_path))

        # Run prediction first
        predictions = model_instance.predict(str(upload_path))

        # Get top class for Grad-CAM
        top_class_idx = np.argmax(list(predictions.values()))

        # Generate Grad-CAM
        cam, overlay_image = generate_gradcam(
            model_instance.model,
            image_tensor,
            original_image,
            target_class=top_class_idx
        )

        # Convert overlay image to base64
        buffered = BytesIO()
        overlay_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Convert CAM to base64
        cam_image = Image.fromarray((cam * 255).astype(np.uint8))
        cam_buffered = BytesIO()
        cam_image.save(cam_buffered, format="PNG")
        cam_str = base64.b64encode(cam_buffered.getvalue()).decode()

        return {
            "heatmap_overlay": f"data:image/png;base64,{img_str}",
            "heatmap_raw": f"data:image/png;base64,{cam_str}",
            "predictions": predictions,
            "top_disease": config.DISEASE_CLASSES[top_class_idx],
            "disease_classes": config.DISEASE_CLASSES
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Complete analysis: prediction + explanation

    Args:
        file: Uploaded X-ray image

    Returns:
        Complete analysis with predictions, top diseases, co-occurrence, and heatmap
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # MOCK MODE - Return simulated data for demo
        if config.MOCK_MODE:
            print("[MOCK MODE] Returning simulated 14-class analysis")
            return get_sample_result(0)

        # REAL MODE - Use actual model
        # Save uploaded file
        upload_path = config.UPLOAD_DIR / file.filename
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Get model
        model_instance = get_model()

        # Run prediction
        predictions = model_instance.predict(str(upload_path))
        top_predictions = model_instance.get_top_predictions(predictions, top_k=5)
        co_occurrence = model_instance.calculate_co_occurrence(predictions)

        # Generate Grad-CAM
        image_tensor, original_image = model_instance.preprocess_image(str(upload_path))
        top_class_idx = np.argmax(list(predictions.values()))

        cam, overlay_image = generate_gradcam(
            model_instance.model,
            image_tensor,
            original_image,
            target_class=top_class_idx
        )

        # Convert images to base64
        buffered = BytesIO()
        overlay_image.save(buffered, format="PNG")
        heatmap_str = base64.b64encode(buffered.getvalue()).decode()

        original_buffered = BytesIO()
        original_image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE)).save(original_buffered, format="PNG")
        original_str = base64.b64encode(original_buffered.getvalue()).decode()

        return {
            "predictions": predictions,
            "top_predictions": [
                {"disease": disease, "probability": prob}
                for disease, prob in top_predictions
            ],
            "co_occurrence": co_occurrence.tolist(),
            "disease_classes": config.DISEASE_CLASSES,
            "heatmap_overlay": f"data:image/png;base64,{heatmap_str}",
            "original_image": f"data:image/png;base64,{original_str}",
            "top_disease": config.DISEASE_CLASSES[top_class_idx],
            "top_disease_probability": float(list(predictions.values())[top_class_idx])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    print(f"Starting CheXCA API server on {config.HOST}:{config.PORT}")
    print(f"Model path: {config.MODEL_PATH}")
    print(f"Disease classes: {len(config.DISEASE_CLASSES)}")
    uvicorn.run(app, host=config.HOST, port=config.PORT)
