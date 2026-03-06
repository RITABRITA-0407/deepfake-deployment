import os
import io
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

from efficientnet_image_detector import DetectorInference

# ==============================
# CONFIGURATION
# ==============================

MODEL_PATH = "../deepfake-model-training/models/final_model_2.pth"
DEVICE = "auto"
IMAGE_SIZE = 380

# ==============================
# LOGGING
# ==============================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

detector = None


# ==============================
# LIFESPAN HANDLER (NEW METHOD)
# ==============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector

    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    logger.info("Loading model...")
    detector = DetectorInference(
        model_path=MODEL_PATH,
        device=DEVICE,
        image_size=IMAGE_SIZE
    )
    logger.info("Model loaded successfully.")

    yield

    logger.info("Shutting down API...")


app = FastAPI(
    title="Deepfake Image Detection API",
    version="1.0.0",
    lifespan=lifespan
)


# ==============================
# HEALTH CHECK
# ==============================

@app.get("/")
def root():
    return {"status": "API running", "model_loaded": detector is not None}


# ==============================
# SINGLE IMAGE PREDICTION
# ==============================

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        transform = detector.transform
        image_tensor = transform(image).unsqueeze(0).to(detector.device)

        detector.model.eval()
        with torch.no_grad():
            logits = detector.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)

        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

        return JSONResponse({
            "filename": file.filename,
            "verdict": "DEEPFAKE" if predicted_class == 1 else "REAL",
            "confidence": float(confidence),
            "real_probability": float(probabilities[0, 0].item()),
            "fake_probability": float(probabilities[0, 1].item())
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# RUN SERVER
# ==============================

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",   # IMPORTANT: match filename
        host="127.0.0.1",
        port=8000,
        reload=True
    )

# import os
# import io
# import logging
# from typing import List
# from contextlib import asynccontextmanager
# import torch
# from fastapi import FastAPI, UploadFile, File, HTTPException, Query
# from fastapi.responses import JSONResponse
# from PIL import Image
# import uvicorn
# from efficientnet_image_detector import DetectorInference
#
# # ==============================
# # CONFIGURATION
# # ==============================
# MODEL_PATH       = "models/final_model_2.pth"
# DEVICE           = "auto"
# IMAGE_SIZE       = 380
# DEFAULT_THRESHOLD = 0.5   # if fake_probability >= threshold → DEEPFAKE
#
# # ==============================
# # LOGGING
# # ==============================
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# detector = None
#
# # ==============================
# # LIFESPAN HANDLER
# # ==============================
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global detector
#     if not os.path.exists(MODEL_PATH):
#         logger.error(f"Model not found at {MODEL_PATH}")
#         raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
#     logger.info("Loading model...")
#     detector = DetectorInference(
#         model_path=MODEL_PATH,
#         device=DEVICE,
#         image_size=IMAGE_SIZE
#     )
#     logger.info("Model loaded successfully.")
#     yield
#     logger.info("Shutting down API...")
#
# app = FastAPI(
#     title="Deepfake Image Detection API",
#     version="1.0.0",
#     lifespan=lifespan
# )
#
# # ==============================
# # THRESHOLD HELPER
# # ==============================
# def apply_threshold(fake_prob: float, threshold: float) -> dict:
#     """
#     Verdict logic:
#       - fake_prob >= threshold          → DEEPFAKE
#       - fake_prob <  threshold          → REAL
#       - near boundary (±0.05)           → flag as low-confidence
#     """
#     is_fake        = fake_prob >= threshold
#     verdict        = "DEEPFAKE" if is_fake else "REAL"
#     confidence     = fake_prob if is_fake else (1.0 - fake_prob)
#     low_confidence = abs(fake_prob - threshold) < 0.05
#
#     return {
#         "verdict":        verdict,
#         "confidence":     round(confidence, 6),
#         "low_confidence": low_confidence,
#         "threshold_used": round(threshold, 4),
#     }
#
# # ==============================
# # HEALTH CHECK
# # ==============================
# @app.get("/")
# def root():
#     return {
#         "status":           "API running",
#         "model_loaded":     detector is not None,
#         "default_threshold": DEFAULT_THRESHOLD,
#     }
#
# # ==============================
# # SINGLE IMAGE PREDICTION
# # ==============================
# @app.post("/predict")
# async def predict_image(
#     file:      UploadFile = File(...),
#     threshold: float      = Query(
#         default=DEFAULT_THRESHOLD,
#         ge=0.0,
#         le=1.0,
#         description="Confidence threshold for DEEPFAKE verdict (0.0–1.0). "
#                     "Raise to reduce false positives; lower to reduce false negatives."
#     )
# ):
#     if detector is None:
#         raise HTTPException(status_code=500, detail="Model not loaded")
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="File must be an image")
#
#     try:
#         contents     = await file.read()
#         image        = Image.open(io.BytesIO(contents)).convert("RGB")
#         image_tensor = detector.transform(image).unsqueeze(0).to(detector.device)
#
#         detector.model.eval()
#         with torch.no_grad():
#             logits        = detector.model(image_tensor)
#             probabilities = torch.softmax(logits, dim=1)
#
#         real_prob = float(probabilities[0, 0].item())
#         fake_prob = float(probabilities[0, 1].item())
#
#         result = apply_threshold(fake_prob, threshold)
#
#         logger.info(
#             f"[{file.filename}] fake_prob={fake_prob:.4f} "
#             f"threshold={threshold} → {result['verdict']}"
#             + (" ⚠️ LOW CONFIDENCE" if result["low_confidence"] else "")
#         )
#
#         return JSONResponse({
#             "filename":         file.filename,
#             "verdict":          result["verdict"],
#             "confidence":       result["confidence"],
#             "low_confidence":   result["low_confidence"],
#             "threshold_used":   result["threshold_used"],
#             "real_probability": round(real_prob, 6),
#             "fake_probability": round(fake_prob, 6),
#         })
#
#     except Exception as e:
#         logger.error(f"Prediction error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
#
# # ==============================
# # BATCH PREDICTION
# # ==============================
# @app.post("/predict/batch")
# async def predict_batch(
#     files:     List[UploadFile] = File(...),
#     threshold: float            = Query(
#         default=DEFAULT_THRESHOLD,
#         ge=0.0,
#         le=1.0,
#         description="Shared confidence threshold applied to all images in the batch."
#     )
# ):
#     if detector is None:
#         raise HTTPException(status_code=500, detail="Model not loaded")
#     if not files:
#         raise HTTPException(status_code=400, detail="No files provided")
#
#     results = []
#     for file in files:
#         if not file.content_type.startswith("image/"):
#             results.append({
#                 "filename": file.filename,
#                 "error":    "Not a valid image file"
#             })
#             continue
#
#         try:
#             contents     = await file.read()
#             image        = Image.open(io.BytesIO(contents)).convert("RGB")
#             image_tensor = detector.transform(image).unsqueeze(0).to(detector.device)
#
#             detector.model.eval()
#             with torch.no_grad():
#                 logits        = detector.model(image_tensor)
#                 probabilities = torch.softmax(logits, dim=1)
#
#             real_prob = float(probabilities[0, 0].item())
#             fake_prob = float(probabilities[0, 1].item())
#             result    = apply_threshold(fake_prob, threshold)
#
#             logger.info(
#                 f"[{file.filename}] fake_prob={fake_prob:.4f} "
#                 f"threshold={threshold} → {result['verdict']}"
#                 + (" ⚠️ LOW CONFIDENCE" if result["low_confidence"] else "")
#             )
#
#             results.append({
#                 "filename":         file.filename,
#                 "verdict":          result["verdict"],
#                 "confidence":       result["confidence"],
#                 "low_confidence":   result["low_confidence"],
#                 "threshold_used":   result["threshold_used"],
#                 "real_probability": round(real_prob, 6),
#                 "fake_probability": round(fake_prob, 6),
#             })
#
#         except Exception as e:
#             logger.error(f"Error processing {file.filename}: {e}")
#             results.append({
#                 "filename": file.filename,
#                 "error":    str(e)
#             })
#
#     summary = {
#         "total":          len(results),
#         "deepfakes":      sum(1 for r in results if r.get("verdict") == "DEEPFAKE"),
#         "real":           sum(1 for r in results if r.get("verdict") == "REAL"),
#         "errors":         sum(1 for r in results if "error" in r),
#         "low_confidence": sum(1 for r in results if r.get("low_confidence")),
#         "threshold_used": threshold,
#     }
#
#     return JSONResponse({"summary": summary, "results": results})
#
# # ==============================
# # RUN SERVER
# # ==============================
# if __name__ == "__main__":
#     uvicorn.run(
#         "api_server:app",
#         host="127.0.0.1",
#         port=8000,
#         reload=True
#     )