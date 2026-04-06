# # # import os
# # # import io
# # # import logging
# # # from contextlib import asynccontextmanager
# # #
# # # import torch
# # # from fastapi import FastAPI, UploadFile, File, HTTPException
# # # from fastapi.responses import JSONResponse
# # # from PIL import Image
# # # import uvicorn
# # #
# # # from efficientnet_image_detector import DetectorInference
# # #
# # # # ==============================
# # # # CONFIGURATION
# # # # ==============================
# # #
# # # MODEL_PATH = "../deepfake-model-training/models/final_model_2.pth"
# # # DEVICE = "auto"
# # # IMAGE_SIZE = 380
# # #
# # # # ==============================
# # # # LOGGING
# # # # ==============================
# # #
# # # logging.basicConfig(level=logging.INFO)
# # # logger = logging.getLogger(__name__)
# # #
# # # detector = None
# # #
# # #
# # # # ==============================
# # # # LIFESPAN HANDLER (NEW METHOD)
# # # # ==============================
# # #
# # # @asynccontextmanager
# # # async def lifespan(app: FastAPI):
# # #     global detector
# # #
# # #     if not os.path.exists(MODEL_PATH):
# # #         logger.error(f"Model not found at {MODEL_PATH}")
# # #         raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
# # #
# # #     logger.info("Loading model...")
# # #     detector = DetectorInference(
# # #         model_path=MODEL_PATH,
# # #         device=DEVICE,
# # #         image_size=IMAGE_SIZE
# # #     )
# # #     logger.info("Model loaded successfully.")
# # #
# # #     yield
# # #
# # #     logger.info("Shutting down API...")
# # #
# # #
# # # app = FastAPI(
# # #     title="Deepfake Image Detection API",
# # #     version="1.0.0",
# # #     lifespan=lifespan
# # # )
# # #
# # #
# # # # ==============================
# # # # HEALTH CHECK
# # # # ==============================
# # #
# # # @app.get("/")
# # # def root():
# # #     return {"status": "API running", "model_loaded": detector is not None}
# # #
# # #
# # # # ==============================
# # # # SINGLE IMAGE PREDICTION
# # # # ==============================
# # #
# # # @app.post("/predict")
# # # async def predict_image(file: UploadFile = File(...)):
# # #     if detector is None:
# # #         raise HTTPException(status_code=500, detail="Model not loaded")
# # #
# # #     if not file.content_type.startswith("image/"):
# # #         raise HTTPException(status_code=400, detail="File must be an image")
# # #
# # #     try:
# # #         contents = await file.read()
# # #         image = Image.open(io.BytesIO(contents)).convert("RGB")
# # #
# # #         transform = detector.transform
# # #         image_tensor = transform(image).unsqueeze(0).to(detector.device)
# # #
# # #         detector.model.eval()
# # #         with torch.no_grad():
# # #             logits = detector.model(image_tensor)
# # #             probabilities = torch.softmax(logits, dim=1)
# # #
# # #         predicted_class = torch.argmax(probabilities, dim=1).item()
# # #         confidence = probabilities[0, predicted_class].item()
# # #
# # #         return JSONResponse({
# # #             "filename": file.filename,
# # #             "verdict": "DEEPFAKE" if predicted_class == 1 else "REAL",
# # #             "confidence": float(confidence),
# # #             "real_probability": float(probabilities[0, 0].item()),
# # #             "fake_probability": float(probabilities[0, 1].item())
# # #         })
# # #
# # #     except Exception as e:
# # #         logger.error(f"Prediction error: {e}")
# # #         raise HTTPException(status_code=500, detail=str(e))
# # #
# # #
# # # # ==============================
# # # # RUN SERVER
# # # # ==============================
# # #
# # # if __name__ == "__main__":
# # #     uvicorn.run(
# # #         "api_server:app",   # IMPORTANT: match filename
# # #         host="127.0.0.1",
# # #         port=8000,
# # #         reload=True
# # #     )
# #
# # import os
# # import io
# # import logging
# # from contextlib import asynccontextmanager
# #
# # import torch
# # from fastapi import FastAPI, UploadFile, File, HTTPException
# # from fastapi.responses import JSONResponse
# # from PIL import Image
# # import uvicorn
# # from huggingface_hub import hf_hub_download
# #
# # from efficientnet_image_detector import DetectorInference
# #
# # # ==============================
# # # CONFIGURATION
# # # ==============================
# #
# # MODEL_PATH = r"C:\Users\RITABRITA\Downloads\PythonProject\PythonProject\deepfake-image-detector\models\final_model_2.pth"
# # DEVICE = "auto"
# # IMAGE_SIZE = 380
# #
# # # ==============================
# # # LOGGING
# # # ==============================
# #
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)
# #
# # detector = None
# #
# #
# # # ==============================
# # # MODEL DOWNLOAD (HuggingFace)
# # # ==============================
# #
# # def ensure_model_exists():
# #     if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
# #         print(">>> Model not found locally. Downloading from Hugging Face...")
# #         os.makedirs("models", exist_ok=True)
# #         hf_hub_download(
# #             repo_id="RK-2910/deepfake-efficientnet-b4",
# #             filename="models/final_model_2.pth",
# #             local_dir="."
# #         )
# #         print(">>> Model downloaded successfully.")
# #     else:
# #         print(">>> Model already exists locally. Skipping download.")
# #
# #
# # # ==============================
# # # LIFESPAN HANDLER
# # # ==============================
# #
# # @asynccontextmanager
# # async def lifespan(app: FastAPI):
# #     global detector
# #
# #     print(">>> Lifespan started")
# #     print(f">>> MODEL_PATH = {MODEL_PATH}")
# #
# #     # Download model if not present
# #     ensure_model_exists()
# #
# #     print(f">>> File exists: {os.path.exists(MODEL_PATH)}")
# #     print(f">>> File size: {os.path.getsize(MODEL_PATH) / 1_000_000:.1f} MB")
# #
# #     print(">>> Loading model...")
# #     try:
# #         detector = DetectorInference(
# #             model_path=MODEL_PATH,
# #             device=DEVICE,
# #             image_size=IMAGE_SIZE
# #         )
# #         print(">>> Model loaded successfully!")
# #     except Exception as e:
# #         print(f">>> CRASH during model load: {e}")
# #         raise
# #
# #     yield
# #
# #     print(">>> Shutting down...")
# #
# #
# # app = FastAPI(
# #     title="Deepfake Image Detection API",
# #     version="1.0.0",
# #     lifespan=lifespan
# # )
# #
# #
# # # ==============================
# # # HEALTH CHECK
# # # ==============================
# #
# # @app.get("/")
# # def root():
# #     return {"status": "API running", "model_loaded": detector is not None}
# #
# #
# # # ==============================
# # # SINGLE IMAGE PREDICTION
# # # ==============================
# #
# # @app.post("/predict")
# # async def predict_image(file: UploadFile = File(...)):
# #     if detector is None:
# #         raise HTTPException(status_code=500, detail="Model not loaded")
# #
# #     if not file.content_type.startswith("image/"):
# #         raise HTTPException(status_code=400, detail="File must be an image")
# #
# #     try:
# #         contents = await file.read()
# #         image = Image.open(io.BytesIO(contents)).convert("RGB")
# #
# #         transform = detector.transform
# #         image_tensor = transform(image).unsqueeze(0).to(detector.device)
# #
# #         detector.model.eval()
# #         with torch.no_grad():
# #             logits = detector.model(image_tensor)
# #             probabilities = torch.softmax(logits, dim=1)
# #
# #         predicted_class = torch.argmax(probabilities, dim=1).item()
# #         confidence = probabilities[0, predicted_class].item()
# #
# #         return JSONResponse({
# #             "filename": file.filename,
# #             "verdict": "DEEPFAKE" if predicted_class == 1 else "REAL",
# #             "probability": float(confidence),
# #             "real_probability": float(probabilities[0, 0].item()),
# #             "fake_probability": float(probabilities[0, 1].item())
# #         })
# #
# #     except Exception as e:
# #         logger.error(f"Prediction error: {e}")
# #         raise HTTPException(status_code=500, detail=str(e))
# #
# #
# # # ==============================
# # # RUN SERVER
# # # ==============================
# #
# # if __name__ == "__main__":
# #     is_production = os.getenv("RAILWAY_ENVIRONMENT") is not None
# #
# #     uvicorn.run(
# #         "api_server:app",
# #         host="0.0.0.0" if is_production else "127.0.0.1",
# #         port=int(os.getenv("PORT", 8000)),
# #         reload=not is_production
# #     )
#
#
#
# # import os
# # import io
# # import logging
# # from contextlib import asynccontextmanager
# #
# # import torch
# # from fastapi import FastAPI, UploadFile, File, HTTPException
# # from fastapi.responses import JSONResponse
# # from PIL import Image
# # import uvicorn
# # from huggingface_hub import hf_hub_download
# #
# # from efficientnet_image_detector import DetectorInference
# #
# # # ==============================
# # # CONFIGURATION
# # # ==============================
# #
# # MODEL_PATH = "models/final_model_2.pth"   # relative path inside container
# # DEVICE = "cpu"                             # HF Spaces free tier = CPU only
# # IMAGE_SIZE = 380
# #
# # # ==============================
# # # LOGGING
# # # ==============================
# #
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)
# #
# # detector = None
# #
# #
# # # ==============================
# # # MODEL DOWNLOAD (HuggingFace)
# # # ==============================
# #
# # def ensure_model_exists():
# #     if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
# #         print(">>> Model not found locally. Downloading from Hugging Face...")
# #         os.makedirs("models", exist_ok=True)
# #         hf_hub_download(
# #             repo_id="RK-2910/deepfake-efficientnet-b4",
# #             filename="models/final_model_2.pth",
# #             local_dir=".",
# #             token=os.environ.get("HF_TOKEN")
# #         )
# #         print(">>> Model downloaded successfully.")
# #     else:
# #         print(">>> Model already exists locally. Skipping download.")
# #
# #
# # # ==============================
# # # LIFESPAN HANDLER
# # # ==============================
# #
# # @asynccontextmanager
# # async def lifespan(app: FastAPI):
# #     global detector
# #
# #     print(">>> Lifespan started")
# #     print(f">>> MODEL_PATH = {MODEL_PATH}")
# #
# #     ensure_model_exists()
# #
# #     print(f">>> File exists: {os.path.exists(MODEL_PATH)}")
# #     print(f">>> File size: {os.path.getsize(MODEL_PATH) / 1_000_000:.1f} MB")
# #
# #     print(">>> Loading model...")
# #     try:
# #         detector = DetectorInference(
# #             model_path=MODEL_PATH,
# #             device=DEVICE,
# #             image_size=IMAGE_SIZE
# #         )
# #         print(">>> Model loaded successfully!")
# #     except Exception as e:
# #         print(f">>> CRASH during model load: {e}")
# #         raise
# #
# #     yield
# #
# #     print(">>> Shutting down...")
# #
# #
# # app = FastAPI(
# #     title="Deepfake Image Detection API",
# #     version="1.0.0",
# #     lifespan=lifespan
# # )
# #
# #
# # # ==============================
# # # HEALTH CHECK
# # # ==============================
# #
# # @app.get("/")
# # def root():
# #     return {"status": "API running", "model_loaded": detector is not None}
# #
# #
# # # ==============================
# # # SINGLE IMAGE PREDICTION
# # # ==============================
# #
# # @app.post("/predict")
# # async def predict_image(file: UploadFile = File(...)):
# #     if detector is None:
# #         raise HTTPException(status_code=500, detail="Model not loaded")
# #
# #     if not file.content_type.startswith("image/"):
# #         raise HTTPException(status_code=400, detail="File must be an image")
# #
# #     try:
# #         contents = await file.read()
# #         image = Image.open(io.BytesIO(contents)).convert("RGB")
# #
# #         transform = detector.transform
# #         image_tensor = transform(image).unsqueeze(0).to(detector.device)
# #
# #         detector.model.eval()
# #         with torch.no_grad():
# #             logits = detector.model(image_tensor)
# #             probabilities = torch.softmax(logits, dim=1)
# #
# #         predicted_class = torch.argmax(probabilities, dim=1).item()
# #
# #         return JSONResponse({
# #             "filename": file.filename,
# #             "verdict": "DEEPFAKE" if predicted_class == 1 else "REAL",
# #             "fake_probability": float(probabilities[0, 1].item())
# #         })
# #
# #     except Exception as e:
# #         logger.error(f"Prediction error: {e}")
# #         raise HTTPException(status_code=500, detail=str(e))
# #
# #
# # # ==============================
# # # RUN SERVER (HF Spaces = port 7860)
# # # ==============================
# #
# # if __name__ == "__main__":
# #     uvicorn.run(
# #         "api_server:app",
# #         host="0.0.0.0",
# #         port=7860,
# #         reload=False
# #     )
#
#
#
# import os
# import io
# import logging
# from contextlib import asynccontextmanager
#
# import torch
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import JSONResponse
# from PIL import Image
# import uvicorn
# from huggingface_hub import hf_hub_download
#
# from efficientnet_image_detector import DetectorInference
#
# # ==============================
# # CONFIGURATION
# # ==============================
#
# MODEL_PATH = "models/final_model_2.pth"
# DEVICE = "cpu"
# IMAGE_SIZE = 380
#
# # ==============================
# # LOGGING
# # ==============================
#
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# detector = None
#
#
# # ==============================
# # MODEL DOWNLOAD (HuggingFace)
# # ==============================
#
# def ensure_model_exists():
#     if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
#         print(">>> Model not found locally. Downloading from Hugging Face...")
#         os.makedirs("models", exist_ok=True)
#         hf_hub_download(
#             repo_id="RK-2910/deepfake-efficientnet-b4",
#             filename="models/final_model_2.pth",
#             local_dir=".",
#             token=os.environ.get("HF_TOKEN")
#         )
#         print(">>> Model downloaded successfully.")
#     else:
#         print(">>> Model already exists locally. Skipping download.")
#
#
# # ==============================
# # LIFESPAN HANDLER
# # ==============================
#
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global detector
#
#     print(">>> Lifespan started")
#     print(f">>> MODEL_PATH = {MODEL_PATH}")
#
#     ensure_model_exists()
#
#     print(f">>> File exists: {os.path.exists(MODEL_PATH)}")
#     print(f">>> File size: {os.path.getsize(MODEL_PATH) / 1_000_000:.1f} MB")
#
#     print(">>> Loading model...")
#     try:
#         detector = DetectorInference(
#             model_path=MODEL_PATH,
#             device=DEVICE,
#             image_size=IMAGE_SIZE
#         )
#         print(">>> Model loaded successfully!")
#     except Exception as e:
#         print(f">>> CRASH during model load: {e}")
#         raise
#
#     yield
#
#     print(">>> Shutting down...")
#
#
# app = FastAPI(
#     title="Deepfake Image Detection API",
#     version="1.0.0",
#     lifespan=lifespan
# )
#
#
# # ==============================
# # HEALTH CHECK
# # ==============================
#
# @app.get("/")
# def root():
#     return {"status": "API running", "model_loaded": detector is not None}
#
#
# # ==============================
# # SINGLE IMAGE PREDICTION
# # ==============================
#
# @app.post("/predict")
# async def predict_image(file: UploadFile = File(...)):
#     if detector is None:
#         raise HTTPException(status_code=500, detail="Model not loaded")
#
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="File must be an image")
#
#     try:
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
#
#         transform = detector.transform
#         image_tensor = transform(image).unsqueeze(0).to(detector.device)
#
#         detector.model.eval()
#         with torch.no_grad():
#             logits = detector.model(image_tensor)
#             probabilities = torch.softmax(logits, dim=1)
#
#         predicted_class = torch.argmax(probabilities, dim=1).item()
#
#         return JSONResponse({
#             "filename": file.filename,
#             "verdict": "DEEPFAKE" if predicted_class == 1 else "REAL",
#             "fake_probability": float(probabilities[0, 1].item())
#         })
#
#     except Exception as e:
#         logger.error(f"Prediction error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# # ==============================
# # RUN SERVER
# # ==============================
#
# if __name__ == "__main__":
#     print(">>> Local test URL : http://localhost:7860")
#     print(">>> Swagger UI     : http://localhost:7860/docs")
#     uvicorn.run(
#         "api_server:app",
#         host="0.0.0.0",   # required for Docker / HF Spaces — use localhost:7860 in browser
#         port=7860,
#         reload=False
#     )


import os
import io
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn
from huggingface_hub import hf_hub_download

from efficientnet_image_detector import DetectorInference

# ==============================
# CONFIGURATION
# ==============================

MODEL_PATH = "models/final_model_2.pth"
DEVICE = "cpu"
IMAGE_SIZE = 380

# ==============================
# LOGGING
# ==============================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

detector = None


# ==============================
# MODEL DOWNLOAD (HuggingFace)
# ==============================

def ensure_model_exists():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
        print(">>> Model not found locally. Downloading from Hugging Face...")
        os.makedirs("models", exist_ok=True)
        hf_hub_download(
            repo_id="RK-2910/deepfake-efficientnet-b4",
            filename="models/final_model_2.pth",
            local_dir=".",
            token=os.environ.get("HF_TOKEN")
        )
        print(">>> Model downloaded successfully.")
    else:
        print(">>> Model already exists locally. Skipping download.")


# ==============================
# LIFESPAN HANDLER
# ==============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector

    print(">>> Lifespan started")
    print(f">>> MODEL_PATH = {MODEL_PATH}")

    ensure_model_exists()

    print(f">>> File exists: {os.path.exists(MODEL_PATH)}")
    print(f">>> File size: {os.path.getsize(MODEL_PATH) / 1_000_000:.1f} MB")

    print(">>> Loading model...")
    try:
        detector = DetectorInference(
            model_path=MODEL_PATH,
            device=DEVICE,
            image_size=IMAGE_SIZE
        )
        print(">>> Model loaded successfully!")
    except Exception as e:
        print(f">>> CRASH during model load: {e}")
        raise

    yield

    print(">>> Shutting down...")


app = FastAPI(
    title="Deepfake Image Detection API",
    version="1.0.0",
    lifespan=lifespan
)

# ==============================
# CORS — allow all origins
# ==============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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

        return JSONResponse({
            "filename": file.filename,
            "verdict": "DEEPFAKE" if predicted_class == 1 else "REAL",
            "fake_probability": float(probabilities[0, 1].item())
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# RUN SERVER
# ==============================

if __name__ == "__main__":
    print(">>> Local test URL : http://localhost:7860")
    print(">>> Swagger UI     : http://localhost:7860/docs")
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",   # required for Docker / HF Spaces — use localhost:7860 in browser
        port=7860,
        reload=False
    )