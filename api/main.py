# api/main.py

import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import numpy as np
import joblib
import pandas as pd
from io import StringIO

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------
# App init
# ----------------------------
app = FastAPI(title="Honey API (Light)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rf_fusion.pkl")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------------
# Load RF Model فقط
# ----------------------------
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    RF_MODEL = None
else:
    RF_MODEL = joblib.load(MODEL_PATH)
    print("✅ RF model loaded")

# ----------------------------
# Dummy features بدل CNN
# ----------------------------
def extract_img_feat():
    return np.zeros(1280, dtype=np.float32)  # مهم نفس الحجم

# ----------------------------
# Prediction
# ----------------------------
def predict_fusion(density: float, ph: float, flow_time: float):

    if RF_MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    img_feat = extract_img_feat()
    num_feat = np.array([density, ph, flow_time], dtype=np.float32)

    X = np.concatenate([img_feat, num_feat]).reshape(1, -1)

    proba = RF_MODEL.predict_proba(X)[0]
    pred = int(np.argmax(proba))

    label = "PURE" if pred == 0 else "ADULTERATED"

    return {
        "label": label,
        "confidence": float(np.max(proba)),
        "p_adulterated": float(proba[1])
    }

# ----------------------------
# API
# ----------------------------
@app.get("/")
def home():
    return {"message": "API is running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    density: float = Form(...),
    ph: float = Form(...),
    flow_time: float = Form(...),
):
    # نحفظ الصورة فقط (بدون استخدامها)
    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)

    content = await image.read()
    with open(path, "wb") as f:
        f.write(content)

    return predict_fusion(density, ph, flow_time)

# ----------------------------
# CSV hyperspectral
# ----------------------------
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "hyperspectral")

try:
    model_class = joblib.load(os.path.join(MODEL_DIR, "model_class.pkl"))
    scaler_class = joblib.load(os.path.join(MODEL_DIR, "scaler_class.pkl"))
    model_reg = joblib.load(os.path.join(MODEL_DIR, "model_reg.pkl"))
    scaler_reg = joblib.load(os.path.join(MODEL_DIR, "scaler_reg.pkl"))
    print("✅ hyperspectral loaded")
except:
    model_class = model_reg = scaler_class = scaler_reg = None

@app.post("/hyperspectral/upload_csv")
async def upload_csv(file: UploadFile = File(...)):

    content = await file.read()
    df = pd.read_csv(StringIO(content.decode("utf-8")), header=None)

    values = df.iloc[0].dropna().tolist()

    if len(values) != 128:
        raise HTTPException(status_code=400, detail="Need 128 values")

    X = np.array(values).reshape(1, -1)

    status = "UNKNOWN"
    pct = 0

    if model_class:
        Xc = scaler_class.transform(X)
        pred = model_class.predict(Xc)[0]
        status = "PURE" if pred == 0 else "ADULTERATED"

        Xr = scaler_reg.transform(X)
        pct = float(model_reg.predict(Xr)[0])

    return {
        "status": status,
        "adulteration_percentage": round(pct, 2)
    }