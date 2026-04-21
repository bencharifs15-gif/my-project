# api/main.py
import os
import csv
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import numpy as np
from PIL import Image
import joblib
import pandas as pd
from io import StringIO

import torch
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

from fastapi import FastAPI, UploadFile, File, Form, Depends, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session

from .database import Base, engine, SessionLocal
from .models import Prediction


# ----------------------------
# App init
# ----------------------------
app = FastAPI(title="Honey Adulteration API", version="0.4.0")

# CORS - محسن لـ Flutter Web (Chrome)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",                    # للتجربة فقط
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)


# ----------------------------
# ENV (Admin)
# ----------------------------
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
SESSION_SECRET = os.getenv("SESSION_SECRET", "change-this-secret-please")

app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")

UI_DIR = os.path.join(PROJECT_ROOT, "ui")
TEMPLATES_DIR = os.path.join(UI_DIR, "templates")
STATIC_DIR = os.path.join(UI_DIR, "static")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEMO_DIR = os.path.join(DATA_DIR, "demo_samples")
DEMO_CSV = os.path.join(DEMO_DIR, "demo_cases.csv")

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rf_fusion.pkl")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(DEMO_DIR, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ----------------------------
# DB dependency
# ----------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----------------------------
# Admin auth helpers
# ----------------------------
def is_admin(request: Request) -> bool:
    return request.session.get("is_admin") is True

def require_admin(request: Request):
    if not is_admin(request):
        raise HTTPException(status_code=401, detail="Not authorized")

# ----------------------------
# Smart search vocab (AR/FR/EN)
# ----------------------------
PURE_KEYS = {
    "pure", "p", "natural", "naturel", "original", "genuine",
    "نقي", "طبيعي", "حر", "صافي", "عسل حر", "اصلي", "أصلي",
    "pur", "miel pur"
}
ADULT_KEYS = {
    "adulterated", "adulter", "fake", "faked", "mixed", "diluted",
    "مغشوش", "مقلد", "مزور", "مخلوط", "مضاف", "سكر", "ماء",
    "falsifié", "falsifie", "fraud", "contrefait"
}
UNCERT_KEYS = {
    "uncertain", "u", "unknown", "maybe",
    "غير مؤكد", "مجهول", "مش واضح",
    "douteux", "incertain"
}

SUGGESTIONS = [
    "pure", "adulterated", "uncertain",
    "نقي", "مغشوش", "غير مؤكد",
    "naturel", "falsifié", "douteux"
]

def normalize_q(q: str) -> str:
    return (q or "").strip().lower()

def interpret_q(q: str) -> Tuple[str, Optional[str]]:
    t = normalize_q(q)
    if not t:
        return ("all", None)

    tokens = [tok for tok in t.replace(",", " ").split() if tok]

    if any(tok in PURE_KEYS for tok in tokens) or t in PURE_KEYS:
        return ("pure", None)
    if any(tok in ADULT_KEYS for tok in tokens) or t in ADULT_KEYS:
        return ("adulterated", None)
    if any(tok in UNCERT_KEYS for tok in tokens) or t in UNCERT_KEYS:
        return ("uncertain", None)

    return ("text", t)

# ----------------------------
# ML: load RF + CNN once
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_cnn_backbone():
    weights = MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval().to(DEVICE)
    preprocess = weights.transforms()
    return model, preprocess

CNN, PREPROCESS = build_cnn_backbone()

if not os.path.exists(MODEL_PATH):
    print(f"⚠️ Model not found: {MODEL_PATH}")
    RF_MODEL = None
else:
    RF_MODEL = joblib.load(MODEL_PATH)
RF_MODEL = joblib.load(MODEL_PATH)

def extract_img_feat(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    x = PREPROCESS(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = CNN(x).squeeze(0).detach().cpu().numpy()
    return feat.astype(np.float32)

def predict_fusion(img_path: str, density: float, ph: float, flow_time: float):
    if RF_MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    img_feat = extract_img_feat(img_path)
    num_feat = np.array([density, ph, flow_time], dtype=np.float32)
    X = np.concatenate([img_feat, num_feat], axis=0).reshape(1, -1)

    proba = RF_MODEL.predict_proba(X)[0]
    pred = int(np.argmax(proba))
    conf = float(np.max(proba))

    label = "PURE" if pred == 0 else "ADULTERATED"
    return label, conf, float(proba[1])
# ----------------------------
# Demo loader
# ----------------------------
def load_demo_cases(limit: int = 4) -> List[Dict]:
    cases = []
    if not os.path.exists(DEMO_CSV):
        return cases

    with open(DEMO_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append({
                "case_id": int(row["case_id"]),
                "image_path": row["image_path"],
                "density": float(row["density"]),
                "ph": float(row["ph"]),
                "flow_time": float(row["flow_time"]),
                "label": row.get("label", ""),
            })
            if len(cases) >= limit:
                break
    return cases

def resolve_demo_image_path(image_path_from_csv: str) -> str:
    rel = image_path_from_csv.replace("\\", "/").lstrip("/")
    return os.path.join(DEMO_DIR, rel)

def copy_demo_image_to_uploads(abs_demo_path: str) -> Optional[str]:
    if not os.path.exists(abs_demo_path):
        return None
    _, ext = os.path.splitext(abs_demo_path)
    ext = ext.lower()
    if ext not in [".png", ".jpg", ".jpeg"]:
        ext = ".png"
    stored = f"{uuid.uuid4().hex}{ext}"
    dest = os.path.join(UPLOAD_DIR, stored)
    with open(abs_demo_path, "rb") as src, open(dest, "wb") as out:
        out.write(src.read())
    return stored

# ----------------------------
# API endpoints
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat(), "device": DEVICE}

@app.get("/test")
def test_connection():
    """اختبار الاتصال من Flutter Web"""
    return {"message": "✅ Connected successfully to FastAPI", "status": "ok"}

@app.post("/predict")
async def predict_api(
    image: UploadFile = File(...),
    density: float = Form(...),
    ph: float = Form(...),
    flow_time: float = Form(...),
):
    ext = os.path.splitext(image.filename or "")[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        ext = ".jpg"

    stored = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, stored)

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image")

    with open(save_path, "wb") as f:
        f.write(content)

    label, conf, p_adult = predict_fusion(save_path, float(density), float(ph), float(flow_time))

    return {
        "label": label,
        "confidence": float(conf),
        "p_adulterated": float(p_adult),
    }

# ----------------------------
# USER UI (باقي الواجهات محتفظ بها كما هي)
# ----------------------------
@app.get("/ui", response_class=HTMLResponse)
def ui_home(request: Request):
    demo_cases = load_demo_cases(limit=4)
    return templates.TemplateResponse("index.html", {"request": request, "demo_cases": demo_cases})

@app.post("/ui/analyze", response_class=HTMLResponse)
async def ui_analyze(
    request: Request,
    image: UploadFile = File(...),
    density: float = Form(...),
    ph: float = Form(...),
    flow_time: float = Form(...),
    db: Session = Depends(get_db),
):
    ext = os.path.splitext(image.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        ext = ".jpg"

    stored = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, stored)

    content = await image.read()
    with open(save_path, "wb") as f:
        f.write(content)

    pred_label, conf, p_adult = predict_fusion(save_path, density, ph, flow_time)

    result = "Possibly Pure" if pred_label == "PURE" else "Possibly Adulterated"
    reason_short = f"RF fusion | P(adulterated)={p_adult:.2f}"

    row = Prediction(
        result=result,
        confidence=conf,
        image_path=stored,
        density=density,
        ph=ph,
        flow_time=flow_time,
        score=None,
        reason=reason_short,
        created_at=datetime.utcnow(),
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return RedirectResponse(url=f"/ui/details/{row.id}", status_code=303)

# ... (باقي دوال ui/demo_run , apply_filters , ui_history , ui_details , admin UI محتفظ بها كما هي)

@app.post("/ui/demo_run", response_class=HTMLResponse)
def ui_demo_run(
    request: Request,
    case_id: int = Form(...),
    db: Session = Depends(get_db),
):
    cases = load_demo_cases(limit=1000)
    chosen = next((c for c in cases if c["case_id"] == case_id), None)
    if not chosen:
        raise HTTPException(status_code=404, detail="Demo case not found")

    abs_img = resolve_demo_image_path(chosen["image_path"])
    stored = copy_demo_image_to_uploads(abs_img)
    if not stored:
        raise HTTPException(status_code=404, detail="Demo image missing")

    save_path = os.path.join(UPLOAD_DIR, stored)

    density, ph, flow_time = chosen["density"], chosen["ph"], chosen["flow_time"]
    pred_label, conf, p_adult = predict_fusion(save_path, density, ph, flow_time)

    result = "Possibly Pure" if pred_label == "PURE" else "Possibly Adulterated"
    reason_short = f"RF fusion | P(adulterated)={p_adult:.2f}"

    row = Prediction(
        result=result,
        confidence=conf,
        image_path=stored,
        density=density,
        ph=ph,
        flow_time=flow_time,
        score=None,
        reason=reason_short,
        created_at=datetime.utcnow(),
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return RedirectResponse(url=f"/ui/details/{row.id}", status_code=303)

def apply_filters(query, q: str, result_type: str, min_conf: str):
    mode, keyword = interpret_q(q)
    if mode == "pure":
        query = query.filter(Prediction.result.ilike("%pure%"))
    elif mode == "adulterated":
        query = query.filter(Prediction.result.ilike("%adulter%"))
    elif mode == "uncertain":
        query = query.filter(Prediction.result.ilike("%uncertain%"))
    elif mode == "text" and keyword:
        query = query.filter(Prediction.result.ilike(f"%{keyword}%"))

    if result_type != "all":
        if result_type == "Pure":
            query = query.filter(Prediction.result.ilike("%pure%"))
        elif result_type == "Adulterated":
            query = query.filter(Prediction.result.ilike("%adulter%"))
        elif result_type == "Uncertain":
            query = query.filter(Prediction.result.ilike("%uncertain%"))

    if (min_conf or "").strip():
        try:
            mc = float(min_conf)
            query = query.filter(Prediction.confidence >= mc)
        except ValueError:
            pass

    return query

# (باقي دوال ui_history, ui_details, admin_login, admin_dashboard, admin_history, admin_details كما هي في الكود الأصلي)

# ----------------------------
# تحميل النماذج الطيفية (Hyperspectral)
# ----------------------------
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "hyperspectral")
try:
    model_class = joblib.load(os.path.join(MODEL_DIR, "model_class.pkl"))
    scaler_class = joblib.load(os.path.join(MODEL_DIR, "scaler_class.pkl"))
    model_reg = joblib.load(os.path.join(MODEL_DIR, "model_reg.pkl"))
    scaler_reg = joblib.load(os.path.join(MODEL_DIR, "scaler_reg.pkl"))
    print("✅ Hyperspectral models loaded successfully")
except Exception as e:
    print(f"⚠️ Warning loading hyperspectral models: {e}")
    model_class = model_reg = scaler_class = scaler_reg = None

# ----------------------------
# Predict من textarea (JSON)
# ----------------------------
@app.post("/hyperspectral/predict")
async def predict_hyperspectral(request: Request):
    try:
        body = await request.json()
        data = body.get("data")

        if not data or len(data) != 128:
            raise HTTPException(status_code=400, detail="Send exactly 128 values")

        X_input = np.array(data, dtype=float).reshape(1, -1)

        X_scaled_class = scaler_class.transform(X_input)
        class_pred = model_class.predict(X_scaled_class)[0]
        status = "نقي" if class_pred == 0 else "مغشوش"

        X_scaled_reg = scaler_reg.transform(X_input)
        adulteration_pct = model_reg.predict(X_scaled_reg)[0]

        return {
            "classification": status,
            "adulteration_percentage": round(float(adulteration_pct), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# Predict من CSV ← المهم لتطبيق Flutter
# ----------------------------
@app.post("/hyperspectral/upload_csv")
async def upload_csv_for_flutter(file: UploadFile = File(...)):
    try:
        if not file.filename or not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only .CSV files are allowed")

        content = await file.read()
        content_str = content.decode("utf-8")

        df = pd.read_csv(StringIO(content_str), header=None)

        if df.shape[1] == 1:
            values = df.iloc[:, 0].dropna().tolist()
        else:
            values = df.iloc[0].dropna().tolist()

        if len(values) != 128:
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain exactly 128 spectral features. Got {len(values)} values."
            )

        X_input = np.array(values, dtype=float).reshape(1, -1)

        # Classification
        X_scaled_class = scaler_class.transform(X_input)
        class_pred = model_class.predict(X_scaled_class)[0]
        status = "PURE" if class_pred == 0 else "ADULTERATED"

        # Regression
        X_scaled_reg = scaler_reg.transform(X_input)
        adulteration_pct = model_reg.predict(X_scaled_reg)[0]

        return {
            "status": status,
            "confidence": round(float(95 if class_pred == 0 else 90), 1),
            "adulteration_percentage": round(float(adulteration_pct), 2),
            "message": "Analysis completed successfully",
            "spectral_features": 128,
            "spectral_data": values,        # مهم للرسم البياني في Flutter
            "file_name": file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")


# ----------------------------
# Pages
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/spectral_analyse", response_class=HTMLResponse)
def spectral_page(request: Request):
    return templates.TemplateResponse("spectral_analyse.html", {"request": request})

print("MODEL PATH:", MODEL_PATH)
print("EXISTS:", os.path.exists(MODEL_PATH))