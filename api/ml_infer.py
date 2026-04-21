import os
import joblib
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# مسار الموديل الذي حفظته
MODEL_PATH = os.path.join("models", "rf_fusion.pkl")

# Cache (باش ما نعاودوش نحمّلو في كل request)
_RF = None
_CNN = None
_PREPROCESS = None
_DEVICE = None


def load_artifacts():
    global _RF, _CNN, _PREPROCESS, _DEVICE

    if _RF is not None:
        return

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}. Run scripts/04_train_rf_fusion.py first.")

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load RF
    _RF = joblib.load(MODEL_PATH)

    # Load CNN feature extractor
    weights = MobileNet_V2_Weights.DEFAULT
    cnn = mobilenet_v2(weights=weights)
    cnn.classifier = torch.nn.Identity()  # output: (1280,)
    cnn.eval()
    cnn.to(_DEVICE)

    _CNN = cnn
    _PREPROCESS = weights.transforms()


@torch.no_grad()
def extract_img_feat(img_path: str) -> np.ndarray:
    load_artifacts()

    img = Image.open(img_path).convert("RGB")
    x = _PREPROCESS(img).unsqueeze(0).to(_DEVICE)   # (1,3,H,W)
    feat = _CNN(x).detach().cpu().numpy().reshape(-1)  # (1280,)
    return feat


def predict_fusion(img_path: str, density: float, ph: float, flow_time: float):
    """
    Returns: (result_str, confidence_float, score_int, reason_str)
    """
    load_artifacts()

    img_feat = extract_img_feat(img_path)  # (1280,)
    num_feat = np.array([density, ph, flow_time], dtype=np.float32)  # (3,)

    X = np.concatenate([img_feat, num_feat], axis=0).reshape(1, -1)  # (1,1283)

    proba_adult = float(_RF.predict_proba(X)[0, 1])
    pred = int(_RF.predict(X)[0])  # 0 pure, 1 adulterated

    if pred == 1:
        result = "AI: Adulterated"
        confidence = proba_adult
    else:
        result = "AI: Pure"
        confidence = 1.0 - proba_adult

    # score في DB كان من -3..+3 في rules. نخليه هنا تقريب مشابه:
    # confidence 0.5..1.0 -> score 0..3
    score = int(round((confidence - 0.5) / 0.5 * 3))
    score = max(0, min(3, score))

    reason = f"AI fusion (CNN+phys): P(adulterated)={proba_adult:.3f}"
    return result, float(confidence), int(score), reason
