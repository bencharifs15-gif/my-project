# 🧠 AI Prediction API (FastAPI)

This project is a machine learning API built باستخدام **FastAPI** لعمل تنبؤات اعتمادًا على نماذج مدربة مسبقًا (مثل `.pkl` أو `PyTorch`).

---

## 🚀 Features

* ⚡ FastAPI backend (سريع وخفيف)
* 🤖 دعم نماذج Machine Learning (Scikit-learn / PyTorch)
* 📡 REST API جاهزة للاستخدام
* 🧪 اختبار محلي باستخدام Uvicorn
* ☁️ قابل للنشر على Render بسهولة

---

## 📁 Project Structure

```
my-project/
│
├── api/
│   └── main.py        # نقطة تشغيل API
│
├── models/
│   └── model.pkl      # الموديل المدرب
│
├── requirements.txt   # المكتبات
└── README.md
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
uvicorn api.main:app --reload
```

ثم افتح:

```
http://127.0.0.1:8000/docs
```

---

## 🌐 Deployment (Render)

* Build Command:

```
pip install -r requirements.txt
```

* Start Command:

```
uvicorn api.main:app --host 0.0.0.0 --port 10000
```

---

## 📡 Example Endpoint

```http
POST /predict
```

### Request:

```json
{
  "data": [1, 2, 3]
}
```

### Response:

```json
{
  "prediction": "result"
}
```

---

## 🧠 Model

* النموذج محفوظ داخل مجلد `models/`
* يتم تحميله باستخدام `joblib` أو `torch`

---

## 📌 Notes

* تأكد أن ملف `.pkl` موجود داخل `models/`
* لا ترفع ملفات كبيرة (>100MB) على GitHub

---

## 👨‍💻 Author

* Developed by: Your Name
* GitHub: https://github.com/your-username

---
