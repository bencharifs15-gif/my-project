# api/models.py
from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime
from .database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    result = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)

    # stored filename in /uploads
    image_path = Column(String, nullable=True)

    density = Column(Float, nullable=True)
    ph = Column(Float, nullable=True)
    flow_time = Column(Float, nullable=True)

    # NEW: for explainability + filtering
    score = Column(Integer, nullable=True)
    reason = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
