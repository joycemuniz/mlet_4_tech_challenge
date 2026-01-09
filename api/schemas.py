from datetime import date
from pydantic import BaseModel, Field
from typing import List, Optional

# -------------------------
# Treinamento
# -------------------------
class TrainRequest(BaseModel):
    ticker: str = Field(..., examples=["BTC-USD"])
    start_date: date = Field(..., examples=["2023-01-01"])
    end_date: Optional[date] = Field(default=None, examples=["2023-12-31"])
    window: int = Field(default=90, ge=10, le=500, examples=[90])
    horizon: int = Field(default=14, ge=1, le=60, examples=[14])
    force: bool = Field(
        default=False,
        description="Se True, retreina mesmo se já existir modelo."
    )

class TrainResponse(BaseModel):
    ticker: str
    window: int
    horizon: int
    trained: bool
    model_path: str
    scaler_path: str
    end_date_used: date

# -------------------------
# Predição
# -------------------------
class PredictRequest(BaseModel):
    ticker: str = Field(..., examples=["BTC-USD"])
    start_date: date = Field(default=date(2018, 1, 1), examples=["2018-01-01"])
    end_date: Optional[date] = Field(default=None, examples=["2026-01-01"])
    window: int = Field(default=90, ge=10, le=500, examples=[90])
    horizon: int = Field(default=14, ge=1, le=60, examples=[14])
    auto_train: bool = Field(default=False, description="Se True, treina automaticamente se não existir modelo.")

class PredictResponse(BaseModel):
    ticker: str
    window: int
    horizon: int
    predictions: List[float]
    model_path: str
    scaler_path: str
    trained_now: bool
