from fastapi import FastAPI, HTTPException
from datetime import date
from pathlib import Path
import os

from api.schemas import PredictRequest, PredictResponse, TrainRequest, TrainResponse
from src.inference import predict_for_ticker
from src.train_one import train_and_save_for_ticker

app = FastAPI(title="Forecast API", version="1.0.0")

WINDOW_FIXED = 90
HORIZON_FIXED = 14

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = str((BASE_DIR / "models").resolve())

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    try:
        end_date_used = req.end_date or date.today()
        safe = req.ticker.upper().replace("/", "-")
        model_guess = os.path.join(MODELS_DIR, f"{safe}_w{WINDOW_FIXED}_h{HORIZON_FIXED}.pkl")
        scaler_guess = os.path.join(MODELS_DIR, f"{safe}_w{WINDOW_FIXED}_h{HORIZON_FIXED}_scaler.pkl")

        if (not req.force) and os.path.exists(model_guess) and os.path.exists(scaler_guess):
            return TrainResponse(
                ticker=req.ticker,
                window=WINDOW_FIXED,
                horizon=HORIZON_FIXED,
                trained=False,
                model_path=model_guess,
                scaler_path=scaler_guess,
                end_date_used=end_date_used,
            )

        train_and_save_for_ticker(
            ticker=req.ticker,
            start_date=req.start_date.isoformat(),
            end_date=end_date_used.isoformat(),
            window_size=WINDOW_FIXED,
            horizon=HORIZON_FIXED,
            models_dir=MODELS_DIR,
        )

        return TrainResponse(
            ticker=req.ticker,
            window=WINDOW_FIXED,
            horizon=HORIZON_FIXED,
            trained=True,
            model_path=model_guess,
            scaler_path=scaler_guess,
            end_date_used=end_date_used,
        )

    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Falha ao treinar: {ex}")

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    trained_now = False

    try:
        out = predict_for_ticker(
            ticker=req.ticker,
            start_date=req.start_date.isoformat(),
            end_date=req.end_date.isoformat() if req.end_date else None,
            window_size=WINDOW_FIXED,
            horizon=HORIZON_FIXED,
            models_dir=MODELS_DIR,
        )

        return PredictResponse(
            ticker=out["ticker"],
            window=out["window_size"],
            horizon=out["horizon"],
            predictions=out["predictions"],
            model_path=out["model_path"],
            scaler_path=out["scaler_path"],
            trained_now=trained_now,
        )

    except FileNotFoundError as e:
        if not req.auto_train:
            raise HTTPException(status_code=404, detail=str(e))

        try:
            end_date_used = req.end_date or date.today()

            train_and_save_for_ticker(
                ticker=req.ticker,
                start_date=req.start_date.isoformat(),
                end_date=end_date_used.isoformat(),
                window_size=WINDOW_FIXED,
                horizon=HORIZON_FIXED,
                models_dir=MODELS_DIR,
            )
            trained_now = True

            out = predict_for_ticker(
                ticker=req.ticker,
                start_date=req.start_date.isoformat(),
                end_date=end_date_used.isoformat(),
                window_size=WINDOW_FIXED,
                horizon=HORIZON_FIXED,
                models_dir=MODELS_DIR,
            )

            return PredictResponse(
                ticker=out["ticker"],
                window=out["window_size"],
                horizon=out["horizon"],
                predictions=out["predictions"],
                model_path=out["model_path"],
                scaler_path=out["scaler_path"],
                trained_now=trained_now,
            )

        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"Falha ao treinar/prever: {ex}")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))