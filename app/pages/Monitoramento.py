import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import yfinance as yf
from datetime import date
import pandas as pd
from plotly import graph_objs as go
import joblib
import tensorflow as tf
import traceback

from src.train_one import sanitize_ticker, train_and_save_for_ticker
from src.monitoring import backtest_metrics_multistep, append_metrics_log

# configurações
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
WINDOW_SIZE = 90
HORIZON_MAX = 14

MAPE_ALERT_THRESHOLD = 25.0
MAPE_RETRAIN_THRESHOLD = 30.0

data_inicio = "2018-01-01"
data_fim = date.today().strftime("%Y-%m-%d")

st.title("🧪 MONITORAMENTO DO MODELO")

def classificar_mape(mape_val: float) -> str:
    if mape_val <= 10:
        return "🟢 Excelente (≤ 10%)"
    elif mape_val <= 15:
        return "🟢 Bom (10–15%)"
    elif mape_val <= 25:
        return "🟡 Razoável (15–25%)"
    else:
        return "🔴 Fraco (> 25%)"

st.sidebar.header("Seleção")
def pegar_dados_criptos():
    path = os.path.join(PROJECT_ROOT, "csv", "criptos.csv")
    return pd.read_csv(path, delimiter=";")

df = pegar_dados_criptos()
nome_cripto_escolhida = st.sidebar.selectbox("Escolha uma criptomoeda:", df["NOME_ACAO"])
cripto_escolhida = df[df["NOME_ACAO"] == nome_cripto_escolhida].iloc[0]["SIGLA"]

st.write(f"Ticker: {nome_cripto_escolhida} ({cripto_escolhida})")

@st.cache_data(ttl=600)
def pegar_dados_yfinance(sigla_cripto):
    dados = yf.download(sigla_cripto, start=data_inicio, end=data_fim)
    dados.reset_index(inplace=True)
    return dados

dados_cripto = pegar_dados_yfinance(cripto_escolhida)

@st.cache_resource
def carregar_modelo_e_scaler(ticker: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    safe = sanitize_ticker(ticker)
    model_path = os.path.join(MODELS_DIR, f"{safe}_w{WINDOW_SIZE}_h{HORIZON_MAX}.keras")
    scaler_path = os.path.join(MODELS_DIR, f"{safe}_scaler.pkl")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler, model_path, scaler_path

    return None, None, model_path, scaler_path

model, scaler, model_path, scaler_path = carregar_modelo_e_scaler(cripto_escolhida)

if model is None:
    st.warning("Modelo ainda não existe para este ticker. Treine na página Home (Previsão).")
    st.stop()

st.success("✅ Modelo carregado")

st.markdown("---")
st.subheader("Backtest e Métricas")
backtest_days = st.slider("Janela de backtest (dias)", 60, 180, 90, step=30)

if st.button("Calcular métricas do modelo"):
    with st.spinner("Calculando..."):
        try:
            close_df = dados_cripto[["Close"]].copy()

            metrics = backtest_metrics_multistep(
                model=model,
                scaler=scaler,
                close_df=close_df,
                window_size=WINDOW_SIZE,
                horizon=HORIZON_MAX,
                backtest_days=backtest_days
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{metrics['mae_global']:.2f}")
            col2.metric("RMSE", f"{metrics['rmse_global']:.2f}")
            col3.metric("MAPE", f"{metrics['mape_global']:.2f}%")

            df_by_day = pd.DataFrame({
                "Dia": list(range(1, HORIZON_MAX + 1)),
                "MAE": metrics["mae_by_day"],
                "RMSE": metrics["rmse_by_day"],
                "MAPE(%)": metrics["mape_by_day"],
            })
            st.dataframe(df_by_day, use_container_width=True)

            log_path = os.path.join(MODELS_DIR, "metrics_log.csv")
            row = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "ticker": cripto_escolhida,
                "window_size": WINDOW_SIZE,
                "horizon": HORIZON_MAX,
                "backtest_days": backtest_days,
                "samples": metrics["samples"],
                "mae_global": metrics["mae_global"],
                "rmse_global": metrics["rmse_global"],
                "mape_global": metrics["mape_global"],
            }
            append_metrics_log(log_path, row)
            st.success("✅ Log atualizado (metrics_log.csv)")
        except Exception as e:
            st.error(str(e))
            st.code(traceback.format_exc())

st.markdown("---")
st.subheader("Tendência do MAPE + Alertas")

log_path = os.path.join(MODELS_DIR, "metrics_log.csv")
if os.path.exists(log_path):
    df_log = pd.read_csv(log_path)
    df_ticker = df_log[df_log["ticker"] == cripto_escolhida].copy()

    if len(df_ticker) == 0:
        st.info("Sem histórico para este ticker ainda.")
    else:
        df_ticker["timestamp"] = pd.to_datetime(df_ticker["timestamp"], errors="coerce")
        df_ticker = df_ticker.dropna(subset=["timestamp"]).sort_values("timestamp")

        last_mape = float(df_ticker["mape_global"].iloc[-1])
        st.write(f"Status atual: **{classificar_mape(last_mape)}**")

        if last_mape > MAPE_RETRAIN_THRESHOLD:
            st.error(f"🚨 ALERTA CRÍTICO: MAPE {last_mape:.2f}% (> {MAPE_RETRAIN_THRESHOLD}%). Re-treinar recomendado.")
        elif last_mape > MAPE_ALERT_THRESHOLD:
            st.warning(f"⚠️ ALERTA: MAPE {last_mape:.2f}% (> {MAPE_ALERT_THRESHOLD}%). Considere re-treinar.")
        else:
            st.success(f"✅ OK: MAPE {last_mape:.2f}% dentro do esperado.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_ticker["timestamp"], y=df_ticker["mape_global"], mode="lines+markers", name="MAPE (%)"))
        fig.update_layout(title="MAPE global (%) ao longo do tempo", xaxis_title="Timestamp", yaxis_title="MAPE (%)")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df_ticker.tail(20), use_container_width=True)

        if st.button("🔁 Re-treinar modelo (atualizar)"):
            with st.spinner("Re-treinando..."):
                try:
                    train_and_save_for_ticker(
                        ticker=cripto_escolhida,
                        start_date="2018-01-01",
                        end_date=date.today().strftime("%Y-%m-%d"),
                        window_size=WINDOW_SIZE,
                        horizon=HORIZON_MAX,
                        models_dir=MODELS_DIR,
                        epochs=30
                    )
                    st.success("✅ Re-treino concluído. Limpando cache e recarregando...")
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
                    st.code(traceback.format_exc())
else:
    st.info("Ainda não existe o log `metrics_log.csv`. Calcule métricas para criar o primeiro registro.")
