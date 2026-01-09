import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import yfinance as yf
from datetime import date
import pandas as pd
from plotly import graph_objs as go
import numpy as np
import joblib
import tensorflow as tf
import traceback

from src.train_one import sanitize_ticker, train_and_save_for_ticker

# configurações
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
WINDOW_SIZE = 90
HORIZON_MAX = 14

# ================================
# SIDEBAR
# ================================

def pegar_dados_criptos():
    path = os.path.join(PROJECT_ROOT, "csv", "criptos.csv")
    return pd.read_csv(path, delimiter=";")

df = pegar_dados_criptos()
cripto = df["NOME_ACAO"]
nome_cripto_escolhida = st.sidebar.selectbox("Escolha uma criptomoeda:", cripto)
n_dias = st.sidebar.slider("Selecione quantidade de dias para a previsão", 3, 14, value=14)

def_cripto = df[df["NOME_ACAO"] == nome_cripto_escolhida]
cripto_escolhida = def_cripto.iloc[0]["SIGLA"]



@st.cache_data(ttl=600)
def pegar_dados_yfinance(sigla_cripto):
    dados = yf.download(sigla_cripto, start=data_inicio, end=data_fim)
    dados.reset_index(inplace=True)
    return dados

data_inicio = "2018-01-01"
data_fim = date.today().strftime("%Y-%m-%d")

st.markdown(
    """
    <h1 style='text-align: center;'>📈 PREVISÃO DE PREÇOS (LSTM)</h1>
    <h1 style='text-align: center;'>🪙 CRIPTOATIVOS</h1>
    """,
    unsafe_allow_html=True
)
st.info(f"Criptomoeda selecionada: {nome_cripto_escolhida} ({cripto_escolhida})")

with st.expander("⚠️ Aviso sobre previsões e riscos de investimento"):
    st.markdown(
        """
        As previsões exibidas neste aplicativo são geradas por **modelos de Machine Learning**
        treinados a partir de **dados históricos**.

        **⚠️ Importante:**
        - As previsões **não constituem recomendação de investimento**.
        - **Não há garantia de ganhos ou retornos financeiros futuros**.
        - Resultados passados **não garantem desempenho futuro**.

        O mercado de **criptoativos é altamente volátil** e envolve riscos significativos.
        Utilize as informações apresentadas **apenas para fins educacionais e analíticos**.
        """
    )

dados_cripto = pegar_dados_yfinance(cripto_escolhida)

# ================================
# HISTÓRICO DE DADOS
# ================================
df_hist = dados_cripto.copy()

#As informações do yfinance pode conter colunas MultiIndex
if isinstance(df_hist.columns, pd.MultiIndex):
    df_hist.columns = [c[0] for c in df_hist.columns]

# =============================
# Gráfico Histórico
# =============================
st.markdown("---")
st.markdown(
    """
    <h2 style="text-align: center; font-weight: bold; color: #00e4ff;">
    INFORMAÇÕES HISTÓRICAS
    </h2>
    """,
    unsafe_allow_html=True
)
st.markdown("---")
st.markdown(
    """
    <h3>📊 Gráfico informações de fechamento</h3>
    """,
    unsafe_allow_html=True
)

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=df_hist["Date"],
    y=df_hist["Close"],
    mode="lines",
    line=dict(color="#00e4ff", width=2),
    name="Close"
))

fig_hist.update_layout(
    xaxis_title="Data",
    yaxis_title="Preço (Fechamento)",
    xaxis=dict(type="date")
)

st.plotly_chart(fig_hist, use_container_width=True)

# ================================
# Informações em tabela histórica
# ================================
st.markdown(
    """
    <h3>📋 Tabela de valores - Últimos 7 dias</h3>
    """,
    unsafe_allow_html=True
)
st.write(dados_cripto.tail(7))

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

def prever(model, scaler, close_series, n_dias_local):
    close_array = np.array(close_series).reshape(-1, 1)
    scaled = scaler.transform(close_array)
    last_window = scaled[-WINDOW_SIZE:, 0].reshape(1, WINDOW_SIZE, 1)
    pred_scaled = model.predict(last_window, verbose=0)
    pred_real = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    return pred_real[:n_dias_local]
st.markdown("---")

# ================================
# INÍCIO ETAPAS DE PREVISÃO
# ================================


st.markdown(
    """
    <h2 style="text-align: center; font-weight: bold; color: #00e4ff;">
    PREVISÃO COM LSTM
    </h2>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# ================================
# CARREGAR / TREINAR MODELO
# ================================
model, scaler, expected_model_path, expected_scaler_path = carregar_modelo_e_scaler(cripto_escolhida)

if model is None:
    st.warning("Modelo ainda não existe para este ticker.")

    if st.button("Treinar modelo dessa cripto (1ª vez)"):
        status = st.empty()
        status.info("⏳ Iniciando treino...")

        with st.spinner("Treinando modelo... (pode levar alguns minutos)"):
            try:
                model_path, scaler_path = train_and_save_for_ticker(
                    ticker=cripto_escolhida,
                    start_date=data_inicio,
                    end_date=data_fim,
                    window_size=WINDOW_SIZE,
                    horizon=HORIZON_MAX,
                    models_dir=MODELS_DIR,
                    epochs=30
                )

                status.success("✅ Treino finalizado! Recarregando...")
                st.success(f"✅ Modelo salvo:\n{model_path}\n{scaler_path}")

                # limpar cache para o Streamlit “ver” o modelo novo
                st.cache_resource.clear()
                st.rerun()

            except Exception as e:
                status.error("❌ Treino falhou")
                st.error(str(e))
                st.code(traceback.format_exc())

else:
    st.success("✅ Modelo carregado")

    # ================================
    # VALIDAR DADOS
    # ================================
    if "Close" not in df_hist.columns:
        st.error("Coluna Close não encontrada no histórico (df_hist).")
        st.stop()

    close_vals = df_hist["Close"].dropna().values
    if len(close_vals) < WINDOW_SIZE:
        st.error(f"Preciso de pelo menos {WINDOW_SIZE} dias para prever.")
        st.stop()

    # ================================
    # GERAR PREVISÃO (n_dias do slider)
    # ================================
    previsao = prever(model, scaler, close_vals, n_dias)

    ultima_data = pd.to_datetime(df_hist["Date"].iloc[-1])
    datas_futuras = pd.date_range(
        start=ultima_data + pd.Timedelta(days=1),
        periods=n_dias,
        freq="D"
    )

    df_prev = pd.DataFrame({
        "Data": datas_futuras,
        "Previsao_Close": previsao
    })

    # ================================
    # GRÁFICO: HISTÓRICO + PREVISÃO
    # ================================
    st.subheader("📊 Gráfico: Histórico + Previsão Fechamento")

    df_hist_plot = df_hist.copy()
    df_hist_plot["Date"] = pd.to_datetime(df_hist_plot["Date"], errors="coerce")

    df_prev_plot = df_prev.copy()
    df_prev_plot["Data"] = pd.to_datetime(df_prev_plot["Data"], errors="coerce")

    start_forecast = df_prev_plot["Data"].min()

    fig_all = go.Figure()

    fig_all.add_trace(go.Scatter(
        x=df_hist_plot["Date"],
        y=df_hist_plot["Close"],
        mode="lines", 
        line=dict(color="#00e4ff", width=2),
        name="Histórico (Close)"
    ))

    fig_all.add_trace(go.Scatter(
        x=df_prev_plot["Data"],
        y=df_prev_plot["Previsao_Close"],
        mode="lines+markers",
        line=dict(color="#005e77", width=2),
        name=f"Previsão ({n_dias} dias)"
    ))

    fig_all.add_vline(
        x=start_forecast,
        line_width=2,
        line=dict(color="#e6f4f1", width=1),
        line_dash="dash"
    )

    fig_all.update_layout(
        xaxis_title="Data",
        yaxis_title="Preço (Fechamento)",
        xaxis=dict(type="date")
    )

    st.plotly_chart(fig_all, use_container_width=True)

    # ================================
    # TABELA: PREVISÃO
    # ================================
    st.subheader("📋 Tabela de valores previstos")
    st.dataframe(df_prev, use_container_width=True)

# =============================
# Rodapé e informações da desenvolvedora
# =============================

st.markdown(
    """
    <hr>
    <div style='text-align: center; font-size: 14px;'>
        Desenvolvido por <b>Joyce Muniz</b><br>
        <a href='https://www.linkedin.com/in/joycemoliveira' target='_blank' style='text-decoration:none; color:gray;'>
            <img src='https://cdn-icons-png.flaticon.com/512/174/174857.png' width='18' style='vertical-align:middle; filter: grayscale(100%); margin-right:6px;'>
            joycemoliveira
        </a><br>
        <a href='https://github.com/joycemuniz' target='_blank' style='text-decoration:none; color:gray;'>
            <img src='https://icones.pro/wp-content/uploads/2021/06/icone-github-grise.png' width='23' style='vertical-align:middle; filter: grayscale(100%); margin-right:6px;'>
            joycemuniz
        </a>
    </div>
    """,
    unsafe_allow_html=True
)



