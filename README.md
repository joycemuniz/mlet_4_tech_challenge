# 📈 Previsão de Preços LSTM - Criptoativos 

Aplicação desenvolvida por **API em FastAPI** e uma **aplicação em Streamlit**, para **previsão de preços de criptoativos** utilizando **modelos de Deep Learning (LSTM)** treinados com dados históricos obtidos via **Yahoo Finance**.


O objetivo do projeto é demonstrar, de forma prática, todo o ciclo de um modelo de ML:
- exploração e preparação de dados
- treinamento e salvamento de modelos
- inferência e monitoramento
- visualização interativa dos resultados

---

## 🧠 Decisão de Arquitetura

Para garantir **consistência**, **estabilidade** e evitar erros entre treino e inferência, os parâmetros abaixo do modelo foram **fixados no código**:

- 📌 **Janela de observação (`window`)**: **90 dias**
- 📌 **Horizonte máximo de previsão (`horizon`)**: **14 dias**

Esses parâmetros **não são configuráveis pelo usuário** apenas para efeito de visualização no app.

## 🧩 Arquitetura Geral

```text
Usuário
  │
  ▼
Streamlit App  ───►  FastAPI  ───►  Modelo LSTM
  │                   │
  │                   ├── /train
  │                   └── /predict
  ▼
Visualização de Previsões

```

> Observação: o projeto também permite **execução local do treino** via Streamlit para fins educacionais e demonstração.

---


## 🖥️ Tecnologias Utilizadas

- Python 3.11
- Streamlit
- TensorFlow / Keras
- Pandas / NumPy
- yFinance
- Plotly
- Joblib

---

## 📂 Estrutura do Projeto

```text
.
├── .streamlit/
│   └── config.toml             # Configurações do Streamlit
├── api/
│   ├── main.py                 # API FastAPI
│   └── schemas.py              # Schemas Pydantic
├── app/
│   ├── Home.py                 # Página principal do Streamlit
│   └── pages/
│       └── Monitoramento.py    # Página de monitoramento
├── csv/
│   └── criptos.csv             # Lista de criptoativos disponíveis
├── models/
│   ├── BTC-USD_w90_h14.keras
│   └── BTC-USD_w90_h14_scaler.pkl
├── notebook/
│   └── lstm_multistep.ipynb    # Notebook exploratório
├── src/
│   ├── __init__.py
│   ├── features.py             # Engenharia de features
│   ├── inference.py            # Inferência do modelo
│   ├── model.py                # Arquitetura LSTM
│   ├── monitoring.py           # Métricas e monitoramento
│   └── train_one.py            # Treinamento e persistência
├── requirements.txt
└── README.md
```

---

## ▶️ Como Executar o Projeto Localmente

### 1️⃣ Criar ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
.venv\Scripts\activate      # Windows
```

### 2️⃣ Instalar dependências

```bash
pip install -r requirements.txt
```

---

## 🔌 Executando a API (FastAPI)

```bash
uvicorn api.main:app --reload
```

Documentação automática:
- Swagger: http://localhost:8000/docs

---

## 🖥️ Executando o App Streamlit

```bash
streamlit run app/Home.py
```

Acesse:
```
http://localhost:8501
```

---

## 🧩 Funcionalidades do Streamlit

- Seleção de criptoativos
- Visualização do histórico de preços
- Treinamento do modelo (quando inexistente)
- Geração de previsões futuras
- Gráficos interativos
- Página dedicada de monitoramento do modelo

---

## 📊 Modelo de Machine Learning

- Tipo: **LSTM (Long Short-Term Memory)**
- Entrada: preços de fechamento históricos
- Normalização via `scaler`
- Persistência:
  - Modelo: `.keras`
  - Scaler: `.pkl`

Padrão de nomenclatura:
```
{TICKER}_w90_h14.keras
{TICKER}_w90_h14_scaler.pkl
```

---

## ⚠️ Aviso Importante

> As previsões apresentadas neste projeto **não constituem recomendação de investimento**.  
> Criptoativos são altamente voláteis e envolvem riscos.  
> Projeto com finalidade **educacional e demonstrativa**.

---

## 🚀 Próximas Evoluções

- Automação de retreino
- Versionamento de modelos
- Observabilidade de modelos

---

## 👩‍💻 Autora

**Joyce Muniz**

- 🔗 LinkedIn: https://www.linkedin.com/in/joycemoliveira  
- 💻 GitHub: https://github.com/joycemuniz
