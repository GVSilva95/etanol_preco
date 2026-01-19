
import os
import base64
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

# ==============================================================================
# CONFIG
# ==============================================================================
st.set_page_config(page_title="Etanol Intelligence Pro", page_icon="⛽", layout="wide")

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_PATH = (PROJECT_ROOT / "data" / "processed" / "dataset_consolidado.csv").resolve()

TARGET = "Preco_Etanol"
BASE_DRIVERS = ["Petroleo_Brent", "Dolar", "Acucar"]
REQUIRED_COLS = BASE_DRIVERS + [TARGET]

TICKERS = {
    "Brent": ("BZ=F", "US$"),
    "Dólar (BRL)": ("BRL=X", "R$"),
    "Açúcar (NY)": ("SB=F", "¢"),
    "Milho (Chicago)": ("ZC=F", "¢"),
    "Gasolina RBOB": ("RB=F", "US$"),
    "Gás Natural": ("NG=F", "US$"),
    "Juros EUA 10Y": ("^TNX", "%"),
}

# ==============================================================================
# HELPERS
# ==============================================================================
def get_img_as_base64(filename: str):
    candidates = [
        APP_DIR / filename,
        PROJECT_ROOT / filename,
        APP_DIR / "images" / filename,
        PROJECT_ROOT / "images" / filename,
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, "rb") as f:
                    return base64.b64encode(f.read()).decode()
            except Exception:
                return None
    return None

def fmt_num(x, decimals=2):
    try:
        return f"{float(x):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "-"

def to_display(v, unidade: str):
    v = float(v)
    return v / 1000.0 if unidade == "R$/L" else v

def unit_label(unidade: str):
    return "/L" if unidade == "R$/L" else "/m³"

@st.cache_data
def carregar_dados(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df

@st.cache_data(ttl=1800)
def get_market_data():
    out = {}
    for name, (ticker, unit) in TICKERS.items():
        try:
            h = yf.Ticker(ticker).history(period="7d").dropna()
            if len(h) >= 2:
                val = float(h["Close"].iloc[-1])
                prev = float(h["Close"].iloc[-2])
                out[name] = {"val": val, "delta": val - prev, "unit": unit}
            elif len(h) == 1:
                val = float(h["Close"].iloc[-1])
                out[name] = {"val": val, "delta": 0.0, "unit": unit}
            else:
                out[name] = {"val": 0.0, "delta": 0.0, "unit": unit}
        except Exception:
            out[name] = {"val": 0.0, "delta": 0.0, "unit": unit}
    return out

def build_features(df: pd.DataFrame, freq_rule: str, max_lag: int, ma_window: int):
    df2 = df.copy()

    missing = [c for c in REQUIRED_COLS if c not in df2.columns]
    if missing:
        raise ValueError(f"Colunas faltando no dataset: {missing}\nDisponíveis: {list(df2.columns)}")

    df2 = df2.sort_index()

    # Alinhar frequência (isso costuma corrigir R² muito ruim)
    df2 = df2.resample(freq_rule).mean()

    df2["Mes"] = df2.index.month

    features = []
    for c in BASE_DRIVERS:
        features.append(c)
        for lag in range(1, max_lag + 1):
            col = f"{c}_lag{lag}"
            df2[col] = df2[c].shift(lag)
            features.append(col)
        ma_col = f"{c}_ma{ma_window}"
        df2[ma_col] = df2[c].rolling(ma_window).mean()
        features.append(ma_col)

    features.append("Mes")

    df2 = df2.dropna(subset=features + [TARGET]).copy()
    return df2, features

@st.cache_resource
def train_model(df: pd.DataFrame, freq_rule: str, max_lag: int, ma_window: int):
    df2, features = build_features(df, freq_rule=freq_rule, max_lag=max_lag, ma_window=ma_window)

    X = df2[features]
    y = df2[TARGET]

    n_splits = 5
    if len(df2) < 60:
        n_splits = 3
    if len(df2) < 30:
        n_splits = 2

    tscv = TimeSeriesSplit(n_splits=n_splits)
    train_idx, test_idx = list(tscv.split(X))[-1]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestRegressor(
        n_estimators=600,
        max_depth=14,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Baseline (naive): usa o último valor observado (t-1)
    y_naive = y_test.shift(1)
    if pd.isna(y_naive.iloc[0]):
        y_naive.iloc[0] = y_train.iloc[-1]

    metrics = {
        "r2_test": r2_score(y_test, y_pred),
        "mae_test": mean_absolute_error(y_test, y_pred),
        "mape_test": mean_absolute_percentage_error(y_test, y_pred),
        "r2_naive": r2_score(y_test, y_naive),
        "mae_naive": mean_absolute_error(y_test, y_naive),
        "mape_naive": mean_absolute_percentage_error(y_test, y_naive),
        "test_start": X_test.index.min(),
        "test_end": X_test.index.max(),
    }

    # Para gráficos
    df2["Preco_Justo_Modelo"] = model.predict(X)
    df2["Spread"] = df2["Preco_Justo_Modelo"] - df2[TARGET]

    bundle = {
        "features": features,
        "y_test": y_test,
        "y_pred": pd.Series(y_pred, index=y_test.index),
    }
    return model, df2, metrics, bundle

def render_ticker_cards(market_data: dict):
    cols = st.columns(7)
    names = list(TICKERS.keys())
    for i, name in enumerate(names):
        d = market_data.get(name, {})
        val = d.get("val", 0.0)
        delta = d.get("delta", 0.0)
        unit = d.get("unit", "")
        sign = "+" if delta >= 0 else ""
        color = "#00FF7F" if delta >= 0 else "#ff4d4d"
        html = f"""
        <div class="glass" style="min-height:92px;">
          <div style="font-size:12px;opacity:.85">{name}</div>
          <div style="font-size:22px;font-weight:900;white-space:nowrap">{unit} {fmt_num(val,2)}</div>
          <div style="font-size:12px;font-weight:800;color:{color};white-space:nowrap">{sign}{fmt_num(delta,2)}</div>
        </div>
        """
        cols[i].markdown(html, unsafe_allow_html=True)

# ==============================================================================
# VISUAL (CSS seguro: sem f-string, sem .format)
# ==============================================================================
bg_base64 = get_img_as_base64("fundo_cana.jpg")
bg_url = (
    f"data:image/jpg;base64,{bg_base64}"
    if bg_base64
    else "https://images.unsplash.com/photo-1633004147966-c1713534327d?q=80&w=1920"
)

css = """
<style>
[data-testid="stAppViewContainer"] {
  background-image: linear-gradient(rgba(0,0,0,0.78), rgba(0,0,0,0.92)), url("__BG_URL__");
  background-size: cover;
  background-attachment: fixed;
}
[data-testid="stSidebar"] {
  background-color: rgba(12,16,12,0.90);
  border-right: 1px solid rgba(0, 255, 127, 0.10);
}
.glass {
  background: rgba(30,30,30,0.45);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 14px 16px;
  backdrop-filter: blur(6px);
}
</style>
""".replace("__BG_URL__", bg_url)

st.markdown(css, unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
logo_base64 = get_img_as_base64("logo_projeto.jpg")
if logo_base64:
    st.sidebar.markdown(
        f'<img src="data:image/jpg;base64,{logo_base64}" style="width:100%; border-radius:14px; margin-bottom:14px;">',
        unsafe_allow_html=True,
    )

st.sidebar.subheader("Configurações")
unidade = st.sidebar.selectbox("Exibir Etanol em", ["R$/m³", "R$/L"], index=0)
freq_opt = st.sidebar.selectbox("Frequência p/ modelagem", ["Semanal", "Mensal", "Diária"], index=0)
freq_rule = {"Semanal": "W-FRI", "Mensal": "M", "Diária": "D"}[freq_opt]
max_lag = st.sidebar.slider("Defasagens (lags)", 1, 8, 4)
ma_window = st.sidebar.slider("Média móvel (janelas)", 3, 16, 4)

# ==============================================================================
# LOAD + TRAIN
# ==============================================================================
df_raw = carregar_dados(DATA_PATH)
if df_raw is None:
    st.error(f"Dataset não encontrado em:\n{DATA_PATH}")
    st.stop()

market = get_market_data()
model, dfm, metrics, bundle = train_model(df_raw, freq_rule=freq_rule, max_lag=max_lag, ma_window=ma_window)

last_date = dfm.index.max()
last_real = float(dfm[TARGET].iloc[-1])
last_fair = float(dfm["Preco_Justo_Modelo"].iloc[-1])
last_spread = float(dfm["Spread"].iloc[-1])

st.sidebar.divider()
st.sidebar.subheader("Modelo (teste temporal)")
st.sidebar.metric("R² (teste)", f"{metrics['r2_test']:.3f}")
st.sidebar.metric("MAPE (teste)", f"{metrics['mape_test']:.1%}")
st.sidebar.caption(f"Teste: {metrics['test_start']:%d/%m/%Y} → {metrics['test_end']:%d/%m/%Y}")

st.sidebar.subheader("Baseline (naive)")
st.sidebar.metric("R² (baseline)", f"{metrics['r2_naive']:.3f}")
st.sidebar.metric("MAPE (baseline)", f"{metrics['mape_naive']:.1%}")

# ==============================================================================
# UI
# ==============================================================================
st.title("⛽ Etanol Intelligence Pro")
st.caption("Valuation + Paridade + Gráficos (com resample + lags + baseline)")
render_ticker_cards(market)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Gráficos", "⚖️ Paridade", "🧠 Diagnóstico do Modelo"])

with tab1:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Etanol (Real) {unit_label(unidade)}", f"R$ {fmt_num(to_display(last_real, unidade), 3)}", help=f"Data: {last_date:%d/%m/%Y}")
    k2.metric(f"Preço justo {unit_label(unidade)}", f"R$ {fmt_num(to_display(last_fair, unidade), 3)}")
    k3.metric(f"Spread {unit_label(unidade)}", f"R$ {fmt_num(to_display(last_spread, unidade), 3)}")
    pct = (last_fair / last_real - 1) * 100 if last_real else 0
    k4.metric("Sinal (%)", f"{fmt_num(pct, 1)}%")

    min_d, max_d = dfm.index.min().date(), dfm.index.max().date()
    d1, d2 = st.date_input("Período", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    dff = dfm.loc[str(d1):str(d2)].copy()

    dff["Etanol_disp"] = dff[TARGET].apply(lambda x: to_display(x, unidade))
    dff["Justo_disp"] = dff["Preco_Justo_Modelo"].apply(lambda x: to_display(x, unidade))
    dff["Spread_disp"] = dff["Spread"].apply(lambda x: to_display(x, unidade))

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=dff.index, y=dff["Etanol_disp"], name="Etanol (Real)", mode="lines"))
    fig1.add_trace(go.Scatter(x=dff.index, y=dff["Justo_disp"], name="Preço Justo (Modelo)", mode="lines"))
    fig1.update_layout(
        title=f"Real vs Preço Justo ({unit_label(unidade)})",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(dff, x=dff.index, y="Spread_disp", title=f"Spread (Justo - Real) ({unit_label(unidade)})", template="plotly_dark")
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

    cols_corr = [c for c in [TARGET] + BASE_DRIVERS if c in dff.columns]
    corr = dff[cols_corr].corr()
    fig3 = px.imshow(corr, text_auto=True, title="Correlação (período selecionado)", template="plotly_dark")
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.subheader("Simulador de bomba")
    c1, c2 = st.columns(2)
    with c1:
        gas = st.number_input("Gasolina (R$/L)", value=5.80, step=0.05)
        eta = st.number_input("Etanol (R$/L)", value=3.60, step=0.05)
        thr = st.slider("Limite (%)", 60, 85, 70)
    with c2:
        ratio = (eta / gas) * 100 if gas else 0
        eta_max = gas * (thr / 100)
        st.metric("Paridade", f"{fmt_num(ratio,1)}%")
        st.metric("Etanol compensa até", f"R$ {fmt_num(eta_max,2)}/L")
        st.success("✅ Etanol vantajoso") if ratio < thr else st.error("❌ Gasolina vantajosa")

with tab3:
    y_test = bundle["y_test"]
    y_pred = bundle["y_pred"]

    st.subheader("Predito vs Real (teste)")
    df_sc = pd.DataFrame({"Real": y_test.values, "Predito": y_pred.values}, index=y_test.index)
    df_sc["Real_disp"] = df_sc["Real"].apply(lambda x: to_display(x, unidade))
    df_sc["Predito_disp"] = df_sc["Predito"].apply(lambda x: to_display(x, unidade))

    fig4 = px.scatter(df_sc, x="Real_disp", y="Predito_disp", title=f"Teste: Predito vs Real ({unit_label(unidade)})", template="plotly_dark")
    mn = float(np.nanmin([df_sc["Real_disp"].min(), df_sc["Predito_disp"].min()]))
    mx = float(np.nanmax([df_sc["Real_disp"].max(), df_sc["Predito_disp"].max()]))
    fig4.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="Linha 45°"))
    fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Métricas")
    a, b, c = st.columns(3)
    a.metric("R² (modelo)", f"{metrics['r2_test']:.3f}")
    b.metric("MAPE (modelo)", f"{metrics['mape_test']:.1%}")
    c.metric("MAE (modelo)", f"{fmt_num(metrics['mae_test'], 3)}")

    a2, b2, c2 = st.columns(3)
    a2.metric("R² (baseline)", f"{metrics['r2_naive']:.3f}")
    b2.metric("MAPE (baseline)", f"{metrics['mape_naive']:.1%}")
    c2.metric("MAE (baseline)", f"{fmt_num(metrics['mae_naive'], 3)}")

