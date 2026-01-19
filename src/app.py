cat > app.py <<'PY'
# --- COLE AQUI O CODIGO LIMPO (SEM <<<<<<<) ---
# Se preferir, apague esta linha e cole o codigo inteiro que vou deixar abaixo.

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


st.set_page_config(
    page_title="Etanol Intelligence Pro",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path("data/processed/dataset_consolidado.csv")

TARGET = "Preco_Etanol"
FEATURES = ["Petroleo_Brent", "Dolar", "Acucar", "Mes"]
REQUIRED_COLS = ["Petroleo_Brent", "Dolar", "Acucar", "Preco_Etanol"]

TICKERS = {
    "Brent": ("BZ=F", "US$"),
    "Dólar (BRL)": ("BRL=X", "R$"),
    "Açúcar (NY)": ("SB=F", "¢"),
    "Milho (Chicago)": ("ZC=F", "¢"),
    "Gasolina RBOB": ("RB=F", "US$"),
    "Gás Natural": ("NG=F", "US$"),
    "Juros EUA 10Y": ("^TNX", "%"),
}


def get_img_as_base64(file_path: str):
    possible_paths = [file_path, os.path.join(".", file_path), os.path.join("..", file_path)]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    return base64.b64encode(f.read()).decode()
            except Exception:
                return None
    return None


bg_base64 = get_img_as_base64("fundo_cana.jpg")
logo_base64 = get_img_as_base64("logo_projeto.jpg")

bg_url = f"data:image/jpg;base64,{bg_base64}" if bg_base64 else "https://images.unsplash.com/photo-1633004147966-c1713534327d?q=80&w=1920"
logo_html = (
    f'<img src="data:image/jpg;base64,{logo_base64}" style="width: 100%; border-radius: 14px; margin-bottom: 18px;">'
    if logo_base64
    else ""
)

st.markdown(
    f"""
<style>
    [data-testid="stAppViewContainer"] {{
        background-image: linear-gradient(rgba(0,0,0,0.78), rgba(0,0,0,0.92)), url("{bg_url}");
        background-size: cover;
        background-attachment: fixed;
    }}

    [data-testid="stSidebar"] {{
        background-color: rgba(12, 16, 12, 0.90);
        border-right: 1px solid rgba(0, 255, 127, 0.10);
    }}

    .glass {{
        background: rgba(30, 30, 30, 0.45);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 14px 16px;
        backdrop-filter: blur(6px);
    }}

    .stButton > button {{
        background: linear-gradient(to bottom, #00FF7F 0%, #00CC66 100%);
        color: #002200;
        font-weight: 900;
        border: none;
        border-bottom: 4px solid #00994D;
        border-radius: 10px;
        padding: 12px 18px;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.12s ease-in-out;
        box-shadow: 0px 6px 14px rgba(0, 255, 127, 0.25);
    }}
    .stButton > button:active {{
        transform: translateY(4px);
        border-bottom: 0px;
        box-shadow: inset 0px 2px 6px rgba(0,0,0,0.25);
    }}
</style>
""",
    unsafe_allow_html=True,
)


def fmt_num(x, decimals=2):
    try:
        return f"{float(x):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "-"


@st.cache_data
def carregar_dados():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)


@st.cache_data(ttl=1800)
def get_market_data():
    out = {}
    for name, (ticker, unit) in TICKERS.items():
        try:
            h = yf.Ticker(ticker).history(period="7d").dropna()
            if len(h) >= 2:
                val = float(h["Close"].iloc[-1])
                prev = float(h["Close"].iloc[-2])
                out[name] = {"val": val, "delta": val - prev, "unit": unit, "date": h.index[-1]}
            elif len(h) == 1:
                val = float(h["Close"].iloc[-1])
                out[name] = {"val": val, "delta": 0.0, "unit": unit, "date": h.index[-1]}
            else:
                out[name] = {"val": 0.0, "delta": 0.0, "unit": unit, "date": None}
        except Exception:
            out[name] = {"val": 0.0, "delta": 0.0, "unit": unit, "date": None}
    return out


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    missing = [c for c in REQUIRED_COLS if c not in df2.columns]
    if missing:
        raise ValueError(f"Colunas faltando no dataset: {missing}")
    df2["Mes"] = df2.index.month
    df2 = df2.dropna(subset=REQUIRED_COLS + ["Mes"])
    return df2


@st.cache_resource
def train_model(df: pd.DataFrame):
    df2 = prepare_df(df)
    X = df2[FEATURES]
    y = df2[TARGET]

    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    metrics = {
        "r2_test": r2_score(y_test, y_pred_test),
        "mae_test": mean_absolute_error(y_test, y_pred_test),
        "mape_test": mean_absolute_percentage_error(y_test, y_pred_test),
        "test_start": X_test.index.min(),
        "test_end": X_test.index.max(),
    }

    last_price = float(df2[TARGET].iloc[-1])
    last_date = df2.index[-1]

    df2["Preco_Justo_Modelo"] = model.predict(X)
    df2["Spread"] = df2["Preco_Justo_Modelo"] - df2[TARGET]

    test_bundle = {"X_test": X_test, "y_test": y_test, "y_pred_test": y_pred_test}
    return model, df2, metrics, last_price, last_date, test_bundle


df_raw = carregar_dados()
market = get_market_data()

if df_raw is None:
    st.error("Dataset não encontrado em data/processed/dataset_consolidado.csv")
    st.stop()

try:
    model, df_model, metrics, last_price, last_date, test_bundle = train_model(df_raw)
except Exception as e:
    st.error("Erro ao preparar modelo/dados")
    st.write(str(e))
    st.stop()


with st.sidebar:
    if logo_base64:
        st.markdown(logo_html, unsafe_allow_html=True)
    else:
        st.header("Etanol Intelligence Pro")

    st.subheader("Navegação")
    page = st.radio(
        "Ir para",
        ["Visão Geral", "Valuation (IA)", "Paridade", "Histórico & Modelo"],
        label_visibility="collapsed",
    )

    st.divider()
    unidade_etanol = st.selectbox(
        "Unidade do preço do etanol no dataset",
        ["R$/L", "R$/m³ (divide por 1000 para R$/L)"],
        index=0,
    )

    def etanol_to_display(v):
        if unidade_etanol.startswith("R$/m³"):
            return float(v) / 1000.0
        return float(v)

    st.divider()
    st.subheader("Modelo (teste temporal)")
    st.metric("R² (teste)", f"{metrics['r2_test']:.3f}")
    st.metric("MAPE (teste)", f"{metrics['mape_test']:.1%}")
    st.caption(f"Teste: {metrics['test_start']:%d/%m/%Y} -> {metrics['test_end']:%d/%m/%Y}")

    st.divider()
    st.caption("Desenvolvido por")
    st.markdown("**Giovanni Silva**")


def render_ticker_cards(market_data: dict):
    cols = st.columns(7)
    names = list(TICKERS.keys())
    for i, name in enumerate(names):
        d = market_data.get(name, {})
        val = d.get("val", 0.0)
        delta = d.get("delta", 0.0)
        unit = d.get("unit", "")
        delta_color = "#00FF7F" if delta >= 0 else "#ff4d4d"
        delta_sign = "+" if delta >= 0 else ""
        html = f"""
        <div class="glass" style="min-height: 92px;">
            <div style="font-size: 13px; opacity: 0.9; margin-bottom: 6px;">{name}</div>
            <div style="font-size: 22px; font-weight: 900; line-height: 1.1;">
                {unit} {fmt_num(val, 2)}
            </div>
            <div style="margin-top: 6px; display: inline-block; padding: 4px 10px; border-radius: 999px;
                        background: rgba(255,255,255,0.08); color: {delta_color}; font-weight: 800; font-size: 12px;">
                {delta_sign}{fmt_num(delta, 2)}
            </div>
        </div>
        """
        cols[i].markdown(html, unsafe_allow_html=True)


st.title("⛽ Etanol Intelligence Pro")
st.caption("Monitoramento de paridade, contexto global e valuation com Machine Learning")
render_ticker_cards(market)
st.markdown("---")


if page == "Visão Geral":
    c1, c2, c3 = st.columns(3)

    last_price_disp = etanol_to_display(last_price)
    c1.markdown(
        f"""
        <div class="glass">
            <div style="opacity:0.85;font-size:13px;">Etanol (último do dataset)</div>
            <div style="font-size:34px;font-weight:900;margin-top:6px;">R$ {fmt_num(last_price_disp, 2)} <span style="font-size:14px;opacity:0.75;">/L</span></div>
            <div style="opacity:0.75;margin-top:4px;">Data: {last_date:%d/%m/%Y}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    gas = c2.number_input("Gasolina (R$/L)", value=5.80, step=0.05)
    eta = c2.number_input("Etanol (R$/L)", value=3.60, step=0.05)
    threshold = c2.slider("Limite de paridade (%)", 60, 85, 70)

    ratio = (eta / gas) * 100 if gas else 0
    eta_max = gas * (threshold / 100)

    verdict = "✅ Etanol vantajoso" if ratio < threshold else "❌ Gasolina vantajosa"
    verdict_color = "#00FF7F" if ratio < threshold else "#ff4d4d"

    c2.markdown(
        f"""
        <div class="glass">
            <div style="opacity:0.85;font-size:13px;">Paridade atual</div>
            <div style="font-size:34px;font-weight:900;margin-top:6px;color:{verdict_color};">{fmt_num(ratio, 1)}%</div>
            <div style="margin-top:6px;font-weight:800;">{verdict}</div>
            <div style="opacity:0.75;margin-top:6px;">Etanol compensa até: <b>R$ {fmt_num(eta_max, 2)}/L</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    last_fair = float(df_model["Preco_Justo_Modelo"].iloc[-1])
    diff = last_fair - last_price
    pct = (last_fair / last_price - 1) * 100 if last_price else 0

    signal = "🚀 Mercado descontado" if diff > 0 else "🔻 Mercado caro"
    signal_color = "#00FF7F" if diff > 0 else "#ff4d4d"

    c3.markdown(
        f"""
        <div class="glass">
            <div style="opacity:0.85;font-size:13px;">Sinal do modelo (último ponto)</div>
            <div style="font-size:28px;font-weight:900;margin-top:6px;">Preço justo: R$ {fmt_num(etanol_to_display(last_fair), 2)}/L</div>
            <div style="font-size:16px;font-weight:900;margin-top:8px;color:{signal_color};">{signal}</div>
            <div style="opacity:0.85;margin-top:6px;">Spread: <b>R$ {fmt_num(etanol_to_display(diff), 2)}/L</b> ({fmt_num(pct,1)}%)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "Valuation (IA)":
    st.subheader("🧮 Valuation (IA) — Simulador de preço justo")
    left, right = st.columns([1, 2])

    with left:
        with st.container(border=True):
            st.markdown("#### Premissas")

            def mkt_or_last(mkt_name, col):
                m = market.get(mkt_name, {}).get("val", 0)
                if m and m > 0:
                    return float(m)
                return float(df_model[col].dropna().iloc[-1])

            p_oil = st.slider("Brent (US$)", 40.0, 150.0, mkt_or_last("Brent", "Petroleo_Brent"))
            p_usd = st.slider("Dólar (R$)", 3.0, 7.5, mkt_or_last("Dólar (BRL)", "Dolar"))
            p_sug = st.slider("Açúcar (cents)", 10.0, 40.0, mkt_or_last("Açúcar (NY)", "Acucar"))
            p_mes = st.selectbox("Mês", list(range(1, 13)), index=int(last_date.month - 1))

            st.write("")
            calc = st.button("CALCULAR PREÇO JUSTO", use_container_width=True)

    with right:
        if not calc:
            st.info("Ajuste as premissas e clique em **CALCULAR PREÇO JUSTO**.")
        else:
            X_in = pd.DataFrame({"Petroleo_Brent": [p_oil], "Dolar": [p_usd], "Acucar": [p_sug], "Mes": [p_mes]})
            pred = float(model.predict(X_in)[0])
            diff = pred - last_price

            a, b, c = st.columns(3)
            a.metric("Preço Justo (Modelo)", f"R$ {fmt_num(etanol_to_display(pred), 2)}/L")
            b.metric(f"Mercado (último) - {last_date:%d/%m/%Y}", f"R$ {fmt_num(etanol_to_display(last_price), 2)}/L")
            c.metric("Spread", f"R$ {fmt_num(etanol_to_display(diff), 2)}/L")

elif page == "Paridade":
    st.subheader("⚖️ Calculadora de Paridade — Simulador de bomba")
    c1, c2 = st.columns([1, 1])

    with c1:
        with st.container(border=True):
            gas = st.number_input("Gasolina (R$/L)", value=5.80, step=0.05)
            eta = st.number_input("Etanol (R$/L)", value=3.60, step=0.05)
            threshold = st.slider("Limite de paridade (%)", 60, 85, 70)

    with c2:
        ratio = (eta / gas) * 100 if gas else 0
        eta_max = gas * (threshold / 100)

        st.metric("Paridade atual", f"{fmt_num(ratio, 1)}%")
        st.metric("Etanol compensa até", f"R$ {fmt_num(eta_max, 2)}/L")

        if ratio < threshold:
            st.success("✅ ETANOL VANTAJOSO")
        else:
            st.error("❌ GASOLINA VANTAJOSA")

elif page == "Histórico & Modelo":
    st.subheader("📊 Histórico & Modelo")

    df2 = df_model.copy()
    df2["Preco_Etanol_disp"] = df2["Preco_Etanol"].apply(etanol_to_display)
    df2["Preco_Justo_disp"] = df2["Preco_Justo_Modelo"].apply(etanol_to_display)
    df2["Spread_disp"] = df2["Spread"].apply(etanol_to_display)

    min_d, max_d = df2.index.min().date(), df2.index.max().date()
    d1, d2 = st.date_input("Período", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    dff = df2.loc[str(d1):str(d2)].copy()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=dff.index, y=dff["Preco_Etanol_disp"], name="Real", mode="lines"))
    fig1.add_trace(go.Scatter(x=dff.index, y=dff["Preco_Justo_disp"], name="Preço Justo (Modelo)", mode="lines"))
    fig1.update_layout(title="Real vs Preço Justo", template="plotly_dark",
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(dff, x=dff.index, y="Spread_disp", title="Spread (Justo - Real)", template="plotly_dark")
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)
PY
