import os
import base64
from pathlib import Path
from datetime import date

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
# 1) CONFIG
# ==============================================================================
st.set_page_config(page_title="Etanol Intelligence Pro", page_icon="⛽", layout="wide")

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_PATH = (PROJECT_ROOT / "data" / "processed" / "dataset_consolidado.csv").resolve()

TARGET = "Preco_Etanol"

# Drivers “base” (antes de criar lags)
BASE_DRIVERS = ["Petroleo_Brent", "Dolar", "Acucar"]
REQUIRED_COLS = BASE_DRIVERS + [TARGET]

# Mercado (yfinance)
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
# 2) VISUAL HELPERS
# ==============================================================================
def get_img_as_base64(filename: str):
    """Procura imagem na pasta do app e na raiz do projeto."""
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
    """Converte valor do dataset para exibição."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    v = float(v)
    # Se o dataset estiver em m³ e o usuário quiser exibir em L, divide por 1000
    return v / 1000.0 if unidade == "R$/L" else v


def unit_label(unidade: str):
    return "/L" if unidade == "R$/L" else "/m³"


# Background + logo
bg_base64 = get_img_as_base64("fundo_cana.jpg")
logo_base64 = get_img_as_base64("logo_projeto.jpg")

bg_url = (
    f"data:image/jpg;base64,{bg_base64}"
    if bg_base64
    else "https://images.unsplash.com/photo-1633004147966-c1713534327d?q=80&w=1920"
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
  background-color: rgba(12,16,12,0.90);
  border-right: 1px solid rgba(0,255,127,0.10);
}}

.glass {{
  background: rgba(30,30,30,0.45);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 14px 16px;
  backdrop-filter: blur(6px);
}}

.small {{
  font-size: 12px;
  opacity: 0.85;
}}

.tickerValue {{
  font-size: 22px;
  font-weight: 900;
  white-space: nowrap;
}}

.tickerDelta {{
  font-size: 12px;
  font-weight: 800;
  white-space: nowrap;
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


# ==============================================================================
# 3) DADOS + MERCADO
# ==============================================================================
@st.cache_data
def carregar_dados(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # garante datetime
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
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
    """
    - Padroniza frequência (resample)
    - Cria features temporais + lags + média móvel
    """
    df2 = df.copy()

    missing = [c for c in REQUIRED_COLS if c not in df2.columns]
    if missing:
        raise ValueError(f"Colunas faltando no dataset: {missing}\nColunas disponíveis: {list(df2.columns)}")

    df2 = df2.sort_index()

    # Resample para alinhar frequência (isso costuma corrigir R² muito negativo)
    df2 = df2.resample(freq_rule).mean()

    # Features temporais simples
    df2["Mes"] = df2.index.month

    # Lags + médias móveis dos drivers
    for c in BASE_DRIVERS:
        for lag in range(1, max_lag + 1):
            df2[f"{c}_lag{lag}"] = df2[c].shift(lag)
        df2[f"{c}_ma{ma_window}"] = df2[c].rolling(ma_window).mean()

    # Lista final de features
    features = []
    for c in BASE_DRIVERS:
        features.append(c)
        for lag in range(1, max_lag + 1):
            features.append(f"{c}_lag{lag}")
        features.append(f"{c}_ma{ma_window}")
    features.append("Mes")

    df2 = df2.dropna(subset=features + [TARGET]).copy()
    return df2, features


@st.cache_resource
def train_model(df: pd.DataFrame, freq_rule: str, max_lag: int, ma_window: int):
    df2, features = build_features(df, freq_rule=freq_rule, max_lag=max_lag, ma_window=ma_window)

    X = df2[features]
    y = df2[TARGET]

    # Split temporal
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

    # Baseline (naive): previsão = último valor observado (t-1)
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

    # Predição no histórico para gráficos
    df2["Preco_Justo_Modelo"] = model.predict(X)
    df2["Spread"] = df2["Preco_Justo_Modelo"] - df2[TARGET]

    bundle = {
        "features": features,
        "X_test": X_test,
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
        <div class="glass" style="min-height: 92px;">
          <div class="small">{name}</div>
          <div class="tickerValue">{unit} {fmt_num(val, 2)}</div>
          <div class="tickerDelta" style="color:{color};">{sign}{fmt_num(delta, 2)}</div>
        </div>
        """
        cols[i].markdown(html, unsafe_allow_html=True)


# ==============================================================================
# 4) LOAD + CONTROLES (SIDEBAR)
# ==============================================================================
df_raw = carregar_dados(DATA_PATH)
market = get_market_data()

st.sidebar.markdown(
    f'<img src="data:image/jpg;base64,{logo_base64}" style="width:100%; border-radius:14px; margin-bottom:14px;">'
    if logo_base64
    else "",
    unsafe_allow_html=True,
)

st.sidebar.subheader("Configurações")
unidade = st.sidebar.selectbox("Exibir Etanol em", ["R$/m³", "R$/L"], index=0)

freq_opt = st.sidebar.selectbox("Frequência para modelagem", ["Semanal", "Mensal", "Diária"], index=0)
freq_rule = {"Semanal": "W-FRI", "Mensal": "M", "Diária": "D"}[freq_opt]

max_lag = st.sidebar.slider("Defasagens (lags)", min_value=1, max_value=8, value=4)
ma_window = st.sidebar.slider("Média móvel (janelas)", min_value=3, max_value=16, value=4)

st.sidebar.divider()

if df_raw is None:
    st.error(f"Dataset não encontrado em:\n{DATA_PATH}")
    st.caption("Confirme se o arquivo existe e se o nome está correto: dataset_consolidado.csv")
    st.stop()

try:
    model, dfm, metrics, bundle = train_model(df_raw, freq_rule=freq_rule, max_lag=max_lag, ma_window=ma_window)
except Exception as e:
    st.error("Erro ao preparar dados/modelo:")
    st.code(str(e))
    st.stop()

# KPIs úteis
last_date = dfm.index.max()
last_real = float(dfm[TARGET].iloc[-1])
last_fair = float(dfm["Preco_Justo_Modelo"].iloc[-1])
last_spread = float(dfm["Spread"].iloc[-1])

st.sidebar.subheader("Modelo (teste temporal)")
st.sidebar.metric("R² (teste)", f"{metrics['r2_test']:.3f}")
st.sidebar.metric("MAPE (teste)", f"{metrics['mape_test']:.1%}")
st.sidebar.caption(f"Teste: {metrics['test_start']:%d/%m/%Y} → {metrics['test_end']:%d/%m/%Y}")

st.sidebar.subheader("Baseline (naive)")
st.sidebar.metric("R² (baseline)", f"{metrics['r2_naive']:.3f}")
st.sidebar.metric("MAPE (baseline)", f"{metrics['mape_naive']:.1%}")

st.sidebar.divider()
st.sidebar.caption("Desenvolvido por")
st.sidebar.markdown("**Giovanni Silva**")


# ==============================================================================
# 5) HEADER
# ==============================================================================
st.title("⛽ Etanol Intelligence Pro")
st.caption("Valuation + Paridade + Gráficos (com validação temporal, resample e lags)")
render_ticker_cards(market)
st.markdown("---")


# ==============================================================================
# 6) TABS
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Gráficos", "🧮 Valuation (IA)", "⚖️ Paridade", "🧠 Diagnóstico do Modelo"])

# --------------------------------------------------------------------------
# TAB 1: GRÁFICOS
# --------------------------------------------------------------------------
with tab1:
    st.markdown("### Visão rápida")
    k1, k2, k3, k4 = st.columns(4)

    k1.metric(f"Etanol (Real) {unit_label(unidade)}", f"R$ {fmt_num(to_display(last_real, unidade), 3)}", help=f"Data: {last_date:%d/%m/%Y}")
    k2.metric(f"Preço justo (Modelo) {unit_label(unidade)}", f"R$ {fmt_num(to_display(last_fair, unidade), 3)}")
    k3.metric(f"Spread (Justo - Real) {unit_label(unidade)}", f"R$ {fmt_num(to_display(last_spread, unidade), 3)}")
    pct = (last_fair / last_real - 1) * 100 if last_real else 0
    k4.metric("Sinal (%)", f"{fmt_num(pct, 1)}%")

    st.markdown("### Período")
    # presets
    preset = st.selectbox("Atalho de período", ["Tudo", "Últimos 5 anos", "Últimos 3 anos", "Último ano"], index=0)
    max_d = dfm.index.max().date()
    if preset == "Último ano":
        min_d = (dfm.index.max() - pd.DateOffset(years=1)).date()
    elif preset == "Últimos 3 anos":
        min_d = (dfm.index.max() - pd.DateOffset(years=3)).date()
    elif preset == "Últimos 5 anos":
        min_d = (dfm.index.max() - pd.DateOffset(years=5)).date()
    else:
        min_d = dfm.index.min().date()

    d1, d2 = st.date_input("Selecione o período", value=(min_d, max_d), min_value=dfm.index.min().date(), max_value=max_d)
    dff = dfm.loc[str(d1):str(d2)].copy()

    # conversão para exibição
    dff["Etanol_disp"] = dff[TARGET].apply(lambda x: to_display(x, unidade))
    dff["Justo_disp"] = dff["Preco_Justo_Modelo"].apply(lambda x: to_display(x, unidade))
    dff["Spread_disp"] = dff["Spread"].apply(lambda x: to_display(x, unidade))

    st.markdown("### Real vs Preço Justo")
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

    st.markdown("### Spread")
    fig2 = px.line(dff, x=dff.index, y="Spread_disp", title=f"Spread (Justo - Real) ({unit_label(unidade)})", template="plotly_dark")
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

    cA, cB = st.columns(2)

    with cA:
        st.markdown("### Correlação")
        cols_corr = [c for c in [TARGET] + BASE_DRIVERS if c in dff.columns]
        corr = dff[cols_corr].corr()
        fig3 = px.imshow(corr, text_auto=True, title="Correlação (período selecionado)", template="plotly_dark")
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

    with cB:
        st.markdown("### Importância das variáveis (RF)")
        imp = pd.DataFrame({"feature": bundle["features"], "importance": model.feature_importances_}).sort_values("importance", ascending=False)
        fig4 = px.bar(imp.head(20), x="feature", y="importance", title="Top 20 features", template="plotly_dark")
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Distribuição do Spread")
    fig5 = px.histogram(dff, x="Spread_disp", nbins=40, title=f"Histograma do Spread ({unit_label(unidade)})", template="plotly_dark")
    fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig5, use_container_width=True)


# --------------------------------------------------------------------------
# TAB 2: VALUATION (IA)
# --------------------------------------------------------------------------
with tab2:
    st.markdown("### Simulador de preço justo")
    left, right = st.columns([1, 2])

    # defaults com base no último valor do df
    last_row = dfm.iloc[-1]

    def mkt_or_df(mkt_name, col):
        m = market.get(mkt_name, {}).get("val", 0)
        if m and m > 0:
            return float(m)
        return float(last_row[col])

    with left:
        with st.container(border=True):
            st.markdown("#### Premissas")
            p_oil = st.slider("Brent (US$)", 40.0, 150.0, float(mkt_or_df("Brent", "Petroleo_Brent")))
            p_usd = st.slider("Dólar (R$)", 3.0, 7.5, float(mkt_or_df("Dólar (BRL)", "Dolar")))
            p_sug = st.slider("Açúcar (cents)", 10.0, 40.0, float(mkt_or_df("Açúcar (NY)", "Acucar")))
            p_mes = st.selectbox("Mês", list(range(1, 13)), index=int(last_date.month - 1))
            st.write("")
            calc = st.button("CALCULAR PREÇO JUSTO", use_container_width=True)

    with right:
        if not calc:
            st.info("Ajuste as premissas e clique em **CALCULAR PREÇO JUSTO**.")
        else:
            # cria uma linha com as features necessárias
            # obs.: para lags/MA, usamos aproximação: replica os valores atuais
            # (para um valuation “do dia” isso funciona como proxy)
            row = {
                "Petroleo_Brent": p_oil,
                "Dolar": p_usd,
                "Acucar": p_sug,
                "Mes": p_mes,
            }
            for c in BASE_DRIVERS:
                for lag in range(1, max_lag + 1):
                    row[f"{c}_lag{lag}"] = row[c]
                row[f"{c}_ma{ma_window}"] = row[c]

            X_in = pd.DataFrame([row])[bundle["features"]]
            pred = float(model.predict(X_in)[0])
            diff = pred - last_real
            pct = (pred / last_real - 1) * 100 if last_real else 0

            a, b, c = st.columns(3)
            a.metric(f"Preço justo (Modelo) {unit_label(unidade)}", f"R$ {fmt_num(to_display(pred, unidade), 3)}")
            b.metric(f"Etanol (Real) {unit_label(unidade)}", f"R$ {fmt_num(to_display(last_real, unidade), 3)}", help=f"Data: {last_date:%d/%m/%Y}")
            c.metric(f"Spread {unit_label(unidade)}", f"R$ {fmt_num(to_display(diff, unidade), 3)}")

            if diff > 0:
                st.success(f"🚀 Mercado descontado em ~{fmt_num(pct, 1)}% vs preço justo.")
            else:
                st.error(f"🔻 Mercado caro em ~{fmt_num(abs(pct), 1)}% vs preço justo.")


# --------------------------------------------------------------------------
# TAB 3: PARIDADE
# --------------------------------------------------------------------------
with tab3:
    st.markdown("### Simulador de bomba")
    c1, c2 = st.columns([1, 1])

    with c1:
        with st.container(border=True):
            gas = st.number_input("Gasolina (R$/L)", value=5.80, step=0.05)
            eta = st.number_input("Etanol (R$/L)", value=3.60, step=0.05)
            thr = st.slider("Limite de paridade (%)", 60, 85, 70)
            st.caption("Regra prática: etanol tende a compensar abaixo de ~70% (depende do carro).")

    with c2:
        ratio = (eta / gas) * 100 if gas else 0
        eta_max = gas * (thr / 100)

        st.metric("Paridade atual", f"{fmt_num(ratio, 1)}%")
        st.metric("Etanol compensa até", f"R$ {fmt_num(eta_max, 2)}/L")

        if ratio < thr:
            st.success("✅ ETANOL VANTAJOSO")
        else:
            st.error("❌ GASOLINA VANTAJOSA")

    dfp = pd.DataFrame({"Tipo": ["Etanol (atual)", f"Etanol (limite {thr}%)"], "R$/L": [eta, eta_max]})
    figp = px.bar(dfp, x="Tipo", y="R$/L", title="Etanol atual vs limite", template="plotly_dark")
    figp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figp, use_container_width=True)


# --------------------------------------------------------------------------
# TAB 4: DIAGNÓSTICO DO MODELO
# --------------------------------------------------------------------------
with tab4:
    st.markdown("### Predito vs Real (teste)")
    y_test = bundle["y_test"]
    y_pred = bundle["y_pred"]

    df_sc = pd.DataFrame(
        {
            "Real": y_test.values,
            "Predito": y_pred.values,
        },
        index=y_test.index,
    )
    df_sc["Real_disp"] = df_sc["Real"].apply(lambda x: to_display(x, unidade))
    df_sc["Predito_disp"] = df_sc["Predito"].apply(lambda x: to_display(x, unidade))

    fig6 = px.scatter(df_sc, x="Real_disp", y="Predito_disp", title=f"Teste: Predito vs Real ({unit_label(unidade)})", template="plotly_dark")
    mn = float(np.nanmin([df_sc["Real_disp"].min(), df_sc["Predito_disp"].min()]))
    mx = float(np.nanmax([df_sc["Real_disp"].max(), df_sc["Predito_disp"].max()]))
    fig6.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="Linha 45°"))
    fig6.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("### Resíduos no tempo (teste)")
    resid = (y_pred - y_test)
    df_res = pd.DataFrame({"Resíduo": resid.values}, index=y_test.index)
    df_res["Resíduo_disp"] = df_res["Resíduo"].apply(lambda x: to_display(x, unidade))
    fig7 = px.line(df_res, x=df_res.index, y="Resíduo_disp", title=f"Resíduo ao longo do tempo ({unit_label(unidade)})", template="plotly_dark")
    fig7.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("### Métricas (comparação com baseline)")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² (modelo)", f"{metrics['r2_test']:.3f}")
    m2.metric("MAPE (modelo)", f"{metrics['mape_test']:.1%}")
    m3.metric("MAE (modelo)", f"{fmt_num(to_display(metrics['mae_test'], unidade), 3)} {unit_label(unidade)}")

    n1, n2, n3 = st.columns(3)
    n1.metric("R² (baseline)", f"{metrics['r2_naive']:.3f}")
    n2.metric("MAPE (baseline)", f"{metrics['mape_naive']:.1%}")
    n3.metric("MAE (baseline)", f"{fmt_num(to_display(metrics['mae_naive'], unidade), 3)} {unit_label(unidade)}")

st.markdown("---")
st.caption("Dica: coloque 'fundo_cana.jpg' e 'logo_projeto.jpg' na raiz do projeto ou em /src/images para personalizar.")
