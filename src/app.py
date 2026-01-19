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
# 1) CONFIG
# ==============================================================================
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


# ==============================================================================
# 2) HELPERS (IMAGENS + CSS)
# ==============================================================================
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

# CSS mais leve + sem cortar números nos cards do topo (usamos HTML próprio pro ticker)
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

    /* Containers com borda suave */
    .glass {{
        background: rgba(30, 30, 30, 0.45);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 14px 16px;
        backdrop-filter: blur(6px);
    }}

    /* Botão premium */
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

    /* Títulos */
    h1, h2, h3 {{
        letter-spacing: 0.2px;
    }}

    /* Remove fundo dos gráficos plotly */
    .js-plotly-plot .plotly .main-svg {{
        border-radius: 14px;
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


# ==============================================================================
# 3) DADOS + MERCADO (CACHE)
# ==============================================================================
@st.cache_data
def carregar_dados():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    return df


@st.cache_data(ttl=1800)  # 30min
def get_market_data():
    out = {}
    for name, (ticker, unit) in TICKERS.items():
        try:
            h = yf.Ticker(ticker).history(period="7d").dropna()
            if len(h) >= 2:
                val = float(h["Close"].iloc[-1])
                prev = float(h["Close"].iloc[-2])
                out[name] = {
                    "val": val,
                    "delta": val - prev,
                    "unit": unit,
                    "date": h.index[-1],
                }
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

    # feature temporal
    df2["Mes"] = df2.index.month

    # drop NA
    df2 = df2.dropna(subset=REQUIRED_COLS + ["Mes"])
    return df2


@st.cache_resource
def train_model(df: pd.DataFrame):
    df2 = prepare_df(df)

    X = df2[FEATURES]
    y = df2[TARGET]

    # split temporal
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

    # predição no histórico (para gráficos)
    df2["Preco_Justo_Modelo"] = model.predict(X)
    df2["Spread"] = df2["Preco_Justo_Modelo"] - df2[TARGET]

    test_bundle = {
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
    }

    return model, df2, metrics, last_price, last_date, test_bundle


# ==============================================================================
# 4) LOAD
# ==============================================================================
df_raw = carregar_dados()
market = get_market_data()

model = None
df_model = None
metrics = None
last_price = None
last_date = None
test_bundle = None
error_msg = None

if df_raw is not None:
    try:
        model, df_model, metrics, last_price, last_date, test_bundle = train_model(df_raw)
    except Exception as e:
        error_msg = str(e)


# ==============================================================================
# 5) SIDEBAR (NAVEGAÇÃO + CONTROLES)
# ==============================================================================
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

    # Unidade do etanol (pra evitar confusão tipo R$ 2909)
    unidade_etanol = st.selectbox(
        "Unidade do preço do etanol no dataset",
        ["R$/L", "R$/m³ (divide por 1000 para R$/L)"],
        index=0,
    )

    def etanol_to_display(v):
        if v is None:
            return None
        if unidade_etanol.startswith("R$/m³"):
            return float(v) / 1000.0
        return float(v)

    def etanol_unit_label():
        return "R$/L" if unidade_etanol.startswith("R$/L") else "R$/m³"

    st.divider()

    st.subheader("Status")
    if df_raw is None:
        st.error("Dataset não encontrado.")
        st.caption("Esperado em: data/processed/dataset_consolidado.csv")
    elif error_msg:
        st.error("Erro ao treinar modelo.")
        st.caption(error_msg)
    else:
        st.success("Dados carregados")
        st.caption(f"Última data do dataset: {last_date:%d/%m/%Y}")

        st.subheader("Modelo (teste temporal)")
        st.metric("R² (teste)", f"{metrics['r2_test']:.3f}")
        st.metric("MAPE (teste)", f"{metrics['mape_test']:.1%}")
        st.caption(f"Janela de teste: {metrics['test_start']:%d/%m/%Y} → {metrics['test_end']:%d/%m/%Y}")

    st.divider()
    st.caption("Desenvolvido por")
    st.markdown("**Giovanni Silva**")


# ==============================================================================
# 6) COMPONENTE: TICKER TAPE (HTML)
# ==============================================================================
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


# ==============================================================================
# 7) HEADER
# ==============================================================================
st.title("⛽ Etanol Intelligence Pro")
st.caption("Monitoramento de paridade, contexto global e valuation com Machine Learning")
render_ticker_cards(market)
st.markdown("---")


# ==============================================================================
# 8) PÁGINAS
# ==============================================================================
if df_raw is None:
    st.error("Não encontrei o arquivo do dataset.")
    st.code("data/processed/dataset_consolidado.csv", language="text")
    st.stop()

if error_msg:
    st.error("O app carregou, mas ocorreu um erro ao preparar o modelo/dados.")
    st.write(error_msg)
    st.stop()


# ----------------------------
# VISÃO GERAL
# ----------------------------
if page == "Visão Geral":
    # Cards principais
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

    # Paridade (input rápido)
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

    # Sinal do modelo (comparando preço justo do último ponto)
    # (usa df_model, que já tem Preco_Justo_Modelo)
    last_fair = float(df_model["Preco_Justo_Modelo"].iloc[-1])
    last_fair_disp = etanol_to_display(last_fair)
    diff = last_fair - last_price
    diff_disp = etanol_to_display(diff)

    pct = (last_fair / last_price - 1) * 100 if last_price else 0
    signal = "🚀 Mercado descontado" if diff > 0 else "🔻 Mercado caro"
    signal_color = "#00FF7F" if diff > 0 else "#ff4d4d"

    c3.markdown(
        f"""
        <div class="glass">
            <div style="opacity:0.85;font-size:13px;">Sinal do modelo (último ponto)</div>
            <div style="font-size:28px;font-weight:900;margin-top:6px;">Preço justo: R$ {fmt_num(last_fair_disp, 2)}/L</div>
            <div style="font-size:16px;font-weight:900;margin-top:8px;color:{signal_color};">{signal}</div>
            <div style="opacity:0.85;margin-top:6px;">Spread: <b>R$ {fmt_num(diff_disp, 2)}/L</b> ({fmt_num(pct,1)}%)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### O que este app faz")
    st.write(
        "- **Contexto global** via yfinance (Brent, dólar, açúcar, etc.)\n"
        "- **Valuation (IA):** estima preço justo do etanol com RandomForest (com validação temporal)\n"
        "- **Paridade:** simula decisão na bomba com limite configurável\n"
        "- **Histórico & Modelo:** gráficos e diagnóstico do modelo"
    )


# ----------------------------
# VALUATION (IA)
# ----------------------------
elif page == "Valuation (IA)":
    st.subheader("🧮 Valuation (IA) — Simulador de preço justo")

    left, right = st.columns([1, 2])

    with left:
        st.markdown("#### Premissas")
        with st.container(border=True):
            # valores padrão (se market tiver, usa)
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
            X_in = pd.DataFrame(
                {"Petroleo_Brent": [p_oil], "Dolar": [p_usd], "Acucar": [p_sug], "Mes": [p_mes]}
            )
            pred = float(model.predict(X_in)[0])
            pred_disp = etanol_to_display(pred)
            last_disp = etanol_to_display(last_price)
            diff = pred - last_price
            diff_disp = etanol_to_display(diff)

            st.markdown("#### Resultado do modelo")
            a, b, c = st.columns(3)
            a.metric("Preço Justo (Modelo)", f"R$ {fmt_num(pred_disp, 2)}/L")
            b.metric(f"Mercado (último do dataset) - {last_date:%d/%m/%Y}", f"R$ {fmt_num(last_disp, 2)}/L")
            c.metric("Spread (Justo - Mercado)", f"R$ {fmt_num(diff_disp, 2)}/L")

            if diff > 0:
                st.success(f"🚀 **OPORTUNIDADE DE COMPRA:** mercado ~{fmt_num((pred/last_price-1)*100,1)}% abaixo do justo.")
            else:
                st.error(f"🔻 **RISCO DE QUEDA:** mercado ~{fmt_num((last_price/pred-1)*100,1)}% acima do justo.")

            # mini sensibilidade (opcional e útil)
            st.markdown("#### Sensibilidade rápida (Brent)")
            brent_grid = np.linspace(max(40, p_oil - 20), min(150, p_oil + 20), 25)
            X_grid = pd.DataFrame(
                {
                    "Petroleo_Brent": brent_grid,
                    "Dolar": [p_usd] * len(brent_grid),
                    "Acucar": [p_sug] * len(brent_grid),
                    "Mes": [p_mes] * len(brent_grid),
                }
            )
            y_grid = model.predict(X_grid)
            y_grid_disp = np.array([etanol_to_display(v) for v in y_grid])

            fig = px.line(
                x=brent_grid,
                y=y_grid_disp,
                labels={"x": "Brent (US$)", "y": "Preço justo estimado (R$/L)"},
                title="Como o preço justo muda quando o Brent varia",
                template="plotly_dark",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# PARIDADE
# ----------------------------
elif page == "Paridade":
    st.subheader("⚖️ Calculadora de Paridade — Simulador de bomba")

    c1, c2 = st.columns([1, 1])
    with c1:
        with st.container(border=True):
            gas = st.number_input("Gasolina (R$/L)", value=5.80, step=0.05)
            eta = st.number_input("Etanol (R$/L)", value=3.60, step=0.05)
            threshold = st.slider("Limite de paridade (%)", 60, 85, 70)
            st.caption("Dica: o limite pode variar por eficiência do carro e condições de uso.")

    with c2:
        ratio = (eta / gas) * 100 if gas else 0
        eta_max = gas * (threshold / 100)

        st.metric("Paridade atual", f"{fmt_num(ratio, 1)}%")
        st.metric("Etanol compensa até", f"R$ {fmt_num(eta_max, 2)}/L")

        if ratio < threshold:
            st.success("✅ **ETANOL VANTAJOSO**")
        else:
            st.error("❌ **GASOLINA VANTAJOSA**")

    st.markdown("### Visual rápido")
    # gráfico simples comparando limites
    dfp = pd.DataFrame(
        {
            "Tipo": ["Etanol (atual)", f"Etanol (limite {threshold}%)"],
            "R$/L": [eta, eta_max],
        }
    )
    fig = px.bar(dfp, x="Tipo", y="R$/L", title="Etanol atual vs limite de paridade", template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# HISTÓRICO & MODELO (GRÁFICOS)
# ----------------------------
elif page == "Histórico & Modelo":
    st.subheader("📊 Histórico & Modelo — gráficos e diagnóstico")

    df2 = df_model.copy()
    # converter para exibição (se o dataset estiver em m³)
    df2["Preco_Etanol_disp"] = df2["Preco_Etanol"].apply(etanol_to_display)
    df2["Preco_Justo_disp"] = df2["Preco_Justo_Modelo"].apply(etanol_to_display)
    df2["Spread_disp"] = df2["Spread"].apply(etanol_to_display)

    # filtro de período
    min_d, max_d = df2.index.min().date(), df2.index.max().date()
    d1, d2 = st.date_input("Período", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    dff = df2.loc[str(d1) : str(d2)].copy()

    tabA, tabB, tabC = st.tabs(["📈 Séries", "🧠 Drivers", "🧪 Qualidade do Modelo"])

    with tabA:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=dff.index, y=dff["Preco_Etanol_disp"], name="Real (Etanol)", mode="lines"))
        fig1.add_trace(go.Scatter(x=dff.index, y=dff["Preco_Justo_disp"], name="Preço Justo (Modelo)", mode="lines"))
        fig1.update_layout(
            title="Real vs Preço Justo (Modelo)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.line(
            dff,
            x=dff.index,
            y="Spread_disp",
            title="Spread (Preço Justo - Real)",
            template="plotly_dark",
            labels={"Spread_disp": "Spread (R$/L)"},
        )
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    with tabB:
        # Correlação
        cols = [c for c in ["Preco_Etanol", "Petroleo_Brent", "Dolar", "Acucar"] if c in dff.columns]
        corr = dff[cols].corr()
        fig3 = px.imshow(corr, text_auto=True, title="Correlação entre variáveis", template="plotly_dark")
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

        # Feature importance
        imp = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_}).sort_values(
            "importance", ascending=False
        )
        fig4 = px.bar(imp, x="feature", y="importance", title="Importância das variáveis (RandomForest)", template="plotly_dark")
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig4, use_container_width=True)

        # Scatter Brent vs Etanol
        fig5 = px.scatter(
            dff,
            x="Petroleo_Brent",
            y="Preco_Etanol_disp",
            color=dff.index.year.astype(str),
            title="Relação histórica: Etanol vs Brent",
            template="plotly_dark",
            labels={"Preco_Etanol_disp": "Etanol (R$/L)"},
        )
        fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig5, use_container_width=True)

    with tabC:
        # Predito vs Real no teste
        y_test = test_bundle["y_test"]
        y_pred = test_bundle["y_pred_test"]

        y_test_disp = np.array([etanol_to_display(v) for v in y_test.values])
        y_pred_disp = np.array([etanol_to_display(v) for v in y_pred])

        df_sc = pd.DataFrame({"Real": y_test_disp, "Predito": y_pred_disp})
        fig6 = px.scatter(df_sc, x="Real", y="Predito", title="Teste: Predito vs Real", template="plotly_dark")
        # linha 45°
        mn = float(np.nanmin([df_sc["Real"].min(), df_sc["Predito"].min()]))
        mx = float(np.nanmax([df_sc["Real"].max(), df_sc["Predito"].max()]))
        fig6.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="Linha 45°"))
        fig6.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig6, use_container_width=True)

        # Resíduos no teste ao longo do tempo
        resid = (y_pred - y_test.values)
        resid_disp = np.array([etanol_to_display(v) for v in resid])

        df_res = pd.DataFrame({"Data": y_test.index, "Resíduo": resid_disp}).set_index("Data")
        fig7 = px.line(df_res, x=df_res.index, y="Resíduo", title="Resíduo no teste ao longo do tempo", template="plotly_dark")
        fig7.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig7, use_container_width=True)

        st.markdown("#### Métricas (teste temporal)")
        m1, m2, m3 = st.columns(3)
        m1.metric("R² (teste)", f"{metrics['r2_test']:.3f}")
        m2.metric("MAE (teste)", f"{fmt_num(etanol_to_display(metrics['mae_test']), 3)} R$/L")
        m3.metric("MAPE (teste)", f"{metrics['mape_test']:.1%}")


# Footer
st.markdown("---")
st.caption("Dica: coloque `logo_projeto.jpg` e `fundo_cana.jpg` na raiz do projeto para personalizar o visual.")
