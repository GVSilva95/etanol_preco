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
# 2) HELPERS (VISUAL / DATA)
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
        return "—"


def to_display(v, unidade: str):
    """Converte valor do dataset para exibição (assumindo que o dataset está em R$/m³)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    v = float(v)
    return v / 1000.0 if unidade == "R$/L" else v


def unit_label(unidade: str):
    return "/L" if unidade == "R$/L" else "/m³"


# ==============================================================================
# 3) CSS (blindado: sem f-string, sem risco de quebrar aspas)
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

.small {
  font-size: 12px;
  opacity: 0.85;
}

.tickerValue {
  font-size: 22px;
  font-weight: 900;
  white-space: nowrap;
}

.tickerDelta {
  font-size: 12px;
  font-weight: 800;
  white-space: nowrap;
}

.stButton > button {
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
}

.stButton > button:active {
  transform: translateY(4px);
  border-bottom: 0px;
  box-shadow: inset 0px 2px 6px rgba(0,0,0,0.25);
}
</style>
""".replace("__BG_URL__", bg_url)

st.markdown(css, unsafe_allow_html=True)


# ==============================================================================
# 4) LOAD + MARKET
# ==============================================================================
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
    """
    Retorna {nome: {val, delta, unit, ok}}
    - ok=False se falhar (para mostrar "N/A" em vez de 0,00)
    """
    out = {}
    for name, (ticker, unit) in TICKERS.items():
        try:
            h = yf.Ticker(ticker).history(period="7d").dropna()
            if len(h) >= 2:
                val = float(h["Close"].iloc[-1])
                prev = float(h["Close"].iloc[-2])
                out[name] = {"val": val, "delta": val - prev, "unit": unit, "ok": True}
            elif len(h) == 1:
                val = float(h["Close"].iloc[-1])
                out[name] = {"val": val, "delta": None, "unit": unit, "ok": True}
            else:
                out[name] = {"val": None, "delta": None, "unit": unit, "ok": False}
        except Exception:
            out[name] = {"val": None, "delta": None, "unit": unit, "ok": False}
    return out


def render_ticker_cards(market_data: dict):
    cols = st.columns(7)
    names = list(TICKERS.keys())

    for i, name in enumerate(names):
        d = market_data.get(name, {})
        unit = d.get("unit", "")
        ok = d.get("ok", False)
        val = d.get("val", None)
        delta = d.get("delta", None)

        if not ok or val is None or (isinstance(val, float) and np.isnan(val)):
            val_str = "—"
            delta_str = "—"
            color = "#aaaaaa"
        else:
            val_str = fmt_num(val, 2)
            if delta is None or (isinstance(delta, float) and np.isnan(delta)):
                delta_str = "—"
                color = "#aaaaaa"
            else:
                sign = "+" if delta >= 0 else ""
                delta_str = f"{sign}{fmt_num(delta, 2)}"
                color = "#00FF7F" if delta >= 0 else "#ff4d4d"

        html = f"""
        <div class="glass" style="min-height: 92px;">
          <div class="small">{name}</div>
          <div class="tickerValue">{unit} {val_str}</div>
          <div class="tickerDelta" style="color:{color};">{delta_str}</div>
        </div>
        """
        cols[i].markdown(html, unsafe_allow_html=True)


# ==============================================================================
# 5) FEATURES + MODEL
# ==============================================================================
def build_features(df: pd.DataFrame, freq_rule: str, max_lag: int, ma_window: int, use_target_lags: bool):
    df2 = df.copy()

    missing = [c for c in REQUIRED_COLS if c not in df2.columns]
    if missing:
        raise ValueError(
            f"Colunas faltando no dataset: {missing}\n"
            f"Colunas disponíveis: {list(df2.columns)}"
        )

    df2 = df2.sort_index()
    df2 = df2.resample(freq_rule).mean()

    # features temporais
    df2["Mes"] = df2.index.month

    features = []

    # drivers + lags + médias móveis
    for c in BASE_DRIVERS:
        features.append(c)
        for lag in range(1, max_lag + 1):
            col = f"{c}_lag{lag}"
            df2[col] = df2[c].shift(lag)
            features.append(col)
        ma_col = f"{c}_ma{ma_window}"
        df2[ma_col] = df2[c].rolling(ma_window).mean()
        features.append(ma_col)

    # lags do próprio etanol (normalmente melhora muito a qualidade vs baseline)
    if use_target_lags:
        for lag in range(1, max_lag + 1):
            col = f"{TARGET}_lag{lag}"
            df2[col] = df2[TARGET].shift(lag)
            features.append(col)
        ma_col = f"{TARGET}_ma{ma_window}"
        df2[ma_col] = df2[TARGET].rolling(ma_window).mean()
        features.append(ma_col)

    features.append("Mes")

    df2 = df2.dropna(subset=features + [TARGET]).copy()

    if len(df2) < 30:
        raise ValueError(
            "Poucos dados após resample/lags. "
            "Tente reduzir lags/janela ou mudar a frequência (ex.: Semanal)."
        )

    return df2, features


@st.cache_resource
def train_model(df: pd.DataFrame, freq_rule: str, max_lag: int, ma_window: int, use_target_lags: bool):
    df2, features = build_features(df, freq_rule, max_lag, ma_window, use_target_lags)

    X = df2[features]
    y = df2[TARGET]

    # split temporal
    n_splits = 5
    if len(df2) < 60:
        n_splits = 3
    if len(df2) < 40:
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

    # baseline naive (último valor observado)
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

    # histórico p/ gráficos
    df2["Preco_Justo_Modelo"] = model.predict(X)
    df2["Spread"] = df2["Preco_Justo_Modelo"] - df2[TARGET]

    bundle = {
        "features": features,
        "y_test": y_test,
        "y_pred": pd.Series(y_pred, index=y_test.index),
    }
    return model, df2, metrics, bundle


# ==============================================================================
# 6) SIDEBAR (clean + avançado recolhível)
# ==============================================================================
logo_base64 = get_img_as_base64("logo_projeto.jpg")
if logo_base64:
    st.sidebar.markdown(
        f'<img src="data:image/jpg;base64,{logo_base64}" style="width:100%; border-radius:14px; margin-bottom:14px;">',
        unsafe_allow_html=True,
    )

st.sidebar.title("Painel")

st.sidebar.markdown(
    "- ✅ Escolha unidade e confira gráficos\n"
    "- ⚙️ Ajustes avançados ficam recolhidos\n"
    "- 📌 Diagnóstico do modelo (opcional)\n"
)

unidade = st.sidebar.selectbox("Exibir etanol em", ["R$/m³", "R$/L"], index=1)

with st.sidebar.expander("⚙️ Configurações avançadas", expanded=False):
    freq_opt = st.selectbox("Frequência p/ modelagem", ["Semanal", "Mensal", "Diária"], index=0)
    freq_rule = {"Semanal": "W-FRI", "Mensal": "M", "Diária": "D"}[freq_opt]

    max_lag = st.slider("Defasagens (lags)", 1, 8, 4)
    ma_window = st.slider("Média móvel (janelas)", 3, 16, 4)
    use_target_lags = st.checkbox("Usar lags do etanol (recomendado)", value=True)

# defaults (se expander não abrir, ainda existem)
try:
    freq_rule
except NameError:
    freq_rule = "W-FRI"
    max_lag = 4
    ma_window = 4
    use_target_lags = True

# ==============================================================================
# 7) LOAD DATA + TRAIN
# ==============================================================================
df_raw = carregar_dados(DATA_PATH)
if df_raw is None:
    st.error("Dataset não encontrado.")
    st.caption(f"Caminho esperado: {DATA_PATH}")
    st.caption("Confirme que o arquivo está no repositório: data/processed/dataset_consolidado.csv")
    st.stop()

market = get_market_data()

try:
    model, dfm, metrics, bundle = train_model(
        df_raw, freq_rule=freq_rule, max_lag=max_lag, ma_window=ma_window, use_target_lags=use_target_lags
    )
except Exception as e:
    st.error("Erro ao preparar dados/modelo:")
    st.code(str(e))
    st.stop()

last_date = dfm.index.max()
last_real = float(dfm[TARGET].iloc[-1])
last_fair = float(dfm["Preco_Justo_Modelo"].iloc[-1])
last_spread = float(dfm["Spread"].iloc[-1])

# sidebar: diagnóstico recolhido
with st.sidebar.expander("🧠 Diagnóstico (modelo vs baseline)", expanded=False):
    st.metric("R² (modelo)", f"{metrics['r2_test']:.3f}")
    st.metric("MAPE (modelo)", f"{metrics['mape_test']:.1%}")
    st.caption(f"Teste: {metrics['test_start']:%d/%m/%Y} → {metrics['test_end']:%d/%m/%Y}")
    st.divider()
    st.metric("R² (baseline)", f"{metrics['r2_naive']:.3f}")
    st.metric("MAPE (baseline)", f"{metrics['mape_naive']:.1%}")

if metrics["r2_naive"] > metrics["r2_test"]:
    st.sidebar.warning(
        "📌 No momento, o baseline está melhor que o modelo.\n"
        "Isso é comum em séries temporais — use o diagnóstico para ajustar lags/frequência."
    )

st.sidebar.divider()
st.sidebar.caption("Desenvolvido por")
st.sidebar.markdown("**Giovanni Silva**")


# ==============================================================================
# 8) MAIN UI
# ==============================================================================
st.title("⛽ Etanol Intelligence Pro")
st.caption("Valuation + Paridade + Gráficos (resample + lags + baseline)")
render_ticker_cards(market)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Gráficos", "⚖️ Paridade", "🧠 Diagnóstico do Modelo"])


# ==============================================================================
# TAB 1: GRÁFICOS
# ==============================================================================
with tab1:
    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        f"Etanol (Real) {unit_label(unidade)}",
        f"R$ {fmt_num(to_display(last_real, unidade), 3)}",
        help=f"Data: {last_date:%d/%m/%Y}",
    )
    k2.metric(
        f"Preço justo (Modelo) {unit_label(unidade)}",
        f"R$ {fmt_num(to_display(last_fair, unidade), 3)}",
    )
    k3.metric(
        f"Spread (Justo - Real) {unit_label(unidade)}",
        f"R$ {fmt_num(to_display(last_spread, unidade), 3)}",
    )
    pct = (last_fair / last_real - 1) * 100 if last_real else 0
    k4.metric("Sinal (%)", f"{fmt_num(pct, 1)}%")

    st.markdown("### Período")
    preset = st.selectbox("Atalho", ["Tudo", "Últimos 5 anos", "Últimos 3 anos", "Último ano"], index=0)
    max_d = dfm.index.max().date()

    if preset == "Último ano":
        min_d = (dfm.index.max() - pd.DateOffset(years=1)).date()
    elif preset == "Últimos 3 anos":
        min_d = (dfm.index.max() - pd.DateOffset(years=3)).date()
    elif preset == "Últimos 5 anos":
        min_d = (dfm.index.max() - pd.DateOffset(years=5)).date()
    else:
        min_d = dfm.index.min().date()

    d1, d2 = st.date_input("Selecione", value=(min_d, max_d), min_value=dfm.index.min().date(), max_value=max_d)
    dff = dfm.loc[str(d1):str(d2)].copy()

    dff["Etanol_disp"] = dff[TARGET].apply(lambda x: to_display(x, unidade))
    dff["Justo_disp"] = dff["Preco_Justo_Modelo"].apply(lambda x: to_display(x, unidade))
    dff["Spread_disp"] = dff["Spread"].apply(lambda x: to_display(x, unidade))

    st.markdown("### Real vs Preço Justo")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=dff.index, y=dff["Etanol_disp"], name="Etanol (Real)", mode="lines"))
    fig1.add_trace(go.Scatter(x=dff.index, y=dff["Justo_disp"], name="Preço Justo (Modelo)", mode="lines"))
    fig1.add_hline(y=to_display(last_real, unidade), line_dash="dot", opacity=0.25)
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
    fig2.add_hline(y=0, line_dash="dash", opacity=0.35)
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

    cA, cB = st.columns(2)

    with cA:
        st.markdown("### Distribuição do Spread")
        figH = px.histogram(dff, x="Spread_disp", nbins=40, title=f"Histograma do Spread ({unit_label(unidade)})", template="plotly_dark")
        figH.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(figH, use_container_width=True)

    with cB:
        st.markdown("### Importância das variáveis (RF)")
        imp = pd.DataFrame({"feature": bundle["features"], "importance": model.feature_importances_}).sort_values("importance", ascending=False)
        figI = px.bar(imp.head(20), x="feature", y="importance", title="Top 20 features", template="plotly_dark")
        figI.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(figI, use_container_width=True)

    st.markdown("### Correlação (drivers)")
    cols_corr = [c for c in [TARGET] + BASE_DRIVERS if c in dff.columns]
    corr = dff[cols_corr].corr()
    figC = px.imshow(corr, text_auto=True, title="Correlação (período selecionado)", template="plotly_dark")
    figC.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figC, use_container_width=True)


# ==============================================================================
# TAB 2: PARIDADE (BUG DO DELTAGENERATOR CORRIGIDO)
# ==============================================================================
with tab2:
    st.header("Simulador de bomba")
    pc1, pc2 = st.columns([1, 1])

    with pc1:
        with st.container(border=True):
            gas = st.number_input("Gasolina (R$/L)", value=5.80, step=0.05)
            eta = st.number_input("Etanol (R$/L)", value=3.60, step=0.05)
            thr = st.slider("Limite (%)", 60, 85, 70)
            st.caption("Regra prática: etanol costuma compensar abaixo de ~70% (depende do veículo).")

    with pc2:
        ratio = (eta / gas) * 100 if gas else 0
        eta_max = gas * (thr / 100)

        st.metric("Paridade", f"{fmt_num(ratio, 1)}%")
        st.metric("Etanol compensa até", f"R$ {fmt_num(eta_max, 2)}/L")

        # ✅ FIX: nunca use ternário com st.success/st.error (vazava DeltaGenerator)
        if ratio < thr:
            st.success("✅ Etanol vantajoso")
        else:
            st.error("❌ Gasolina vantajosa")


# ==============================================================================
# TAB 3: DIAGNÓSTICO
# ==============================================================================
with tab3:
    y_test = bundle["y_test"]
    y_pred = bundle["y_pred"]

    st.subheader("Predito vs Real (teste)")
    df_sc = pd.DataFrame({"Real": y_test.values, "Predito": y_pred.values}, index=y_test.index)
    df_sc["Real_disp"] = df_sc["Real"].apply(lambda x: to_display(x, unidade))
    df_sc["Predito_disp"] = df_sc["Predito"].apply(lambda x: to_display(x, unidade))

    fig4 = px.scatter(
        df_sc,
        x="Real_disp",
        y="Predito_disp",
        title=f"Teste: Predito vs Real ({unit_label(unidade)})",
        template="plotly_dark",
    )
    mn = float(np.nanmin([df_sc["Real_disp"].min(), df_sc["Predito_disp"].min()]))
    mx = float(np.nanmax([df_sc["Real_disp"].max(), df_sc["Predito_disp"].max()]))
    fig4.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="Linha 45°"))
    fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Resíduo no tempo (teste)")
    resid = (y_pred - y_test)
    df_res = pd.DataFrame({"Resíduo": resid.values}, index=y_test.index)
    df_res["Resíduo_disp"] = df_res["Resíduo"].apply(lambda x: to_display(x, unidade))
    figR = px.line(df_res, x=df_res.index, y="Resíduo_disp", title=f"Resíduo ({unit_label(unidade)})", template="plotly_dark")
    figR.add_hline(y=0, line_dash="dash", opacity=0.35)
    figR.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figR, use_container_width=True)

    st.subheader("Métricas")
    a, b, c = st.columns(3)
    a.metric("R² (modelo)", f"{metrics['r2_test']:.3f}")
    b.metric("MAPE (modelo)", f"{metrics['mape_test']:.1%}")
    c.metric("MAE (modelo)", f"{fmt_num(metrics['mae_test'], 3)}")

    a2, b2, c2 = st.columns(3)
    a2.metric("R² (baseline)", f"{metrics['r2_naive']:.3f}")
    b2.metric("MAPE (baseline)", f"{metrics['mape_naive']:.1%}")
    c2.metric("MAE (baseline)", f"{fmt_num(metrics['mae_naive'], 3)}")

st.markdown("---")
st.caption("Obs.: Se algum ticker aparecer como '—', é indisponibilidade temporária do yfinance para aquele ativo.")
