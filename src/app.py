import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import base64
import os

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Etanol Intelligence Pro",
    page_icon="🎋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNÇÃO PARA CARREGAR IMAGEM LOCAL ---
def get_img_as_base64(file_path):
    possible_paths = [file_path, os.path.join("..", file_path), os.path.join(".", file_path)]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = f.read()
                return base64.b64encode(data).decode()
            except: pass
    return None

img_base64 = get_img_as_base64("fundo_cana.jpg")
bg_image_url = f"data:image/jpg;base64,{img_base64}" if img_base64 else "https://images.unsplash.com/photo-1633004147966-c1713534327d?q=80&w=1920&auto=format&fit=crop"

st.markdown(f"""
<style>
    [data-testid="stAppViewContainer"] {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.9), rgba(0, 0, 0, 0.95)), url("{bg_image_url}");
        background-size: cover;
        background-attachment: fixed;
    }}
    [data-testid="stSidebar"] {{ background-color: rgba(10, 15, 10, 0.9); border-right: 1px solid #333; }}
    div[data-testid="stMetric"] {{
        background-color: rgba(30, 30, 30, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 8px;
    }}
    div[data-testid="stMetricValue"] {{ font-size: 1.4rem !important; color: #fff; }}
    div[data-testid="stMetricLabel"] {{ color: #aaa; font-size: 0.8rem; }}
    .stButton>button {{
        background-color: #00FF7F; color: black; font-weight: bold; border-radius: 8px; border: none;
    }}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CARREGAMENTO DE DADOS
# ==============================================================================

@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv('data/processed/dataset_consolidado.csv', index_col=0, parse_dates=True)
        return df
    except: return None

def get_market_data():
    # Lista expandida de ativos
    tickers = {
        'Petróleo Brent': 'BZ=F',
        'Dólar (BRL)': 'BRL=X',
        'Açúcar (NY)': 'SB=F',
        'Milho (Chicago)': 'ZC=F',
        'Gasolina RBOB': 'RB=F',  # Novo: Gasolina internacional
        'Gás Natural': 'NG=F',    # Novo: Custo de energia
        'Juros EUA 10Y': '^TNX'   # Novo: Macroeconomia
    }
    
    data = {}
    try:
        for name, ticker in tickers.items():
            t = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if len(hist) > 1:
                curr = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                data[name] = {'val': curr, 'delta': curr - prev}
            else:
                data[name] = {'val': 0.0, 'delta': 0.0}
    except: pass
    return data

df = carregar_dados()
market = get_market_data()

# Modelo IA
@st.cache_resource
def train_model(df):
    if df is None: return None, 0
    if 'Mes' not in df.columns: df['Mes'] = df.index.month
    df_clean = df.dropna()
    features = ['Petroleo_Brent', 'Dolar', 'Acucar']
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(df_clean[features + ['Mes']], df_clean['Preco_Etanol'])
    return model, model.score(df_clean[features + ['Mes']], df_clean['Preco_Etanol'])

model = None
score = 0
last_price = 0

if df is not None:
    model, score = train_model(df)
    last_price = df['Preco_Etanol'].iloc[-1]

# ==============================================================================
# 3. SIDEBAR
# ==============================================================================
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1596739268306-692795c7394c?q=80&w=400", caption="Agro Analytics")
    st.header("Painel de Controle")
    st.info(f"Modelo Ativo (R²: {score:.1%})")
    st.markdown("---")
    st.caption("Desenvolvido por Giovanni Silva")

# ==============================================================================
# 4. DASHBOARD PRINCIPAL
# ==============================================================================
st.title("⛽ Etanol Intelligence Pro")
st.markdown("##### 💹 Monitoramento de Paridade, Arbitragem & Macroeconomia")

# --- TICKER TAPE (Linha de Cotações Expandida) ---
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
def metric(col, label, key, prefix="", suffix=""):
    d = market.get(key, {})
    col.metric(label, f"{prefix}{d.get('val',0):.2f}{suffix}", f"{d.get('delta',0):.2f}")

if market:
    metric(c1, "🛢️ Brent", 'Petróleo Brent', "US$ ")
    metric(c2, "💵 Dólar", 'Dólar (BRL)', "R$ ")
    metric(c3, "🍬 Açúcar", 'Açúcar (NY)', "¢")
    metric(c4, "🌽 Milho", 'Milho (Chicago)', "¢")
    metric(c5, "⛽ Gasolina", 'Gasolina RBOB', "US$ ")
    metric(c6, "🔥 Gás Nat.", 'Gás Natural', "US$ ")
    metric(c7, "🏦 Juros 10Y", 'Juros EUA 10Y', "", "%")

st.markdown("---")

# --- ABAS ---
tab1, tab2, tab3, tab4 = st.tabs(["🧮 Valuation (IA)", "⚖️ Calculadora de Paridade", "🌍 Macro & Energia", "📊 Histórico"])

# ABA 1: VALUATION IA
with tab1:
    st.header("Preço Justo via Inteligência Artificial")
    col_in, col_out = st.columns([1, 2])
    
    with col_in:
        with st.container(border=True):
            st.subheader("Cenário Base")
            def get_v(k, c): return float(market.get(k, {}).get('val', df[c].iloc[-1] if df is not None else 0))
            
            p_oil = st.slider("Brent (US$)", 40.0, 150.0, get_v('Petróleo Brent', 'Petroleo_Brent'))
            p_usd = st.slider("Dólar (R$)", 3.0, 7.0, get_v('Dólar (BRL)', 'Dolar'))
            p_sug = st.slider("Açúcar (cents)", 10.0, 40.0, get_v('Açúcar (NY)', 'Acucar'))
            p_mes = st.selectbox("Mês", range(1, 13), index=int(df.index[-1].month-1) if df is not None else 0)
            
            calc = st.button("🔄 Calcular IA", use_container_width=True)

    with col_out:
        if model:
            pred = model.predict(pd.DataFrame({'Petroleo_Brent':[p_oil], 'Dolar':[p_usd], 'Acucar':[p_sug], 'Mes':[p_mes]}))[0]
            diff = pred - last_price
            
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Preço Justo (Modelo)", f"R$ {pred:.2f}")
            rc2.metric("Mercado Hoje", f"R$ {last_price:.2f}")
            rc3.metric("Spread", f"R$ {diff:.2f}", delta_color="normal")
            
            if pred > last_price:
                st.success(f"🚀 **OPORTUNIDADE DE COMPRA:** Upside de {((pred/last_price)-1)*100:.1f}%")
            else:
                st.error(f"🔻 **RISCO DE QUEDA:** Mercado sobrevalorizado em {((last_price/pred)-1)*100:.1f}%")

# ABA 2: PARIDADE (NOVO!)
with tab2:
    st.header("⚖️ Arbitragem na Bomba (Gasolina vs Etanol)")
    st.markdown("Simule a competitividade do Etanol na ponta final (Postos).")
    
    pc1, pc2 = st.columns(2)
    with pc1:
        gasolina_bomba = st.number_input("Preço da Gasolina no Posto (R$/L)", value=5.80, step=0.10)
        etanol_bomba = st.number_input("Preço do Etanol no Posto (R$/L)", value=3.60, step=0.10)
    
    with pc2:
        ratio = (etanol_bomba / gasolina_bomba) * 100
        st.metric("Paridade Atual", f"{ratio:.1f}%")
        
        if ratio < 70:
            st.success("✅ **ETANOL É VANTAJOSO:** Abaixo de 70%. Consumo deve aumentar.")
        else:
            st.error("❌ **GASOLINA É VANTAJOSA:** Acima de 70%. Demanda por etanol deve cair.")
            
    # Gráfico de gauge simples
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = ratio,
        title = {'text': "Competitividade do Etanol"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "white"},
            'steps': [
                {'range': [0, 70], 'color': "#00FF7F"},
                {'range': [70, 100], 'color': "#FF4B4B"}],
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 70}}))
    st.plotly_chart(fig_gauge, use_container_width=True)

# ABA 3: MACRO & ENERGIA (NOVO!)
with tab3:
    st.header("🌍 Contexto Macroeconômico & Energético")
    
    m1, m2 = st.columns(2)
    with m1:
        st.subheader("Inflação e Juros (EUA)")
        st.markdown(f"""
        **Treasury Yield 10Y:** `{market.get('Juros EUA 10Y', {}).get('val',0):.2f}%`
        * **Impacto:** Juros americanos altos fortalecem o Dólar frente ao Real, encarecendo a gasolina importada e abrindo margem para o Etanol subir.
        """)
        st.image("https://images.unsplash.com/photo-1611974765270-ca1258634369?q=80&w=600", caption="Wall Street")
        
    with m2:
        st.subheader("Custo de Energia")
        st.markdown(f"""
        **Gás Natural:** `US$ {market.get('Gás Natural', {}).get('val',0):.2f}`
        * **Impacto:** O gás natural é um custo industrial chave para fertilizantes e processamento. Alta no gás = Alta no custo de produção da cana/milho.
        """)
        st.image("https://images.unsplash.com/photo-1595246140625-573b715d11dc?q=80&w=600", caption="Gasoduto")

# ABA 4: HISTÓRICO
with tab4:
    if df is not None:
        st.subheader("Histórico de Preços")
        fig = px.line(df, y=['Preco_Etanol', 'Petroleo_Brent'], title="Evolução Comparativa")
        st.plotly_chart(fig, use_container_width=True)
