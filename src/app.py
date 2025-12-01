import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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

# --- FUNÇÃO PARA CARREGAR IMAGENS LOCAIS (Background e Logo) ---
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

# Carrega as imagens
bg_base64 = get_img_as_base64("fundo_cana.jpg")
logo_base64 = get_img_as_base64("logo_projeto.jpg")

# Define URLs (Local ou Fallback Online)
bg_url = f"data:image/jpg;base64,{bg_base64}" if bg_base64 else "https://images.unsplash.com/photo-1633004147966-c1713534327d?q=80&w=1920"
logo_html = f'<img src="data:image/jpg;base64,{logo_base64}" style="width: 100%; border-radius: 15px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,255,127,0.2);">' if logo_base64 else ""

# CSS CUSTOMIZADO (Design Premium + Botões 3D)
st.markdown(f"""
<style>
    /* Fundo Realista */
    [data-testid="stAppViewContainer"] {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.95)), url("{bg_url}");
        background-size: cover;
        background-attachment: fixed;
    }}

    /* Barra Lateral */
    [data-testid="stSidebar"] {{
        background-color: rgba(12, 16, 12, 0.95);
        border-right: 1px solid rgba(0, 255, 127, 0.1);
    }}

    /* Métricas Glassmorphism */
    div[data-testid="stMetric"] {{
        background-color: rgba(30, 30, 30, 0.5);
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        transition: transform 0.2s;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-5px);
        border-color: #00FF7F;
    }}
    div[data-testid="stMetricValue"] {{ font-size: 1.6rem !important; color: #fff; }}
    div[data-testid="stMetricLabel"] {{ color: #aaa; }}

    /* --- NOVOS BOTÕES 3D INTERATIVOS --- */
    /* Estilo Normal */
    .stButton > button {{
        background: linear-gradient(180deg, #00FF7F 0%, #00CC66 100%);
        color: #003300;
        font-weight: 800;
        border: none;
        border-bottom: 4px solid #00994D; /* A borda inferior cria o volume 3D */
        border-radius: 8px;
        padding: 12px 24px;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.1s ease-in-out;
        box-shadow: 0px 5px 15px rgba(0, 255, 127, 0.2);
    }}
    
    /* Efeito Hover (Passar o mouse) */
    .stButton > button:hover {{
        filter: brightness(1.1);
        transform: translateY(-1px);
    }}

    /* Efeito Active (Clicar - Afunda o botão) */
    .stButton > button:active {{
        transform: translateY(4px); /* Move para baixo ocupando o espaço da borda */
        border-bottom: 0px solid #00994D; /* Remove a borda para parecer que entrou na tela */
        margin-bottom: 4px; /* Compensa o layout */
        box-shadow: inset 0px 3px 5px rgba(0,0,0,0.2); /* Sombra interna */
    }}

</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. LÓGICA DE DADOS
# ==============================================================================

@st.cache_data
def carregar_dados():
    try:
        return pd.read_csv('data/processed/dataset_consolidado.csv', index_col=0, parse_dates=True)
    except: return None

def get_market_data():
    tickers = {'Petróleo Brent': 'BZ=F', 'Dólar (BRL)': 'BRL=X', 'Açúcar (NY)': 'SB=F', 'Milho (Chicago)': 'ZC=F', 'Gasolina RBOB': 'RB=F', 'Gás Natural': 'NG=F', 'Juros EUA 10Y': '^TNX'}
    data = {}
    try:
        for name, t in tickers.items():
            h = yf.Ticker(t).history(period="5d")
            if len(h)>1: data[name] = {'val': h['Close'].iloc[-1], 'delta': h['Close'].iloc[-1]-h['Close'].iloc[-2]}
            else: data[name] = {'val': 0.0, 'delta': 0.0}
    except: pass
    return data

df = carregar_dados()
market = get_market_data()

@st.cache_resource
def train_model(df):
    if df is None: return None, 0
    if 'Mes' not in df.columns: df['Mes'] = df.index.month
    df_c = df.dropna()
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    fts = ['Petroleo_Brent', 'Dolar', 'Acucar', 'Mes']
    model.fit(df_c[fts], df_c['Preco_Etanol'])
    return model, model.score(df_c[fts], df_c['Preco_Etanol'])

model, score = (None, 0)
if df is not None:
    model, score = train_model(df)
    last_price = df['Preco_Etanol'].iloc[-1]
    data_ref = df.index[-1].strftime('%d/%m/%Y')

# ==============================================================================
# 3. BARRA LATERAL (Com Logo Personalizada)
# ==============================================================================
with st.sidebar:
    # --- ÁREA DA LOGO (Topo Esquerdo) ---
    if logo_base64:
        st.markdown(logo_html, unsafe_allow_html=True)
    else:
        # Fallback se a imagem não carregar
        st.header("Agro Analytics")
        st.caption("Adicione 'logo_projeto.jpg' na raiz")

    st.subheader("Painel de Controle")
    if model:
        col1, col2 = st.columns(2)
        col1.metric("Modelo", "R. Forest", help="Algoritmo de Machine Learning")
        col2.metric("Precisão", f"{score:.1%}")
        
    st.markdown("### Navegação")
    st.info("Utilize as abas acima para alternar entre Simulador e Gráficos.")
    
    # Espaçador grande para empurrar o rodapé para o fundo da tela
    st.markdown("<br>" * 8, unsafe_allow_html=True)
    
    # --- RODAPÉ (Inferior Esquerdo) ---
    st.markdown("---")
    st.markdown("### 👨‍💻 Desenvolvedor")
    st.markdown("**Giovanni Silva**")
    st.caption("Especialista em Inteligência de Mercado | Data Science Aplicado ao Agro")

# ==============================================================================
# 4. DASHBOARD PRINCIPAL
# ==============================================================================
st.title("⛽ Etanol Intelligence Pro")
st.markdown("##### 💹 Monitoramento de Paridade & Arbitragem")

# Ticker Tape
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
def metric(col, label, key, prefix="", suffix=""):
    d = market.get(key, {})
    col.metric(label, f"{prefix}{d.get('val',0):.2f}{suffix}", f"{d.get('delta',0):.2f}")

if market:
    metric(c1, "🛢️ Brent", 'Petróleo Brent', "US$ ")
    metric(c2, "💵 Dólar", 'Dólar (BRL)', "R$ ")
    metric(c3, "🍬 Açúcar", 'Açúcar (NY)', "US$ ")
    metric(c4, "🌽 Milho", 'Milho (Chicago)', "US$ ")
    metric(c5, "⛽ Gasolina", 'Gasolina RBOB', "US$ ")
    metric(c6, "🔥 Gás Nat.", 'Gás Natural', "US$ ")
    metric(c7, "🏦 Juros 10Y", 'Juros EUA 10Y', "", "%")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🧮 Valuation (IA)", "⚖️ Calculadora de Paridade", "📊 Dados Históricos"])

# ABA 1: VALUATION
with tab1:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.container(border=True):
            st.markdown("### Premissas")
            def get_v(k, c): return float(market.get(k, {}).get('val', df[c].iloc[-1] if df is not None else 0))
            
            p_oil = st.slider("Brent (US$)", 40.0, 150.0, get_v('Petróleo Brent', 'Petroleo_Brent'))
            p_usd = st.slider("Dólar (R$)", 3.0, 7.0, get_v('Dólar (BRL)', 'Dolar'))
            p_sug = st.slider("Açúcar (cents)", 10.0, 40.0, get_v('Açúcar (NY)', 'Acucar'))
            p_mes = st.selectbox("Mês", range(1, 13), index=int(df.index[-1].month-1) if df is not None else 0)
            
            # BOTÃO COM NOVO ESTILO 3D
            st.write("")
            calc = st.button("CALCULAR PREÇO JUSTO", use_container_width=True)

    with col_out:
        if model:
            pred = model.predict(pd.DataFrame({'Petroleo_Brent':[p_oil], 'Dolar':[p_usd], 'Acucar':[p_sug], 'Mes':[p_mes]}))[0]
            diff = pred - last_price
            
            # Resultado Visual
            st.markdown("### Resultado da Inteligência Artificial")
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Preço Justo (Modelo)", f"R$ {pred:.2f}")
            rc2.metric("Mercado Hoje (CEPEA)", f"R$ {last_price:.2f}")
            rc3.metric("Spread (Diferença)", f"R$ {diff:.2f}", delta_color="normal")
            
            if pred > last_price:
                st.success(f"🚀 **OPORTUNIDADE DE COMPRA:** O mercado está descontado em {((pred/last_price)-1)*100:.1f}%.")
            else:
                st.error(f"🔻 **RISCO DE QUEDA:** O mercado está caro em {((last_price/pred)-1)*100:.1f}%.")

# ABA 2: PARIDADE
with tab2:
    st.header("Simulador de Bomba")
    pc1, pc2 = st.columns(2)
    with pc1:
        gas = st.number_input("Gasolina (R$/L)", value=5.80, step=0.10)
        eta = st.number_input("Etanol (R$/L)", value=3.60, step=0.10)
        
        # Botão com estilo 3D
        st.write("")
        st.button("Verificar Paridade", use_container_width=True)
        
    with pc2:
        ratio = (eta / gas) * 100
        st.metric("Paridade Atual", f"{ratio:.1f}%")
        if ratio < 70: st.success("✅ **ETANOL VANTAJOSO** (Abaixo de 70%)")
        else: st.error("❌ **GASOLINA VANTAJOSA** (Acima de 70%)")

# ABA 3: HISTÓRICO
with tab3:
    if df is not None:
        fig = px.scatter(df, x='Petroleo_Brent', y='Preco_Etanol', color=df.index.year, title="Correlação Histórica", template="plotly_dark")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
