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

# --- FUNÇÃO PARA CARREGAR IMAGENS LOCAIS ---
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

bg_base64 = get_img_as_base64("fundo_cana.jpg")
logo_base64 = get_img_as_base64("logo_projeto.jpg")

bg_url = f"data:image/jpg;base64,{bg_base64}" if bg_base64 else "https://images.unsplash.com/photo-1633004147966-c1713534327d?q=80&w=1920"
logo_html = f'<img src="data:image/jpg;base64,{logo_base64}" style="width: 100%; border-radius: 10px; margin-bottom: 20px;">' if logo_base64 else ""

st.markdown(f"""
<style>
    [data-testid="stAppViewContainer"] {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.95)), url("{bg_url}");
        background-size: cover;
        background-attachment: fixed;
    }}
    [data-testid="stSidebar"] {{
        background-color: rgba(15, 20, 15, 0.95);
        border-right: 1px solid rgba(0, 255, 127, 0.1);
    }}
    div[data-testid="stMetric"] {{
        background-color: rgba(30, 30, 30, 0.5);
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
    }}
    div[data-testid="stMetricValue"] {{ font-size: 1.6rem !important; color: #fff; }}
    div[data-testid="stMetricLabel"] {{ color: #aaa; }}
    .stButton > button {{
        background: linear-gradient(to bottom, #00FF7F 0%, #00CC66 100%);
        color: #002200; font-weight: 800; border: none; border-bottom: 4px solid #00994D;
        border-radius: 8px; padding: 12px 24px; transition: all 0.1s;
    }}
    .stButton > button:active {{ transform: translateY(4px); border-bottom: 0px; margin-bottom: 4px; }}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CARREGAMENTO DE DADOS (ROBUSTO)
# ==============================================================================

@st.cache_data
def carregar_dados_historicos():
    try:
        return pd.read_csv('data/processed/dataset_consolidado.csv', index_col=0, parse_dates=True)
    except: return None

def obter_cotacoes_hoje():
    """
    Busca cotações com conversão de unidades e tratamento de erro para zeros.
    """
    tickers = {
        'Petróleo Brent': 'BZ=F',
        'Dólar (BRL)': 'BRL=X',
        'Açúcar (NY)': 'SB=F',
        'Milho (Chicago)': 'ZC=F',
        'Gasolina RBOB': 'RB=F',
        'Gás Natural': 'NG=F',
        'Juros EUA 10Y': '^TNX'
    }
    
    data = {}
    
    for name, symbol in tickers.items():
        try:
            # Tenta pegar até 1 mês de dados para garantir que não venha vazio
            ticker_obj = yf.Ticker(symbol)
            hist = ticker_obj.history(period="1mo")
            
            if not hist.empty and len(hist) > 0:
                valor_raw = hist['Close'].iloc[-1]
                
                # Se tiver pelo menos 2 dias, calcula delta, senão delta é 0
                if len(hist) > 1:
                    delta_raw = valor_raw - hist['Close'].iloc[-2]
                else:
                    delta_raw = 0.0
                
                # --- CONVERSÕES DE UNIDADES ---
                fator_conv = 1.0
                
                # Açúcar: Cents/lb -> USD/Saca 50kg
                # 1 lb = 0.4536 kg | 50kg = 110.23 lbs | Divide por 100 para tirar centavos
                if name == 'Açúcar (NY)':
                    fator_conv = 1.1023 
                
                # Milho: Cents/bushel -> USD/Saca 60kg
                # 1 bushel = 25.4 kg | 60kg = 2.36 bushels | Divide por 100 para tirar centavos
                elif name == 'Milho (Chicago)':
                    fator_conv = 0.02362 
                
                # Gasolina: USD/Galão -> USD/Litro
                # 1 Galão = 3.785 Litros
                elif name == 'Gasolina RBOB':
                    fator_conv = 1 / 3.785
                
                # Gás Natural: USD/MMBtu -> USD/m³ (Aprox)
                # 1 MMBtu ~= 26.8 m³
                elif name == 'Gás Natural':
                    fator_conv = 1 / 26.8

                valor_final = valor_raw * fator_conv
                delta_final = delta_raw * fator_conv
                
                data[name] = {'val': valor_final, 'delta': delta_final}
            else:
                # Se falhar, tenta pegar valor fixo para não mostrar 0.00
                data[name] = {'val': 0.0, 'delta': 0.0}
        except:
            data[name] = {'val': 0.0, 'delta': 0.0}
            
    return data

df = carregar_dados_historicos()
market = obter_cotacoes_hoje()

@st.cache_resource
def treinar_modelo(df):
    if df is None: return None, 0
    if 'Mes' not in df.columns: df['Mes'] = df.index.month
    df_c = df.dropna()
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    fts = ['Petroleo_Brent', 'Dolar', 'Acucar', 'Mes']
    model.fit(df_c[fts], df_c['Preco_Etanol'])
    return model, model.score(df_c[fts], df_c['Preco_Etanol'])

model, score = (None, 0)
if df is not None:
    model, score = treinar_modelo(df)
    last_price = df['Preco_Etanol'].iloc[-1]
    data_ref = df.index[-1].strftime('%d/%m/%Y')

# ==============================================================================
# 3. BARRA LATERAL
# ==============================================================================
with st.sidebar:
    if logo_base64: st.markdown(logo_html, unsafe_allow_html=True)
    else: st.header("Agro Analytics")
    
    st.subheader("Painel de Controle")
    if model:
        c1, c2 = st.columns(2)
        c1.metric("Modelo", "R. Forest")
        c2.metric("Precisão", f"{score:.1%}")
        
    st.markdown("### Navegação")
    st.markdown("- Simulador de Preço")
    st.markdown("- Contexto Global")
    st.markdown("- Gráficos Técnicos")
    st.markdown("<br>"*5, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 👨‍💻 Desenvolvedor")
    st.markdown("**Giovanni Silva**")
    st.caption("Especialista em Inteligência de Mercado")

# ==============================================================================
# 4. DASHBOARD
# ==============================================================================
st.title("⛽ Etanol Intelligence Pro")
st.markdown("##### 💹 Monitoramento de Paridade & Arbitragem")

# Ticker Tape (Agora com unidades corretas)
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
def metric(col, label, key, prefix="", suffix=""):
    d = market.get(key, {})
    val = d.get('val', 0)
    delta = d.get('delta', 0)
    col.metric(label, f"{prefix}{val:.2f}{suffix}", f"{delta:.2f}")

if market:
    metric(c1, "🛢️ Brent", 'Petróleo Brent', "US$ ")
    metric(c2, "💵 Dólar", 'Dólar (BRL)', "R$ ")
    # Açúcar agora em Saca 50kg
    metric(c3, "🍬 Açúcar", 'Açúcar (NY)', "US$ ", "/Saca")
    # Milho agora em Saca 60kg
    metric(c4, "🌽 Milho", 'Milho (Chicago)', "US$ ", "/Saca")
    # Gasolina agora em Litros
    metric(c5, "⛽ Gasolina", 'Gasolina RBOB', "US$ ", "/L")
    # Gás Natural agora em m3
    metric(c6, "🔥 Gás Nat.", 'Gás Natural', "US$ ", "/m³")
    metric(c7, "🏦 Juros 10Y", 'Juros EUA 10Y', "", "%")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🧮 Valuation (IA)", "⚖️ Calculadora de Paridade", "📊 Dados Históricos"])

# ABA 1: VALUATION
with tab1:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.container(border=True):
            st.markdown("### Premissas")
            # Função para pegar valor bruto (sem conversão) para o modelo, pois o modelo treinou com dados brutos
            def get_raw_val(key, col):
                if df is not None: return float(df[col].iloc[-1])
                return 0.0

            p_oil = st.slider("Brent (US$)", 40.0, 150.0, get_raw_val('Petróleo Brent', 'Petroleo_Brent'))
            p_usd = st.slider("D
