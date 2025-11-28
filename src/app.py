import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA E DESIGN "SUCRO-PREMIUM"
# ==============================================================================
st.set_page_config(
    page_title="Etanol Intelligence Pro",
    page_icon="🎋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS CUSTOMIZADO (Estética de Vidro + Fundo de Cana)
st.markdown("""
<style>
    /* 1. Imagem de Fundo (Canavial Autêntico) */
    [data-testid="stAppViewContainer"] {
        /* Imagem de canavial ao pôr do sol, escurecida */
        background-image: linear-gradient(rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.9)), 
                          url("https://images.unsplash.com/photo-1633004147966-c1713534327d?q=80&w=1920&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* 2. Barra Lateral (Glassmorphism Escuro) */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 15, 10, 0.85);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0, 255, 127, 0.1);
    }

    /* 3. Métricas com Efeito de Vidro */
    div[data-testid="stMetric"] {
        background-color: rgba(30, 30, 30, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: #00FF7F;
        background-color: rgba(30, 30, 30, 0.8);
    }
    
    /* 4. Títulos das Métricas */
    div[data-testid="stMetricLabel"] {
        color: #A0A0A0 !important;
        font-size: 0.9rem !important;
        font-weight: 500;
    }

    /* 5. Valores das Métricas */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        color: #FFFFFF !important;
        font-weight: 700;
    }

    /* 6. Botões Estilizados (Verde Cana) */
    .stButton > button {
        background-color: #00FF7F;
        color: #002200;
        font-weight: 800;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #33FF99;
        box-shadow: 0 0 15px rgba(0, 255, 127, 0.5);
        color: #000000;
    }

    /* 7. Abas (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.03);
        border-radius: 8px;
        padding: 10px 25px;
        color: #CCCCCC;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 255, 127, 0.1) !important;
        color: #00FF7F !important;
        border-color: #00FF7F !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. FUNÇÕES DE CARREGAMENTO (Backend)
# ==============================================================================

@st.cache_data
def carregar_dados_historicos():
    try:
        df = pd.read_csv('data/processed/dataset_consolidado.csv', index_col=0, parse_dates=True)
        return df
    except FileNotFoundError:
        return None

def obter_cotacoes_hoje():
    """Busca cotações em tempo real de múltiplos ativos."""
    tickers = {
        'Petróleo Brent': 'BZ=F',
        'Dólar (BRL)': 'BRL=X',
        'Açúcar (NY)': 'SB=F',
        'Milho (Chicago)': 'ZC=F',
        'Etanol (Chicago)': 'CU=F'
    }
    
    dados_live = {}
    try:
        for nome, ticker in tickers.items():
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period="5d")
            
            if len(hist) > 1:
                atual = hist['Close'].iloc[-1]
                anterior = hist['Close'].iloc[-2]
                delta = atual - anterior
                dados_live[nome] = {'valor': atual, 'delta': delta}
            else:
                dados_live[nome] = {'valor': 0.0, 'delta': 0.0}
    except:
        pass
    return dados_live

df = carregar_dados_historicos()
cotacoes = obter_cotacoes_hoje()

@st.cache_resource
def treinar_modelo(df):
    if df is None: return None, 0
    if 'Mes' not in df.columns:
        df['Mes'] = df.index.month
    
    df_clean = df.dropna()
    features = ['Petroleo_Brent', 'Dolar', 'Acucar']
    
    X = df_clean[features + ['Mes']]
    y = df_clean['Preco_Etanol']
    
    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    model.fit(X, y)
    score = model.score(X, y)
    return model, score

model = None
score = 0
ultimo_preco = 0
data_ref = "N/A"

if df is not None:
    model, score = treinar_modelo(df)
    ultimo_preco = df['Preco_Etanol'].iloc[-1]
    data_ref = df.index[-1].strftime('%d/%m/%Y')

# ==============================================================================
# 3. BARRA LATERAL (Sidebar)
# ==============================================================================
with st.sidebar:
    # Imagem de Capa (Cana de Açúcar Close-up)
    st.image("https://images.unsplash.com/photo-1596739268306-692795c7394c?q=80&w=400&auto=format&fit=crop", caption="Agro Business Intelligence")
    
    st.header("Painel de Controle")
    st.info("Ferramenta avançada para precificação de Etanol Hidratado (Paulínia/SP).")
    
    st.markdown("---")
    
    if model:
        col_kpi1, col_kpi2 = st.columns(2)
        with col_kpi1:
            st.metric("Modelo", "R. Forest")
        with col_kpi2:
            st.metric("Precisão", f"{score:.1%}")
            
        st.caption(f"📅 Dados atualizados até: {data_ref}")
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Desenvolvedor")
    st.markdown("**Giovanni Silva**")
    st.caption("Especialista em Inteligência de Mercado")

# ==============================================================================
# 4. CORPO PRINCIPAL
# ==============================================================================

st.title("⛽ Etanol Intelligence Pro")
st.markdown("##### 💹 Monitoramento Estratégico de Commodities & Biocombustíveis")
st.markdown("---")

# --- BANNER DE COTAÇÕES ---
cols = st.columns(5)

def exibir_metrica(col, titulo, chave, prefixo="US$"):
    dado = cotacoes.get(chave, {})
    valor = dado.get('valor', 0.0)
    delta = dado.get('delta', 0.0)
    col.metric(titulo, f"{prefixo} {valor:.2f}", f"{delta:.2f}")

if cotacoes:
    exibir_metrica(cols[0], "🛢️ Petróleo", 'Petróleo Brent')
    exibir_metrica(cols[1], "💵 Dólar", 'Dólar (BRL)', "R$")
    exibir_metrica(cols[2], "🍬 Açúcar", 'Açúcar (NY)', "¢")
    exibir_metrica(cols[3], "🌽 Milho", 'Milho (Chicago)', "¢")
    exibir_metrica(cols[4], "🇺🇸 Etanol EUA", 'Etanol (Chicago)', "$")

st.markdown("---")

# --- ABAS DE NAVEGAÇÃO ---
tab1, tab2, tab3 = st.tabs(["🧮 Simulador de Valuation", "🌍 Contexto de Mercado", "📊 Análise Técnica"])

# === ABA 1: SIMULADOR ===
with tab1:
    st.header("Simulador de Paridade & Arbitragem")
    
    if model:
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.markdown("### 1. Definir Cenário")
            
            # Pega valores padrão
            def get_val(key, col):
                val_live = float(cotacoes.get(key, {}).get('valor', 0.0))
                val_hist = float(df[col].iloc[-1]) if df is not None else 0.0
                return val_live if val_live > 0 else val_hist

            with st.container(border=True):
                petroleo = st.slider("Petróleo Brent (US$)", 40.0, 150.0, get_val('Petróleo Brent', 'Petroleo_Brent'))
                dolar = st.slider("Dólar (R$)", 3.0, 7.0, get_val('Dólar (BRL)', 'Dolar'))
                acucar = st.slider("Açúcar (cents/lb)", 10.0, 40.0, get_val('Açúcar (NY)', 'Acucar'))
                mes = st.selectbox("Mês de Safra", range(1, 13), index=int(df.index[-1].month - 1) if df is not None else 0)
                
                calcular = st.button("🔄 Calcular Preço Justo", use_container_width=True)

        with c2:
            st.markdown("### 2. Análise de Preço Justo")
            
            # Previsão (Reativa)
            cenario = pd.DataFrame({
                'Petroleo_Brent': [petroleo],
                'Dolar': [dolar],
                'Acucar': [acucar],
                'Mes': [mes]
            })
            preco_justo = model.predict(cenario)[0]
            diff = preco_justo - ultimo_preco
            
            # Cards de Resultado
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.info("🎯 Preço Justo (Modelo)")
                st.metric("Fair Value", f"R$ {preco_justo:.2f}")
            
            with res_col2:
                st.info("📉 Spread vs Mercado")
                st.metric("Diferença", f"R$ {diff:.2f}", delta_color="normal")
            
            st.markdown("#### Veredito da IA:")
            if preco_justo > ultimo_preco:
                st.success(f"🚀 **OPORTUNIDADE DE COMPRA (UPSIDE)**\n\nO modelo indica que o Etanol está descontado frente aos fundamentos globais.")
            else:
                st.error(f"🔻 **OPORTUNIDADE DE VENDA (DOWNSIDE)**\n\nO Etanol está caro. O modelo sugere correção de preço para baixo.")
            
            # Gráfico de termómetro
            st.caption(f"Preço Atual de Mercado (CEPEA): R$ {ultimo_preco:.2f}")
            progress_val = min(max((preco_justo / 4000), 0.0), 1.0)
            st.progress(progress_val)

# === ABA 2: CONTEXTO GLOBAL ===
with tab2:
    st.header("Panorama Global do Setor")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        # IMAGEM: Indústria de Etanol (Para não mostrar plantação de milho)
        st.image("https://images.unsplash.com/photo-1616401784845-180882ba9ba8?q=80&w=600&auto=format&fit=crop", use_container_width=True)
        st.subheader("🇺🇸 Estados Unidos (Indústria)")
        st.write("""
        Os EUA são os maiores produtores mundiais, utilizando principalmente **Milho** como matéria-prima. 
        A cotação do milho em Chicago afeta indiretamente o Brasil através da paridade de exportação.
        """)
        
    with col_g2:
        # IMAGEM: Colheita de Cana (Brasil)
        st.image("https://images.unsplash.com/photo-1533596593406-3c224213793e?q=80&w=600&auto=format&fit=crop", use_container_width=True)
        st.subheader("🇧🇷 Brasil & Índia (Cana)")
        st.write("""
        No Brasil, a **Cana-de-Açúcar** domina. A decisão das usinas entre produzir Açúcar ou Etanol (Mix Produtivo) 
        é o principal driver de oferta interna, balizado pelos preços internacionais do Açúcar em NY.
        """)

# === ABA 3: GRÁFICOS ===
with tab3:
    if df is not None:
        st.header("Análise Técnica Histórica")
        
        # Gráfico Scatter
        fig_scatter = px.scatter(
            df, x='Petroleo_Brent', y='Preco_Etanol', 
            color=df.index.year,
            size_max=10,
            color_continuous_scale='Turbo',
            template='plotly_dark',
            title="Correlação: Petróleo x Etanol (2015-2025)"
        )
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white")
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
