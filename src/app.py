import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Etanol Intelligence",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título Principal com Estilo
st.title("⛽ Etanol Intelligence: Dashboard & Valuation")
st.markdown("---")

# ==============================================================================
# 2. FUNÇÕES DE CARREGAMENTO (Backend)
# ==============================================================================

@st.cache_data
def carregar_dados_historicos():
    try:
        df = pd.read_csv('data/processed/dataset_consolidado.csv', index_col=0, parse_dates=True)
        return df
    except:
        st.error("Erro: Base de dados histórica não encontrada.")
        return None

def obter_cotacoes_hoje():
    # Tickers: Petróleo Brent, Dólar, Açúcar, Milho (Correlato)
    tickers = {
        'Petróleo Brent': 'BZ=F',
        'Dólar (USD/BRL)': 'BRL=X',
        'Açúcar No.11': 'SB=F',
        'Milho (Corn)': 'ZC=F'
    }
    
    dados_live = {}
    
    try:
        # Baixa dados de hoje e ontem para calcular variação
        for nome, ticker in tickers.items():
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period="5d") # Pega 5 dias para garantir
            
            if len(hist) > 1:
                preco_atual = hist['Close'].iloc[-1]
                preco_anterior = hist['Close'].iloc[-2]
                delta = preco_atual - preco_anterior
                delta_pct = (delta / preco_anterior) * 100
                
                dados_live[nome] = {
                    'valor': preco_atual,
                    'delta': delta
                }
            else:
                dados_live[nome] = {'valor': 0.0, 'delta': 0.0}
                
    except Exception as e:
        st.warning(f"Não foi possível buscar cotações online agora. ({e})")
    
    return dados_live

# Carregando os dados
df = carregar_dados_historicos()
cotacoes = obter_cotacoes_hoje()

# Treinando o Modelo (Cacheado para ser rápido)
@st.cache_resource
def treinar_modelo(df):
    features_base = ['Petroleo_Brent', 'Dolar', 'Acucar']
    target = 'Preco_Etanol'
    
    # Criando feature sazonalidade se não existir
    if 'Mes' not in df.columns:
        df['Mes'] = df.index.month
        
    X = df[features_base + ['Mes']]
    y = df[target]
    
    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    model.fit(X, y)
    score = model.score(X, y)
    return model, score

if df is not None:
    model, score = treinar_modelo(df)
    ultimo_preco_etanol = df['Preco_Etanol'].iloc[-1]
    data_etanol = df.index[-1].strftime('%d/%m/%Y')

# ==============================================================================
# 3. INTERFACE VISUAL (Frontend)
# ==============================================================================

# --- BANNER DE COTAÇÕES (Topo da Página) ---
# Mostra os preços do mercado internacional AGORA
st.subheader("🌍 Mercado Agora (Cotações em Tempo Real)")
col1, col2, col3, col4 = st.columns(4)

if cotacoes:
    with col1:
        st.metric("🛢️ Petróleo Brent", 
                  f"US$ {cotacoes.get('Petróleo Brent', {}).get('valor', 0):.2f}", 
                  f"{cotacoes.get('Petróleo Brent', {}).get('delta', 0):.2f}")
    with col2:
        st.metric("💵 Dólar", 
                  f"R$ {cotacoes.get('Dólar (USD/BRL)', {}).get('valor', 0):.3f}", 
                  f"{cotacoes.get('Dólar (USD/BRL)', {}).get('delta', 0):.3f}")
    with col3:
        st.metric("🍬 Açúcar (NY)", 
                  f"US$ {cotacoes.get('Açúcar No.11', {}).get('valor', 0):.2f}", 
                  f"{cotacoes.get('Açúcar No.11', {}).get('delta', 0):.2f}")
    with col4:
        # Etanol não tem ticker live fácil, usamos o último fechamento do CEPEA
        st.metric(f"⛽ Etanol (CEPEA - {data_etanol})", 
                  f"R$ {ultimo_preco_etanol:.2f}", 
                  help="Último fechamento disponível na base de dados CEPEA")

st.markdown("---")

# --- SISTEMA DE ABAS ---
tab1, tab2, tab3 = st.tabs(["🧮 Simulador de Preço Justo", "📈 Panorama Histórico", "ℹ️ Sobre o Modelo"])

# === ABA 1: O SIMULADOR (Seu código original melhorado) ===
with tab1:
    st.markdown("### 🤖 Calculadora de Valuation com IA")
    st.info(f"O modelo de Inteligência Artificial tem uma precisão de **{score:.1%}** baseada em 10 anos de histórico.")
    
    col_input, col_result = st.columns([1, 2])
    
    with col_input:
        st.markdown("#### Premissas de Cenário")
        
        # Valores iniciais pegando do Live ou do Histórico
        val_petroleo = cotacoes.get('Petróleo Brent', {}).get('valor', df['Petroleo_Brent'].iloc[-1])
        val_dolar = cotacoes.get('Dólar (USD/BRL)', {}).get('valor', df['Dolar'].iloc[-1])
        val_acucar = cotacoes.get('Açúcar No.11', {}).get('valor', df['Acucar'].iloc[-1])

        user_petroleo = st.slider("Petróleo Brent (US$)", 40.0, 150.0, float(val_petroleo))
        user_dolar = st.slider("Dólar (R$)", 3.0, 7.0, float(val_dolar))
        user_acucar = st.slider("Açúcar (cents/lb)", 10.0, 40.0, float(val_acucar))
        user_mes = st.selectbox("Mês de Referência", range(1, 13), index=int(df.index[-1].month - 1))

    with col_result:
        # Previsão
        cenario = pd.DataFrame({
            'Petroleo_Brent': [user_petroleo],
            'Dolar': [user_dolar],
            'Acucar': [user_acucar],
            'Mes': [user_mes]
        })
        preco_justo = model.predict(cenario)[0]
        spread = preco_justo - ultimo_preco_etanol
        
        # Cartão de Resultado Grande
        st.markdown("#### Resultado da Simulação")
        
        c1, c2 = st.columns(2)
        c1.metric("Preço Justo (Fair Value)", f"R$ {preco_justo:.2f}", help="Preço sugerido pelo modelo matemático")
        c2.metric("Potencial Upside/Downside", f"R$ {spread:.2f}", delta_color="normal")
        
        if preco_justo > ultimo_preco_etanol:
            st.success(f"📢 **OPORTUNIDADE DE COMPRA:** O Etanol está barato. Deveria custar R$ {preco_justo:.2f}, mas está R$ {ultimo_preco_etanol:.2f}.")
        else:
            st.error(f"📢 **OPORTUNIDADE DE VENDA:** O Etanol está caro. O preço justo seria R$ {preco_justo:.2f}.")

# === ABA 2: GRÁFICOS ===
with tab2:
    st.markdown("### 📊 Correlações Históricas")
    
    # Gráfico interativo com Plotly
    fig = px.scatter(df, x='Petroleo_Brent', y='Preco_Etanol', color=df.index.year,
                     title="Correlação: Petróleo x Etanol (2015-2025)",
                     labels={'Petroleo_Brent': 'Petróleo (US$)', 'Preco_Etanol': 'Etanol (R$)'},
                     color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("Este gráfico comprova que, historicamente, aumentos no petróleo puxam o preço do etanol para cima.")

# === ABA 3: SOBRE ===
with tab3:
    st.markdown("""
    ### Metodologia
    Este projeto utiliza dados públicos para oferecer transparência ao mercado sucroenergético.
    
    * **Fonte de Dados:** CEPEA/ESALQ e Yahoo Finance API.
    * **Modelo:** Random Forest Regressor (Machine Learning).
    * **Atualização:** Os dados históricos vão até a última atualização do arquivo CSV. As cotações do topo são em tempo real (delay de 15 min).
    
    **Desenvolvido por Giovanni Silva.**
    """)


