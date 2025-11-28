import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA E DESIGN
# ==============================================================================
st.set_page_config(
    page_title="Etanol Intelligence Pro",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado para métricas
st.markdown("""
<style>
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. FUNÇÕES DE CARREGAMENTO (Backend)
# ==============================================================================

@st.cache_data
def carregar_dados_historicos():
    try:
        # Tenta carregar o CSV.
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

# Inicialização
df = carregar_dados_historicos()
cotacoes = obter_cotacoes_hoje()

# Treinamento do Modelo
@st.cache_resource
def treinar_modelo(df):
    if df is None: return None, 0
    
    # Garante a feature sazonal
    if 'Mes' not in df.columns:
        df['Mes'] = df.index.month
    
    df_clean = df.dropna()
    # Features usadas no treino
    features = ['Petroleo_Brent', 'Dolar', 'Acucar']
    
    # Se o modelo foi treinado com 'Mes', precisamos garantir que ele entre
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
    st.image("https://images.unsplash.com/photo-1597850239592-3d7790c50720?q=80&w=400&auto=format&fit=crop", caption="Setor Sucroenergético")
    st.header("Painel de Controle")
    st.info("Este dashboard utiliza IA para calcular o preço justo do etanol com base em commodities globais.")
    st.markdown("---")
    if model:
        st.write(f"**Modelo:** Random Forest")
        st.write(f"**Acurácia:** {score:.1%}")
        st.write(f"**Dados até:** {data_ref}")

# ==============================================================================
# 4. CORPO PRINCIPAL
# ==============================================================================

st.title("⛽ Etanol Intelligence: Global Dashboard")
st.markdown("### Monitorização de Mercado em Tempo Real")

# --- BANNER DE COTAÇÕES (5 Colunas agora) ---
cols = st.columns(5)

# Função auxiliar para exibir métrica segura
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
tab1, tab2, tab3 = st.tabs(["🧮 Simulador de Preço", "🌍 Contexto Global", "📊 Gráficos Históricos"])

# === ABA 1: SIMULADOR ===
with tab1:
    st.header("Simulador de Paridade & Preço Justo")
    
    if model:
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.subheader("Cenário")
            
            # Pega valores padrão
            def get_val(key, col):
                val_live = float(cotacoes.get(key, {}).get('valor', 0.0))
                val_hist = float(df[col].iloc[-1]) if df is not None else 0.0
                return val_live if val_live > 0 else val_hist

            petroleo = st.slider("Petróleo Brent (US$)", 40.0, 150.0, get_val('Petróleo Brent', 'Petroleo_Brent'))
            dolar = st.slider("Dólar (R$)", 3.0, 7.0, get_val('Dólar (BRL)', 'Dolar'))
            acucar = st.slider("Açúcar (cents/lb)", 10.0, 40.0, get_val('Açúcar (NY)', 'Acucar'))
            
            idx_mes = 0
            if df is not None:
                idx_mes = int(df.index[-1].month - 1)
            mes = st.selectbox("Mês de Safra", range(1, 13), index=idx_mes)

        with c2:
            # Previsão
            cenario = pd.DataFrame({
                'Petroleo_Brent': [petroleo],
                'Dolar': [dolar],
                'Acucar': [acucar],
                'Mes': [mes]
            })
            preco_justo = model.predict(cenario)[0]
            diff = preco_justo - ultimo_preco
            
            st.subheader("Resultado da IA")
            res_col1, res_col2 = st.columns(2)
            
            res_col1.metric("Preço Justo (Paulínia)", f"R$ {preco_justo:.2f}")
            res_col2.metric("Diferença Mercado", f"R$ {diff:.2f}", delta_color="normal")
            
            if preco_justo > ultimo_preco:
                st.success("📢 **SINAL DE COMPRA:** O mercado está abaixo do preço justo calculado.")
            else:
                st.error("📢 **SINAL DE VENDA:** O mercado está acima do preço justo calculado.")
                
            # Gráfico de termómetro simples com barra de progresso
            st.write("Termómetro de Preço:")
            percentual = min(max((preco_justo / 4000) * 100, 0), 100) # Normalizando para barra 0-100
            st.progress(int(percentual))
            st.caption("Escala visual de preço (0 a R$ 4.000)")
    else:
        st.warning("A aguardar dados para carregar o simulador...")

# === ABA 2: CONTEXTO GLOBAL (NOVO!) ===
with tab2:
    st.header("Panorama Global do Etanol")
    st.markdown("O preço do etanol brasileiro não depende apenas de nós. Entenda os grandes players:")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.image("https://images.unsplash.com/photo-1632219782522-a7229a438722?q=80&w=600&auto=format&fit=crop", caption="Milho nos EUA")
        st.subheader("🇺🇸 Estados Unidos (Milho)")
        st.write("""
        * **Matéria-prima:** Milho (Corn Ethanol).
        * **Influência:** É o maior produtor mundial. Se a safra de milho nos EUA quebra, o preço do etanol global sobe.
        * **Relação:** Acompanhe a cotação do Milho (ZC=F) no topo da página.
        """)
        
    with col_g2:
        st.image("https://images.unsplash.com/photo-1605000797499-95a51c5269ae?q=80&w=600&auto=format&fit=crop", caption="Cana na Índia e Brasil")
        st.subheader("🇮🇳 Índia & 🇧🇷 Brasil (Cana)")
        st.write("""
        * **Matéria-prima:** Cana-de-Açúcar.
        * **Índia:** Está a aumentar a mistura de etanol na gasolina (E20), o que retira açúcar do mercado global.
        * **Brasil:** O mix produtivo (Açúcar vs Etanol) define a oferta. Se o açúcar paga mais, produz-se menos etanol.
        """)

# === ABA 3: GRÁFICOS ===
with tab3:
    if df is not None:
        st.header("Correlação Histórica (10 Anos)")
        
        # Gráfico Scatter
        fig_scatter = px.scatter(
            df, x='Petroleo_Brent', y='Preco_Etanol', 
            color=df.index.year,
            size_max=10,
            color_continuous_scale='Turbo',
            title="Matriz de Dispersão: Petróleo vs Etanol"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Gráfico de Linha Comparativo (Normalizado)
        st.subheader("Tendência Relativa (Normalizada)")
        df_norm = df[['Preco_Etanol', 'Petroleo_Brent']].copy()
        df_norm = df_norm / df_norm.iloc[0] * 100 # Base 100
        
        fig_line = px.line(df_norm, title="Quem subiu mais? (Base 100 = Início da Série)")
        st.plotly_chart(fig_line, use_container_width=True)
