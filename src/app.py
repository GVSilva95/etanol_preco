import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Configuração da Página (Título e Layout)
st.set_page_config(page_title="Simulador de Preço de Etanol", layout="wide")

# ==============================================================================
# 1. CARREGAMENTO E TREINAMENTO (O Cérebro da App)
# ==============================================================================
@st.cache_data # Isso faz o site ficar rápido (não recarrega os dados toda hora)
def carregar_e_treinar():
    # Carregar dados
    try:
        # Ajustando caminho para rodar da raiz do projeto
        df = pd.read_csv('data/processed/dataset_consolidado.csv', index_col=0, parse_dates=True)
    except:
        st.error("Erro: Não achei o arquivo 'dataset_consolidado.csv'. Verifique a pasta 'data/processed'.")
        return None, None, None

    # Engenharia de Features (Igual ao seu Notebook vencedor)
    features_base = ['Petroleo_Brent', 'Dolar', 'Acucar']
    target = 'Preco_Etanol'
    df['Mes'] = df.index.month
    
    # Treinamento do Modelo
    X = df[features_base + ['Mes']]
    y = df[target]
    
    # Treinando com TODOS os dados para o simulador ficar esperto
    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    model.fit(X, y)
    
    # Calcular acurácia só para mostrar
    model_score = model.score(X, y)
    
    return model, df, model_score

# Carrega a IA
model, df, score = carregar_e_treinar()

# ==============================================================================
# 2. BARRA LATERAL (Controles do Usuário)
# ==============================================================================
st.sidebar.header("🎛️ Painel de Controle")
st.sidebar.markdown("Crie seus próprios cenários:")

# Pegar os últimos valores reais para usar de padrão
ultimo_petroleo = df['Petroleo_Brent'].iloc[-1]
ultimo_dolar = df['Dolar'].iloc[-1]
ultimo_acucar = df['Acucar'].iloc[-1]

# Sliders para simulação
user_petroleo = st.sidebar.slider("🛢️ Petróleo Brent (US$)", 
                                  min_value=40.0, max_value=150.0, 
                                  value=float(ultimo_petroleo))

user_dolar = st.sidebar.slider("💵 Taxa de Câmbio (R$)", 
                               min_value=3.0, max_value=7.0, 
                               value=float(ultimo_dolar))

user_acucar = st.sidebar.slider("🍬 Açúcar (US$ cents/lb)", 
                                min_value=10.0, max_value=30.0, 
                                value=float(ultimo_acucar))

user_mes = st.sidebar.selectbox("📅 Mês da Safra", range(1, 13), index=int(df.index[-1].month - 1))

# ==============================================================================
# 3. CORPO PRINCIPAL (Resultados)
# ==============================================================================
st.title("⛽ Simulador de Preços: Etanol Hidratado")
st.markdown(f"**Inteligência Artificial Calibrada** (Precisão do Modelo: `{score:.1%}`)")
st.markdown("---")

# Fazer a Previsão com os dados do usuário
cenario = pd.DataFrame({
    'Petroleo_Brent': [user_petroleo],
    'Dolar': [user_dolar],
    'Acucar': [user_acucar],
    'Mes': [user_mes]
})

preco_previsto = model.predict(cenario)[0]
preco_atual_mercado = df['Preco_Etanol'].iloc[-1]

# Exibindo os números grandes (KPIs)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Preço Justo (Calculado pela IA)", f"R$ {preco_previsto:.4f}")

with col2:
    variacao = preco_previsto - preco_atual_mercado
    st.metric("Diferença para o Hoje", f"R$ {variacao:.4f}", delta_color="inverse")

with col3:
    status = "CARO (Vender)" if preco_atual_mercado > preco_previsto else "BARATO (Comprar)"
    cor = "red" if "CARO" in status else "green"
    st.markdown(f"### Status: :{cor}[{status}]")

# Gráfico de Sensibilidade
st.markdown("---")
st.subheader("📈 Análise de Sensibilidade: Impacto do Petróleo")

# Criar dados falsos para plotar a linha de tendência
faixa_petroleo = np.linspace(40, 150, 50)
dados_simulados = []
for p in faixa_petroleo:
    dados_simulados.append([p, user_dolar, user_acucar, user_mes])
    
df_simulado = pd.DataFrame(dados_simulados, columns=['Petroleo_Brent', 'Dolar', 'Acucar', 'Mes'])
df_simulado['Preco_Estimado'] = model.predict(df_simulado)

fig = px.line(df_simulado, x='Petroleo_Brent', y='Preco_Estimado', 
              title=f"Como o preço do Etanol muda se o Petróleo subir? (Dólar fixo em R$ {user_dolar})",
              labels={'Petroleo_Brent': 'Preço do Barril de Petróleo (US$)', 'Preco_Estimado': 'Preço do Etanol (R$)'})

# Adiciona um ponto vermelho onde o usuário escolheu
fig.add_scatter(x=[user_petroleo], y=[preco_previsto], mode='markers', marker=dict(size=15, color='red'), name='Cenário Atual')

st.plotly_chart(fig, use_container_width=True)