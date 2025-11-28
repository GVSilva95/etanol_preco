⛽ Predição de Preços de Etanol com Machine Learning

Uma ferramenta de inteligência de mercado que utiliza Inteligência Artificial para calcular o "Preço Justo" do Etanol Hidratado com base no Petróleo, Dólar e Açúcar.

📊 O Problema

O mercado de commodities é altamente volátil. Para usinas e traders, saber se o preço atual do etanol está "caro" ou "barato" é o diferencial entre o lucro e o prejuízo. A precificação depende de uma complexa teia de fatores globais (Petróleo Brent, Câmbio) e locais (Safra, Açúcar).

🧠 A Solução

Desenvolvi um pipeline de dados End-to-End que:

Coleta dados históricos de 10 anos (CEPEA/ESALQ e Yahoo Finance).

Processa e limpa os dados, corrigindo disparidades e sincronizando mercados.

Treina um modelo de Machine Learning (Random Forest) para entender a correlação entre as variáveis.

Disponibiliza um Dashboard interativo para simulação de cenários.

🚀 Funcionalidades da Aplicação

Cálculo de Preço Justo: O modelo diz quanto o Etanol deveria custar hoje.

Indicador de Arbitragem: Alerta se o mercado está em oportunidade de COMPRA ou VENDA.

Simulador de Cenários: O utilizador pode testar hipóteses (ex: "Qual o impacto se o Petróleo subir para $100?").

Análise de Sensibilidade: Gráficos interativos que mostram a correlação histórica.

🛠️ Tecnologias Utilizadas

Linguagem: Python 3.11

Análise de Dados: Pandas, NumPy

Machine Learning: Scikit-Learn (Random Forest Regressor)

Visualização: Plotly, Matplotlib, Tableau

Web App: Streamlit

Fonte de Dados: yfinance API e Dados Públicos do CEPEA.

📈 Resultados Alcançados

O modelo final atingiu uma performance excepcional nos dados de teste:

Acurácia (R² Score): 99.6%

Principal Driver: Petróleo Brent (confirmando a paridade econômica).

⚙️ Como Executar o Projeto Localmente

Clone o repositório:

git clone [https://github.com/SEU_USUARIO/previsao-etanol-ia.git](https://github.com/SEU_USUARIO/previsao-etanol-ia.git)
cd previsao-etanol-ia



Crie um ambiente virtual (Opcional, mas recomendado):

python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate



Instale as dependências:

pip install -r requirements.txt



Execute a aplicação:

streamlit run src/app.py



📂 Estrutura do Projeto

agro_precos_etanol/
├── data/
│   ├── raw/          # Dados brutos (Excel do CEPEA)
│   └── processed/    # CSV final tratado (dataset_consolidado.csv)
├── notebooks/        # Jupyter Notebooks de análise e treino
├── src/              # Código fonte da aplicação (app.py)
├── images/           # Imagens para apresentação e README
├── requirements.txt  # Lista de bibliotecas
└── README.md         # Documentação



🤝 Autor

Giovanni Silva

LinkedIn https://www.linkedin.com/in/giovannivitorsilva/

Portfólio

Este projeto foi desenvolvido para fins educacionais e de portfólio na área de Data Science e Agronegócio.
