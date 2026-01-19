# Etanol Intelligence Pro ⛽🌱  
**Dashboard interativo para analisar o preço do etanol, paridade com gasolina e sinais de “preço justo” usando séries temporais.**  
👉 App: https://etanol-invest-giovanni.streamlit.app/

---

## 🎯 Objetivo
Investigar a relação entre o preço do **etanol** e variáveis de mercado (ex.: **Brent**, **USD/BRL**, **açúcar**) e entregar um app que:
- Mostra **tendência histórica** e **correlações**
- Simula **paridade etanol vs gasolina**
- Estima um **preço justo** com Machine Learning (comparando com baseline)

---

## 🧠 O que tem no app
- **KPIs** (preço real, preço justo, spread e sinal)
- **Gráficos**: Real vs Preço Justo, Spread, Histograma do Spread, Correlação, Importância das variáveis
- **Paridade na bomba** (regra do 70% configurável)
- **Diagnóstico do modelo** (Predito vs Real, Resíduos, métricas e baseline)

---

## 📌 Metodologia (resumo)
1. Consolidação e limpeza do dataset
2. Padronização temporal (**resample**) para evitar desalinhamento entre séries
3. Criação de features:
   - Sazonalidade (mês)
   - Defasagens (**lags**) e médias móveis
4. Treino com validação temporal (TimeSeriesSplit)
5. Comparação com um baseline forte: **último valor observado (naive)**

> ⚠️ Observação: Em séries temporais, o baseline naive é um benchmark difícil de bater e é normal em alguns cenários ele superar modelos mais complexos.

---

## 📊 Métricas
O app exibe (no diagnóstico):
- **R² e MAPE do modelo**
- **R² e MAPE do baseline (naive)**
- Período real de teste temporal

---

## 🗂️ Estrutura do projeto (sugestão)
etanol_preco/
├─ src/
│ └─ app.py
├─ data/
│ └─ processed/
│ └─ dataset_consolidado.csv
├─ images/ (opcional)
├─ requirements.txt
└─ README.md
