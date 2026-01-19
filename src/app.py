from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

DATA_PATH = Path("data/processed/dataset_consolidado.csv")
FEATURES = ["Petroleo_Brent", "Dolar", "Acucar", "Mes"]
TARGET = "Preco_Etanol"
REQUIRED_COLS = ["Petroleo_Brent", "Dolar", "Acucar", "Preco_Etanol"]

@st.cache_data
def carregar_dados():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    return df

@st.cache_data(ttl=1800)  # 30 min
def get_market_data():
    tickers = {
        "Petróleo Brent": "BZ=F",
        "Dólar (BRL)": "BRL=X",
        "Açúcar (NY)": "SB=F",
        "Milho (Chicago)": "ZC=F",
        "Gasolina RBOB": "RB=F",
        "Gás Natural": "NG=F",
        "Juros EUA 10Y": "^TNX",
    }
    out = {}
    for name, t in tickers.items():
        try:
            h = yf.Ticker(t).history(period="7d")
            h = h.dropna()
            if len(h) >= 2:
                out[name] = {
                    "val": float(h["Close"].iloc[-1]),
                    "delta": float(h["Close"].iloc[-1] - h["Close"].iloc[-2]),
                    "date": h.index[-1],
                }
            else:
                out[name] = {"val": 0.0, "delta": 0.0, "date": None}
        except Exception:
            out[name] = {"val": 0.0, "delta": 0.0, "date": None}
    return out

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # valida colunas
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas faltando no dataset: {missing}")
    # feature temporal
    df["Mes"] = df.index.month
    # remove linhas com NA nas colunas importantes
    df = df.dropna(subset=REQUIRED_COLS + ["Mes"])
    return df

@st.cache_resource
def train_model(df: pd.DataFrame):
    df2 = prepare_df(df)
    X = df2[FEATURES]
    y = df2[TARGET]

    tscv = TimeSeriesSplit(n_splits=5)
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    # pega o ÚLTIMO split como teste (mais realista)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "r2_test": r2_score(y_test, y_pred),
        "mae_test": mean_absolute_error(y_test, y_pred),
        "mape_test": mean_absolute_percentage_error(y_test, y_pred),
        "test_start": X_test.index.min(),
        "test_end": X_test.index.max(),
    }

    last_price = float(df2[TARGET].iloc[-1])
    last_date = df2.index[-1]

    return model, metrics, last_price, last_date
