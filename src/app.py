import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

with tab3:
    if df is None or model is None:
        st.warning("Sem dados/modelo para exibir gráficos.")
    else:
        df2 = df.copy()
        df2["Mes"] = df2.index.month
        FEATURES = ["Petroleo_Brent", "Dolar", "Acucar", "Mes"]

        # Filtro de período
        min_d, max_d = df2.index.min().date(), df2.index.max().date()
        d1, d2 = st.date_input("Período", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        dff = df2.loc[str(d1):str(d2)].dropna(subset=FEATURES + ["Preco_Etanol"]).copy()

        # 1) Série temporal Etanol
        fig1 = px.line(dff, x=dff.index, y="Preco_Etanol", title="Preço do Etanol (Histórico)", template="plotly_dark")
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig1, use_container_width=True)

        # 2) Real vs Preço Justo (predição no histórico)
        dff["Preco_Justo_Modelo"] = model.predict(dff[FEATURES])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dff.index, y=dff["Preco_Etanol"], name="Real", mode="lines"))
        fig2.add_trace(go.Scatter(x=dff.index, y=dff["Preco_Justo_Modelo"], name="Preço Justo (Modelo)", mode="lines"))
        fig2.update_layout(title="Real vs Preço Justo (Modelo)", template="plotly_dark",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

        # 3) Spread
        dff["Spread"] = dff["Preco_Justo_Modelo"] - dff["Preco_Etanol"]
        fig3 = px.line(dff, x=dff.index, y="Spread", title="Spread (Preço Justo - Real)", template="plotly_dark")
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

        # 4) Feature importance
        imp = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_}) \
                .sort_values("importance", ascending=False)
        fig4 = px.bar(imp, x="feature", y="importance", title="Importância das Variáveis (RandomForest)", template="plotly_dark")
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig4, use_container_width=True)

        # 5) Correlação
        cols = [c for c in ["Preco_Etanol","Petroleo_Brent","Dolar","Acucar"] if c in dff.columns]
        corr = dff[cols].corr()
        fig5 = px.imshow(corr, text_auto=True, title="Correlação entre variáveis", template="plotly_dark")
        fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig5, use_container_width=True)
