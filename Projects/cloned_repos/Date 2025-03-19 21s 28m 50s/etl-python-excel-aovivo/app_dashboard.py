import streamlit as st
import pandas as pd
import plotly.express as px


# Título do dashboard
st.title('Análise de KPIs de Anúncios')

# Upload do arquivo CSV
uploaded_file = st.file_uploader("Envie um arquivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Convertendo colunas para os tipos adequados
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Amount_spent'] = df['Amount_spent'].replace({'R$ ': ''}, regex=True).astype(float)
    df['Link_clicks'] = pd.to_numeric(df['Link_clicks'], errors='coerce').fillna(0).astype(int)
    df['Conversions'] = pd.to_numeric(df['Conversions'], errors='coerce').fillna(0).astype(int)
    
    # Cálculo de KPIs
    kpi1 = df.groupby(df['Date'].dt.strftime('%Y-%m'))['Amount_spent'].sum()
    kpi2 = df.groupby(df['Date'].dt.strftime('%Y-%m'))['Conversions'].sum()
    kpi3 = df.groupby(df['Date'].dt.strftime('%Y-%m'))['Link_clicks'].sum()
    kpi4 = (df.groupby(df['Date'].dt.strftime('%Y-%m'))['Amount_spent'].sum() / df.groupby(df['Date'].dt.strftime('%Y-%m'))['Conversions'].sum()).fillna(0)
    
    # Exibição dos dados
    st.write("### Amostra dos Dados")
    st.dataframe(df.head())

    # Exibição dos KPIs lado a lado
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Mês com Maior Gasto", value=str(kpi1.idxmax()))
    with col2:
        st.metric(label="Total de Conversões no Mês com Mais Gasto", value=int(kpi2.max()))
    with col3:
        st.metric(label="Total de Cliques no Mês com Mais Gasto", value=int(kpi3.max()))
    with col4:
        st.metric(label="Custo por Conversão Médio", value=f"R$ {kpi4.mean():.2f}")
    
    # Gráfico de Gasto por Data usando o Streamlit
    st.write("### Gasto Diário com Marketing")
    st.line_chart(df.groupby('Date')['Amount_spent'].sum())
    
    # Gráfico de Gasto por Segmentação
    st.write("### Gasto por Segmentação")
    segmentacao_gasto = df.groupby('Segmentação')['Amount_spent'].sum().sort_values(ascending=False)
    st.bar_chart(segmentacao_gasto)

        # KPI calculations
    df["CPC"] = (df["Amount_spent"] / df["Link_clicks"]).replace(
        [float("inf"), float("nan")], 0
    )
    df["CPM"] = (df["Amount_spent"] / df["Impressions"] * 1000).replace(
        [float("inf"), float("nan")], 0
    )
    df["CPA"] = (df["Amount_spent"] / df["Conversions"]).replace(
        [float("inf"), float("nan")], 0
    )
    df["CTR (%)"] = (df["Link_clicks"] / df["Impressions"] * 100).replace(
        [float("inf"), float("nan")], 0
    )
    df["Conversion Rate (%)"] = (df["Conversions"] / df["Link_clicks"] * 100).replace(
        [float("inf"), float("nan")], 0
    )

    # Interactive Monthly Analysis
    st.subheader("🔍 Interactive Monthly Analysis")
    df["Month"] = df["Date"].dt.month_name()
    months = df["Month"].unique().tolist()
    selected_month = st.selectbox("Select Month for Analysis", months)

    column_options = ["Amount_spent", "Link_clicks", "Impressions", "Conversions"]
    selected_column = st.selectbox("Select KPI for Analysis", column_options)

    monthly_df = df[df["Month"] == selected_month]
    daily_summary = (
        monthly_df.groupby(df["Date"].dt.day)[selected_column].sum().reset_index()
    )
    daily_summary.columns = ["Day", selected_column]

    fig_monthly = px.bar(
        daily_summary,
        x="Day",
        y=selected_column,
        title=f"Daily {selected_column} in {selected_month}",
        labels={"Day": "Day of Month", selected_column: selected_column},
    )

    st.plotly_chart(fig_monthly)

else:
    st.write("Por favor, envie um arquivo CSV para análise.")
