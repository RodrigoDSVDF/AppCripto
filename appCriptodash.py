import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.stats import norm
import altair as alt

# Configuração da página
st.set_page_config(page_title='Minha Dashboard Interativa', layout='wide')

# Funções para calcular indicadores técnicos
def calculate_moving_averages(df, window=30):
    df['SMA'] = df['fechamento'].rolling(window=window).mean()
    return df

def calculate_rsi(df, periods=14):
    delta = df['fechamento'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=periods, min_periods=periods).mean()
    avg_loss = loss.rolling(window=periods, min_periods=periods).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_macd(df, span1=12, span2=26, signal=9):
    df['MACD_line'] = df['fechamento'].ewm(span=span1, adjust=False).mean() - df['fechamento'].ewm(span=span2, adjust=False).mean()
    df['MACD_signal'] = df['MACD_line'].ewm(span=signal, adjust=False).mean()
    return df

# Carregamento dos dados
@st.cache_data
def importar_dados():
    caminho_arquivo_csv = "dados_cripto.csv"
    df = pd.read_csv(caminho_arquivo_csv)
    df['tempo'] = pd.to_datetime(df['tempo'])
    df['retorno_diario'] = df.groupby('moeda')['fechamento'].pct_change()  # Calcula retorno diário aqui
    df.dropna(subset=['retorno_diario'], inplace=True)  # Remove NaN values in 'retorno_diario'
    return df

df = importar_dados()

# Interface
st.title('Análises de Cripto Moedas')
st.sidebar.header('Menu')

opcoes = ['Home', 'Visualização', 'Análise', 'Sobre']
escolha = st.sidebar.selectbox("Escolha uma opção", opcoes)

if escolha == 'Home':
    url_da_imagem = 'https://images.unsplash.com/photo-1516245834210-c4c142787335?q=80&w=1469&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'
    st.image(url_da_imagem, use_column_width=True)


if escolha == 'Visualização':
    url_da_imagem = 'post_thumbnail-55a60f34beddda4324a2e11c4503b6f8.jpeg'
    st.image(url_da_imagem, use_column_width=True)

if escolha == 'Visualização':
    st.subheader('Visualização de Dados')
    criptomoedas = df['moeda'].unique()
    for moeda in criptomoedas:
        df_moeda = df[df['moeda'] == moeda]
        fig = px.line(df_moeda, x='tempo', y='fechamento', title=f'Preço de Fechamento ao Longo do Tempo para {moeda}')
        st.plotly_chart(fig)

elif escolha == 'Análise':
    st.subheader('Análise de Correlação e Indicadores de Mercado')
    criptomoedas = df['moeda'].unique()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Correlação Geral", "Médias Móveis", "RSI", "MACD", "Probabilidade de Variação", "Gráfico de Dispersão"])
    
    with tab1:
        for moeda in criptomoedas:
            df_moeda = df[df['moeda'] == moeda]
            fig = px.scatter(df_moeda, x='volume', y='fechamento', trendline="ols",
                             labels={'volume': 'Volume de Negociação', 'fechamento': 'Preço de Fechamento'},
                             title=f'Correlação entre Volume e Preço de Fechamento para {moeda}')
            st.plotly_chart(fig)

    with tab2:
        for moeda in criptomoedas:
            df_moeda = calculate_moving_averages(df[df['moeda'] == moeda].copy())
            fig = px.line(df_moeda, x='tempo', y=['fechamento', 'SMA'],
                          labels={'value': 'Preço', 'variable': 'Indicadores'},
                          title=f'Médias Móveis para {moeda}')
            st.plotly_chart(fig)
    
    with tab3:
        for moeda in criptomoedas:
            df_moeda = calculate_rsi(df[df['moeda'] == moeda].copy())
            fig = px.line(df_moeda, x='tempo', y='RSI', title=f'RSI para {moeda}')
            st.plotly_chart(fig)
    
    with tab4:
        for moeda in criptomoedas:
            df_moeda = calculate_macd(df[df['moeda'] == moeda].copy())
            fig = px.line(df_moeda, x='tempo', y=['MACD_line', 'MACD_signal'],
                          labels={'value': 'MACD', 'variable': 'Linhas MACD'},
                          title=f'MACD para {moeda}')
            st.plotly_chart(fig)
    
    with tab5:
        for moeda in criptomoedas:
            df_moeda = df[df['moeda'] == moeda]
            mu, std = norm.fit(df_moeda['retorno_diario'])
            x = np.linspace(df_moeda['retorno_diario'].min(), df_moeda['retorno_diario'].max(), 100)
            p = norm.pdf(x, mu, std)
            df_normal = pd.DataFrame({'x': x, 'p': p})
            histogram = alt.Chart(df_moeda).transform_density(
                'retorno_diario',
                as_=['retorno_diario', 'density'],
                extent=[df_moeda['retorno_diario'].min(), df_moeda['retorno_diario'].max()]
            ).mark_area(opacity=0.5).encode(
                alt.X('retorno_diario:Q'),
                alt.Y('density:Q'),
            )
            normal_curve = alt.Chart(df_normal).mark_line(color='red').encode(
                alt.X('x'),
                alt.Y('p')
            )
            st.altair_chart(histogram + normal_curve, use_container_width=True)
            st.write(f"{moeda} - Retorno Médio: {df_moeda['retorno_diario'].mean()}, Risco: {df_moeda['retorno_diario'].std()}")
    
    with tab6:
        fig = go.Figure()
        for moeda in criptomoedas:
            df_moeda = df[df['moeda'] == moeda]
            fig.add_trace(go.Scatter(x=[df_moeda['retorno_diario'].mean()], y=[df_moeda['retorno_diario'].std()], mode='markers', name=moeda))
        fig.update_xaxes(title='Média esperada retorno diário', showgrid=True)
        fig.update_yaxes(title='Risco diário', showgrid=True)
        st.plotly_chart(fig)

# Este é um exemplo completo, assegure-se de adaptar caminhos, colunas e dados conforme necessário.
