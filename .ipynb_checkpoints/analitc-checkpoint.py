import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração da página
st.set_page_config(
    page_title="Dashboard de Análise de Leite",
    page_icon="🥛",
    layout="wide"
)

# Título do Dashboard
st.title("🥛 Dashboard de Análise Exploratória de Amostras de Leite")
st.markdown("### Insights para Tomada de Decisão na Indústria de Laticínios")
st.markdown("---")

# Função para carregar e processar os dados
@st.cache_data
def load_and_process_data():
    df = pd.read_parquet("dados.parquet")

    # Lidar com valores ausentes: Imputar com a média para colunas numéricas
    numerical_cols_with_nans = [col for col in ["Caseína", "Densidade"] if df[col].isnull().any()]
    for col in numerical_cols_with_nans:
        df[col] = df[col].fillna(df[col].mean())

    # Feature Engineering: Extrair informações da coluna DATAANALISE
    df["Ano"] = df["DATAANALISE"].dt.year
    df["Mes"] = df["DATAANALISE"].dt.month
    df["DiaSemana"] = df["DATAANALISE"].dt.dayofweek # 0=Segunda, 6=Domingo
    df["Trimestre"] = df["DATAANALISE"].dt.quarter

    # Converter \'Estacão\' para tipo categórico, se ainda não for
    df["Estacão"] = df["Estacão"].astype("category")
    df["CIDADE"] = df["CIDADE"].astype("category")
    
    return df

df = load_and_process_data()

# Função para gerar e salvar as visualizações
@st.cache_data
def generate_visualizations(df_full):
    # --- Configurações de Visualização ---
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12

    # --- Visualização 1: Distribuição dos Componentes Principais do Leite ---
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df_full["Gordura"], kde=True, color="skyblue", ax=axes1[0])
    axes1[0].set_title("Distribuição de Gordura")
    axes1[0].set_xlabel("Gordura (%)")
    axes1[0].set_ylabel("Frequência")

    sns.histplot(df_full["Proteina"], kde=True, color="lightcoral", ax=axes1[1])
    axes1[1].set_title("Distribuição de Proteína")
    axes1[1].set_xlabel("Proteína (%)")
    axes1[1].set_ylabel("Frequência")

    sns.histplot(df_full["Lactose"], kde=True, color="lightgreen", ax=axes1[2])
    axes1[2].set_title("Distribuição de Lactose")
    axes1[2].set_xlabel("Lactose (%)")
    axes1[2].set_ylabel("Frequência")
    plt.tight_layout()
    fig1.savefig("distribuicao_componentes.png")
    plt.close(fig1)

    # --- Visualização 2: Gordura Média por Estação ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    gordura_por_estacao = df_full.groupby("Estacão", observed=False)["Gordura"].mean().sort_values(ascending=False)
    sns.barplot(x=gordura_por_estacao.index, y=gordura_por_estacao.values, palette="viridis", ax=ax2)
    ax2.set_title("Gordura Média por Estação do Ano")
    ax2.set_xlabel("Estação")
    ax2.set_ylabel("Gordura Média (%)")
    fig2.savefig("gordura_por_estacao.png")
    plt.close(fig2)

    # --- Visualização 3: Proteína Média por Cidade (Top N) ---
    top_n = st.sidebar.slider('Quantas cidades mostrar no gráfico de proteína média?', min_value=5, max_value=30, value=10)
    proteina_por_cidade = df_full.groupby("CIDADE", observed=False)["Proteina"].mean().sort_values(ascending=False).head(top_n)
    fig3, ax3 = plt.subplots(figsize=(10, top_n * 0.6))
    bars = sns.barplot(x=proteina_por_cidade.values, y=proteina_por_cidade.index, palette="magma", ax=ax3)
    ax3.set_title(f"Top {top_n} Cidades com Maior Proteína Média")
    ax3.set_xlabel("Proteína Média (%)")
    ax3.set_ylabel("Cidade")
    # Adicionar valores nas barras
    for i, v in enumerate(proteina_por_cidade.values):
        ax3.text(v + 0.01, i, f"{v:.2f}", color='black', va='center', fontsize=10)
    plt.tight_layout()
    fig3.savefig("proteina_por_cidade.png")
    st.pyplot(fig3)
    plt.close(fig3)

    # --- Visualização 4: Tendência Anual da Gordura ---
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    gordura_por_ano = df_full.groupby("Ano")["Gordura"].mean()
    sns.lineplot(x=gordura_por_ano.index, y=gordura_por_ano.values, marker="o", color="darkblue", ax=ax4)
    ax4.set_title("Tendência Anual da Gordura Média no Leite")
    ax4.set_xlabel("Ano")
    ax4.set_ylabel("Gordura Média (%)")
    plt.xticks(rotation=45)
    fig4.savefig("tendencia_gordura_anual.png")
    plt.close(fig4)

    # --- Visualização 5: Matriz de Correlação ---
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    correlation_matrix = df_full[["Gordura", "Proteina", "Lactose", "ST", "Log CCS", "CCS", "Caseína", "Densidade", "SNG"]].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, ax=ax5)
    ax5.set_title("Matriz de Correlação das Variáveis Numéricas")
    plt.tight_layout()
    fig5.savefig("matriz_correlacao.png")
    plt.close(fig5)

# Gerar as visualizações (serão cacheadas)
generate_visualizations(df)

# Sidebar para filtros
st.sidebar.header("🔍 Filtros de Dados")

# Filtro por Estação
estacoes_selecionadas = st.sidebar.multiselect(
    "Selecione as Estações",
    options=df["Estacão"].unique(),
    default=df["Estacão"].unique()
)

# Filtro por Ano
anos_selecionados = st.sidebar.multiselect(
    "Selecione os Anos",
    options=sorted(df["Ano"].unique()),
    default=sorted(df["Ano"].unique())
)

# Aplicar filtros
df_filtered = df[
    (df["Estacão"].isin(estacoes_selecionadas)) &
    (df["Ano"].isin(anos_selecionados))
]

# Exibir estatísticas gerais
st.header("📊 Estatísticas Gerais do Dataset Filtrado")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total de Amostras", f"{len(df_filtered):,}")
with col2:
    st.metric("Gordura Média", f"{df_filtered["Gordura"].mean():.2f}%")
with col3:
    st.metric("Proteína Média", f"{df_filtered["Proteina"].mean():.2f}%")
with col4:
    st.metric("Lactose Média", f"{df_filtered["Lactose"].mean():.2f}%")

st.markdown("---")

# Seção de visualizações
st.header("📈 Visualizações e Insights")

# Tabs para organizar as visualizações
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Distribuição dos Componentes",
    "Análise Sazonal",
    "Análise Regional",
    "Tendências Temporais",
    "Correlações"
])

with tab1:
    st.subheader("Distribuição dos Componentes Principais do Leite")
    st.image("distribuicao_componentes.png", use_container_width=True)
    
    st.markdown("""
    **Insights:**
    - A distribuição de **Gordura** mostra uma concentração em torno de 3.5% a 4.0%, com uma distribuição aproximadamente normal.
    - A **Proteína** apresenta uma distribuição mais estreita, centrada em torno de 3.2%.
    - A **Lactose** tem uma distribuição mais ampla, indicando maior variabilidade entre as amostras.
    """)

with tab2:
    st.subheader("Gordura Média por Estação do Ano")
    st.image("gordura_por_estacao.png", use_container_width=True)
    
    # Calcular estatísticas para o dataset filtrado
    gordura_estacao = df_filtered.groupby("Estacão", observed=False)["Gordura"].mean().sort_values(ascending=False)
    
    st.markdown("""
    **Insights:**
    - O **Outono** e o **Inverno** apresentam os maiores teores médios de gordura no leite.
    - Durante a **Primavera** e o **Verão**, o teor de gordura tende a ser ligeiramente menor.
    - **Recomendação:** Ajustar a alimentação do gado durante a primavera e o verão para manter níveis consistentes de gordura.
    """)
    
    st.dataframe(gordura_estacao, use_container_width=True)

with tab3:
    st.subheader("Top 10 Cidades com Maior Proteína Média")
    st.image("proteina_por_cidade.png", use_container_width=True)
    
    # Calcular estatísticas para o dataset filtrado
    proteina_cidade = df_filtered.groupby("CIDADE", observed=False)["Proteina"].mean().sort_values(ascending=False).head(10)
    
    st.markdown("""
    **Insights:**
    - Cidades como **CARLOS BARBOSA-RS** e **JAQUIRANA-RS** se destacam com os maiores teores de proteína.
    - Essas regiões podem ter práticas de manejo superiores ou condições ambientais favoráveis.
    - **Recomendação:** Investigar as práticas dessas regiões para replicar em outras áreas.
    """)
    
    st.dataframe(proteina_cidade, use_container_width=True)

with tab4:
    st.subheader("Tendência Anual da Gordura Média no Leite")
    st.image("tendencia_gordura_anual.png", use_container_width=True)
    
    # Calcular estatísticas para o dataset filtrado
    gordura_ano = df_filtered.groupby("Ano")["Gordura"].mean()
    
    st.markdown("""
    **Insights:**
    - A gordura média permanece relativamente estável ao longo dos anos, com pequenas variações.
    - Não há uma tendência clara de aumento ou diminuição significativa.
    - **Recomendação:** Manter o monitoramento contínuo para detectar mudanças futuras.
    """)
    
    st.dataframe(gordura_ano, use_container_width=True)

with tab5:
    st.subheader("Matriz de Correlação das Variáveis Numéricas")
    st.image("matriz_correlacao.png", use_container_width=True)
    
    st.markdown("""
    **Insights:**
    - **Proteína** e **Caseína** têm uma correlação muito forte (0.93), o que é esperado biologicamente.
    - **Gordura** e **Sólidos Totais (ST)** também apresentam alta correlação (0.88).
    - **Lactose** tem correlação negativa com **Log CCS** e **CCS**, sugerindo que amostras com maior contagem de células somáticas tendem a ter menor lactose.
    - **Recomendação:** Utilizar essas correlações para otimizar formulações de produtos lácteos e controle de qualidade.
    """)

# Seção de análise detalhada
st.markdown("---")
st.header("🔬 Análise Detalhada")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Estatísticas Descritivas")
    st.dataframe(
        df_filtered[["Gordura", "Proteina", "Lactose", "ST", "CCS", "Caseína", "Densidade", "SNG"]].describe(),
        use_container_width=True
    )

with col2:
    st.subheader("Distribuição por Estação")
    estacao_counts = df_filtered["Estacão"].value_counts()
    st.bar_chart(estacao_counts)

# Footer
st.markdown("---")
st.markdown("**Desenvolvido por:** Manus AI | **Dataset:** Análise de Amostras de Leite | **Foco:** Tratamento de Dados e Análise Exploratória para Tomada de Decisão")
