import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("milk_sessions.csv")

milk_quality_cols = [
    "DailyFat_P", "DailyProtein_P", "Fat", "Protein", "Lactose", "LogScc",
    "Cf", "Blood", "Component7", "Casein", "Mufa", "Pufa", "Sfa", "Ufa",
    "Pa", "Sa", "Oa"
]

# Converter 'Date' para datetime
df["Date"] = pd.to_datetime(df["Date"])

# 1. Estatísticas descritivas para as colunas de qualidade do leite
print("### Estatísticas Descritivas das Variáveis de Qualidade do Leite ###")
print(df[milk_quality_cols].describe())

# 2. Verificação de valores ausentes
print("\n### Valores Ausentes nas Variáveis de Qualidade do Leite ###")
print(df[milk_quality_cols].isnull().sum())

# 3. Correlação entre as variáveis de qualidade do leite
print("\n### Matriz de Correlação das Variáveis de Qualidade do Leite ###")
correlation_matrix = df[milk_quality_cols].corr()
print(correlation_matrix)

# Plotar e salvar o heatmap da matriz de correlação
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title("Matriz de Correlação das Variáveis de Qualidade do Leite")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# 4. Visualizações (exemplos para algumas variáveis chave)
# Histograma da Gordura Diária
plt.figure(figsize=(10, 6))
sns.histplot(df["DailyFat_P"], kde=True)
plt.title("Distribuição de Gordura Diária no Leite (%)")
plt.xlabel("Gordura Diária (%)")
plt.ylabel("Frequência")
plt.savefig("daily_fat_distribution.png")
plt.close()

# Histograma da Proteína Diária
plt.figure(figsize=(10, 6))
sns.histplot(df["DailyProtein_P"], kde=True)
plt.title("Distribuição de Proteína Diária no Leite (%)")
plt.xlabel("Proteína Diária (%)")
plt.ylabel("Frequência")
plt.savefig("daily_protein_distribution.png")
plt.close()

# Scatter plot: Gordura vs Proteína
plt.figure(figsize=(10, 6))
sns.scatterplot(x="DailyFat_P", y="DailyProtein_P", data=df, alpha=0.5)
plt.title("Gordura Diária vs Proteína Diária no Leite")
plt.xlabel("Gordura Diária (%)")
plt.ylabel("Proteína Diária (%)")
plt.savefig("fat_vs_protein_scatterplot.png")
plt.close()

# Boxplot da Gordura Diária por LactationNumber
plt.figure(figsize=(12, 7))
sns.boxplot(x="LactationNumber", y="DailyFat_P", data=df)
plt.title("Gordura Diária por Número de Lactação")
plt.xlabel("Número de Lactação")
plt.ylabel("Gordura Diária (%)")
plt.savefig("daily_fat_by_lactation.png")
plt.close()

# Boxplot da Proteína Diária por LactationNumber
plt.figure(figsize=(12, 7))
sns.boxplot(x="LactationNumber", y="DailyProtein_P", data=df)
plt.title("Proteína Diária por Número de Lactação")
plt.xlabel("Número de Lactação")
plt.ylabel("Proteína Diária (%)")
plt.savefig("daily_protein_by_lactation.png")
plt.close()

# Análise temporal da produção de leite (exemplo)
# Agrupar por data e calcular a média da produção diária
daily_avg_yield = df.groupby("Date")["DailyYield_KG"].mean().reset_index()
plt.figure(figsize=(14, 7))
sns.lineplot(x="Date", y="DailyYield_KG", data=daily_avg_yield)
plt.title("Média Diária de Produção de Leite ao Longo do Tempo")
plt.xlabel("Data")
plt.ylabel("Média Diária de Produção de Leite (KG)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("daily_yield_over_time.png")
plt.close()

print("Análise exploratória concluída. Gráficos e matriz de correlação salvos.")