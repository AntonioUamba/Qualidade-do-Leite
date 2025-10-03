import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Extração de Dados
file_path = 'dados.parquet'
df = pd.read_parquet(file_path)

# 2. Limpeza e Transformação de Dados
# Renomear colunas para facilitar o uso
df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('ã', 'a').str.replace('ç', 'c').str.replace('ó', 'o')
# Corrigir FutureWarning do pandas
df["caseína"] = df["caseína"].fillna(df["caseína"].mean())
df['densidade'] = df['densidade'].fillna(df['densidade'].mean())
# Criar novas features, se aplicável (ex: ano, mês)
df['ano'] = df['dataanalise'].dt.year
df['mes'] = df['dataanalise'].dt.month

# 3. Análise Exploratória de Dados (EDA)
# Estatísticas descritivas
print("\nEstatísticas Descritivas:")
print(df.describe())

# Contagem de valores únicos para colunas categóricas
print("\nValores únicos por cidade:")
print(df['cidade'].value_counts())
print("\nValores únicos por estação:")
print(df['estacao'].value_counts())

# Correlação entre variáveis numéricas
print("\nMatriz de Correlação:")
print(df[['gordura', 'proteina', 'lactose', 'st', 'log_ccs', 'ccs', 'caseína', 'densidade', 'sng']].corr())

# 4. Visualização de Dados

# Histograma da Gordura
plt.figure(figsize=(10, 6))
sns.histplot(df['gordura'], bins=30, kde=True)
plt.title('Distribuição de Gordura no Leite')
plt.xlabel('Gordura (%)')
plt.ylabel('Frequência')
plt.savefig('gordura_distribuicao.png')
plt.close()

# Boxplot da Proteína por Estação
plt.figure(figsize=(12, 7))
sns.boxplot(x='estacao', y='proteina', data=df)
plt.title('Distribuição de Proteína por Estação')
plt.xlabel('Estação')
plt.ylabel('Proteína (%)')
plt.savefig('proteina_por_estacao.png')
plt.close()

# Média de CCS (Contagem de Células Somáticas) ao longo do tempo (por ano)
plt.figure(figsize=(14, 7))
df.groupby('ano')['ccs'].mean().plot(kind='line', marker='o')
plt.title('Média de CCS por Ano')
plt.xlabel('Ano')
plt.ylabel('Média de CCS')
plt.grid(True)
plt.savefig('ccs_por_ano.png')
plt.close()

# Dispersão entre Gordura e Proteína
plt.figure(figsize=(10, 6))
sns.scatterplot(x='gordura', y='proteina', data=df, alpha=0.5)
plt.title('Relação entre Gordura e Proteína')
plt.xlabel('Gordura (%)')
plt.ylabel('Proteína (%)')
plt.savefig('gordura_proteina_dispersao.png')
plt.close()

# 5. Machine Learning - Regressão Linear para prever gordura a partir das demais variáveis
# Selecionar features e target
features = [col for col in df.columns if col not in ['gordura', 'dataanalise', 'cidade', 'estacao']]
X = df[features]
y = df['gordura']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline com padronização e Regressão Linear
pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])
pipeline_lr.fit(X_train, y_train)
y_pred_lr = pipeline_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f'\nRegressão Linear - Resultados para predição de Gordura:')
print(f'Mean Squared Error (MSE): {mse_lr:.2f}')
print(f'R-squared (R2): {r2_lr:.2f}')

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_lr, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Gordura Real')
plt.ylabel('Gordura Prevista')
plt.title('Regressão Linear: Gordura Real vs Prevista')
plt.grid(True)
plt.savefig('linear_real_vs_prevista.png')
plt.close()
print('Gráfico linear_real_vs_prevista.png salvo.')

print("Análise e visualizações concluídas. Imagens salvas como: gordura_distribuicao.png, proteina_por_estacao.png, ccs_por_ano.png, gordura_proteina_dispersao.png")