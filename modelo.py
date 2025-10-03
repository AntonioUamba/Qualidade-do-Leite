import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("milk_sessions.csv")

# Converter 'Date' para datetime e extrair features temporais
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek

# Colunas a serem removidas devido a muitos valores nulos ou baixa variância
drop_cols = [
    "DailyRestPerBout", "DailyRestRatio", "CurrentLDA", "CurrentMAST",
    "CurrentEdma", "MilkTemperature", "Date"
]
df = df.drop(columns=drop_cols)

# Converter colunas booleanas para inteiros (0 ou 1)
for col in ["Disease", "DIM_50-175", "DIM_<50", "DIM_>=175"]:
    df[col] = df[col].astype(int)

# Definir as variáveis alvo
target_fat = "DailyFat_P"
target_protein = "DailyProtein_P"

# Features (todas as colunas exceto as alvo e as removidas)
features = [col for col in df.columns if col not in [target_fat, target_protein, "Fat", "Protein"]]

X = df[features]
y_fat = df[target_fat]
y_protein = df[target_protein]

# Escalonar as features numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

# Salvar o scaler
joblib.dump(scaler, 'scaler.pkl')

# Dividir os dados em conjuntos de treino e teste para DailyFat_P
X_train_fat, X_test_fat, y_train_fat, y_test_fat = train_test_split(
    X_scaled, y_fat, test_size=0.2, random_state=42
)

# Treinar o modelo para DailyFat_P
model_fat = LinearRegression()
model_fat.fit(X_train_fat, y_train_fat)
y_pred_fat = model_fat.predict(X_test_fat)

# Salvar o modelo de gordura
joblib.dump(model_fat, 'model_fat.pkl')

# Avaliar o modelo para DailyFat_P
mse_fat = mean_squared_error(y_test_fat, y_pred_fat)
r2_fat = r2_score(y_test_fat, y_pred_fat)

print(f"\n--- Modelo para DailyFat_P ---")
print(f"Mean Squared Error: {mse_fat:.4f}")
print(f"R-squared: {r2_fat:.4f}")

###### Dividir os dados em conjuntos de treino e teste para DailyProtein_P 
X_train_protein, X_test_protein, y_train_protein, y_test_protein = train_test_split(
    X_scaled, y_protein, test_size=0.2, random_state=42
)

# Treinar o modelo para DailyProtein_P
model_protein = LinearRegression()
model_protein.fit(X_train_protein, y_train_protein)
y_pred_protein = model_protein.predict(X_test_protein)

###### Salvar o modelo de proteína ######
joblib.dump(model_protein, 'model_protein.pkl')

###### Avaliar o modelo para DailyProtein_P ######
mse_protein = mean_squared_error(y_test_protein, y_pred_protein)
r2_protein = r2_score(y_test_protein, y_pred_protein)

print(f"\n--- Modelo para DailyProtein_P ---")
print(f"Mean Squared Error: {mse_protein:.4f}")
print(f"R-squared: {r2_protein:.4f}")

print("\nPreparação de dados e modelagem concluídas.")