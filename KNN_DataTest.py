# -----------------------------
# KNN con dataset BankChurners
# -----------------------------
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar el archivo CSV
csv_path = "BankChurners.csv"  # asegúrate que está en la misma carpeta que este script
data = pd.read_csv(csv_path)

print("Datos cargados correctamente ✅")
print(data.head())

# 2. Preparación de los datos
df = data.copy()

# Eliminar columna identificador
if "CLIENTNUM" in df.columns:
    df = df.drop(columns=["CLIENTNUM"])

# Codificar variables categóricas a numéricas
encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = encoder.fit_transform(df[col])

# Separar variables predictoras y variable objetivo
X = df.drop("Attrition_Flag", axis=1)
y = df["Attrition_Flag"]

# 3. División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Entrenamiento del modelo KNN
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# 5. Predicciones
y_pred = knn.predict(X_test)

# 6. Evaluación
print("\n--- Resultados ---")
print(f"Precisión del modelo: {accuracy_score(y_test, y_pred):.4f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# 7. Matriz de confusión (visual)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Se fue", "Se quedó"],
    yticklabels=["Se fue", "Se quedó"]
)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de confusión KNN (BankChurners)")
plt.show()
