# ========================
# 1. Importar librerías
# ========================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler

# ========================
# 2. Cargar dataset CSV
# ========================
print("Cargando datos")
data = pd.read_csv("diabetes.csv")

print("Primeras filas del dataset:")
print(data.head())

# ========================
# 3. Preprocesamiento
# ========================

# --- One-Hot Encoding para variables categóricas ---
data_encoded = pd.get_dummies(data, drop_first=True)

# Variables predictoras (X) y variable objetivo (y)
X = data_encoded.drop("diabetes", axis=1)
y = data_encoded["diabetes"]

# --- Escalado de datos numéricos ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Balanceo de clases con RandomUnderSampler ---
rus = RandomUnderSampler(random_state=42)
X_bal, y_bal = rus.fit_resample(X_scaled, y)

# División en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal
)

# ========================
# 4. GridSearch para KNN
# ========================
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='accuracy')
grid_knn.fit(X_train, y_train)

print("\n=== KNN ===")
print("Mejores hiperparámetros:", grid_knn.best_params_)
print("Mejor score en CV:", grid_knn.best_score_)
print("Precisión en test:", grid_knn.score(X_test, y_test))
print("\nReporte KNN:\n", classification_report(y_test, grid_knn.predict(X_test)))
print("Matriz de confusión KNN:\n", confusion_matrix(y_test, grid_knn.predict(X_test)))

# ========================
# 5. GridSearch para Árbol de Decisión
# ========================
param_grid_tree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_tree = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_tree, cv=5, scoring='accuracy')
grid_tree.fit(X_train, y_train)

print("\n=== Árbol de Decisión ===")
print("Mejores hiperparámetros:", grid_tree.best_params_)
print("Mejor score en CV:", grid_tree.best_score_)
print("Precisión en test:", grid_tree.score(X_test, y_test))
print("\nReporte Árbol de Decisión:\n", classification_report(y_test, grid_tree.predict(X_test)))
print("Matriz de confusión Árbol:\n", confusion_matrix(y_test, grid_tree.predict(X_test)))

# ========================
# 6. Importancia de variables en el Árbol
# ========================
best_tree = grid_tree.best_estimator_
importances = best_tree.feature_importances_
features = X.columns

plt.bar(features, importances)
plt.title("Importancia de Variables (Árbol de Decisión)")
plt.ylabel("Peso relativo")
plt.xticks(rotation=90)
plt.show()

# ========================
# 7. Visualización del Árbol
# ========================
# plt.figure(figsize=(14,8))
# plot_tree(best_tree, feature_names=features, class_names=["No Diabetes","Diabetes"], filled=True)
# plt.show()
