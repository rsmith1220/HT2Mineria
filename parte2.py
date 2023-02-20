import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("cleaned_file.csv")

#Prepara el dataset para el modelado
data.dropna(subset=["class"], inplace=True)
data["class"] = (data["class"] == 4).astype(int)

X = data.drop("class", axis=1)
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de Regresión Logística
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_cm = confusion_matrix(y_test, lr_pred)
lr_report = classification_report(y_test, lr_pred)

# Modelo K-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
knn_cm = confusion_matrix(y_test, knn_pred)
knn_report = classification_report(y_test, knn_pred)

# Resultado de los modelos: Matrices de confusión, precisión, etc
print("Matriz de confusión - Regresión Logística:")
print(lr_cm)
print("Precisión: {:.2f}".format(lr_accuracy))
print("Métricas adicionales:\n", lr_report)

print("Matriz de confusión - K-NN:")
print(knn_cm)
print("Precisión: {:.2f}".format(knn_accuracy))
print("Métricas adicionales:\n", knn_report)