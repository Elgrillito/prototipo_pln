import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# CARGAR DATASET



print("        ENTRENAMIENTO DEL MODELO NLP")
print("-" * 60)

archivo = "dataset/consultas_180.csv"

df = pd.read_csv(archivo)

print(f"\nRegistros cargados: {len(df)}")



# SEPARAR ENTRADAS Y CATEGORÍAS


X = df["texto"]
y = df["categoria"]



# DIVIDIR DATASET


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Datos para entrenamiento: {len(X_train)}")
print(f"Datos para prueba: {len(X_test)}")



# CONVERTIR TEXTO A NÚMEROS CON TF-IDF


vectorizador = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizador.fit_transform(X_train)
X_test_tfidf = vectorizador.transform(X_test)

print("\nTF-IDF generado correctamente.")



# CREAR Y ENTRENAR CLASIFICADOR


clasificador = MultinomialNB()

clasificador.fit(X_train_tfidf, y_train)

print("Modelo entrenado correctamente.")



# REALIZAR PREDICCIONES


y_pred = clasificador.predict(X_test_tfidf)



# EVALUAR MODELO


precision = accuracy_score(y_test, y_pred)

print("\n" + "-" * 60)
print("              RESULTADOS")
print("-" * 60)

print(f"\nPrecisión general: {precision * 100:.2f}%")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))



# GUARDAR MODELO Y VECTORIZADOR


joblib.dump(clasificador, "modelos/clasificador.pkl")
joblib.dump(vectorizador, "modelos/vectorizador.pkl")

print("\nModelo guardado en:")
print("modelo/clasificador.pkl")

print("\nVectorizador guardado en:")
print("modelo/vectorizador.pkl")

print("\n" + "-" * 60)
print("        ENTRENAMIENTO FINALIZADO")
print("-" * 60)