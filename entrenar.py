"""Entrenamiento del modelo NLP con mejoras quick-wins (sin nuevas dependencias)."""
import joblib
import pandas as pd

from preprocesado import STOPWORDS_ES, normalizar_texto

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC


# VECTORIZADOR: usa normalizar_texto de preprocesado.py (modulo compartido
# para que asistente.py pueda deserializar el pickle sin errores).


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


# CONVERTIR TEXTO A NÚMEROS CON TF-IDF MEJORADO


vectorizador = TfidfVectorizer(
    preprocessor=normalizar_texto,
    stop_words=STOPWORDS_ES,
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.95,
    sublinear_tf=True,
)

X_train_tfidf = vectorizador.fit_transform(X_train)
X_test_tfidf = vectorizador.transform(X_test)

print("\nTF-IDF generado correctamente.")
print(f"Vocabulario: {len(vectorizador.vocabulary_)} terminos")


# COMPARAR CLASIFICADORES CON VALIDACION CRUZADA + GRIDSEARCH


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidatos = {
    "LogisticRegression": (
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        {"C": [0.5, 1.0, 2.0, 5.0]},
    ),
    "LinearSVC_calibrado": (
        CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3),
        {"estimator__C": [0.5, 1.0, 2.0]},
    ),
    "ComplementNB": (
        ComplementNB(),
        {"alpha": [0.1, 0.5, 1.0]},
    ),
    "MultinomialNB_baseline": (
        MultinomialNB(),
        {"alpha": [0.1, 0.5, 1.0]},
    ),
}

mejor_nombre = None
mejor_score = -1.0
mejor_clasificador = None
mejor_params = None

print("\nBusqueda de hiperparametros (CV-5, metrica F1-macro):")
for nombre, (modelo, grilla) in candidatos.items():
    busqueda = GridSearchCV(modelo, grilla, cv=cv, scoring="f1_macro", n_jobs=-1)
    busqueda.fit(X_train_tfidf, y_train)
    print(f" - {nombre}: F1-macro CV={busqueda.best_score_:.4f} params={busqueda.best_params_}")
    if busqueda.best_score_ > mejor_score:
        mejor_score = busqueda.best_score_
        mejor_nombre = nombre
        mejor_clasificador = busqueda.best_estimator_
        mejor_params = busqueda.best_params_

clasificador = mejor_clasificador

print(f"\nMejor modelo: {mejor_nombre} {mejor_params}")
print("Modelo entrenado correctamente.")


# REALIZAR PREDICCIONES


y_pred = clasificador.predict(X_test_tfidf)


# EVALUAR MODELO


precision = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")

print("\n" + "-" * 60)
print("              RESULTADOS")
print("-" * 60)

print(f"\nModelo seleccionado: {mejor_nombre}")
print(f"Precisión general (accuracy): {precision * 100:.2f}%")
print(f"F1-macro: {f1_macro * 100:.2f}%")
print(f"F1-macro CV-5 (train): {mejor_score * 100:.2f}%")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))


# GUARDAR MODELO Y VECTORIZADOR


joblib.dump(clasificador, "modelos/clasificador.pkl")
joblib.dump(vectorizador, "modelos/vectorizador.pkl")

print("\nModelo guardado en:")
print("modelos/clasificador.pkl")

print("\nVectorizador guardado en:")
print("modelos/vectorizador.pkl")

print("\n" + "-" * 60)
print("        ENTRENAMIENTO FINALIZADO")
print("-" * 60)
