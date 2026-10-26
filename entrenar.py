import json

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from preprocesado import normalizar

ARCHIVO_DATASET = "dataset/consultas.csv"
ARCHIVO_MODELO = "modelos/modelo.pkl"
ARCHIVO_METRICAS = "modelos/metricas.json"
SEMILLA = 42


def crear_pipeline():
    """TF-IDF de n-gramas de caracteres + regresión logística.

    Los n-gramas de caracteres toleran faltas de ortografía y variantes
    de una palabra, algo que las palabras sueltas no hacen con tan pocos datos.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=normalizar,
            analyzer="char_wb",
            ngram_range=(2, 5),
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(C=10, max_iter=3000)),
    ])


def main():
    print("        ENTRENAMIENTO DEL MODELO NLP")
    print("-" * 60)

    df = pd.read_csv(ARCHIVO_DATASET)
    X = df["texto"]
    y = df["categoria"]

    print(f"\nRegistros cargados: {len(df)}")

    # VALIDACIÓN CRUZADA (estimación principal de la precisión)
    # Con ~180 ejemplos un solo 80/20 es muy ruidoso: cada ejemplo de
    # prueba pesa casi 3 puntos. Se repite 5 folds x 3 veces.

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEMILLA)
    metricas = cross_validate(
        crear_pipeline(), X, y, cv=cv,
        scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
    )
    puntajes = metricas["test_accuracy"]

    print("\n" + "-" * 60)
    print("              RESULTADOS")
    print("-" * 60)
    print(f"\nPrecisión (validación cruzada 5x3): "
          f"{puntajes.mean() * 100:.2f}% ± {puntajes.std() * 100:.2f}")

    # HOLDOUT 80/20 (para ver errores concretos)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEMILLA, stratify=y
    )
    prueba = crear_pipeline().fit(X_train, y_train)
    y_pred = prueba.predict(X_test)

    print(f"Precisión en prueba ({len(X_test)} ejemplos): "
          f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))

    etiquetas = sorted(y.unique())
    matriz = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=etiquetas),
        index=[f"real:{e[:6]}" for e in etiquetas],
        columns=[f"pred:{e[:6]}" for e in etiquetas],
    )
    print("Matriz de confusión:")
    print(matriz.to_string())

    errores = [(t, r, p) for t, r, p in zip(X_test, y_test, y_pred) if r != p]
    if errores:
        print("\nErrores en prueba:")
        for texto, real, pred in errores:
            print(f" - {texto!r}: real={real} predicho={pred}")

    # MODELO FINAL: se reentrena con todos los datos antes de guardarlo

    modelo = crear_pipeline().fit(X, y)
    joblib.dump(modelo, ARCHIVO_MODELO)

    # La página web muestra estas métricas
    with open(ARCHIVO_METRICAS, "w", encoding="utf-8") as f:
        json.dump({
            "n_registros": len(df),
            "accuracy": float(puntajes.mean() * 100),
            "precision": float(metricas["test_precision_macro"].mean() * 100),
            "recall": float(metricas["test_recall_macro"].mean() * 100),
            "f1": float(metricas["test_f1_macro"].mean() * 100),
        }, f, indent=2)

    print(f"\nModelo (entrenado con los {len(df)} registros) guardado en:")
    print(ARCHIVO_MODELO)

    print("\n" + "-" * 60)
    print("        ENTRENAMIENTO FINALIZADO")
    print("-" * 60)


if __name__ == "__main__":
    main()
