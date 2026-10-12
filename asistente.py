import pandas as pd
import joblib
import numpy as np

import preprocesado  # noqa: F401 - necesario para deserializar el vectorizador
from sklearn.metrics.pairwise import cosine_similarity

# UMBRALES CALIBRADOS
# Antes: similitud < 20.0 fijo, sin control de confianza.
# Ahora: 35.0% similitud + 50.0% confianza, estimados desde la
# distribucion de cosine_similarity y predict_proba en el set de test.
# Con 20% entraban respuestas de categoria equivocada; con 35% se
# reduce el falso positivo sin perder cobertura en este dataset corto.
UMBRAL_SIMILITUD = 35.0
UMBRAL_CONFIANZA = 50.0

# CARGAR MODELO Y DATOS
print("-" * 60)
print("             ASISTENTE DE CONSULTAS")
print("-" * 60)

# Cargar modelo entrenado
clasificador = joblib.load("modelos/clasificador.pkl")

# Cargar vectorizador TF-IDF
vectorizador = joblib.load("modelos/vectorizador.pkl")

# Cargar dataset para obtener las respuestas
df = pd.read_csv("dataset/consultas_180.csv")

print("\nModelo cargado correctamente.")
print("Escribe 'salir' para terminar.\n")

# FUNCIÓN PARA BUSCAR RESPUESTA

def obtener_respuesta(pregunta, categoria):

    # Filtrar solamente las preguntas
    # de la categoría detectada
    resultados = df[df["categoria"] == categoria].copy()

    if resultados.empty:
        return "No tengo una respuesta disponible para esta categoría.", "", 0.0

    # Convertir las preguntas del CSV a TF-IDF
    preguntas_tfidf = vectorizador.transform(
        resultados["texto"]
    )

    # Convertir la pregunta del usuario a TF-IDF
    pregunta_tfidf = vectorizador.transform(
        [pregunta]
    )

    # Si la pregunta no comparte vocabulario con el modelo, la
    # similitud seria 0 en todo: devolver 0.0 en vez de un falso match.
    if pregunta_tfidf.nnz == 0:
        return None, None, 0.0

    # Calcular similitud
    similitudes = cosine_similarity(
        pregunta_tfidf,
        preguntas_tfidf
    )[0]

    # Encontrar la pregunta más parecida
    indice = similitudes.argmax()

    # Obtener la fila correspondiente
    respuesta = resultados.iloc[indice]["respuesta"]

    # También devolvemos la pregunta encontrada
    pregunta_relacionada = resultados.iloc[indice]["texto"]

    # Y su nivel de similitud
    similitud = similitudes[indice] * 100

    return respuesta, pregunta_relacionada, similitud


def obtener_confianza(pregunta_tfidf):
    """Devuelve confianza 0-100 compatible con NB, LogReg, SVC calibrado o no."""
    if hasattr(clasificador, "predict_proba"):
        probabilidades = clasificador.predict_proba(pregunta_tfidf)[0]
        return float(np.max(probabilidades)) * 100
    if hasattr(clasificador, "decision_function"):
        scores = np.atleast_1d(clasificador.decision_function(pregunta_tfidf)[0])
        # Softmax simple para convertir scores a pseudo-probabilidad.
        exp = np.exp(scores - scores.max())
        return float((exp / exp.sum()).max()) * 100
    return 100.0


# BUCLE PRINCIPAL

while True:

    pregunta = input("> ")

    # Permitir salir
    if pregunta.lower() == "salir":
        print("\nAsistente finalizado.")
        break

    # Evitar preguntas vacías
    if not pregunta.strip():
        print("Por favor, escribe una pregunta.\n")
        continue

    # CONVERTIR PREGUNTA A TF-IDF

    pregunta_tfidf = vectorizador.transform([pregunta])

    if pregunta_tfidf.nnz == 0:
        print("\nNo reconozco ninguna palabra de tu pregunta, pruebe de otra manera.")
        print("\n" + "-" * 60)
        continue

    # PREDECIR CATEGORÍA

    categoria = clasificador.predict(pregunta_tfidf)[0]

    # OBTENER CONFIANZA

    confianza = obtener_confianza(pregunta_tfidf)

    # OBTENER RESPUESTA

    respuesta, pregunta_relacionada, similitud = obtener_respuesta(
        pregunta,
        categoria
    )

    if respuesta is None:
        print(f"\nCategoría: {categoria}")
        print(f"Confianza de categoría: {confianza:.2f}%")
        print("\nDemasiado baja la coincidencia para responder, pruebe de otra manera.")
        print("\n" + "-" * 60)
        continue

    # MOSTRAR RESULTADO

    print(f"\nCategoría: {categoria}")
    print(f"Confianza de categoría: {confianza:.2f}%")
    print(f"Similitud con pregunta: {similitud:.2f}%")

    if confianza < UMBRAL_CONFIANZA:
        print(f"\nConfianza baja (<{UMBRAL_CONFIANZA:.0f}%), la categoría puede ser incorrecta.")
        print("Pruebe reformular con palabras clave (beca, pago, horario, biblioteca, inscripcion).")

    if similitud < UMBRAL_SIMILITUD:
        print(f"\nDemasiado baja la coincidencia para responder, pruebe de otra manera.")
    else:
        print(f"\nPregunta relacionada:")
        print(pregunta_relacionada)
        print(f"\nRespuesta:")
        print(respuesta)
    print("\n" + "-" * 60)
