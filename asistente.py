import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity

import preprocesado  # noqa: F401 - necesario para cargar el modelo guardado

ARCHIVO_MODELO = "modelos/modelo.pkl"
ARCHIVO_DATASET = "dataset/consultas.csv"

# Umbrales (en %) estimados con consultas de prueba dentro y fuera del
# dominio. Por debajo de ellos el asistente prefiere no responder.
UMBRAL_CONFIANZA = 60.0
UMBRAL_SIMILITUD = 35.0

# Categoría para saludos y consultas fuera del dominio del asistente
CATEGORIA_FUERA_DE_TEMA = "OTRO"


def cargar_recursos():
    try:
        modelo = joblib.load(ARCHIVO_MODELO)
        df = pd.read_csv(ARCHIVO_DATASET)
    except FileNotFoundError as error:
        raise SystemExit(
            f"No se encontró {error.filename}. "
            "Ejecuta primero: python entrenar.py"
        )

    # Se vectorizan las preguntas del dataset una sola vez
    preguntas_tfidf = modelo["tfidf"].transform(df["texto"])
    return modelo, df, preguntas_tfidf


# FUNCIÓN PARA BUSCAR RESPUESTA

def obtener_respuesta(pregunta, categoria, modelo, df, preguntas_tfidf):

    # Filtrar solamente las preguntas
    # de la categoría detectada
    mascara = (df["categoria"] == categoria).to_numpy()
    resultados = df[mascara]

    if resultados.empty:
        return None, None, 0.0

    # Calcular similitud
    pregunta_tfidf = modelo["tfidf"].transform([pregunta])
    similitudes = cosine_similarity(
        pregunta_tfidf,
        preguntas_tfidf[mascara]
    )[0]

    # Encontrar la pregunta más parecida
    indice = similitudes.argmax()

    respuesta = resultados.iloc[indice]["respuesta"]
    pregunta_relacionada = resultados.iloc[indice]["texto"]
    similitud = similitudes[indice] * 100

    return respuesta, pregunta_relacionada, similitud


def main():
    print("-" * 60)
    print("             ASISTENTE DE CONSULTAS")
    print("-" * 60)

    modelo, df, preguntas_tfidf = cargar_recursos()

    print("\nModelo cargado correctamente.")
    print("Escribe 'salir' para terminar.\n")

    while True:

        pregunta = input("> ")

        # Permitir salir
        if pregunta.strip().lower() == "salir":
            print("\nAsistente finalizado.")
            break

        # Evitar preguntas vacías
        if not pregunta.strip():
            print("Por favor, escribe una pregunta.\n")
            continue

        # PREDECIR CATEGORÍA Y CONFIANZA

        probabilidades = modelo.predict_proba([pregunta])[0]
        categoria = modelo.classes_[probabilidades.argmax()]
        confianza = probabilidades.max() * 100

        # CONSULTAS FUERA DE TEMA

        if categoria == CATEGORIA_FUERA_DE_TEMA:
            print(f"\nCategoría: {categoria}")
            print(f"Confianza de categoría: {confianza:.2f}%")
            print("\n" + df[df["categoria"] == categoria]["respuesta"].iloc[0])
            print("\n" + "-" * 60)
            continue

        # OBTENER RESPUESTA

        respuesta, pregunta_relacionada, similitud = obtener_respuesta(
            pregunta,
            categoria,
            modelo,
            df,
            preguntas_tfidf
        )

        # MOSTRAR RESULTADO

        print(f"\nCategoría: {categoria}")
        print(f"Confianza de categoría: {confianza:.2f}%")
        print(f"Similitud con pregunta: {similitud:.2f}%")

        if (
            respuesta is None
            or confianza < UMBRAL_CONFIANZA
            or similitud < UMBRAL_SIMILITUD
        ):
            print("\nDemasiado baja la coincidencia para responder, pruebe de otra manera.")
        else:
            print("\nPregunta relacionada:")
            print(pregunta_relacionada)
            print("\nRespuesta:")
            print(respuesta)
        print("\n" + "-" * 60)


if __name__ == "__main__":
    main()
