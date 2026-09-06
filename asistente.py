import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity

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
        return "No tengo una respuesta disponible para esta categoría."

    # Convertir las preguntas del CSV a TF-IDF
    preguntas_tfidf = vectorizador.transform(
        resultados["texto"]
    )

    # Convertir la pregunta del usuario a TF-IDF
    pregunta_tfidf = vectorizador.transform(
        [pregunta]
    )

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
    
    # PREDECIR CATEGORÍA    

    categoria = clasificador.predict(pregunta_tfidf)[0]
    
    # OBTENER CONFIANZA    

    probabilidades = clasificador.predict_proba(pregunta_tfidf)[0]

    confianza = max(probabilidades) * 100
    
    # OBTENER RESPUESTA   

    respuesta, pregunta_relacionada, similitud = obtener_respuesta(
        pregunta,
        categoria
    )
    
    # MOSTRAR RESULTADO    

    print(f"\nCategoría: {categoria}")
    print(f"Confianza de categoría: {confianza:.2f}%")    
    print(f"Similitud con pregunta: {similitud:.2f}%")

    if similitud<20.0 :
        print(f"\nDemasiado baja la coincidencia para responder, pruebe de otra manera.")
    else :
        print(f"\nPregunta relacionada:")
        print(pregunta_relacionada)
        print(f"\nRespuesta:")
        print(respuesta)
    print("\n" + "-" * 60)