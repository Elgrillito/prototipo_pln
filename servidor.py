import json
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

import asistente
from entrenar import ARCHIVO_METRICAS

MAX_CARACTERES = 500

app = Flask(__name__)

# El modelo y el dataset se cargan una sola vez al arrancar
modelo, df, preguntas_tfidf = asistente.cargar_recursos()


@app.get("/")
def inicio():
    return send_from_directory("web", "index.html")


@app.get("/api/info")
def info():
    ruta = Path(ARCHIVO_METRICAS)
    metricas = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None
    conteo = df["categoria"].value_counts().to_dict()
    conteo.pop(asistente.CATEGORIA_FUERA_DE_TEMA, None)
    return jsonify(
        metricas=metricas,
        categorias=dict(sorted(conteo.items())),
        umbral_confianza=asistente.UMBRAL_CONFIANZA,
        umbral_similitud=asistente.UMBRAL_SIMILITUD,
    )


@app.post("/api/consultar")
def consultar():
    datos = request.get_json(silent=True) or {}
    texto = datos.get("texto")
    if not isinstance(texto, str) or not texto.strip():
        return jsonify(error="Escribe una consulta."), 400
    if len(texto) > MAX_CARACTERES:
        return jsonify(error=f"La consulta supera {MAX_CARACTERES} caracteres."), 400

    umbral = datos.get("umbral", asistente.UMBRAL_CONFIANZA)
    if isinstance(umbral, bool) or not isinstance(umbral, (int, float)):
        return jsonify(error="El umbral debe ser un número."), 400
    umbral = min(max(float(umbral), 0.0), 100.0)

    return jsonify(asistente.consultar(texto, modelo, df, preguntas_tfidf, umbral))


if __name__ == "__main__":
    # Solo localhost y sin debug: es una demo, no un servicio público
    app.run(host="127.0.0.1", port=5000)
