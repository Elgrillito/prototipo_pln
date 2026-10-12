"""Preprocesado compartido ES (importado por entrenar.py y asistente.py).

Se extrae a modulo propio para que joblib pueda serializar/deserializar
el TfidfVectorizer(preprocessor=normalizar_texto): si la funcion viviera
en entrenar.py (__main__), asistente.py fallaria con
AttributeError: module '__main__' has no attribute 'normalizar_texto'.
"""
import re
import unicodedata


def normalizar_texto(texto: str) -> str:
    """Minusculas + sin tildes + sin signos (¿?¡!.,) para reducir dispersion."""
    if not isinstance(texto, str):
        texto = str(texto)
    texto = texto.lower()
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    texto = re.sub(r"[^a-z0-9 ]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


# Stopwords manual minimo: palabras que aparecen en todas las categorias
# y no discriminan (evita instalar nltk/spacy).
STOPWORDS_ES = [
    "que", "como", "cuando", "donde", "donde", "cual", "cuales",
    "para", "por", "para", "con", "sin", "sobre", "entre",
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "al", "en", "y", "o", "u", "mi", "mis",
    "me", "se", "hay", "son", "es", "puedo", "puede",
]
