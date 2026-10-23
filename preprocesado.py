import re
import unicodedata


def normalizar(texto):
    """Minúsculas, sin tildes ni signos, con espacios simples.

    Se usa tanto al entrenar como al consultar, por eso vive en su propio
    módulo: el modelo guardado necesita poder importarlo al cargarse.
    """
    texto = unicodedata.normalize("NFD", str(texto).lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^a-z0-9 ]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()
