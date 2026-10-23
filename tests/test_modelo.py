from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score

import asistente
from entrenar import ARCHIVO_DATASET, ARCHIVO_MODELO, crear_pipeline
from preprocesado import normalizar

# Frases que no están en el dataset: sirven para detectar regresiones.
FRASES_CONTROL = [
    ("como pido una beca", "BECAS"),
    ("k horario tiene biblioteca", "BIBLIOTECA"),
    ("necesito ayuda con mi inscripcion", "INSCRIPCIONES"),
    ("hasta cuando puedo pagar", "PAGOS"),
    ("puedo llevarme libros a casa", "BIBLIOTECA"),
    ("que talleres hay", "EXTRAESCOLARES"),
    ("a que hora entramos", "HORARIOS"),
    ("me pueden prestar un libro", "BIBLIOTECA"),
    ("buenas tardes", "OTRO"),
    ("cuanto cuesta una pizza", "OTRO"),
    ("quien gano el partido de ayer", "OTRO"),
]


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(ARCHIVO_DATASET)


@pytest.fixture(scope="module")
def modelo(df):
    return crear_pipeline().fit(df["texto"], df["categoria"])


def test_normalizar():
    assert normalizar("¿Qué Máquina, Año?") == "que maquina ano"
    assert normalizar("  hola   mundo ") == "hola mundo"


def test_dataset_valido(df):
    assert {"id", "texto", "categoria", "respuesta"} <= set(df.columns)
    assert not df[["texto", "categoria", "respuesta"]].isnull().any().any()
    normalizados = df["texto"].map(normalizar)
    assert not normalizados.duplicated().any()
    assert (df.groupby(normalizados)["categoria"].nunique() == 1).all()


def test_precision_validacion_cruzada(df):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    puntajes = cross_val_score(crear_pipeline(), df["texto"], df["categoria"], cv=cv)
    assert puntajes.mean() >= 0.90


@pytest.mark.parametrize("texto,esperada", FRASES_CONTROL)
def test_frases_control(modelo, texto, esperada):
    assert modelo.predict([texto])[0] == esperada


def test_modelo_guardado_carga_y_predice():
    if not Path(ARCHIVO_MODELO).exists():
        pytest.skip("Ejecuta python entrenar.py para generar el modelo")
    guardado = joblib.load(ARCHIVO_MODELO)
    assert guardado.predict(["como pido una beca"])[0] == "BECAS"


def test_fuera_de_tema_se_clasifica_como_otro(modelo):
    for texto in ["recomiendame una pelicula", "cual es la capital de italia"]:
        assert modelo.predict([texto])[0] == asistente.CATEGORIA_FUERA_DE_TEMA


def test_asistente_responde_con_respuesta_del_dataset(modelo, df):
    preguntas = modelo["tfidf"].transform(df["texto"])
    respuesta, _, similitud = asistente.obtener_respuesta(
        "como pido una beca", "BECAS", modelo, df, preguntas
    )
    assert respuesta in set(df.loc[df["categoria"] == "BECAS", "respuesta"])
    assert similitud >= asistente.UMBRAL_SIMILITUD
