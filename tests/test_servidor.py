import pytest

import servidor


@pytest.fixture
def cliente():
    return servidor.app.test_client()


def consultar(cliente, texto, **extra):
    return cliente.post("/api/consultar", json={"texto": texto, **extra})


def test_pagina_principal(cliente):
    respuesta = cliente.get("/")
    assert respuesta.status_code == 200
    assert b"ITLAC" in respuesta.data


def test_info(cliente):
    info = cliente.get("/api/info").get_json()
    assert set(info["categorias"]) == {
        "BECAS", "BIBLIOTECA", "EXTRAESCOLARES", "HORARIOS", "INSCRIPCIONES", "PAGOS",
    }
    assert info["umbral_confianza"] == 60.0
    assert info["metricas"] is None or info["metricas"]["accuracy"] > 80


def test_consulta_respondida(cliente):
    r = consultar(cliente, "¿Qué requisitos necesito para pedir una beca?").get_json()
    assert r["decision"] == "respondida"
    assert r["categoria"] == "BECAS"
    assert r["respuesta"]
    assert r["texto_normalizado"] == "que requisitos necesito para pedir una beca"


def test_consulta_fuera_de_tema(cliente):
    r = consultar(cliente, "¿Cuál es la capital de Francia?").get_json()
    assert r["decision"] == "fuera_de_tema"
    assert r["categoria"] == "OTRO"


def test_consulta_derivada_por_baja_confianza(cliente):
    r = consultar(cliente, "bekas disponibles").get_json()
    assert r["decision"] == "derivada"
    assert r["respuesta"] is None


def test_umbral_alto_deriva_una_consulta_respondible(cliente):
    texto = "¿Dónde puedo pagar la colegiatura?"
    assert consultar(cliente, texto).get_json()["decision"] == "respondida"
    assert consultar(cliente, texto, umbral=100).get_json()["decision"] == "derivada"


@pytest.mark.parametrize("cuerpo", [
    {},
    {"texto": ""},
    {"texto": "   "},
    {"texto": 5},
    {"texto": "x" * 501},
    {"texto": "hola", "umbral": "alto"},
    {"texto": "hola", "umbral": True},
])
def test_entradas_invalidas(cliente, cuerpo):
    assert cliente.post("/api/consultar", json=cuerpo).status_code == 400
