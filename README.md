# Asistente de PLN

Proyecto de Procesamiento de Lenguaje Natural (PLN) desarrollado en Python.

El proyecto implementa un asistente capaz de analizar consultas escritas por el usuario, identificar la categoría a la que pertenecen y proporcionar una respuesta relacionada.

## Estructura del proyecto

```text
pln/
├── dataset/
├── modelos/
├── asistente.py
├── entrenar.py
├── revisar_dataset.py
├── .gitignore
└── README.md

## Archivos principales

  asistente.py: Ejecuta el asistente y procesa las consultas del usuario.
  entrenar.py: entrena el modelo utilizando el dataset.
  revisar_dataset.py: permite revisar y analizar los datos utilizados para el entrenamiento.
  dataset/: contiene los datos utilizados para entrenar y evaluar el sistema.
  modelos/: contiene los modelos y archivos generados durante el entrenamiento.

## Tecnologías
  Python
  Scikit-learn
  Procesamiento de Lenguaje Natural
  TF-IDF
  Clasificación de texto

## Instalacion
Clonar el repositorio:

  git clone git@github.com:elgrillito/protoripo_pln.git
  cd prototipo_pln

Crear el entorno virtual:

  python -m venv .venv

Activarlo:

  source .venv/bin/activate

Instalar las dependencias:

  pip install -r requirements.txt

## Uso

Para entrenar el modelo:

  python entrenar.py

Para ejecutar el asistente:

  python asistente.py

Para revisar el dataset:

  python revisar_dataset.py

