# Asistente de PLN

Proyecto de Procesamiento de Lenguaje Natural (PLN) desarrollado en Python.

El proyecto implementa un asistente capaz de analizar consultas escritas por el usuario, identificar la categoría a la que pertenecen y proporcionar una respuesta relacionada.

## Estructura del proyecto

```text
proyecto_pln/
├── dataset/
├── modelos/
├── tests/
├── web/
├── asistente.py
├── entrenar.py
├── preprocesado.py
├── revisar_dataset.py
├── servidor.py
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```

## Archivos principales

- `asistente.py`: ejecuta el asistente y procesa las consultas del usuario.
- `entrenar.py`: entrena el modelo utilizando el dataset y muestra su precisión.
- `preprocesado.py`: normalización de texto (minúsculas, sin tildes ni signos) compartida por el entrenamiento y el asistente.
- `revisar_dataset.py`: permite revisar y analizar los datos utilizados para el entrenamiento.
- `dataset/consultas.csv`: datos de entrenamiento (columnas `id`, `texto`, `categoria`, `respuesta`). Además de las 6 categorías temáticas incluye `OTRO`, para saludos y consultas fuera del dominio, que el asistente rechaza con un mensaje fijo.
- `modelos/`: contiene el modelo generado durante el entrenamiento (`modelo.pkl`).
- `servidor.py`: servidor Flask que expone el asistente a la página web (`/api/consultar`, `/api/info`).
- `web/index.html`: interfaz web del asistente. Muestra el pipeline de PLN, el umbral de confianza y las métricas reales del modelo.
- `tests/`: pruebas automáticas (`pytest`).

## Tecnologías

- Python
- Scikit-learn
- Procesamiento de Lenguaje Natural
- Flask (servidor web)
- TF-IDF de n-gramas de caracteres y regresión logística
- Clasificación de texto

## Instalación

Clonar el repositorio:

```bash
git clone git@github.com:elgrillito/protoripo_pln.git
cd protoripo_pln
```

Crear el entorno virtual:

```bash
python -m venv .venv
```

Activarlo:

```bash
source .venv/bin/activate
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Uso

Para entrenar el modelo:

```bash
python entrenar.py
```

Para ejecutar el asistente:

```bash
python asistente.py
```

Para abrir la interfaz web (primero hay que haber entrenado el modelo):

```bash
python servidor.py
```

Y abrir <http://127.0.0.1:5000>. La página carga Tailwind, FontAwesome y la fuente Inter desde Internet.

Para revisar el dataset:

```bash
python revisar_dataset.py
```

Para ejecutar las pruebas:

```bash
pip install -r requirements-dev.txt
pytest
```

> El archivo `modelos/modelo.pkl` depende de la versión de scikit-learn con que se generó. Si al ejecutar el asistente aparece una advertencia de versión, vuelve a ejecutar `python entrenar.py`.
