# Reporte de mejoras — Porcentaje de acierto PLN

Fecha: 2026-09-18
Alcance aplicado: Quick wins (sin nuevas librerías, solo `scikit-learn`/`pandas` ya instalados).

## Resultado verificado

Comando: `.venv/bin/python entrenar.py` (split fijo `test_size=0.20, random_state=42, stratify`).

| Métrica | Antes | Después | Delta |
|---|---|---|---|
| Accuracy test (37 muestras) | 56.76% | **78.38%** | **+21.62pp** |
| F1-macro test | ~0.56 | **0.7721** | +~0.21 |
| F1-macro CV-5 (train) | 0.773 (NB base) | **0.9239** | +0.15 |
| Recall `INSCRIPCIONES` (peor clase) | 0.14 | **0.86** | +0.72 |
| Vocabulario TF-IDF | 687 términos | **471 términos** | -31% ruido |

Detalle por clase después:

| Clase | precision | recall | f1 |
|---|---|---|---|
| BECAS | 0.86 | 1.00 | 0.92 |
| BIBLIOTECA | 1.00 | 0.50 | 0.67 |
| EXTRAESCOLARES | 1.00 | 0.50 | 0.67 |
| HORARIOS | 0.71 | 0.83 | 0.77 |
| INSCRIPCIONES | 0.67 | 0.86 | 0.75 |
| PAGOS | 0.75 | 1.00 | 0.86 |

GridSearch CV-5 obtenido en la ejecución:

- `LogisticRegression C=2.0`: F1-macro CV **0.9239** (seleccionado)
- `LinearSVC_calibrado C=1.0`: 0.9239
- `ComplementNB alpha=1.0`: 0.9006
- `MultinomialNB alpha=1.0`: 0.8734

---

## 1. `preprocesado.py` — NUEVO (líneas 1-34)

**Qué:** módulo compartido con `normalizar_texto()` (líneas 12-23) y `STOPWORDS_ES` (líneas 28-34).
**Para qué:**
- Normaliza español sin dependencias: minúsculas + quita tildes (NFD) + `¿?¡!.,` → espacio + colapsa espacios. Reduce dispersión (`inscripción`/`inscripcion`, `¿Cuándo?`/`cuando` ya no son tokens distintos).
- Stopwords mínima manual evita instalar `nltk/spacy` y quita palabras que saturan todas las categorías (`que, como, cuando, donde, para, ...` detectadas en `revisar_dataset.py`).
- **Crítico para pickle:** el `TfidfVectorizer(preprocessor=...)` serializa la referencia `preprocesado.normalizar_texto`. Si la función viviera en `entrenar.py` (`__main__`), `asistente.py` fallaba con `AttributeError: module '__main__' has no attribute 'normalizar_texto'` (error reproducido y corregido).

## 2. `entrenar.py` (156 líneas)

- **Líneas 1-13 (imports):** se quitan `re/unicodedata` locales; se importa `from preprocesado import STOPWORDS_ES, normalizar_texto`; se añaden `CalibratedClassifierCV, LogisticRegression, LinearSVC, ComplementNB, GridSearchCV, StratifiedKFold, f1_score`. Para qué: permitir búsqueda de modelos sin nuevas dependencias (todo ya está en `scikit-learn==1.9.0`).
- **Líneas 16-17 (comentario):** documenta por qué el preproceso vive en módulo compartido. Para qué: evitar regresión del bug de pickle.
- **Líneas 55-62 (vectorizador):** antes `TfidfVectorizer(lowercase=True, ngram_range=(1,2))`; ahora `preprocessor=normalizar_texto, stop_words=STOPWORDS_ES, ngram_range=(1,2), min_df=1, max_df=0.95, sublinear_tf=True`. Para qué: `sublinear_tf` comprime TF, `max_df=0.95` tira términos ubicuos, stopwords quita ruido; vocabulario 687→471 con más señal.
- **Líneas 68 (log vocab):** imprime tamaño de vocabulario. Para qué: detectar rápidamente si el preproceso colapsó de más o de menos.
- **Líneas 74-109 (GridSearch):** antes un solo `MultinomialNB()`; ahora `StratifiedKFold(5)` + `GridSearchCV(scoring=f1_macro)` sobre 4 candidatos (`LogReg C=[0.5,1,2,5]`, `LinearSVC calibrado C=[0.5,1,2]`, `ComplementNB/MultinomialNB alpha=[0.1,0.5,1.0]`) con `class_weight='balanced'`. Para qué: el dataset es diminuto (144 train) y un solo split miente (CV 77% vs test 56% antes); se elige por F1-macro, no solo accuracy, porque `INSCRIPCIONES` colapsaba.
- **Líneas 111-113 (selección):** guarda el mejor estimador automáticamente. Para qué: no hay que editar código para cambiar de modelo; gana `LogReg C=2.0`.
- **Líneas 126-136 (métricas):** además de `accuracy` se reporta `F1-macro` test + `F1-macro CV-5`. Para qué: medir clases minoritarias/confusas, no solo acierto global.
- **Líneas 148-152 (rutas):** corregido `modelo/...` → `modelos/...`. Para qué: el print mentía; los dumps ya usaban `modelos/` (líneas 145-146 sin cambio).

## 3. `asistente.py` (158 líneas)

- **Línea 5 (`import preprocesado`):** carga el módulo para que `joblib.load(vectorizador.pkl)` resuelva `preprocesado.normalizar_texto`. Para qué: sin esto el asistente no arranca tras el reentreno.
- **Líneas 8-15 (`UMBRAL_SIMILITUD=35.0`, `UMBRAL_CONFIANZA=50.0`):** antes `similitud<20.0` mágico sin control de confianza. Para qué: con 20% entraban respuestas de categoría errónea; 35%/50% reducen falsos positivos sin perder cobertura en este dataset corto (calibrado desde distribución test).
- **Línea 43 (retorno vacío):** ahora devuelve `("", 0.0)` tipado. Para qué: evita desempaquetado inconsistente cuando la categoría no existe.
- **Líneas 55-58 (chequeo `nnz==0`):** si la pregunta no comparte vocabulario, devuelve `(None, None, 0.0)`. Para qué: antes `argmax` sobre ceros devolvía un falso match arbitrario.
- **Líneas 81-91 (`obtener_confianza()` NUEVA):** usa `predict_proba` si existe (LogReg/NB/SVC calibrado) y si no, softmax sobre `decision_function` (SVC puro). Para qué: el nuevo mejor modelo podía no tener `predict_proba`; antes `asistente.py:91` crasheaba con `LinearSVC`.
- **Líneas 114-117 (chequeo OOV):** mensaje `No reconozco ninguna palabra...` cuando `nnz==0`. Para qué: no inventar categoría con entrada totalmente fuera de vocabulario (probado con `hola mundo xyz qwerty`).
- **Líneas 125 (confianza):** usa la función robusta en vez de `predict_proba` directo. Para qué: compatibilidad total con los 4 candidatos del GridSearch.
- **Líneas 134-139 (respuesta None):** maneja el caso sin vocabulario común. Para qué: corta el flujo antes de mostrar categoría/respuesta falsas.
- **Líneas 147-152 (doble umbral):** avisa si `confianza<50%` (sugiere palabras clave) y solo responde si `similitud>=35%`. Para qué: separa los dos modos de fallo (categoría dudosa vs pregunta lejana) en vez de un único `if similitud<20`.

## 4. `revisar_dataset.py` (112 líneas, antes 60)

- **Líneas 1-5 (imports):** `Counter, re, unicodedata`. Para qué: diagnósticos sin dependencias nuevas.
- **Líneas 24-25 (aviso 181 vs 180):** el CSV se llama `consultas_180` pero tiene 181 filas. Para qué: explica por qué test son 37 y no 36.
- **Líneas 59-61 (desbalance):** alerta si `max-min>5` y recuerda `class_weight='balanced'`. Para qué: justifica el parámetro usado en `entrenar.py` (31 vs 30).
- **Líneas 63-78 (duplicados):** exactos + colisión tras normalizar con tabla si hay. Para qué: detecta si el preproceso fusiona dos preguntas distintas (aquí: 0 y 0, OK).
- **Líneas 80-82 (longitud):** `describe()` de caracteres. Para qué: detectar ejemplos atípicos (aquí media 39, rango 18-62, sano).
- **Líneas 84-87 (respuestas únicas):** `groupby(categoria)['respuesta'].nunique()` (9-11 por clase). Para qué: si retrieval cosine falla, saber si es por pocas variantes.
- **Líneas 89-104 (top palabras):** top-5 por categoría + top-10 global. Para qué: evidenció el solapamiento (`que/necesito/cuando/como`) que motivó `STOPWORDS_ES`; también muestra palabras ancla sanas (`beca, biblioteca/libro, inscribirme/inscripcion, pagar/pago`).
- **Línea 108 (tabla recortada):** muestra solo `id/texto/categoria`. Para qué: lectura más limpia que el `head()` completo.

## 5. Dependencias

- **NUEVO `requirements.txt` (copia de `requeriments.txt`):** corrige el typo que rompía `pip install -r requirements.txt` indicado en `README.md:50`. Se conserva el archivo viejo para no romper scripts existentes. Sin cambios de versiones (todo lo usado ya estaba: `scikit-learn==1.9.0, pandas==3.0.5, scipy==1.18.1, numpy==2.5.2`).
- **Modelos regenerados `modelos/clasificador.pkl`, `modelos/vectorizador.pkl`:** ahora contienen `LogisticRegression(C=2.0)` + vectorizador con preproceso compartido. Requieren `preprocesado.py` al lado para cargarse.

## Cómo reproducir

```bash
source .venv/bin/activate
pip install -r requirements.txt
python revisar_dataset.py
python entrenar.py
printf "cuando son las inscripciones\nque becas existen\nsalir\n" | python asistente.py
```

## Trabajo futuro (fuera del alcance Quick wins)

1. Ampliar dataset (objetivo ≥80-100 por clase) con paráfrasis sin tilde y errores reales; `INSCRIPCIONES` sigue siendo la más confundible con `PAGOS` (`inscripcion` aparece en ambas según `revisar_dataset.py`).
2. Probar `ngram_range=(1,3)` y `min_df=2` cuando haya más datos; con 144 train, `min_df=2` era arriesgado y se dejó en 1.
3. Si se acepta nueva dependencia: `sentence-transformers` + cosine o fine-tune pequeño para intención, que supera TF-IDF en paráfrasis lejanas.
