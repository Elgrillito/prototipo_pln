import pandas as pd
from pathlib import Path
from collections import Counter
import re
import unicodedata

# Ruta del archivo CSV
archivo = Path("dataset/consultas_180.csv")

# Verificar que el archivo exista
if not archivo.exists():
    print(f"ERROR: No se encontró el archivo: {archivo}")
    exit()

# Leer el CSV
df = pd.read_csv(archivo)

print("=" * 50)
print("      REVISIÓN DEL DATASET")
print("=" * 50)

# Información general
print(f"\nCantidad de registros: {len(df)}")
if len(df) != 180:
    print(f"AVISO: el archivo se llama *_180.csv pero contiene {len(df)} filas.")

print("\nColumnas encontradas:")
for columna in df.columns:
    print(f" - {columna}")

# Verificar columnas necesarias
columnas_necesarias = ["id", "texto", "categoria", "respuesta"]

faltantes = [
    columna for columna in columnas_necesarias
    if columna not in df.columns
]

if faltantes:
    print("\nERROR: Faltan las siguientes columnas:")
    for columna in faltantes:
        print(f" - {columna}")
    exit()

# Buscar valores vacíos
print("\nValores vacíos:")
vacios = df[columnas_necesarias].isnull().sum()

for columna, cantidad in vacios.items():
    print(f" - {columna}: {cantidad}")

# Mostrar categorías
print("\nCantidad de ejemplos por categoría:")
conteo = df["categoria"].value_counts()

for categoria, cantidad in conteo.items():
    print(f" - {categoria}: {cantidad}")

min_c, max_c = conteo.min(), conteo.max()
if max_c - min_c > 5:
    print(f"AVISO: desbalance leve ({min_c}-{max_c} por clase). Se usa class_weight='balanced'.")

# Duplicados exactos y casi-duplicados (tras normalizar)
print("\nDuplicados exactos en 'texto':", int(df.duplicated(subset=["texto"]).sum()))


def norm(s):
    s = str(s).lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


df["_norm"] = df["texto"].map(norm)
dup_norm = df.duplicated(subset=["_norm"], keep=False)
print(f"Textos que colisionan tras normalizar (acentos/signos): {int(dup_norm.sum())}")
if dup_norm.any():
    print(df[dup_norm].sort_values("_norm")[["texto", "categoria"]].to_string(index=False))

# Longitud de textos (detecta ejemplos atipicos)
print("\nLongitud de textos (caracteres):")
print(df["texto"].str.len().describe().to_string())

# Respuestas por categoria (si hay muchas preguntas para 1 respuesta, el retrieval es facil;
# si hay 1 pregunta por respuesta, el clasificador sufre mas)
print("\nRespuestas unicas por categoria:")
print(df.groupby("categoria")["respuesta"].nunique().to_string())

# Palabras mas frecuentes por categoria vs globales (detecta solapamiento:
# 'que/necesito/cuando/como' aparecen en todas y no discriminan -> stopwords)
print("\nTop-5 palabras por categoria (normalizadas, con stopwords basicas):")
stop = {"que", "como", "cuando", "donde", "para", "por", "con", "una", "para",
        "el", "la", "los", "las", "de", "del", "en", "y", "mi", "hay", "puedo"}
global_counter = Counter()
for categoria, grupo in df.groupby("categoria"):
    tokens = []
    for t in grupo["_norm"]:
        tokens += [w for w in t.split() if w not in stop and len(w) > 2]
    cnt = Counter(tokens)
    global_counter.update(tokens)
    print(f" - {categoria}: {', '.join(f'{w}({c})' for w, c in cnt.most_common(5))}")

print("\nPalabras mas comunes en TODO el dataset (candidatas a stopwords si saturan):")
print(", ".join(f"{w}({c})" for w, c in global_counter.most_common(10)))

# Mostrar ejemplos
print("\nPrimeros 5 registros:")
print(df[["id", "texto", "categoria"]].head().to_string(index=False))

print("\n" + "=" * 50)
print("Revisión terminada.")
print("=" * 50)
