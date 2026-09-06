import pandas as pd
from pathlib import Path

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

# Mostrar ejemplos
print("\nPrimeros 5 registros:")
print(df.head().to_string(index=False))

print("\n" + "=" * 50)
print("Revisión terminada.")
print("=" * 50)