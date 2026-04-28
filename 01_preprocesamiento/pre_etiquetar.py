"""
pre_etiquetar.py
Fase 2 – Pre-etiquetado automático del corpus (Supervisión débil)

Toma los textos del corpus (Cartas CORESPUC + Granma) y busca en cada oración
los lemas del diccionario de cubanismos. Genera un CSV de oraciones anotadas
con los cubanismos detectados para su posterior revisión manual.

Columnas del CSV de salida:
  archivo, oracion_id, oracion, cubanismo_detectado, lema_bd, categoria_gramatical, es_cubanismo
  (la última columna se dejará vacía para que el anotador la rellene: 1 = sí, 0 = no)
"""

import os
import re
import csv
import sqlite3
from pathlib import Path

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
DB_PATH    = "diccionario_cubanismos.db"
OUTPUT_CSV = "corpus_preetiquetado.csv"

CORPUS_DIRS = [
    r"Cartas CORESPUC",
    r"granma-txt\txt-files-new",
    r"granma-txt\txt-files-old",
]
# ──────────────────────────────────────────────────────────────────────────────


def cargar_lemas(db_path):
    """Carga todos los lemas del diccionario y devuelve dict lema->categoria."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT lema, categoria_gramatical FROM cubanismos WHERE lema != ''")
    rows = cursor.fetchall()
    conn.close()

    lemas = {}
    for lema, cat in rows:
        lema_clean = lema.strip().lower()
        # Ignorar lemas muy cortos (1-2 chars) para evitar falsos positivos
        if len(lema_clean) >= 3:
            lemas[lema_clean] = cat
    print(f"✓ Cargados {len(lemas)} lemas del diccionario.")
    return lemas


def segmentar_oraciones(texto):
    """Divide el texto en oraciones básicas por puntuación."""
    oraciones = re.split(r'(?<=[.!?])\s+', texto.strip())
    return [o.strip() for o in oraciones if len(o.strip()) > 10]


def detectar_cubanismos(oracion, lemas):
    """
    Busca coincidencias de lemas del diccionario en la oración.
    Retorna lista de (token_encontrado, lema_bd, categoria).
    Usa coincidencia de palabra completa para reducir falsos positivos.
    """
    encontrados = []
    palabras = re.findall(r'\b[a-záéíóúüñ]+\b', oracion.lower())
    for palabra in palabras:
        if palabra in lemas:
            encontrados.append((palabra, palabra, lemas[palabra]))
    return encontrados


def leer_texto_archivo(filepath):
    """Lee el contenido de texto de un archivo, maneja codificaciones."""
    for enc in ('utf-8', 'latin-1', 'cp1252'):
        try:
            content = Path(filepath).read_text(encoding=enc)
            # Los archivos CORESPUC y Granma tienen: línea 1 = título, línea 2 = cuerpo
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            if len(lines) >= 2:
                return lines[1]   # Usamos el cuerpo, no el título
            elif len(lines) == 1:
                return lines[0]
            return ""
        except Exception:
            continue
    return ""


def procesar_corpus(corpus_dirs, lemas, output_csv):
    total_oraciones = 0
    total_detecciones = 0
    archivos_procesados = 0

    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow([
            'archivo', 'oracion_id', 'oracion',
            'cubanismo_detectado', 'lema_bd', 'categoria_gramatical',
            'es_cubanismo'   # ← columna para anotación manual: 1=sí, 0=no
        ])

        for corpus_dir in corpus_dirs:
            dir_path = Path(corpus_dir)
            if not dir_path.exists():
                print(f"⚠ No encontrado: {corpus_dir}")
                continue
            
            txt_files = list(dir_path.rglob("*.txt"))
            print(f"\nProcesando {corpus_dir} — {len(txt_files)} archivos...")

            for filepath in txt_files:
                texto = leer_texto_archivo(filepath)
                if not texto:
                    continue

                oraciones = segmentar_oraciones(texto)
                archivos_procesados += 1

                for i, oracion in enumerate(oraciones, 1):
                    total_oraciones += 1
                    detecciones = detectar_cubanismos(oracion, lemas)

                    for token, lema, cat in detecciones:
                        writer.writerow([
                            filepath.name,
                            i,
                            oracion,
                            token,
                            lema,
                            cat,
                            ''   # es_cubanismo: a rellenar manualmente
                        ])
                        total_detecciones += 1

    print(f"\n{'─'*50}")
    print(f"✓ Archivos procesados:  {archivos_procesados}")
    print(f"✓ Oraciones procesadas: {total_oraciones}")
    print(f"✓ Detecciones pre-etiquetadas: {total_detecciones}")
    print(f"✓ CSV guardado en: {output_csv}")


if __name__ == "__main__":
    lemas = cargar_lemas(DB_PATH)
    procesar_corpus(CORPUS_DIRS, lemas, OUTPUT_CSV)
