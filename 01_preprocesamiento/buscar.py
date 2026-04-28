"""
buscar.py — Buscador interactivo del Diccionario de Cubanismos
Uso:
    python buscar.py              → modo interactivo (escribe palabras)
    python buscar.py guagua       → búsqueda directa desde la terminal
"""

import sqlite3
import sys
import re

DB_PATH = "diccionario_cubanismos.db"
ANCHO   = 72


def normalizar(texto):
    """Elimina tildes para búsqueda más flexible."""
    reemplazos = str.maketrans("áéíóúüÁÉÍÓÚÜ", "aeiouuAEIOUU")
    return texto.lower().strip().translate(reemplazos)


def buscar(termino):
    termino_norm = normalizar(termino)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Buscar por coincidencia exacta primero, luego por prefijo (LIKE)
    cursor.execute(
        "SELECT lema, categoria_gramatical, definicion, ejemplo FROM cubanismos "
        "WHERE LOWER(lema) = ? OR LOWER(lema) LIKE ?",
        (termino.lower(), f"{termino.lower()}%")
    )
    resultados = cursor.fetchall()
    conn.close()

    if not resultados:
        # Segundo intento: búsqueda sin tildes sobre todos los lemas
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT lema, categoria_gramatical, definicion, ejemplo FROM cubanismos")
        todos = cursor.fetchall()
        conn.close()
        resultados = [r for r in todos if normalizar(r[0]).startswith(termino_norm)]

    return resultados


def mostrar(resultados, termino):
    if not resultados:
        print(f"\n  ✗ No se encontró '{termino}' en el diccionario.\n")
        return

    print("\n" + "═" * ANCHO)
    for lema, cat, definicion, ejemplo in resultados:
        print(f"  📖 {lema.upper()}  [{cat}]")
        print("─" * ANCHO)
        # Imprimir definición con saltos de línea cada ~70 chars
        if definicion:
            palabras = (definicion or "").split()
            linea = "  "
            for p in palabras:
                if len(linea) + len(p) + 1 > ANCHO:
                    print(linea)
                    linea = "  " + p + " "
                else:
                    linea += p + " "
            if linea.strip():
                print(linea)
        if ejemplo:
            print(f"\n  Ej: {ejemplo[:200]}")
        print("═" * ANCHO)
    print(f"  {len(resultados)} resultado(s) para '{termino}'\n")


def main():
    print("╔" + "═" * (ANCHO - 2) + "╗")
    print("║  DICCIONARIO DE CUBANISMOS — buscador interactivo".ljust(ANCHO - 1) + "║")
    print("║  Escribe una palabra para buscar · 'salir' para terminar".ljust(ANCHO - 1) + "║")
    print("╚" + "═" * (ANCHO - 2) + "╝\n")

    if len(sys.argv) > 1:
        # Modo de una sola búsqueda desde la terminal
        termino = " ".join(sys.argv[1:])
        mostrar(buscar(termino), termino)
        return

    while True:
        try:
            termino = input("  🔍 Buscar: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Hasta luego.")
            break

        if not termino:
            continue
        if termino.lower() in ("salir", "exit", "q"):
            print("\n  Hasta luego.\n")
            break

        mostrar(buscar(termino), termino)


if __name__ == "__main__":
    main()
