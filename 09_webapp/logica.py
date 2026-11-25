"""Lógica sin Streamlit de la webapp, separada para poder testearla."""

import os
import sqlite3

import pandas as pd

DB_POR_DEFECTO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "03_datos", "diccionario_cubanismos.db")
)

COLOR_FONDO = "#E2E8F0"
COLOR_TEXTO = "#0F172A"


def ruta_db():
    """Ruta del diccionario; CUBANISMOS_DB_PATH permite usar una copia (tests)."""
    return os.environ.get("CUBANISMOS_DB_PATH", DB_POR_DEFECTO)


def densidad_por_mil(n_hallazgos, n_palabras):
    """Cubanismos por cada 1000 palabras.

    Antes se calculaba n / max(1, palabras/1000): con menos de 1000 palabras
    el denominador valía 1 y la "densidad" era el conteo bruto (4 hallazgos
    en 14 palabras daban 4.00 en vez de 285.71).
    """
    return n_hallazgos * 1000 / n_palabras if n_palabras else 0.0


def construir_anotaciones(texto, hallazgos):
    """Segmentos para annotated_text a partir de los offsets de cada hallazgo.

    Solo se resalta la aparición concreta que BETO aceptó: si 'guagua' sale
    dos veces y solo una es cubanismo, la otra queda como texto plano (antes
    se resaltaban todas las apariciones con el mismo texto).
    """
    segmentos = []
    cursor = 0
    for h in sorted(hallazgos, key=lambda h: h["inicio"]):
        inicio, fin = h["inicio"], h["fin"]
        if inicio < cursor:  # solapado; no debería ocurrir con tokens
            continue
        if inicio > cursor:
            segmentos.append(texto[cursor:inicio])
        segmentos.append((texto[inicio:fin], f"➔ {h.get('equivalente', 'N/A')}", COLOR_FONDO, COLOR_TEXTO))
        cursor = fin
    if cursor < len(texto):
        segmentos.append(texto[cursor:])
    return segmentos


def _celda(valor):
    if valor is None or (not isinstance(valor, str) and pd.isna(valor)):
        return ""
    return str(valor).strip()


def normalizar_filas_carga(df):
    """Convierte la hoja subida en filas (lema, definicion, traduccion, ejemplo).

    Devuelve (filas, omitidas). Lanza ValueError si faltan las columnas
    obligatorias. Una fila sin lema o sin definición se omite: antes una
    definición vacía se guardaba como el texto 'nan'.
    """
    columnas = {str(c).lower().strip(): c for c in df.columns}
    alias = {"definición": "definicion", "traducción": "traduccion"}
    for con_tilde, sin_tilde in alias.items():
        if con_tilde in columnas and sin_tilde not in columnas:
            columnas[sin_tilde] = columnas[con_tilde]
    if "lema" not in columnas or "definicion" not in columnas:
        raise ValueError("Faltan las columnas obligatorias 'Lema' y 'Definicion'.")

    c_trad = columnas.get("traduccion")
    c_ejem = columnas.get("ejemplo")
    filas, omitidas = [], 0
    for _, row in df.iterrows():
        lema = _celda(row[columnas["lema"]]).lower()
        definicion = _celda(row[columnas["definicion"]])
        if not lema or not definicion:
            omitidas += 1
            continue
        filas.append((
            lema,
            definicion,
            _celda(row[c_trad]) if c_trad else "",
            _celda(row[c_ejem]) if c_ejem else "",
        ))
    return filas, omitidas


def existe_entrada(conn, lema, definicion):
    """True si ya hay una acepción con el mismo lema y la misma definición."""
    cur = conn.execute(
        "SELECT 1 FROM cubanismos WHERE lower(trim(lema)) = ? AND trim(definicion) = ? LIMIT 1",
        (lema.lower().strip(), definicion.strip()),
    )
    return cur.fetchone() is not None


def insertar_entradas(conn, filas):
    """Inserta filas evitando duplicados exactos. Devuelve (insertadas, duplicadas)."""
    insertadas, duplicadas = [], 0
    vistas = set()
    for lema, definicion, trad, ejem in filas:
        clave = (lema, definicion)
        if clave in vistas or existe_entrada(conn, lema, definicion):
            duplicadas += 1
            continue
        vistas.add(clave)
        conn.execute(
            "INSERT INTO cubanismos (lema, definicion, traduccion, ejemplo) VALUES (?, ?, ?, ?)",
            (lema, definicion, trad, ejem),
        )
        insertadas.append((lema, definicion, trad, ejem))
    conn.commit()
    return insertadas, duplicadas


def registrar_en_memoria(candidatos, lema, traduccion):
    """Actualiza el diccionario en memoria del detector sin pisar una
    traducción existente con un valor vacío."""
    if traduccion:
        candidatos[lema] = traduccion
    else:
        candidatos.setdefault(lema, "Sin traducción")


def buscar(conn, consulta):
    """Búsqueda del explorador. Sin consulta, devuelve las 100 últimas entradas."""
    columnas = ("SELECT id, lema AS Lema, definicion AS Definición, "
                "traduccion AS Traducción, ejemplo AS Ejemplo FROM cubanismos ")
    consulta = consulta.strip().lower()
    if not consulta:
        return pd.read_sql_query(columnas + "WHERE lema IS NOT NULL ORDER BY id DESC LIMIT 100", conn)
    contiene, empieza = f"%{consulta}%", f"{consulta}%"
    sql = columnas + """
        WHERE lema IS NOT NULL AND (lema LIKE ? OR definicion LIKE ? OR traduccion LIKE ?)
        ORDER BY CASE WHEN lower(lema) = ? THEN 1 WHEN lema LIKE ? THEN 2 ELSE 3 END, lema ASC
    """
    return pd.read_sql_query(sql, conn, params=(contiene, contiene, contiene, consulta, empieza))
