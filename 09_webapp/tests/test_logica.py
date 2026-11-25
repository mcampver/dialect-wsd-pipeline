import sqlite3

import pandas as pd
import pytest

import logica


def _h(texto, palabra, desde=0, equivalente="x"):
    i = texto.index(palabra, desde)
    return {"inicio": i, "fin": i + len(palabra), "equivalente": equivalente}


def test_anotaciones_reconstruyen_el_texto_exacto():
    texto = "Ayer esperé la guagua, asere.\n¡Qué bolá!"
    segs = logica.construir_anotaciones(texto, [_h(texto, "asere"), _h(texto, "guagua")])
    plano = "".join(s if isinstance(s, str) else s[0] for s in segs)
    assert plano == texto
    marcadas = [s[0] for s in segs if isinstance(s, tuple)]
    assert marcadas == ["guagua", "asere"]


def test_solo_se_resalta_la_aparicion_aceptada():
    # Regresión: antes se resaltaban todas las apariciones con el mismo texto.
    texto = "Una guagua de juguete. Cogí la guagua."
    segunda = _h(texto, "guagua", desde=10)
    segs = logica.construir_anotaciones(texto, [segunda])
    marcadas = [s for s in segs if isinstance(s, tuple)]
    assert len(marcadas) == 1
    assert segs[0] == "Una guagua de juguete. Cogí la "


@pytest.mark.parametrize("n,palabras,esperado", [
    (4, 14, 285.714),   # regresión: daba 4.00 con max(1, palabras/1000)
    (3, 1000, 3.0),
    (10, 5000, 2.0),
    (0, 0, 0.0),
])
def test_densidad_por_mil(n, palabras, esperado):
    assert logica.densidad_por_mil(n, palabras) == pytest.approx(esperado, abs=1e-3)


def test_anotaciones_sin_hallazgos_devuelven_texto():
    assert logica.construir_anotaciones("hola", []) == ["hola"]


def test_normalizar_omite_definicion_vacia_en_vez_de_guardar_nan():
    df = pd.DataFrame({
        " LEMA ": ["Fula", "jama", None, "pepe"],
        "Definición": ["Dólar.", None, "x", "  "],
        "Traduccion": ["dólar", "comida", None, None],
    })
    filas, omitidas = logica.normalizar_filas_carga(df)
    assert filas == [("fula", "Dólar.", "dólar", "")]
    assert omitidas == 3
    assert all("nan" not in campo for f in filas for campo in f)


def test_normalizar_exige_columnas():
    with pytest.raises(ValueError):
        logica.normalizar_filas_carga(pd.DataFrame({"Palabra": ["x"]}))


def test_insertar_evita_duplicados_exactos(db_temporal):
    conn = sqlite3.connect(db_temporal)
    filas = [("guagua", "Autobús.", "", ""),        # ya existe
             ("guagua", "Insecto.", "bicho", ""),   # acepción nueva: válida
             ("fula", "Dólar.", "", ""),
             ("fula", "Dólar.", "", "")]            # repetida en el propio archivo
    insertadas, duplicadas = logica.insertar_entradas(conn, filas)
    assert [f[:2] for f in insertadas] == [("guagua", "Insecto."), ("fula", "Dólar.")]
    assert duplicadas == 2
    assert conn.execute("SELECT COUNT(*) FROM cubanismos").fetchone()[0] == 5
    conn.close()


def test_registrar_en_memoria_no_pisa_traduccion_con_vacio():
    cands = {"guagua": "autobús"}
    logica.registrar_en_memoria(cands, "guagua", "")
    logica.registrar_en_memoria(cands, "fula", "")
    logica.registrar_en_memoria(cands, "asere", "amigo")
    assert cands == {"guagua": "autobús", "fula": "Sin traducción", "asere": "amigo"}


def test_buscar_prioriza_coincidencia_exacta(db_temporal):
    conn = sqlite3.connect(db_temporal)
    conn.execute("INSERT INTO cubanismos (lema, definicion) VALUES ('guaguero', 'Chofer de guagua.')")
    df = logica.buscar(conn, "  GUAGUA ")
    assert df["Lema"].tolist()[0] == "guagua"
    assert set(df["Lema"]) == {"guagua", "guaguero"}
    assert len(logica.buscar(conn, "")) == 4
    conn.close()


def test_ruta_db_respeta_variable(db_temporal):
    assert logica.ruta_db() == str(db_temporal)
