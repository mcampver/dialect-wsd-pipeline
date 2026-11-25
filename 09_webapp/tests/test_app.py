"""Prueba de humo de la app completa con streamlit.testing (detector simulado)."""

import os
import sqlite3
import sys
import types

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")


class DetectorSimulado:
    def __init__(self):
        self.candidatos_traduccion = {"guagua": "autobús"}

    def analizar_texto(self, texto, on_progreso=None):
        hallazgos = []
        i = texto.find("guagua")
        while i != -1:
            hallazgos.append({"palabra": "guagua", "confianza": 97.5, "pos": "NOUN", "dep": "obj",
                              "lema": "guagua", "equivalente": "autobús", "sentimiento": "Neutro",
                              "inicio": i, "fin": i + 6, "contexto": texto.strip()})
            i = texto.find("guagua", i + 1)
        if on_progreso:
            on_progreso(1, 1)
        return hallazgos


@pytest.fixture
def app(db_temporal, monkeypatch):
    modulo = types.ModuleType("clasificador_final")
    modulo.DetectorCubano = DetectorSimulado
    monkeypatch.setitem(sys.modules, "clasificador_final", modulo)
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def _ir_a(at, etiqueta):
    at.sidebar.radio[0].set_value(etiqueta).run()
    assert not at.exception, at.exception


def test_analisis_rapido_detecta_y_muestra_tabla(app):
    app.text_area[0].input("Me monté en la guagua.").run()
    app.button[0].click().run()
    assert not app.exception
    assert "Se encontraron 1 posibles cubanismo(s)." in [s.value for s in app.success]
    assert len(app.dataframe) == 1


def test_analisis_rapido_texto_vacio_avisa(app):
    app.button[0].click().run()
    assert app.warning[0].value.startswith("Por favor")


def test_explorador_busca_en_db_temporal(app):
    _ir_a(app, "➕ Gestión del Diccionario")
    app.text_input[0].input("guagua").run()
    metricas = {m.label: m.value for m in app.metric}
    assert metricas["Resultados de la Búsqueda"] == "1"
    assert metricas["Total en Diccionario Completo"] == "3"


def test_alta_manual_y_duplicado(app, db_temporal):
    _ir_a(app, "➕ Gestión del Diccionario")
    formulario = {t.label: t for t in app.text_input}
    formulario["Lema / Palabra Original (Requerido)*"].input("Fula")
    app.text_area[0].input("Dólar.")
    app.button(key="FormSubmitter:form_nuevo_cubanismo-Guardar en Base de Datos").click().run()
    assert not app.exception
    assert any("se ha guardado" in s.value for s in app.success)

    app.button(key="FormSubmitter:form_nuevo_cubanismo-Guardar en Base de Datos").click().run()
    assert any("ya existe" in w.value for w in app.warning)

    conn = sqlite3.connect(db_temporal)
    assert conn.execute("SELECT COUNT(*) FROM cubanismos WHERE lema='fula'").fetchone()[0] == 1
    conn.close()


def test_batch_sin_archivos_no_falla(app):
    _ir_a(app, "📚 Laboratorio Batch (Múltiples Archivos)")
    assert not app.exception
