import os
import sqlite3
import sys
from types import SimpleNamespace

import pytest

WEBAPP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APLICACION = os.path.abspath(os.path.join(WEBAPP, "..", "06_aplicacion"))
for p in (WEBAPP, APLICACION):
    if p not in sys.path:
        sys.path.insert(0, p)

SCHEMA = """
CREATE TABLE cubanismos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lema TEXT NOT NULL,
    categoria_gramatical TEXT,
    funcion_sintactica TEXT,
    marcas_dialectales TEXT,
    definicion TEXT NOT NULL,
    ejemplo TEXT,
    fuente TEXT,
    traduccion TEXT
);
"""


@pytest.fixture
def db_temporal(tmp_path, monkeypatch):
    """Copia mínima del diccionario; nunca se toca 03_datos/ en los tests."""
    ruta = tmp_path / "dic.db"
    conn = sqlite3.connect(ruta)
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT INTO cubanismos (lema, definicion, traduccion) VALUES (?, ?, ?)",
        [("guagua", "Autobús.", "autobús"), ("asere", "Amigo.", "amigo"), ("Jama", "Comida.", None)],
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("CUBANISMOS_DB_PATH", str(ruta))
    return ruta


class ArchivoSubido(SimpleNamespace):
    """Imita el UploadedFile de Streamlit (name + getvalue)."""

    def getvalue(self):
        return self.data


@pytest.fixture
def archivo():
    return lambda name, data: ArchivoSubido(name=name, data=data)
