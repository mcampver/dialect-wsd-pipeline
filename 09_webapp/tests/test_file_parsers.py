import io

import docx
import pytest

from utils_file_parsers import FileParseError, extract_text_from_file


def test_txt_utf8_con_bom(archivo):
    f = archivo("a.txt", "﻿Cogí la guagua.".encode("utf-8"))
    assert extract_text_from_file(f) == "Cogí la guagua."


def test_txt_ansi_windows(archivo):
    # Regresión: un .txt guardado como ANSI lanzaba UnicodeDecodeError y el
    # mensaje de error se analizaba como si fuera el texto.
    f = archivo("a.txt", "Qué bolá, asere. ¿Cómo está la jama?".encode("cp1252"))
    assert extract_text_from_file(f) == "Qué bolá, asere. ¿Cómo está la jama?"


def test_docx(archivo):
    d = docx.Document()
    d.add_paragraph("Primera línea.")
    d.add_paragraph("La guagua pasó.")
    buf = io.BytesIO()
    d.save(buf)
    assert extract_text_from_file(archivo("Doc.DOCX", buf.getvalue())) == "Primera línea.\nLa guagua pasó."


@pytest.mark.parametrize("nombre,datos,mensaje", [
    ("a.rtf", b"x", "no soportado"),
    ("sin_extension", b"x", "no soportado"),
    ("vacio.txt", b"  \n ", "no contiene texto extraíble$"),  # sin pista de PDF
    ("roto.pdf", b"esto no es un pdf", "ilegible"),
    ("roto.docx", b"esto no es un docx", "ilegible"),
])
def test_errores_se_lanzan_no_se_devuelven_como_texto(archivo, nombre, datos, mensaje):
    with pytest.raises(FileParseError, match=mensaje):
        extract_text_from_file(archivo(nombre, datos))
