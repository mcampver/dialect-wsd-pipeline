import io

import docx
import pdfplumber

EXTENSIONES_SOPORTADAS = ("txt", "pdf", "docx")


class FileParseError(Exception):
    """El archivo no se pudo convertir en texto analizable."""


def _decodificar_txt(datos):
    # utf-8-sig quita el BOM que añade el Bloc de notas; cp1252 cubre los .txt
    # guardados como "ANSI" en Windows (latin-1 nunca falla: último recurso).
    for codificacion in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return datos.decode(codificacion)
        except UnicodeDecodeError:
            continue
    raise FileParseError("codificación de texto no reconocida")  # inalcanzable con latin-1


def extract_text_from_file(uploaded_file):
    """Extrae el texto de un .txt, .pdf o .docx subido con Streamlit.

    Lanza FileParseError si el formato no es soportado, el archivo está dañado
    o no contiene texto extraíble (p. ej. un PDF escaneado sin OCR). Antes se
    devolvía el mensaje de error como si fuera el texto, y ese mensaje acababa
    analizado y contado como palabras del corpus.
    """
    nombre = uploaded_file.name
    ext = nombre.rsplit(".", 1)[-1].lower() if "." in nombre else ""
    if ext not in EXTENSIONES_SOPORTADAS:
        raise FileParseError(f"formato no soportado ({ext or 'sin extensión'})")

    datos = uploaded_file.getvalue()
    try:
        if ext == "txt":
            texto = _decodificar_txt(datos)
        elif ext == "pdf":
            with pdfplumber.open(io.BytesIO(datos)) as pdf:
                paginas = [p.extract_text() for p in pdf.pages]
            texto = "\n".join(p for p in paginas if p)
        else:  # docx
            documento = docx.Document(io.BytesIO(datos))
            texto = "\n".join(p.text for p in documento.paragraphs)
    except FileParseError:
        raise
    except Exception as e:
        raise FileParseError(f"archivo dañado o ilegible: {e}") from e

    if not texto.strip():
        pista = " (¿PDF escaneado sin OCR?)" if ext == "pdf" else ""
        raise FileParseError(f"no contiene texto extraíble{pista}")
    return texto
