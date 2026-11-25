"""
CLASIFICADOR FINAL DE CUBANISMOS - TESIS (v2.2)
===============================================

v2.2:
- `analizar_texto` analiza oración por oración (BETO se entrenó con pares
  oración + palabra; pasarle un texto largo truncado a 128 tokens perdía el
  contexto y no detectaba nada).
- Cada hallazgo incluye `inicio`/`fin` (offsets de carácter) y `contexto`.
- Sin prints con emoji: en consolas Windows cp1252 lanzaban UnicodeEncodeError.
- Ruta de la base de datos configurable con la variable CUBANISMOS_DB_PATH.
"""

import logging
import os
import sqlite3
import sys

import pandas as pd
import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_APP_DIR = os.path.dirname(os.path.abspath(__file__))
WEBAPP_DIR = os.path.join(BASE_APP_DIR, "..", "09_webapp")
sys.path.append(os.path.abspath(WEBAPP_DIR))
try:
    from utils_nlp import analizar_sentimiento_basico
except ImportError:  # fuera de su contexto
    def analizar_sentimiento_basico(_texto):
        return "Neutro"

logger = logging.getLogger("clasificador_final")

MODEL_PATH = os.path.join(BASE_APP_DIR, "modelo_cubanismos_final")
DB_PATH = os.environ.get(
    "CUBANISMOS_DB_PATH",
    os.path.abspath(os.path.join(BASE_APP_DIR, "..", "03_datos", "diccionario_cubanismos.db")),
)

UMBRAL = 0.35          # probabilidad mínima para aceptar un cubanismo
MAX_TOKENS_BETO = 128  # longitud usada en el entrenamiento (fase5)
MAX_CHARS_TROZO = 100_000  # spaCy admite 1M; trozos pequeños acotan la memoria

RESPALDO = {
    "asere": "amigo/socio", "acere": "amigo/socio",
    "guagua": "autobús", "máquina": "automóvil antiguo", "maquina": "automóvil antiguo",
    "bola": "rumor", "pinchar": "trabajar", "finca": "prisión",
    "ómnibus": "autobús", "que bola": "hola / qué tal",
}


def trocear_texto(texto, max_chars=MAX_CHARS_TROZO):
    """Divide `texto` en trozos de como mucho `max_chars` caracteres, cortando
    en saltos de línea cuando es posible. Devuelve [(offset, trozo), ...] tal
    que texto[offset:offset+len(trozo)] == trozo y los trozos cubren todo."""
    if len(texto) <= max_chars:
        return [(0, texto)] if texto else []
    trozos = []
    inicio = 0
    n = len(texto)
    while inicio < n:
        fin = min(inicio + max_chars, n)
        if fin < n:
            corte = texto.rfind("\n", inicio, fin)
            if corte > inicio:
                fin = corte + 1
        trozos.append((inicio, texto[inicio:fin]))
        inicio = fin
    return trozos


def cargar_candidatos(db_path):
    """Lee {lema: traducción} desde la base de datos SQLite."""
    candidatos = {}
    if not os.path.exists(db_path):
        logger.warning("No existe la base de datos %s; uso solo el respaldo.", db_path)
        return candidatos
    try:
        conn = sqlite3.connect(db_path)
        try:
            try:
                df = pd.read_sql_query("SELECT lema, traduccion FROM cubanismos", conn)
            except Exception:
                df = pd.read_sql_query("SELECT DISTINCT lema FROM cubanismos", conn)
                df["traduccion"] = None
        finally:
            conn.close()
        for lema, trad in zip(df["lema"], df["traduccion"]):
            if lema is None or pd.isna(lema):
                continue
            clave = str(lema).lower().strip()
            if not clave:
                continue
            if trad is None or pd.isna(trad) or not str(trad).strip():
                trad = "Sin equivalente registrado"
            candidatos[clave] = trad
    except Exception as e:  # base corrupta, tabla ausente...
        logger.error("Error cargando la base de datos (%s); uso solo el respaldo.", e)
    return candidatos


class DetectorCubano:
    def __init__(self, nlp=None, tokenizer=None, model=None, candidatos=None):
        """Los parámetros opcionales permiten inyectar dobles en los tests."""
        logger.info("Iniciando Detector de Cubanismos (v2.2)")

        if nlp is None:
            try:
                nlp = spacy.load("es_core_news_lg")
            except OSError:
                os.system(f'"{sys.executable}" -m spacy download es_core_news_lg')
                nlp = spacy.load("es_core_news_lg")
        self.nlp = nlp

        if tokenizer is None or model is None:
            logger.info("Cargando modelo BETO desde %s", MODEL_PATH)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()

        self.candidatos_traduccion = cargar_candidatos(DB_PATH) if candidatos is None else dict(candidatos)
        for k, v in RESPALDO.items():
            self.candidatos_traduccion.setdefault(k, v)
        logger.info("Diccionario cargado: %d términos.", len(self.candidatos_traduccion))

    # ------------------------------------------------------------------ #
    def _traduccion(self, token):
        lema = token.lemma_.lower()
        if lema in self.candidatos_traduccion:
            return self.candidatos_traduccion[lema]
        return self.candidatos_traduccion.get(token.text.lower())

    def _probabilidades(self, contexto, palabras):
        """P(cubanismo) para cada palabra en `contexto`, en una sola pasada."""
        inputs = self.tokenizer(
            [contexto] * len(palabras),
            text_pair=palabras,
            return_tensors="pt",
            truncation="only_first",  # nunca recortar la palabra objetivo
            padding=True,
            max_length=MAX_TOKENS_BETO,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return torch.nn.functional.softmax(logits, dim=-1)[:, 1].tolist()

    def _evaluar_tramo(self, tokens, contexto, offset):
        """Evalúa los tokens de un tramo (una oración) con `contexto` como
        entrada de BETO. `offset` se suma a token.idx para dar posiciones
        absolutas en el texto original."""
        candidatos = [(t, trad) for t in tokens if (trad := self._traduccion(t)) is not None]
        if not candidatos:
            return []
        contexto_limpio = contexto.strip()
        probs = self._probabilidades(contexto_limpio, [t.text for t, _ in candidatos])
        sentimiento = None
        hallazgos = []
        for (token, trad), prob in zip(candidatos, probs):
            if prob <= UMBRAL:
                continue
            if sentimiento is None:
                sentimiento = analizar_sentimiento_basico(contexto_limpio)
            inicio = offset + token.idx
            hallazgos.append({
                "palabra": token.text,
                "confianza": round(prob * 100, 1),
                "pos": token.pos_,
                "dep": token.dep_,
                "lema": token.lemma_,
                "equivalente": trad,
                "sentimiento": sentimiento,
                "inicio": inicio,
                "fin": inicio + len(token.text),
                "contexto": contexto_limpio,
            })
        return hallazgos

    # ------------------------------------------------------------------ #
    def analizar_frase(self, texto):
        """Analiza `texto` como una sola unidad de contexto (una oración)."""
        doc = self.nlp(texto)
        hallazgos = self._evaluar_tramo(list(doc), texto, 0)
        logger.debug("analizar_frase: %d hallazgos", len(hallazgos))
        return hallazgos

    def analizar_texto(self, texto, on_progreso=None):
        """Analiza un texto de cualquier longitud oración por oración.

        `on_progreso(hechas, total)` se llama tras cada oración (opcional).
        Los offsets `inicio`/`fin` son relativos a `texto`.
        """
        trozos = trocear_texto(texto)
        docs = list(self.nlp.pipe(t for _, t in trozos)) if trozos else []
        total = sum(1 for d in docs for _ in d.sents)
        hechas = 0
        hallazgos = []
        for (offset, _), doc in zip(trozos, docs):
            for sent in doc.sents:
                if sent.text.strip():
                    hallazgos.extend(self._evaluar_tramo(list(sent), sent.text, offset))
                hechas += 1
                if on_progreso:
                    on_progreso(hechas, total)
        return hallazgos


def main():
    logging.basicConfig(level=logging.INFO)
    app = DetectorCubano()
    print("\nListo. Escribe una frase (o 'salir').")
    while True:
        frase = input("\nAnalizar frase > ")
        if frase.lower() in ("salir", "exit"):
            break
        if not frase.strip():
            continue
        for h in app.analizar_texto(frase):
            print(f"* '{h['palabra']}' ({h['confianza']}%) {h['pos']}/{h['dep']} -> {h['equivalente']}")


if __name__ == "__main__":
    main()
