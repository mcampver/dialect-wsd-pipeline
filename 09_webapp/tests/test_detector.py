"""Tests del DetectorCubano con dobles de spaCy y BETO (no carga el modelo real)."""

from types import SimpleNamespace

import pytest
import spacy
import torch

import clasificador_final as cf

CUBANAS = {"guagua", "asere"}


class TokenizerFalso:
    def __init__(self):
        self.llamadas = []

    def __call__(self, contextos, text_pair, **kw):
        self.llamadas.append({"contextos": contextos, "palabras": text_pair, **kw})
        return {"palabras": text_pair}


class ModeloFalso:
    """P(cubanismo)=0.9 para palabras en CUBANAS, 0.1 para el resto."""

    def eval(self):
        return self

    def __call__(self, palabras):
        logits = [[0.0, 2.2] if p.lower() in CUBANAS else [2.2, 0.0] for p in palabras]
        return SimpleNamespace(logits=torch.tensor(logits))


@pytest.fixture
def detector():
    nlp = spacy.blank("es")
    nlp.add_pipe("sentencizer")
    tok = TokenizerFalso()
    d = cf.DetectorCubano(nlp=nlp, tokenizer=tok, model=ModeloFalso(),
                          candidatos={"guagua": "autobús", "asere": "amigo", "casa": "hogar"})
    d.tok = tok
    return d


def test_texto_largo_se_analiza_por_oracion(detector):
    # Regresión: el texto completo se truncaba a 128 tokens y BETO no veía la
    # oración con el cubanismo. Ahora cada llamada recibe solo su oración.
    relleno = "Esto es una oración normal sobre economía nacional y política. " * 20
    texto = relleno + "Me monté en la guagua con mi asere."
    hallazgos = detector.analizar_texto(texto)
    assert [h["palabra"] for h in hallazgos] == ["guagua", "asere"]
    llamada = detector.tok.llamadas[-1]
    assert set(llamada["contextos"]) == {"Me monté en la guagua con mi asere."}
    assert llamada["truncation"] == "only_first"
    assert llamada["max_length"] == cf.MAX_TOKENS_BETO


def test_offsets_apuntan_a_la_palabra_en_el_texto(detector):
    texto = "La casa es grande.\n\nCogí la guagua. Mi asere llegó."
    for h in detector.analizar_texto(texto):
        assert texto[h["inicio"]:h["fin"]] == h["palabra"]
        assert h["palabra"] in h["contexto"]


def test_candidato_rechazado_por_beto_no_aparece(detector):
    assert detector.analizar_texto("Mi casa es tu casa.") == []


def test_texto_vacio(detector):
    assert detector.analizar_texto("") == []
    assert detector.analizar_texto("   ") == []


def test_progreso_llega_al_total(detector):
    vistos = []
    detector.analizar_texto("Uno. Dos. Tres guagua.", on_progreso=lambda h, t: vistos.append((h, t)))
    assert vistos[-1][0] == vistos[-1][1] == 3


def test_offsets_correctos_entre_trozos(detector, monkeypatch):
    monkeypatch.setattr(cf, "MAX_CHARS_TROZO", 40)
    texto = "\n".join(["Una línea normal de relleno aquí."] * 5 + ["Cogí la guagua ayer."])
    original = cf.trocear_texto
    monkeypatch.setattr(cf, "trocear_texto", lambda t: original(t, 40))
    hallazgos = detector.analizar_texto(texto)
    assert len(hallazgos) == 1
    assert texto[hallazgos[0]["inicio"]:hallazgos[0]["fin"]] == "guagua"


@pytest.mark.parametrize("texto,maximo", [
    ("", 10), ("corto", 10), ("a\nb\nc\nd\ne\nf", 4), ("x" * 25, 10), ("ab\n" * 50, 7),
])
def test_trocear_cubre_todo_sin_pasarse(texto, maximo):
    trozos = cf.trocear_texto(texto, maximo)
    assert "".join(t for _, t in trozos) == texto
    for off, t in trozos:
        assert texto[off:off + len(t)] == t
        assert len(t) <= maximo


def test_sin_print_con_emoji(detector, capsys):
    # Regresión: print('📍 ...') lanzaba UnicodeEncodeError en consolas cp1252.
    detector.analizar_frase("Cogí la guagua.")
    salida = capsys.readouterr().out
    salida.encode("cp1252")  # no debe lanzar


def test_cargar_candidatos_desde_db(db_temporal, monkeypatch):
    cands = cf.cargar_candidatos(str(db_temporal))
    assert cands == {"guagua": "autobús", "asere": "amigo", "jama": "Sin equivalente registrado"}


def test_cargar_candidatos_db_inexistente(tmp_path):
    assert cf.cargar_candidatos(str(tmp_path / "no.db")) == {}
