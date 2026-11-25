import pytest

from utils_nlp import analizar_sentimiento_basico


@pytest.mark.parametrize("texto,esperado", [
    ("La jama estaba buena.", "Positivo"),      # regresión: 'buena' era neutra
    ("La jama estuvo mala.", "Negativo"),       # regresión: 'mala' era neutra
    ("Tenemos muchos problemas.", "Negativo"),
    ("Las cosas están lindas, gracias.", "Positivo"),
    ("Fue una película pésima.", "Negativo"),
    ("Cogí la guagua.", "Neutro"),
    ("Bueno, pero el viaje fue malo y lento.", "Negativo"),
])
def test_sentimiento(texto, esperado):
    assert analizar_sentimiento_basico(texto) == esperado
