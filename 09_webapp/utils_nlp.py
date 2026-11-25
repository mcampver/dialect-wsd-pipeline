# Módulo ultraligero para análisis de sentimiento offline basado en lexicón
# Diseñado para funcionar sin internet y sin modelos pesados.

_POSITIVAS_BASE = {
    "bueno", "bien", "excelente", "mejor", "gracias", "feliz", "alegre", "amor",
    "ayuda", "útil", "genial", "lindo", "hermoso", "rico", "sabroso", "éxito",
    "victoria", "amigo", "hermano", "socio", "resolver", "resuelto", "rápido",
    "fácil", "positivo", "suerte", "maravilla", "ganar", "ganamos", "risa"
}

_NEGATIVAS_BASE = {
    "malo", "mal", "peor", "pésimo", "triste", "enojado", "odio", "problema",
    "error", "horrible", "feo", "asco", "mierda", "coño", "carajo", "joder",
    "dolor", "difícil", "queja", "molesto", "lento", "tarde", "nunca", "hambre",
    "roto", "negativo", "perder", "perdimos", "llorar", "cansado", "fatal", "duro"
}


def _flexionar(palabras):
    """Añade femenino y plurales a los adjetivos/sustantivos en -o, y plural
    al resto. Sin esto 'buena', 'mala' o 'problemas' puntuaban como neutras."""
    formas = set(palabras)
    for p in palabras:
        if p.endswith("o"):
            formas.update({p[:-1] + "a", p + "s", p[:-1] + "as"})
        elif p[-1] in "aeiouáéíóú":
            formas.add(p + "s")
        else:  # error -> errores; las formas sobrantes (p. ej. 'ganares') no estorban
            formas.add(p + "es")
    return formas


PALABRAS_POSITIVAS = _flexionar(_POSITIVAS_BASE)
PALABRAS_NEGATIVAS = _flexionar(_NEGATIVAS_BASE)


def analizar_sentimiento_basico(texto):
    """
    Analiza el sentimiento de una oración contando palabras positivas y negativas.
    Retorna: 'Positivo', 'Negativo' o 'Neutro'
    """
    puntos = 0
    for p in texto.lower().split():
        p_clean = ''.join(e for e in p if e.isalnum())
        if p_clean in PALABRAS_POSITIVAS:
            puntos += 1
        elif p_clean in PALABRAS_NEGATIVAS:
            puntos -= 1

    if puntos > 0:
        return "Positivo"
    elif puntos < 0:
        return "Negativo"
    else:
        return "Neutro"
