# Módulo ultraligero para análisis de sentimiento offline basado en lexicón
# Diseñado para funcionar sin internet y sin modelos pesados.

PALABRAS_POSITIVAS = {
    "bueno", "bien", "excelente", "mejor", "gracias", "feliz", "alegre", "amor", 
    "ayuda", "útil", "genial", "lindo", "hermoso", "rico", "sabroso", "éxito", 
    "victoria", "amigo", "hermano", "socio", "resolver", "resuelto", "rápido",
    "fácil", "positivo", "suerte", "maravilla", "ganar", "ganamos", "risa"
}

PALABRAS_NEGATIVAS = {
    "malo", "mal", "peor", "pésimo", "triste", "enojado", "odio", "problema", 
    "error", "horrible", "feo", "asco", "mierda", "coño", "carajo", "joder", 
    "dolor", "difícil", "queja", "molesto", "lento", "tarde", "nunca", "hambre", 
    "roto", "negativo", "perder", "perdimos", "llorar", "cansado", "fatal", "duro"
}

def analizar_sentimiento_basico(texto):
    """
    Analiza el sentimiento de una oración contando palabras positivas y negativas.
    Retorna: 'Positivo', 'Negativo' o 'Neutro'
    """
    texto_lower = texto.lower()
    palabras = texto_lower.split()
    
    puntos = 0
    for p in palabras:
        # Limpiar puntuación básica
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
