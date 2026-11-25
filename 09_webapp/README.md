# Webapp: Analizador de Cubanismos

Interfaz Streamlit sobre `06_aplicacion/clasificador_final.py` (spaCy + BETO).

## Ejecutar

```powershell
python -m streamlit run 09_webapp/app.py
```

La primera carga tarda unos 30 s (spaCy `es_core_news_lg` + BETO).

Para probar sin tocar el diccionario real, apunta a una copia:

```powershell
$env:CUBANISMOS_DB_PATH = "C:\ruta\copia.db"; python -m streamlit run 09_webapp/app.py
```

## Estructura

| Archivo | Qué hace |
|---|---|
| `app.py` | UI: análisis rápido, laboratorio batch, gestión del diccionario |
| `logica.py` | Lógica sin Streamlit: anotaciones por offsets, densidad, carga masiva, duplicados, búsqueda |
| `utils_file_parsers.py` | `.txt` (UTF-8/ANSI), `.pdf`, `.docx`; lanza `FileParseError` |
| `utils_nlp.py` | Sentimiento por lexicón (con femeninos y plurales) |

El detector analiza **oración por oración** (`DetectorCubano.analizar_texto`), igual que
se entrenó BETO (pares oración + palabra, 128 tokens).

## Tests (gate, sin modelo real)

```powershell
python -m pytest 09_webapp/tests -q
```

Usan dobles de spaCy/BETO y una BD temporal; nunca escriben en `03_datos/`.
`test_app.py` ejecuta la app completa con `streamlit.testing.AppTest`.

## Eval (periódico, modelo real, ~7 min)

```powershell
python 09_webapp/evals/eval_detector.py
```

Mide F1 de BETO sobre `03_datos/split_test.csv` y el recall en textos largos
(oración positiva tras ~250 tokens de relleno). Sale con código 1 si baja de los umbrales.
Línea base (2026-11-25, n=300): F1 0.765; recall en textos largos 1.0 (modo antiguo: 0.0).
