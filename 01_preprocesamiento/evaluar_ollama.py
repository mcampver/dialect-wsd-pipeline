import pandas as pd
import subprocess
import re
import time
import sqlite3
from sklearn.metrics import classification_report, confusion_matrix

MODEL_NAME = "qwen3.5:2b"

def formatear_prompt(palabra, oracion, definicion):
    return f"""Analiza la siguiente palabra en su contexto oracional. Muestra únicamente tu decisión en el formato exacto requerido, sin introducción ni saludos.

Palabra a analizar: {palabra}
Oración: {oracion}

Toma en cuenta la siguiente definición del Diccionario de Cubanismos:
{definicion}

Pregunta: En la oración de contexto proporcionada, ¿la palabra significa SÍ o SÍ lo que indica la Definición del Diccionario?

Si el significado contextual de la palabra coincide con la definición (ES UN CUBANISMO), escribe:
DECISION: 1

Si es una palabra de uso común normal en el idioma español u otro significado que NO sea el dialectal descrito (ES UNA COINCIDENCIA / FALSO POSITIVO), escribe:
DECISION: 0

RAZON: [Explicación de 1 sola línea]"""

def llamar_ollama(prompt):
    try:
        cmd = ["ollama", "run", MODEL_NAME, prompt]
        process = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=45)
        return process.stdout.strip()
    except subprocess.TimeoutExpired:
        return "DECISION: ?\nRAZON: timeout"
    except Exception as e:
        return f"DECISION: ?\nRAZON: {e}"

def extraer_decision(respuesta):
    # Eliminar bloques hipotéticos <think> si los hay
    respuesta = re.sub(r"<think>.*?</think>", "", respuesta, flags=re.IGNORECASE | re.DOTALL).strip()
    
    match_dec = re.search(r"DECISION[:\*]?\s*(\d)", respuesta, re.IGNORECASE)
    if match_dec:
        return int(match_dec.group(1))
        
    match_fallback = re.search(r"^\s*[\*\-]?\s*([01])", respuesta)
    if match_fallback:
        return int(match_fallback.group(1))
        
    return -1 # Desconocido/Error

def main():
    print("==================================================")
    print(f"🤖 FASE 4: EXPERIMENTO ZERO-SHOT - {MODEL_NAME.upper()}")
    print("==================================================")
    
    try:
        # Cargar el mismo test set exacto que falló/acertó el RandomForest
        df_test = pd.read_csv('resultados_ml_test.csv', sep=';', encoding='utf-8-sig')
    except FileNotFoundError:
        print("Error: No se encuentra 'resultados_ml_test.csv'. Ejecuta la Fase 4 (ML) primero.")
        return

    print(f"Test Set cargado: Evaluando {len(df_test)} oraciones contra el LLM...\n")
    
    # Conexión a SQLite para el RAG básico (obtener la definición real del lema)
    conn = sqlite3.connect('diccionario_cubanismos.db')
    cursor = conn.cursor()
    
    y_true = df_test['y_true'].tolist()
    y_pred_llm = []
    respuestas_crudas = []
    
    start = time.time()
    
    # Progreso iterativo
    for idx, row in df_test.iterrows():
        palabra = row['palabra']
        oracion = row['oracion']
        
        # Buscar la definición de la palabra candidata en el diccionario
        cursor.execute("SELECT definicion FROM cubanismos WHERE lema = ? LIMIT 1", (palabra,))
        res = cursor.fetchone()
        definicion_real = res[0] if res else "Acepción dialectal desconocida."
        
        prompt = formatear_prompt(palabra, oracion, definicion_real)
        
        salida_llm = llamar_ollama(prompt)
        decision = extraer_decision(salida_llm)
        
        y_pred_llm.append(decision if decision in [0, 1] else 0) # Si falla, asumimos 0
        respuestas_crudas.append(salida_llm.replace('\n', ' '))
        
        # Opcional: imprimir el progreso cada 10 oraciones
        if (idx+1) % 10 == 0:
            print(f"[{idx+1}/{len(df_test)}] Procesados... T. transcurrido: {time.time()-start:.1f}s")
            
    print(f"\nEvaluación finalizada en {time.time()-start:.1f} segundos.")
    
    # Guardar análisis en el DataFrame para revisión
    df_test['prediccion_ollama'] = y_pred_llm
    df_test['ollama_crudo'] = respuestas_crudas
    df_test.to_csv("resultados_llm_zero_shot.csv", sep=";", index=False, encoding="utf-8-sig")
    
    print("\n--- MÉTRICAS DEL LLM ZERO-SHOT ---")
    print(classification_report(y_true, y_pred_llm, target_names=['0 (Falso Positivo)', '1 (Cubanismo Real)'], zero_division=0))
    print("Matriz de confusión (Ollama):\n", confusion_matrix(y_true, y_pred_llm))
    
    conn.close()
    
if __name__ == "__main__":
    main()
