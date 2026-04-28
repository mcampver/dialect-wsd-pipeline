import re
import requests
import time
import subprocess

MODEL_NAME = "qwen3.5:2b"

def test_llamar_ollama(palabra, oracion, definicion):
    prompt = f"""Responde solo siguiendo el formato.
Palabra: {palabra}
Oración: {oracion}
Definición: {definicion}

Si es cubanismo escribe: DECISION: 1
Si no lo es escribe: DECISION: 0
RAZON: [explicación breve]"""

    print(f"\n--- Testing '{palabra}' ---")
    
    try:
        start_time = time.time()
        
        cmd = ["ollama", "run", MODEL_NAME, prompt]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        
        content = ""
        for line in process.stdout:
            content += line
            print(line, end="")
            
        process.wait()
        
        end_time = time.time()
        content = content.strip()
        
        print(f"\n✅ Tiempo total de Ollama (CLI): {end_time - start_time:.2f} segundos")
        print("---------------------------")
        
        # Parseo
        content_no_think = re.sub(r"<think>.*?</think>", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
        if not content_no_think: content_no_think = content

        match_decision = re.search(r"DECISION[:\*]?\s*(\d)", content_no_think, re.IGNORECASE)
        if match_decision:
            decision = match_decision.group(1)
        else:
            match_fallback = re.search(r"^\s*[\*\-]?\s*([01])", content_no_think)
            decision = match_fallback.group(1) if match_fallback else "?"

        match_razon = re.search(r"RAZ[OÓ]N[:\*]?\s*(.*)", content_no_think, re.IGNORECASE | re.DOTALL)
        if match_razon:
            razon = match_razon.group(1).strip()
        else:
            razon = re.sub(r"(?i)\**DECISION[:\*]?\s*\d\**\s*", "", content_no_think).strip()
    
        razon = razon.replace("\n", " ").replace("\r", " ")

        print("PARSED DECISION:", decision)
        print("PARSED RAZON:", razon[:150])

        content = re.sub(r"<think>.*?</think>", "", content, flags=re.IGNORECASE | re.DOTALL).strip()

        match_decision = re.search(r"DECISION[:\*]?\s*(\d)", content, re.IGNORECASE)
        if match_decision:
            decision = match_decision.group(1)
        else:
            match_fallback = re.search(r"^\s*[\*\-]?\s*([01])", content)
            decision = match_fallback.group(1) if match_fallback else "?"

        match_razon = re.search(r"RAZ[OÓ]N[:\*]?\s*(.*)", content, re.IGNORECASE | re.DOTALL)
        if match_razon:
            razon = match_razon.group(1).strip()
        else:
            razon = re.sub(r"(?i)\**DECISION[:\*]?\s*\d\**\s*", "", content).strip()
    
        razon = razon.replace("\n", " ").replace("\r", " ")

        print("PARSED DECISION:", decision)
        print("PARSED RAZON:", razon[:150])
        print("="*40)
        
    except Exception as e:
        print("Error:", e)

test_llamar_ollama(
    "finca",
    "Asimismo se informó sobre otra escena en la finca Montecristo en Tepecoyo, La Libertad.",
    "[Sustantivo] Propiedad rústica o urbana."
)

test_llamar_ollama(
    "miércoles",
    "Desde las 8:00 de la noche de este martes hasta las 5:00 de la mañana del miércoles estará restringido...",
    "[sust(m/f)/adj] Día de la semana."
)

test_llamar_ollama(
    "doble",
    "Un doble homicidio se ha reportado este viernes en el caserío Los Trejos...",
    "[Sustantivo] Que contiene dos veces una cantidad."
)
