import requests
import json
import sys

def test_stream():
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen3.5:2b",
        "prompt": "Responde solo siguiendo el formato.\nPalabra: asere\nOración: que bola asere\nDefinición: [Sustantivo] Amigo\n\nSi es cubanismo escribe: DECISION: 1\nSi no lo es escribe: DECISION: 0\nRAZON: [explicación breve]",
        "stream": True,
        "options": {"temperature": 0.0}
    }

    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        content = ""
        sys.stdout.write("  [Ollama]: ")
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                text = chunk.get("response", "")
                content += text
                sys.stdout.write(text)
                sys.stdout.flush()
        print("\n\nFINAL CONTENT CAPTURED:")
        print(repr(content[:100] + "..."))
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_stream()
