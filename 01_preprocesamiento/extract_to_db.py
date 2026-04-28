import fitz
import re
import sqlite3

DB_PATH = "diccionario_cubanismos.db"

# Marcadores de categoría gramatical válidos en este diccionario
# Nota: 'I' y 'II' son numerales romanos dentro de la definición, NO categorías
CAT_MARKERS = r'(m/f|sust\([^)]+\)/adj|sust\([^)]+\)|sust|m|f|v|adj|adv|pron|interj|prep|conj)'

# Una nueva entrada del diccionario comienza así (al inicio del texto del bloque):
# LEMA [LEMA...] CATEGORIA_GRAMATICAL texto...
# El lema puede incluir letras, tildes, guiones, '~', ':', ',', '!', '?'
# y tiene una longitud razonable (máx. ~50 chars) antes de la categoría.
NEW_ENTRY_RE = re.compile(
    r'^([A-Za-záéíóúüñÁÉÍÓÚÜÑ~:,!?\-\s()./\\"\']{1,50}?)\s+' + CAT_MARKERS + r'(\s+.*)?$'
)

# Líneas que deben descartarse (cabeceras, números de página, letras de sección aisladas)
SKIP_LINE_RE = re.compile(
    r'^(\d+|[A-ZÁÉÍÓÚÜÑ]|Diccionario del Español|Madrid, Gredos|Gisela Cárdenas|Antonia María|Tristá Pérez)$'
)


def extract_text_from_pdfs(pdf_paths):
    """Extrae el texto de todos los PDFs, limpiando cabeceras/pies de página."""
    full_text = ""
    for pdf_path in pdf_paths:
        print(f"Leyendo {pdf_path}...")
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text = page.get_text()
                lines = text.split('\n')
                cleaned = []
                for line in lines:
                    stripped = line.strip()
                    # Descartar líneas que son sólo números de página, letras de sección o cabeceras
                    if not stripped or SKIP_LINE_RE.match(stripped):
                        # Insertar línea vacía para conservar separadores de entrada
                        cleaned.append('')
                    else:
                        cleaned.append(line)
                full_text += '\n'.join(cleaned) + '\n'
        except Exception as e:
            print(f"Error procesando {pdf_path}: {e}")
    return full_text


def parse_entries(full_text):
    """Divide el texto en entradas de diccionario, uniendo las que se cortan entre páginas."""
    raw_blocks = re.split(r'\n{2,}', full_text)

    entries = []
    current = ""

    for block in raw_blocks:
        text = ' '.join(block.split())
        if not text or len(text) < 3:
            continue

        if NEW_ENTRY_RE.match(text):
            if current:
                entries.append(current)
            current = text
        else:
            # Continuación de la entrada anterior (corte de página)
            if current:
                current += ' ' + text

    if current:
        entries.append(current)

    return entries


def parse_entry_fields(entry):
    """Extrae lema, categoría gramatical, definición y ejemplos de una entrada."""
    match = NEW_ENTRY_RE.match(entry)
    if not match:
        return entry.strip(), '', entry.strip(), ''

    lema = match.group(1).strip()
    cat_gramatical = match.group(2).strip()
    definicion = (match.group(3) or '').strip()

    # Extraer ejemplos desde llaves {} y equivalencias desde corchetes []
    ejemplos_list = re.findall(r'(\{[^}]+\})', definicion)
    equivalencias_list = re.findall(r'(\[[^\]]+\])', definicion)
    ejemplos = ' | '.join(ejemplos_list + equivalencias_list)

    return lema, cat_gramatical, definicion, ejemplos


def parse_and_insert(full_text):
    entries = parse_entries(full_text)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Limpiar tabla y resetear autoincrement
    cursor.execute('DELETE FROM cubanismos')
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='cubanismos'")

    inserted_count = 0
    for entry in entries:
        lema, cat, definicion, ejemplos = parse_entry_fields(entry)
        cursor.execute(
            'INSERT INTO cubanismos (lema, categoria_gramatical, definicion, ejemplo) VALUES (?, ?, ?, ?)',
            (lema, cat, definicion, ejemplos)
        )
        inserted_count += 1

    conn.commit()
    conn.close()
    print(f'Extracción completada. {inserted_count} registros insertados en {DB_PATH}.')


if __name__ == '__main__':
    pdfs = ['CUBA1.pdf', 'CUBA2.pdf']
    text = extract_text_from_pdfs(pdfs)
    parse_and_insert(text)
