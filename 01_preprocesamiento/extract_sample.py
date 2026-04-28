import fitz

def extract_sample(pdf_path, output_path, num_pages=5):
    try:
        doc = fitz.open(pdf_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(min(num_pages, len(doc))):
                text = doc.load_page(i).get_text()
                f.write(f"--- PÁGINA {i+1} ---\n")
                f.write(text)
                f.write("\n")
        print(f"Muestra guardada en {output_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    extract_sample("CUBA1.pdf", "sample_cuba1.txt")
    extract_sample("CUBA2.pdf", "sample_cuba2.txt")
