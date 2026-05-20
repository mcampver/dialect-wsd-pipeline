import pdfplumber
import docx
import io

def extract_text_from_file(uploaded_file):
    """
    Extrae texto de archivos TXT, PDF o DOCX subidos vía Streamlit.
    """
    file_name = uploaded_file.name
    file_ext = file_name.split('.')[-1].lower()
    
    texto = ""
    try:
        if file_ext == "txt":
            texto = uploaded_file.getvalue().decode("utf-8")
        elif file_ext == "pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                paginas = [page.extract_text() for page in pdf.pages if page.extract_text() is not None]
                texto = "\n".join(paginas)
        elif file_ext in ["doc", "docx"]:
            # Usar io.BytesIO para leer el buffer
            doc = docx.Document(io.BytesIO(uploaded_file.getvalue()))
            texto = "\n".join([para.text for para in doc.paragraphs])
        else:
            texto = f"Formato no soportado: {file_ext}"
            
    except Exception as e:
        texto = f"[Error al procesar el archivo {file_name}: {str(e)}]"
        
    return texto