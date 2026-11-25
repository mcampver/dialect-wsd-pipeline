import io
import os
import sqlite3
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
from annotated_text import annotated_text

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logica
from utils_file_parsers import FileParseError, extract_text_from_file

# Añadir la carpeta de la aplicación al path de Python para importar DetectorCubano
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "06_aplicacion")))
from clasificador_final import DetectorCubano

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Detector  de Cubanismos", page_icon="🇨🇺", layout="wide")


# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
    <style>
    /* Fondo general blanco y texto oscuro de alto contraste */
    .main, [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF;
        color: #1A202C;
    }
    [data-testid="stSidebar"] {
        background-color: #F0F4F8;
        border-right: 2px solid #CBD5E0;
    }

    /* Tipografía más grande para facilitar la lectura */
    body, p, div, span, label {
        font-size: 16px !important;
        line-height: 1.6 !important;
    }
    h1 { font-size: 2rem !important; color: #1A56A0 !important; }
    h2, h3 { font-size: 1.4rem !important; color: #1A202C !important; }

    /* KPI Cards — fondo blanco con borde azul, texto oscuro legible */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 2px solid #1A56A0;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(26, 86, 160, 0.12);
    }
    div[data-testid="metric-container"] > label {
        color: #2D3748 !important;
        font-weight: 700;
        font-size: 14px !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #1A56A0 !important;
        font-size: 1.8rem !important;
        font-weight: 800;
    }

    /* Botones primarios con buen contraste */
    .stButton > button[kind="primary"] {
        background-color: #1A56A0;
        color: #FFFFFF;
        font-size: 16px !important;
        font-weight: 700;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #12407A;
    }

    /* Inputs y text areas con borde visible */
    .stTextArea textarea, .stTextInput input {
        border: 2px solid #CBD5E0 !important;
        border-radius: 6px !important;
        font-size: 16px !important;
        color: #1A202C !important;
        background-color: #FFFFFF !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #1A56A0 !important;
        box-shadow: 0 0 0 2px rgba(26, 86, 160, 0.2) !important;
    }

    /* Tabs con contraste claro */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #F0F4F8;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #2D3748 !important;
        font-weight: 600;
        font-size: 15px !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1A56A0 !important;
        color: #FFFFFF !important;
        border-radius: 6px;
    }

    /* Tooltip accesible */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 2px dotted #1A56A0;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 140px;
        background-color: #1A202C;
        color: #FFFFFF;
        text-align: center;
        border-radius: 6px;
        padding: 6px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -70px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 14px !important;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
""", unsafe_allow_html=True)

# --- INICIALIZAR EL MODELO (Singleton Decorator para no recrearlo en cada ejecución) ---
@st.cache_resource(show_spinner="Cargando modelos lingüísticos y cerebro de IA (BETO)... (puede tardar un minuto la primera vez)")
def load_model():
    return DetectorCubano()

detector = load_model()

TRADUCCION_POS = {"NOUN": "Sustantivo", "VERB": "Verbo", "ADJ": "Adjetivo", "INTJ": "Interjección",
                  "ADV": "Adverbio", "PRON": "Pronombre", "PROPN": "Nombre Propio"}

COLUMNAS_HALLAZGOS = {
    "palabra": st.column_config.TextColumn("Palabra Detectada", width="medium"),
    "equivalente": st.column_config.TextColumn("Estándar", width="medium"),
    "confianza": st.column_config.ProgressColumn(
        "Confianza IA",
        help="Probabilidad calculada por BETO",
        format="%.1f %%",
        min_value=0,
        max_value=100,
    ),
    "sentimiento": st.column_config.TextColumn("Sentimiento"),
    "pos": st.column_config.TextColumn("Categoría"),
    "dep": st.column_config.TextColumn("Función"),
    "lema": st.column_config.TextColumn("Lema"),
    "contexto": st.column_config.TextColumn("Oración", width="large"),
    "archivo": st.column_config.TextColumn("Archivo"),
}


def tabla_hallazgos(df, columnas):
    df = df.copy()
    df["pos"] = df["pos"].map(lambda p: TRADUCCION_POS.get(p, p))
    st.dataframe(df, width="stretch", hide_index=True,
                 column_order=columnas, column_config=COLUMNAS_HALLAZGOS)


# --- HEADER APP ---
st.title("🇨🇺 Analizador Lexicométrico de Cubanismos")
st.markdown("Herramienta de asistencia apoyada en IA para identificar cubanismos en textos en tiempo real. **Desarrollado para la tesis.**")

# --- MENÚ LATERAL (TABS O SIDEBAR) ---
st.sidebar.title("Navegación")
opcion = st.sidebar.radio("Elige una herramienta:",
                          ["📝 Grammarly Rápido", "📚 Laboratorio Batch (Múltiples Archivos)", "➕ Gestión del Diccionario"])

st.sidebar.markdown("---")
st.sidebar.info("Este modelo evalúa el contexto morfosintáctico de la palabra para determinar si está siendo utilizada como un **cubanismo** en la oración.")

if opcion == "📝 Grammarly Rápido":
    st.header("Análisis Rápido de Texto")

    texto_input = st.text_area("Introduce tu texto aquí:", height=200,
                               placeholder="Ejemplo: Ayer esperé la guagua por una hora, asere.")

    if st.button("Analizar Texto", type="primary"):
        if not texto_input.strip():
            st.warning("Por favor, introduce un texto para analizar.")
        else:
            with st.spinner("Analizando morfosintaxis..."):
                hallazgos = detector.analizar_texto(texto_input)

            if not hallazgos:
                st.success("No se encontraron cubanismos en el texto analizado.")
            else:
                st.success(f"Se encontraron {len(hallazgos)} posibles cubanismo(s).")
                st.markdown("### Resultado Interactivo")
                annotated_text(*logica.construir_anotaciones(texto_input, hallazgos))

                st.markdown("### Detalles de Extracción")
                tabla_hallazgos(pd.DataFrame(hallazgos),
                                ["palabra", "equivalente", "confianza", "sentimiento", "pos", "contexto"])

elif opcion == "📚 Laboratorio Batch (Múltiples Archivos)":
    st.header("Procesamiento por Lotes (Laboratorio)")
    st.markdown("Sube múltiples archivos `.txt`, `.pdf` o `.docx` y obtén analíticas agregadas.")

    uploaded_files = st.file_uploader("Arrastra aquí tus archivos", type=['txt', 'pdf', 'docx'], accept_multiple_files=True)

    if uploaded_files and st.button("Procesar Archivos", type="primary"):
        resultados_globales = []
        errores = []
        conteo_textos = 0
        total_palabras = 0

        progress_bar = st.progress(0.0)
        status_text = st.empty()
        n_archivos = len(uploaded_files)

        for i, file in enumerate(uploaded_files):
            status_text.text(f"Procesando: {file.name} | Extrayendo texto...")
            try:
                texto = extract_text_from_file(file)
            except FileParseError as e:
                errores.append(f"{file.name}: {e}")
                progress_bar.progress((i + 1) / n_archivos)
                continue

            total_palabras += len(texto.split())

            def progreso(hechas, total, nombre=file.name, i=i):
                if hechas % 5 == 0 or hechas == total:  # no saturar la UI
                    status_text.text(f"Procesando: {nombre} | Analizando oraciones... ({hechas}/{total})")
                    progress_bar.progress(min(1.0, (i + hechas / max(1, total)) / n_archivos))

            for h in detector.analizar_texto(texto, on_progreso=progreso):
                h['archivo'] = file.name
                resultados_globales.append(h)

            conteo_textos += 1
            progress_bar.progress((i + 1) / n_archivos)

        status_text.text("¡Procesamiento Completado!")
        # Guardar en sesión: el botón de descarga provoca un rerun y, sin esto,
        # los resultados desaparecían al exportar el CSV.
        st.session_state["batch"] = {
            "resultados": resultados_globales,
            "errores": errores,
            "documentos": conteo_textos,
            "palabras": total_palabras,
        }

    batch = st.session_state.get("batch")
    if batch:
        for err in batch["errores"]:
            st.error(f"❌ No se pudo procesar {err}")

        if batch["documentos"] == 0:
            pass
        elif len(batch["resultados"]) == 0:
            st.info("No se encontraron cubanismos en los documentos subidos.")
        else:
            df_resultados = pd.DataFrame(batch["resultados"])
            total_palabras = batch["palabras"]

            # -- DASHBOARDS MACRO --
            st.header("📊 Macro-Estadísticas")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total de Documentos", batch["documentos"])
            col2.metric("Total de Palabras Procesadas", f"{total_palabras:,}")
            densidad = logica.densidad_por_mil(len(df_resultados), total_palabras)
            col3.metric("Densidad Dialectal (por 1000 pals)", f"{densidad:.2f}")

            st.markdown("---")

            col_chart1, col_chart2 = st.columns(2)

            # Frecuencias Top 10
            with col_chart1:
                st.subheader("Top 10 Cubanismos más Usados")
                top_10 = df_resultados['lema'].str.lower().value_counts().head(10).reset_index()
                top_10.columns = ['Cubanismo', 'Frecuencia']
                fig_bar = px.bar(top_10, x='Cubanismo', y='Frecuencia', color='Frecuencia', color_continuous_scale="Blues")
                fig_bar.update_layout(
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    font=dict(color="#1A202C", size=13),
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_title="",
                    yaxis_title="Ocurrencias",
                    xaxis=dict(tickfont=dict(color="#1A202C")),
                    yaxis=dict(tickfont=dict(color="#1A202C"), gridcolor="#E2E8F0"),
                )
                st.plotly_chart(fig_bar, width="stretch")

            # Distribución de Sentimiento
            with col_chart2:
                st.subheader("Análisis de Sentimiento Contextual")
                sent_counts = df_resultados['sentimiento'].value_counts().reset_index()
                sent_counts.columns = ['Contexto Emocional', 'Cantidad']
                color_map = {"Positivo": "#4A90E2", "Neutro": "#8795A1", "Negativo": "#2C3E50"}
                fig_pie = px.pie(sent_counts, values='Cantidad', names='Contexto Emocional', color='Contexto Emocional', color_discrete_map=color_map, hole=0.5)
                fig_pie.update_layout(
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    font=dict(color="#1A202C", size=13),
                    margin=dict(l=20, r=20, t=30, b=20),
                    legend=dict(font=dict(color="#1A202C")),
                )
                st.plotly_chart(fig_pie, width="stretch")

            # Tabla Completa
            st.subheader("Base de Datos Generada (Gold Standard)")
            tabla_hallazgos(df_resultados, ["archivo", "palabra", "equivalente", "confianza",
                                            "sentimiento", "pos", "dep", "lema", "contexto"])

            # Exportador
            csv = df_resultados.drop(columns=["inicio", "fin"]).to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Exportar Resultados a CSV",
                data=csv,
                file_name='analisis_cubanismos.csv',
                mime='text/csv',
                on_click="ignore",
            )

elif opcion == "➕ Gestión del Diccionario":
    st.header("➕ Gestión y Crecimiento del Diccionario")
    st.markdown("Alimenta la base de datos con nuevos cubanismos. Nuestro modelo aprenderá inmediatamente de tus aportes.")

    tab_search, tab1, tab2 = st.tabs(["🔍 Explorador / Buscador", "✍️ Añadir Manualmente", "📂 Carga Masiva (Excel/CSV)"])

    with tab_search:
        st.subheader("Buscador Interactivo de Cubanismos")
        search_query = st.text_input("🔍 Buscar palabra, definición o traducción...", "")

        try:
            conn = sqlite3.connect(logica.ruta_db())
            try:
                df_db = logica.buscar(conn, search_query)
                total_db = conn.execute("SELECT COUNT(*) FROM cubanismos").fetchone()[0]
            finally:
                conn.close()

            col_metric1, col_metric2 = st.columns(2)
            col_metric1.metric("Resultados de la Búsqueda", len(df_db))
            col_metric2.metric("Total en Diccionario Completo", total_db)

            st.dataframe(df_db, width="stretch", hide_index=True)

        except Exception as e:
            st.error(f"Error al conectar con la base de datos: {e}")

    with tab1:
        st.subheader("Registrar un nuevo dialectismo")
        with st.form("form_nuevo_cubanismo"):
            lema = st.text_input("Lema / Palabra Original (Requerido)*", placeholder="Ej: fula")
            definicion = st.text_area("Definición de la RAE / Academia (Requerido)*", placeholder="Ej: Moneda estadounidense; persona falsa o de mala calidad.")
            traduccion = st.text_input("Traducción / Equivalente Estándar (Opcional)", placeholder="Ej: dólar / falso")
            ejemplo = st.text_input("Ejemplo de uso (Opcional)", placeholder="Ese tipo es un fula.")

            submitted = st.form_submit_button("Guardar en Base de Datos")
            if submitted:
                if not lema.strip() or not definicion.strip():
                    st.error("❌ El LEMA y la DEFINICIÓN son campos obligatorios.")
                else:
                    fila = (lema.strip().lower(), definicion.strip(), traduccion.strip(), ejemplo.strip())
                    try:
                        conn = sqlite3.connect(logica.ruta_db())
                        try:
                            insertadas, _ = logica.insertar_entradas(conn, [fila])
                        finally:
                            conn.close()

                        if not insertadas:
                            st.warning(f"⚠️ '{fila[0]}' ya existe con esa misma definición. No se ha duplicado.")
                        else:
                            # Actualizar en memoria el diccionario del detector
                            logica.registrar_en_memoria(detector.candidatos_traduccion, fila[0], fila[2])
                            if " " in fila[0]:
                                st.info("ℹ️ El detector analiza palabra a palabra: las locuciones de varias palabras se guardan pero no se detectan en el texto.")
                            st.success(f"✅ ¡El cubanismo '{fila[0]}' se ha guardado exitosamente!")
                            st.balloons()
                    except Exception as e:
                        st.error(f"Error al guardar: {e}")

    with tab2:
        st.subheader("Carga Masiva de Cubanismos")
        st.info("Sube un Excel o CSV. El modelo buscará automáticamente las columnas requeridas para inyectarlas directamente al corpus.")

        # Plantilla al vuelo
        df_plantilla = pd.DataFrame({
            "Lema": ["asere", "guagua", "fula"],
            "Definicion": ["Amigo cercano.", "Autobús.", "Dinero / Persona falsa."],
            "Traduccion": ["amigo", "autobús", "falso"],
            "Ejemplo": ["¿Qué bolá asere?", "La guagua pasó llena.", "Esa gente son fula."]
        })

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_plantilla.to_excel(writer, index=False, sheet_name="Plantilla")

        col_down, col_info = st.columns([1, 2])
        with col_down:
            st.download_button(
                label="⬇️ Descargar Plantilla (.xlsx)",
                data=buffer.getvalue(),
                file_name="plantilla_cubanismos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                on_click="ignore",
            )

        uploaded_db = st.file_uploader("Sube tu archivo validado", type=['xlsx', 'csv'])

        if uploaded_db and st.button("🚀 Inyectar Palabras al Corpus", type="primary"):
            try:
                if uploaded_db.name.lower().endswith('.csv'):
                    # sep=None detecta ',' o ';' (Excel en español exporta con ';')
                    df_nuevos = pd.read_csv(uploaded_db, sep=None, engine="python", encoding="utf-8-sig")
                else:
                    df_nuevos = pd.read_excel(uploaded_db)

                filas, omitidas = logica.normalizar_filas_carga(df_nuevos)
            except ValueError as e:
                st.error(f"❌ {e} Revisa el encabezado de tu archivo.")
            except Exception as e:
                st.error(f"Ocurrió un error leyendo el archivo: {e}")
            else:
                try:
                    conn = sqlite3.connect(logica.ruta_db())
                    try:
                        insertadas, duplicadas = logica.insertar_entradas(conn, filas)
                    finally:
                        conn.close()
                    for lema_row, _, trad_row, _ in insertadas:
                        logica.registrar_en_memoria(detector.candidatos_traduccion, lema_row, trad_row)

                    st.success(f"🎉 ¡Inyección completada! Se añadieron {len(insertadas)} nuevos cubanismos al conocimiento del IA.")
                    if duplicadas or omitidas:
                        st.warning(f"Se omitieron {duplicadas} fila(s) duplicadas y {omitidas} fila(s) sin lema o definición.")
                    if insertadas:
                        st.balloons()
                except Exception as e:
                    st.error(f"Ocurrió un error guardando en la base de datos: {e}")
