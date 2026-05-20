import streamlit as st
import pandas as pd
import sys
import os
import io
import sqlite3
import plotly.express as px
from annotated_text import annotated_text
from utils_file_parsers import extract_text_from_file

# --- CONFIGURAR RUTAS PARA IMPORTAR EL CLASIFICADOR Y DB ---
DB_CUBANISMOS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "03_datos", "diccionario_cubanismos.db"))

# Añadir la carpeta de la aplicación al path de Python para importar DetectorCubano
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "06_aplicacion")))
from clasificador_final import DetectorCubano

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Detector de Cubanismos", page_icon="🇨🇺", layout="wide")

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted black;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 120px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px 0;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* KPI Cards */
    div[data-testid="metric-container"] {
        background-color: #1E293B; /* Fondo azul oscuro/gris para modo oscuro */
        border: 1px solid #334155;
        padding: 5% 5% 5% 10%;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="metric-container"] > label {
        color: #94A3B8 !important; /* Letra más sutil para el título de la métrica */
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# --- INICIALIZAR EL MODELO (Singleton Decorator para no recrearlo en cada ejecución) ---
@st.cache_resource
def load_model():
    with st.spinner('Cargando modelos lingüísticos y cerebro de IA (BETO)... (puede tardar un minuto la primera vez)'):
        return DetectorCubano()

detector = load_model()

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
                hallazgos = detector.analizar_frase(texto_input)
                
                if not hallazgos:
                    st.success("No se encontraron cubanismos en el texto analizado.")
                else:
                    st.success(f"Se encontraron {len(hallazgos)} posibles cubanismo(s).")
                    
                    # Generar texto anotado
                    doc = detector.nlp(texto_input)
                    annotated_data = []
                    
                    # Para saber rápido si un token es hallazgo
                    indices_cubanismos = {h['palabra']: h for h in hallazgos}
                    
                    for token in doc:
                        if token.text in indices_cubanismos:
                            h = indices_cubanismos[token.text]
                            
                            # Traducción de etiquetas spaCy para el usuario final
                            traduccion_pos = {"NOUN": "Sustantivo", "VERB": "Verbo", "ADJ": "Adjetivo", "INTJ": "Interjección", "ADV": "Adverbio", "PRON": "Pronombre", "PROPN": "Nombre Propio"}
                            traduccion_dep = {"obj": "Objeto Directo", "nsubj": "Sujeto", "root": "Núcleo Oracional", "amod": "Modificador", "advmod": "Modificador"}
                            
                            pos_amigable = traduccion_pos.get(h['pos'], h['pos'])
                            dep_amigable = traduccion_dep.get(h['dep'], h['dep'])
                            sentimiento = h.get('sentimiento', 'Neutro')
                            equivalente = h.get('equivalente', 'N/A')
                            
                            # annotated_text format: ("texto", "anotacion", "background_color", "text_color")
                            # Anotación compacta y elegante
                            anotacion_corta = f"➔ {equivalente}"
                            annotated_data.append((token.text + token.whitespace_, anotacion_corta, "#E2E8F0", "#0F172A"))
                        else:
                            annotated_data.append(token.text + token.whitespace_)
                            
                    st.markdown("### Resultado Interactivo")
                    annotated_text(*annotated_data)
                    
                    st.markdown("### Detalles de Extracción")
                    df_hallazgos = pd.DataFrame(hallazgos)
                    st.dataframe(
                        df_hallazgos,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "palabra": st.column_config.TextColumn("Palabra Detectada", width="medium"),
                            "equivalente": st.column_config.TextColumn("Estándar", width="medium"),
                            "confianza": st.column_config.ProgressColumn(
                                "Confianza IA",
                                help="Probabilidad calculada por BETO",
                                format="%.1f %%",
                                min_value=0,
                                max_value=100,
                            ),
                            "sentimiento": st.column_config.TextColumn("Sentimiento")
                        }
                    )

elif opcion == "📚 Laboratorio Batch (Múltiples Archivos)":
    st.header("Procesamiento por Lotes (Laboratorio)")
    st.markdown("Sube múltiples archivos `.txt`, `.pdf` o `.docx` y obtén analíticas agregadas.")
    
    uploaded_files = st.file_uploader("Arrastra aquí tus archivos", type=['txt', 'pdf', 'docx'], accept_multiple_files=True)
    
    if uploaded_files and st.button("Procesar Archivos", type="primary"):
        resultados_globales = []
        conteo_textos = 0
        total_palabras = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Procesando: {file.name} | Extrayendo texto...")
            
            # Extraer el texto completo
            texto = extract_text_from_file(file)
            total_palabras += len(texto.split())
            
            doc_completo = detector.nlp(texto)
            oraciones = list(doc_completo.sents)
            total_oraciones = len(oraciones)
            
            for index_o, oracion in enumerate(oraciones):
                # Actualizar barra de progreso con el número de oración
                if index_o % 5 == 0:  # Actualizar cada 5 oraciones para no alentar la UI
                    status_text.text(f"Procesando: {file.name} | Analizando oraciones... ({index_o}/{total_oraciones})")
                    
                hallazgos_oracion = detector.analizar_frase(oracion.text)
                if hallazgos_oracion:
                    for h in hallazgos_oracion:
                        h['archivo'] = file.name
                        h['contexto'] = oracion.text
                        resultados_globales.append(h)
            
            conteo_textos += 1
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        status_text.text("¡Procesamiento Completado!")
        
        if len(resultados_globales) == 0:
            st.info("No se encontraron cubanismos en los documentos subidos.")
        else:
            df_resultados = pd.DataFrame(resultados_globales)
            
            # -- DASHBOARDS MACRO --
            st.header("📊 Macro-Estadísticas")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total de Documentos", conteo_textos)
            col2.metric("Total de Palabras Procesadas", f"{total_palabras:,}")
            densidad = len(df_resultados) / max(1, (total_palabras/1000))
            col3.metric("Densidad Dialectal (por 1000 pals)", f"{densidad:.2f}")
            
            st.markdown("---")
            
            col_chart1, col_chart2 = st.columns(2)
            
            # Frecuencias Top 10
            with col_chart1:
                st.subheader("Top 10 Cubanismos más Usados")
                top_10 = df_resultados['lema'].value_counts().head(10).reset_index()
                top_10.columns = ['Cubanismo', 'Frecuencia']
                fig_bar = px.bar(top_10, x='Cubanismo', y='Frecuencia', color='Frecuencia', color_continuous_scale="Blues")
                fig_bar.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", 
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_title="", 
                    yaxis_title="Ocurrencias"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
            # Distribución de Sentimiento
            with col_chart2:
                st.subheader("Análisis de Sentimiento Contextual")
                if 'sentimiento' in df_resultados.columns:
                    sent_counts = df_resultados['sentimiento'].value_counts().reset_index()
                    sent_counts.columns = ['Contexto Emocional', 'Cantidad']
                    color_map = {"Positivo": "#4A90E2", "Neutro": "#8795A1", "Negativo": "#2C3E50"}
                    fig_pie = px.pie(sent_counts, values='Cantidad', names='Contexto Emocional', color='Contexto Emocional', color_discrete_map=color_map, hole=0.5)
                    fig_pie.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=20, r=20, t=30, b=20)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            # Tabla Completa
            st.subheader("Base de Datos Generada (Gold Standard)")
            st.dataframe(
                df_resultados,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "palabra": st.column_config.TextColumn("Palabra Detectada", width="medium"),
                    "equivalente": st.column_config.TextColumn("Estándar", width="medium"),
                    "confianza": st.column_config.ProgressColumn(
                        "Confianza IA",
                        help="Probabilidad calculada por BETO",
                        format="%.1f %%",
                        min_value=0,
                        max_value=100,
                    ),
                    "sentimiento": st.column_config.TextColumn("Sentimiento")
                }
            )
            
            # Exportador
            csv = df_resultados.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Exportar Resultados a CSV",
                data=csv,
                file_name='analisis_cubanismos.csv',
                mime='text/csv',
            )

elif opcion == "➕ Gestión del Diccionario":
    st.header("➕ Gestión y Crecimiento del Diccionario")
    st.markdown("Alimenta la base de datos con nuevos cubanismos. Nuestro modelo aprenderá inmediatamente de tus aportes.")
    
    tab_search, tab1, tab2 = st.tabs(["🔍 Explorador / Buscador", "✍️ Añadir Manualmente", "📂 Carga Masiva (Excel/CSV)"])
    
    with tab_search:
        st.subheader("Buscador Interactivo de Cubanismos")
        search_query = st.text_input("🔍 Buscar palabra, definición o traducción...", "")
        
        try:
            conn = sqlite3.connect(DB_CUBANISMOS_PATH)
            if search_query.strip():
                exact = search_query.strip().lower()
                starts = f"{exact}%"
                contains = f"%{exact}%"
                
                sql_query = """
                    SELECT id, lema AS Lema, definicion AS Definición, traduccion AS Traducción, ejemplo AS Ejemplo 
                    FROM cubanismos 
                    WHERE lema IS NOT NULL AND (lema LIKE ? OR definicion LIKE ? OR traduccion LIKE ?)
                    ORDER BY 
                        CASE 
                            WHEN lema = ? THEN 1
                            WHEN lema LIKE ? THEN 2
                            WHEN lema LIKE ? THEN 3
                            ELSE 4
                        END, 
                        lema ASC
                """
                df_db = pd.read_sql_query(sql_query, conn, params=(contains, contains, contains, exact, starts, contains))
            else:
                # Mostrar total u overview reciente
                df_db = pd.read_sql_query(
                    "SELECT id, lema AS Lema, definicion AS Definición, traduccion AS Traducción, ejemplo AS Ejemplo "
                    "FROM cubanismos WHERE lema NOT NULL ORDER BY id DESC LIMIT 100", 
                    conn
                )
            
            # Obtener el total de la base de datos original
            total_db = pd.read_sql_query("SELECT COUNT(*) as total FROM cubanismos", conn).iloc[0]['total']
            conn.close()
            
            col_metric1, col_metric2 = st.columns(2)
            col_metric1.metric("Resultados de la Búsqueda", len(df_db))
            col_metric2.metric("Total en Diccionario Completo", total_db)
            
            st.dataframe(df_db, use_container_width=True, hide_index=True)
            
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
                    try:
                        conn = sqlite3.connect(DB_CUBANISMOS_PATH)
                        cursor = conn.cursor()
                        cursor.execute("INSERT INTO cubanismos (lema, definicion, traduccion, ejemplo) VALUES (?, ?, ?, ?)", 
                                       (lema.strip().lower(), definicion.strip(), traduccion.strip(), ejemplo.strip()))
                        conn.commit()
                        conn.close()
                        
                        # Actualizar en memoria el diccionario de nuestro modelo (DetectorCubano) actual
                        detector.candidatos_traduccion[lema.strip().lower()] = traduccion.strip() if traduccion.strip() else "Sin traducción"
                        
                        st.success(f"✅ ¡El cubanismo '{lema}' se ha guardado exitosamente!")
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
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        uploaded_db = st.file_uploader("Sube tu archivo validado", type=['xlsx', 'csv'])
        
        if uploaded_db and st.button("🚀 Inyectar Palabras al CorpusMasivo", type="primary"):
            try:
                if uploaded_db.name.endswith('.csv'):
                    df_nuevos = pd.read_csv(uploaded_db)
                else:
                    df_nuevos = pd.read_excel(uploaded_db)
                    
                # Validar la existencia de columnas
                columnas_validas = {c.lower().strip(): c for c in df_nuevos.columns}
                
                if "lema" not in columnas_validas or "definicion" not in columnas_validas:
                    st.error("❌ Tu Excel debe contener obligatoriamente las columnas 'Lema' y 'Definicion' en el encabezado.")
                else:
                    col_lema = columnas_validas['lema']
                    col_def = columnas_validas['definicion']
                    col_trad = columnas_validas.get('traduccion', None)
                    col_ejem = columnas_validas.get('ejemplo', None)
                    
                    conn = sqlite3.connect(DB_CUBANISMOS_PATH)
                    cursor = conn.cursor()
                    agregados = 0
                    
                    for index, row in df_nuevos.iterrows():
                        lema_row = str(row[col_lema]).strip().lower()
                        def_row = str(row[col_def]).strip()
                        
                        if pd.isna(row[col_lema]) or not lema_row or lema_row == 'nan':
                            continue
                            
                        trad_row = str(row[col_trad]).strip() if col_trad and not pd.isna(row[col_trad]) else ''
                        ejem_row = str(row[col_ejem]).strip() if col_ejem and not pd.isna(row[col_ejem]) else ''
                        
                        cursor.execute("INSERT INTO cubanismos (lema, definicion, traduccion, ejemplo) VALUES (?, ?, ?, ?)", 
                                       (lema_row, def_row, trad_row, ejem_row))
                        agregados += 1
                        
                        # Actualizar en memoria viva
                        detector.candidatos_traduccion[lema_row] = trad_row if trad_row else "Sin traducción"
                    
                    conn.commit()
                    conn.close()
                    
                    st.success(f"🎉 ¡Inyección completada! Se añadieron {agregados} nuevos cubanismos al conocimiento del IA.")
                    st.balloons()
            except Exception as e:
                st.error(f"Ocurrió un error leyendo o subiendo el archivo: {e}")