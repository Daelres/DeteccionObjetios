import streamlit as st
import os
import PIL.Image

# Intento de importación segura para diagnóstico
try:
    import cv2
    import ultralytics
    OPENCV_STATUS = "✅ Sistema: Listo"
except ImportError as e:
    OPENCV_STATUS = f"❌ Error: {str(e)}"
    st.error(f"Error de dependencias: {e}")
    st.stop()

try:
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"Error crítico al cargar YOLO: {e}")
    st.stop()

# Configuración de la página
st.set_page_config(
    page_title="Control de Acceso PPE",
    page_icon="⛑️",
    layout="centered"
)

def load_model():
    model_path = 'modelo_ppe_yolov8.pt'
    if os.path.exists(model_path):
        try:
            # Diagnóstico de tamaño
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            if size_mb < 0.1:
                st.error(f"⚠️ El archivo del modelo parece estar corrupto o es un puntero (Tamaño: {size_mb:.4f} MB).")
                return None
            return YOLO(model_path)
        except EOFError:
            st.error("❌ Error de lectura (EOFError): El archivo del modelo está incompleto o dañado. Por favor, vuelve a subirlo a GitHub.")
            return None
        except Exception as e:
            st.error(f"❌ Error inesperado al cargar el modelo: {e}")
            return None
    return None

def main():
    # Estilos CSS para mejorar la interfaz
    st.markdown("""
        <style>
        .status-box {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .authorized {
            background-color: #d4edda;
            color: #155724;
            border: 2px solid #c3e6cb;
        }
        .denied {
            background-color: #f8d7da;
            color: #721c24;
            border: 2px solid #f5c6cb;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar simplificada
    st.sidebar.header("⚙️ Configuración")
    st.sidebar.write(OPENCV_STATUS)
    conf_threshold = st.sidebar.slider("Umbral de Confianza", 0.0, 1.0, 0.45)
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Regla de Negocio:**
    Para ingresar a la obra es obligatorio el uso de:
    - ⛑️ **Casco** (Helmet)
    - 🦺 **Chaleco** (Vest)
    """)

    # Título principal
    st.title("🛡️ Control de Acceso Inteligente")
    st.write("Verificación automática de Equipo de Protección Personal (EPP)")

    model = load_model()
    if model is None:
        st.error("⚠️ Archivo del modelo no encontrado.")
        return

    # Área de carga de imagen
    uploaded_file = st.file_uploader("Seleccione la imagen del trabajador...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = PIL.Image.open(uploaded_file)
        
        # Botón de acción centrado
        if st.button("🚀 Validar Requisitos de Ingreso", use_container_width=True):
            with st.spinner("Analizando imagen..."):
                results = model.predict(image, conf=conf_threshold)
                
                # Obtener imagen anotada
                res_plotted = results[0].plot()
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                # Procesar resultados
                classes = results[0].names
                counts = {}
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    cls_name = classes[cls_id]
                    counts[cls_name] = counts.get(cls_name, 0) + 1
                
                # Lógica de ingreso
                # Nota: Los nombres de clases en el modelo son 'helmet' y 'vest' en minúsculas
                tiene_casco = counts.get('helmet', 0) > 0
                tiene_chaleco = counts.get('vest', 0) > 0

                # Mostrar resultado visual
                st.image(res_rgb, caption='Resultado de la Detección', use_container_width=True)

                if tiene_casco and tiene_chaleco:
                    st.markdown('<div class="status-box authorized">✅ INGRESO PERMITIDO<br><small>Se detectó casco y chaleco correctamente.</small></div>', unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown('<div class="status-box denied">❌ INGRESO DENEGADO<br><small>Faltan elementos de seguridad obligatorios.</small></div>', unsafe_allow_html=True)
                    
                    faltantes = []
                    if not tiene_casco: faltantes.append("⛑️ Casco (Helmet)")
                    if not tiene_chaleco: faltantes.append("🦺 Chaleco (Vest)")
                    
                    st.error(f"**Atención:** Falta {', '.join(faltantes)}")

                # Mostrar otros elementos detectados
                with st.expander("Ver inventario completo de EPP detectado"):
                    if counts:
                        for item, qty in counts.items():
                            st.write(f"- **{item.capitalize()}**: {qty}")
                    else:
                        st.write("No se detectó ningún elemento de seguridad.")
    else:
        st.info("👆 Por favor, suba una fotografía para realizar la validación de seguridad.")

if __name__ == "__main__":
    main()
