import streamlit as st
import sys
import os

# Intento de importación segura para diagnóstico
try:
    import cv2
    import ultralytics
    OPENCV_STATUS = f"✅ OpenCV {cv2.__version__} | Ultralytics {ultralytics.__version__}"
except ImportError as e:
    OPENCV_STATUS = f"❌ Error de Importación: {str(e)}"
except Exception as e:
    OPENCV_STATUS = f"❌ Error inesperado: {str(e)}"

# Intentar cargar YOLO de forma que capturemos el error exacto de librería compartida
try:
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"Error crítico al cargar Ultralytics/OpenCV: {e}")
    st.info("Sugerencia: Asegúrate de que 'opencv-python-headless' esté en requirements.txt y no haya conflictos en packages.txt")
    st.stop()
import PIL.Image
import numpy as np
import cv2
import os

def load_model():
    model_path = 'modelo_ppe_yolov8.pt'
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

def main():
    st.set_page_config(page_title="Detección de Objetos PPE", page_icon="🔍")
    
    st.title("🛡️ Aplicación de Detección de PPE")
    st.write("Esta aplicación permite detectar equipos de protección personal en imágenes.")
    
    st.sidebar.header("⚙️ Configuración")
    st.sidebar.write(f"Sistema: {OPENCV_STATUS}")
    conf_threshold = st.sidebar.slider("Umbral de Confianza", 0.0, 1.0, 0.4)
    st.sidebar.markdown("---")
    opcion = st.sidebar.selectbox("Selecciona una opción", ["Inicio", "Detección", "Acerca de"])
    
    model = load_model()
    
    if model is None:
        st.error("⚠️ No se encontró el archivo del modelo 'modelo_ppe_yolov8.pt'. Asegúrate de entrenar el modelo y exportarlo primero.")
        st.stop()

    if opcion == "Inicio":
        st.subheader("Bienvenido")
        st.info("Navega a la sección de 'Detección' en el menú lateral para subir una imagen.")
        st.markdown("""
        ### Objetos que detecta el modelo:
        - 🦺 Chaleco (Vest)
        - 🥾 Calzado de Seguridad (Safety Shoe)
        - 😷 Tapabocas (Mask)
        - ⛑️ Casco (Helmet)
        - 🥽 Gafas (Goggles)
        - 🧤 Guantes (Gloves)
        """)
        
    elif opcion == "Detección":
        st.subheader("🕵️ Detección en Imágenes")
        
        uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = PIL.Image.open(uploaded_file)
            st.image(image, caption='Imagen subida', use_column_width=True)
            
            if st.button("Ejecutar Detección"):
                with st.spinner("Analizando imagen..."):
                    # Convertir imagen para YOLO
                    results = model.predict(image, conf=conf_threshold)
                    
                    # El primer resultado contiene la imagen anotada
                    res_plotted = results[0].plot()
                    
                    # Convertir BGR (OpenCV) a RGB (Streamlit/PIL)
                    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    
                    st.subheader("Resultado:")
                    st.image(res_rgb, caption='Detecciones realizadas', use_column_width=True)
                    
                    # Mostrar conteo
                    st.write("### Objetos detectados:")
                    classes = results[0].names
                    counts = {}
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        cls_name = classes[cls_id]
                        counts[cls_name] = counts.get(cls_name, 0) + 1
                    
                    if counts:
                        for name, count in counts.items():
                            st.write(f"- **{name}**: {count}")
                    else:
                        st.write("No se detectaron objetos PPE.")
            
    elif opcion == "Acerca de":
        st.subheader("Información del Proyecto")
        st.write("Este proyecto utiliza **YOLOv8** para la detección de Equipos de Protección Personal.")
        st.write("Desarrollado para la UNAB Digital.")

if __name__ == "__main__":
    main()
