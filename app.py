import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# 1. Configuración de la página
st.set_page_config(page_title="Digits App", layout="centered")

st.title("Reconocimiento de Dígitos (MNIST)")
st.write("Sube una imagen de un número para que la IA lo detecte.")

# 2. Cargar el modelo
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('digits_model.h5')

try:
    model = load_my_model()
    st.success("✅ Modelo cargado correctamente.")
except:
    st.error("❌ No se encuentra 'digits_model.h5'. Asegúrate de subirlo al repositorio.")

# 3. Subir la imagen (Aquí definimos la variable 'file')
file = st.file_uploader("Sube tu imagen", type=["jpg", "png", "jpeg"])

# 4. Lógica de predicción
if file:
    image = Image.open(file).convert('L')
    st.image(image, width=150, caption="Imagen Original cargada")
    
    if st.button("Detectar número"):
        
        # A. Redimensionar a 28x28
        img_resized = image.resize((28, 28))
        
        # B. Convertir a array
        img_array = np.array(img_resized)
        
        # C. Invertir colores si el fondo es blanco (El modelo espera fondo negro)
        if img_array.mean() > 127:
            st.warning("⚠️ Se detectó fondo claro. Invirtiendo colores para la IA...")
            img_array = 255 - img_array
            
        # D. Normalizar (0 a 1)
        img_normalized = img_array.astype('float32') / 255.0
        
        # --- E. MOSTRAR LO QUE VE LA IA ---
        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("👁️ **La IA ve esto:**")
            # Mostramos la imagen procesada. clamp=True asegura que se vea bien
            st.image(img_normalized, width=100, clamp=True, channels='GRAY')
        with col2:
            st.info("Esta es la imagen de 28x28 píxeles que entra al modelo. Si aquí el número se ve mal, la predicción fallará.")
        st.divider()
        # -------------------------------

        # F. Preparar forma para el modelo (1, 28, 28)
        model_input = img_normalized.reshape(1, 28, 28)
        
        # G. Predicción
        pred = model.predict(model_input)
        resultado = np.argmax(pred)
        confianza = np.max(pred) * 100
        
        st.write(f"### Es un: {resultado}")
        st.write(f"Confianza: {confianza:.2f}%")
        st.bar_chart(pred[0])