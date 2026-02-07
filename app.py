import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="Digits App", layout="centered")

st.title("Reconocimiento de Dígitos (MNIST)")
st.write("Sube una imagen de un número o dibújalo en fondo blanco.")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('digits_model.h5')

try:
    model = load_my_model()
    st.success("Modelo cargado.")
except:
    st.error("No se encuentra 'digits_model.h5'. Súbelo al repositorio.")

file = st.file_uploader("Sube tu imagen", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file).convert('L')
    st.image(image, width=150, caption="Original")
    
    if st.button("Detectar número"):
        img = image.resize((28, 28))
        img_array = np.array(img)
        
        if img_array.mean() > 127:
            img_array = 255 - img_array
            
        img_array = img_array.astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28)
        
        pred = model.predict(img_array)
        resultado = np.argmax(pred)
        confianza = np.max(pred) * 100
        
        st.write(f"### Es un: {resultado}")
        st.write(f"Confianza: {confianza:.2f}%")
        st.bar_chart(pred[0])