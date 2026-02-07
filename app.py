import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")
st.title("🔢 Reconocimiento de Dígitos")
st.write("Dibuja un número del 0 al 9 en el recuadro y la IA intentará adivinarlo.")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('digits_model.h5')

model = load_my_model()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Dibuja aquí")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype(np.uint8)
    img = Image.fromarray(img).convert("L")
    img = img.resize((28, 28))
    
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    with col2:
        st.markdown("### Predicción")
        if st.button("Analizar dibujo"):
            prediction = model.predict(img_array)
            digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            st.metric(label="Número detectado", value=str(digit))
            st.write(f"Confianza: {confidence:.2%}")
            st.bar_chart(prediction[0])