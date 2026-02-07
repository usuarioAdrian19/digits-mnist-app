if file:
    image = Image.open(file).convert('L')
    st.image(image, width=150, caption="Imagen Original cargada")
    
    # Quitamos el botón para que calcule directo al subir la foto (más estable)
    # O si prefieres botón, asegúrate de pulsarlo.
    if st.button("Detectar número"):
        
        # 1. Redimensionar a 28x28
        img_resized = image.resize((28, 28))
        
        # 2. Convertir a array para manipular
        img_array = np.array(img_resized)
        
        # 3. Invertir colores si el fondo es blanco
        # (El modelo necesita: Fondo Negro (0) y Número Blanco (255))
        if img_array.mean() > 127:
            st.warning("Se detectó fondo claro. Invirtiendo colores...")
            img_array = 255 - img_array
            
        # 4. Normalizar para el modelo (0 a 1)
        img_normalized = img_array.astype('float32') / 255.0
        
        # --- MOSTRAR LO QUE VE LA IA ---
        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("👁️ **La IA ve esto:**")
            # Mostramos la imagen procesada. Usamos clamp=True para asegurar que se ve
            st.image(img_normalized, width=100, clamp=True, channels='GRAY')
        with col2:
            st.info("Esta es la imagen de 28x28 píxeles que entra al modelo. Si aquí el número se ve mal, la predicción fallará.")
        st.divider()
        # -------------------------------

        # 5. Preparar forma para el modelo (1, 28, 28)
        model_input = img_normalized.reshape(1, 28, 28)
        
        # 6. Predicción
        pred = model.predict(model_input)
        resultado = np.argmax(pred)
        confianza = np.max(pred) * 100
        
        st.write(f"### Es un: {resultado}")
        st.write(f"Confianza: {confianza:.2f}%")
        st.bar_chart(pred[0])