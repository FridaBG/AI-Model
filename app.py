import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar modelo
model = tf.keras.models.load_model("modelo_bien.h5")
IMG_SIZE = (150, 150)
CLASSES = ['billiard', 'bowling', 'cricket', 'football', 'golf', 'soccer', 'tennis']

# Preprocesamiento
def preprocess(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Función de predicción (¡ya no regresa la imagen!)
def predict_image(img):
    processed = preprocess(img)
    predictions = model.predict(processed)[0]
    return {CLASSES[i]: float(predictions[i]) for i in range(len(CLASSES))}

# Interfaz
with gr.Blocks(css="""
    body {
        background-color: #027353;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .container {
        max-width: 500px;
        margin: auto;
    }
    .gr-box, .gr-image, .gr-label {
        margin: auto !important;
    }
    .gr-button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 10px;
        padding: 12px 24px;
        font-size: 16px;
        margin: 5px;
    }
    .clear-btn {
        background-color: #d9534f !important;
        color: white !important;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 15px;
        margin: 5px;
    }
""") as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown("## 🎾 Clasificador de Balones Deportivos")
        gr.Markdown("""
        <p style='text-align:center; max-width: 600px; margin: auto'>
        Sube una imagen y deja que el modelo detecte el deporte correspondiente.<br>
        <b>Compatible con:</b> <i>billar, boliche, cricket, football, golf, soccer, tennis</i>.
        </p>
        """)

        image_input = gr.Image(type="pil", label="📷 Sube tu imagen")
        label_output = gr.Label(num_top_classes=3, label="🎯 Predicción")

        with gr.Row():
            clear_btn = gr.Button("🧹 Limpiar todo", elem_classes="clear-btn")
            submit_btn = gr.Button("🔍 Analizar", elem_classes="clear-btn")

        # Botón analizar (ya no usa image_output)
        submit_btn.click(
            fn=predict_image,
            inputs=image_input,
            outputs=label_output
        )

        # Botón limpiar
        clear_btn.click(
            fn=lambda: (None, None),
            outputs=[image_input, label_output]
        )

demo.launch(share=True)
