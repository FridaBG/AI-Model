import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Cargar modelo
model = load_model('mi_modelo.h5')

# Nombres de las clases (ajústalos si cambian)
class_labels = ['cricket', 'football', 'soccer', 'tennis']

# Preprocesar imagen como en el entrenamiento
def preprocess_image(image):
    image = image.resize((150, 150))  # Ajustar tamaño
    image = img_to_array(image)
    image = image / 255.0  # Escalado
    image = np.expand_dims(image, axis=0)  # Añadir dimensión batch
    return image

# Función para predecir
def predict(image):
    img = preprocess_image(image)
    pred_probs = model.predict(img)[0]
    result = {class_labels[i]: float(pred_probs[i]) for i in range(len(class_labels))}
    return result

# Interfaz
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=4),
    title="Clasificador de Balones Deportivos",
    description="Sube una imagen de una pelota (tenis, soccer, cricket, football) y el modelo la clasificará automáticamente."
)

demo.launch(share=True)