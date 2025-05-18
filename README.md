# Clasificador de Imágenes

&nbsp;

## 🖼️ Generación del set de datos

### 1. Obtener set de datos

El conjunto de datos utilizado fue tomado de Kaggle: 🔗 [_Sports Balls – Multiclass Image Classification_](https://www.kaggle.com/datasets/samuelcortinhas/sports-balls-multiclass-image-classification/data)

Aunque el dataset original incluye una variedad más amplia de clases, se seleccionaron únicamente **5 categorías** con el objetivo de reducir el tiempo de entrenamiento, se seleccionaron las siguientes clases:

- 🏈 Football
- 🎳 Boliche
- 🏏 Cricket
- ⚽ Soccer
- 🎾 Tenis

Se agregaron imágenes extra a la clase Football y Boliche para asegurar un buen balance entre las clases.

> **¿Está balanceado?** Sí, hay la misma cantidad de imágenes por clase (600 cada una)

> **¿Es correcto?** Sí, las imágenes están correctamente etiquetadas y clasificadas.

> **¿Es representativo?** Sí, incluye variedad suficiente de imágenes para cada clase.

### 2. División del det de datos

  <img src="https://velog.velcdn.com/images/iguv/post/8ae842e3-f2b6-44c5-b7bf-a1f74b3a9124/image.png" width="300"/>
  
  Para entrenar un modelo que realmente aprenda a clasificar y no solo memorice, es importante dividir el dataset en tres partes:

- **Training**
  - _500 imágenes por clase (80%)_
  - Aquí es donde el modelo “aprende” haciendo los ajustes internos necesarios para reducir el error.
  - Entrena directamente el modelo
- **Validation**
  - _50 imágenes por clase (10%)_
  - Se usa mientras el modelo entrena, pero no se le muestra nunca para aprender.
    Ayuda a medir si el modelo está empezando a "overfitting" y permite ajustar o detener el entrenamiento en el mejor momento.
  - Sirve para monitorear el rendimiento sin hacer trampa.
- **Testing**
  - _50 imágenes por clase (10%)_
  - Se guarda hasta el final. Es un grupo de datos nunca visto por el modelo ni durante el entrenamiento ni durante la validación.
  - Sirve para saber qué tan bien clasifica el modelo a datos nuevos.

## ⚙️ Preprocesado de los datos

### 1. Técnicas de escalamiento

- `rescale=1./255`: Convierte los valores de píxeles de [0, 255] a [0, 1]
- Acelera el entrenamiento
- Mejora la estabilidad del modelo

### 2. Preprocesado de los datos

Técnicas de preprocesamiento de datos para mejorar la categorización del modelo:

- `rotation_range=30`: gira aleatoriamente la imagen
- `width_shift_range=0.2`: desplaza horizontalmente
- `height_shift_range=0.2`: desplaza verticalmente
- `shear_range=0.2`: aplica corte en la imágen
- `zoom_range=0.3`: hace zoom aleatorio
- `brightness_range=[0.8, 1.2]`: cambia el brillo
- `fill_mode='nearest'`: rellena los huecos generados por estas transformaciones

  <img src="https://i.postimg.cc/XYycfTm8/Captura-de-pantalla-2025-05-18-a-la-s-2-42-15-p-m.png" width="800"/>


🔗 [Documentación Tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
