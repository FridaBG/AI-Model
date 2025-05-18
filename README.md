# Clasificador de Imágenes

## Generación del set de datos

### Obtener set de datos

El conjunto de datos utilizado fue tomado de Kaggle:

🔗 [_Sports Balls – Multiclass Image Classification_](https://www.kaggle.com/datasets/samuelcortinhas/sports-balls-multiclass-image-classification/data)

Aunque el dataset original incluye una variedad más amplia de clases, se seleccionaron únicamente **5 categorías** con el objetivo de reducir el tiempo de entrenamiento y enfocarse en un problema de clasificación multiclase más acotado:

- 🏈 Fútbol Americano
- 🎳 Boliche
- 🏏 Cricket
- ⚽ Fútbol Soccer
- 🎾 Tenis

Estas clases fueron elegidas por su **diversidad visual** y porque presentan un buen desafío para modelos de clasificación sin ser demasiado complejas.

### Set de entrenamiento y pruebas

Train
Test
Validation

## Preprocesado de los datos

### Escalamiento y Preprocesado de datos

_ESCALAMIETO:_
`rescale=1./255`
Convierte los valores de píxeles de [0, 255] a [0, 1], esto:

- Acelera el entrenamiento
- Mejora la estabilidad del modelo

_PROCESADO DE DATOS:_
Técnicas de preprocesamiento o aumento de datos para mejorar la generalización del modelo:

- `rotation_range=30`: gira aleatoriamente la imagen
- `width_shift_range=0.2`: desplaza horizontalmente
- `height_shift_range=0.2`: desplaza verticalmente
- `shear_range=0.2`: aplica corte en la imágen
- `zoom_range=0.3`: hace zoom aleatorio
- `brightness_range=[0.8, 1.2]`: cambia el brillo
- `fill_mode='nearest'`: rellena los huecos generados por estas transformaciones

[Documentación Tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
