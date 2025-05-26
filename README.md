# Clasificador de Imágenes

&nbsp;

## 🖼️ Generación del set de datos

### 1. Obtener set de datos

El conjunto de datos utilizado fue tomado de Kaggle: 🔗 [_Sports Balls – Multiclass Image Classification_](https://www.kaggle.com/datasets/samuelcortinhas/sports-balls-multiclass-image-classification/data)

Aunque el dataset original incluye una variedad más amplia de clases, se seleccionaron únicamente **4 categorías** con el objetivo de reducir el tiempo de entrenamiento, se seleccionaron las siguientes clases:

- 🏈 Football
- 🏏 Cricket
- ⚽ Soccer
- 🎾 Tenis

Se agregaron imágenes extra a todas las clases para asegurar un buen balance y tener un mejor rendimiento al aumentar el dataset.

> **¿Está balanceado?** Sí, hay la misma cantidad de imágenes por clase (840 cada una)

> **¿Es correcto?** Sí, las imágenes están correctamente etiquetadas y clasificadas.

> **¿Es representativo?** Sí, incluye variedad suficiente de imágenes para cada clase.

### 2. División del det de datos

  <img src="https://velog.velcdn.com/images/iguv/post/8ae842e3-f2b6-44c5-b7bf-a1f74b3a9124/image.png" width="300"/>
  
  Para entrenar un modelo que realmente aprenda a clasificar y no solo memorice, es importante dividir el dataset en tres partes:

- **Training**
  - _700 imágenes por clase (80%)_
  - Aquí es donde el modelo “aprende” haciendo los ajustes internos necesarios para reducir el error.
  - Entrena directamente el modelo
- **Validation**
  - _70 imágenes por clase (10%)_
  - Se usa mientras el modelo entrena, pero no se le muestra nunca para aprender.
    Ayuda a medir si el modelo está empezando a "overfit" y permite ajustar o detener el entrenamiento en el mejor momento.
  - Sirve para monitorear el rendimiento sin hacer trampa.
- **Testing**
  - _70 imágenes por clase (10%)_
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
- `width_shift_range=0.1`: desplaza horizontalmente
- `height_shift_range=0.1`: desplaza verticalmente
- `shear_range=0.2`: aplica corte en la imágen
- `zoom_range=0.4`: hace zoom aleatorio
- `brightness_range=[0.8, 1.1]`: cambia el brillo
- `fill_mode='nearest'`: rellena los huecos generados por estas transformaciones

  <img src="https://i.postimg.cc/022CjTWN/Captura-de-pantalla-2025-05-24-a-la-s-7-00-36-p-m.png" width="800"/>

  🔗 [Documentación Tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)

## 🚧 Implementación del modelo

### 1. Seleccionar un modelo

Para esta implementación me basé en el modelo ICNN-BNDOA propuesto en el artículo **"Improved CNN Based on Batch Normalization and Adam"** publicado en Advances in Intelligent Systems and Computing (Springer). Esta arquitectura se diseñó específicamente para mejorar el rendimiento de las CNN en tareas de clasificación mediante la combinación de tres técnicas principales:

- Batch Normalization (BN) para estabilizar y acelerar el entrenamiento.

- Dropout (DO) como técnica para mitigar el overfitting.

- Optimización con Adam, dada su eficiencia y adaptabilidad.

> “This research created a seven-layer CNN model that includes an input layer with BN and DO layers, two hidden layers with BN and DO layers composed of convolution and pool layers, a flatten layer with DO layer (0.3), a fully connected layer, two dense layers with BN and DO layers, and an output layer with sigmoid activation.”

Esta propuesta utiliza una red con múltiples capas convolucionales intercaladas con BN y DO, seguida de capas densas totalmente conectadas con activación ReLU y salida sigmoid para tareas binarias (en mi caso, adaptada a softmax para clasificación multiclase).

### 2. Implementar el modelo seleccionado

Inspirado en la estructura general del modelo ICNN-BNDOA, implementé una red CNN adaptada a mi conjunto de datos. Dado el tamaño relativamente pequeño del dataset, decidí no incluir Batch Normalization, por las siguiente razón:

- **Peor rendimiento observado empíricamente:** al implementar BN después de cada capa convolucional (como lo sugiere el artículo), noté un descenso en la precisión tanto en el conjunto de validación como de prueba. Esto podría deberse a que BN introduce una estimación de media y varianza por batch, lo cual puede no generalizar bien en datasets pequeños, como indican varios estudios:

> "Batch normalization works best with large datasets and batch sizes, and its effectiveness can be limited when data is scarce or imbalanced." (Goodfellow et al., Deep Learning, 2016)

- En su lugar, utilicé una tasa de Dropout del 30%, lo cual proporcionó un mejor control del overfitting

La arquitectura final incluye:

- 3 bloques Conv2D + MaxPooling2D, con activación ReLU.
- Dropout(0.3) aplicado después de las dos últimas capas convolucionales.
- 1 capa Flatten para vectorizar.
- Dropout(0.3) adicional antes de la capa densa final.
- 1 capa Dense(64) con activación ReLU.
- 1 capa Dense final con activación softmax para clasificación en 4 clases.
- Entrenamiento con el optimizador Adam, con una tasa de aprendizaje de 1e-4.

Esta configuración fue seleccionada tras pruebas comparativas, logrando mejorar la precisión del modelo y reducir el riesgo de overfitting, sin necesidad de aumentar la complejidad de la arquitectura.

## 📈 Evaluación inicial del modelo

### 1. Seleccionar métricas adecuadas

Para evaluar el desempeño de mi modelo de clasificación multiclase, seleccioné la métrica **accuracy**. Esta elección está respaldada por el artículo **"A comprehensive survey of loss functions and metrics in deep learning"** , donde se destaca que:

> “Accuracy remains the most widely used metric for classification tasks, especially when the dataset is balanced and the misclassification costs are uniform across classes.”

Dado que mi conjunto de datos está equilibrado entre las cuatro clases y no existen penalizaciones diferenciadas por errores de clasificación, la métrica accuracy proporciona una evaluación clara y directa del rendimiento general del modelo.

En cuanto a la función de pérdida, utilicé **categorical_crossentropy**, que es la opción estándar para problemas de clasificación multiclase. Esta elección también está respaldada por el artículo mencionado anteriormente, que indica:

> “Categorical cross-entropy is the default loss function for multi-class classification tasks, offering a probabilistic interpretation and aligning well with the softmax activation in the output layer.”

Esta función de pérdida mide la divergencia entre las distribuciones de probabilidad predichas y las verdaderas, lo que la hace especialmente adecuada para modelos con una capa de salida softmax, como es el caso de mi implementación.

### 2. Reporte de resultados obtenidos e interpretación.

El modelo alcanzó un accuracy final en el conjunto de prueba de 0.789 (78.9%), como se muestra tanto en la métrica global como en las gráficas de desempeño. La curva de accuracy muestra una mejora constante en el entrenamiento, y aunque el data de validation es más inestable, ambas curvas están en valores altos, lo que indica que el modelo logra aprender sin caer en un overfitting grave.

En contraste, la curva de loss se comporta de forma distinta. Esto es normal, ya que la función de pérdida (categorical_crossentropy) no siempre se realciona directamente con el accuracy, especialmente en clasificación multiclase. Además la **val loss** fluctúa porque el conjunto de validación es pequeño, y basta que falle una clase para subir el valor de loss.

<img src="https://i.postimg.cc/Lsw1ngb9/Captura-de-pantalla-2025-05-25-a-la-s-5-43-05-p-m.png" width="400"/>
<img src="https://i.postimg.cc/nrYjRRcr/Captura-de-pantalla-2025-05-25-a-la-s-5-43-42-p-m.png" width="400"/>

La **matriz de confusión** muestra que el modelo logra un buen desempeño general, con las cuatro clases bien representadas. Tennis fue la clase mejor clasificada, con 62 de 70 ejemplos correctamente identificados. Football también mostró alta precisión, con 56 aciertos. Sin embargo, se observa que cricket fue confundida con soccer y tennis en 10 casos cada una, lo que indica similitudes visuales entre estas clases. Por otro lado, soccer presentó errores distribuidos entre las demás clases. Aun así, el modelo mantiene un buen equilibrio de clasificación entre todas las clases.

<img src="https://i.postimg.cc/65tG2XYv/Captura-de-pantalla-2025-05-25-a-la-s-5-44-26-p-m.png" width="400"/>

---

Ogundokun, R.O., Maskeliunas, R., Misra, S., Damaševičius, R. (2022). Improved CNN Based on Batch Normalization and Adam Optimizer. In: Gervasi, O., Murgante, B., Misra, S., Rocha, A.M.A.C., Garau, C. (eds) Computational Science and Its Applications – ICCSA 2022 Workshops. ICCSA 2022. Lecture Notes in Computer Science, vol 13381. Springer, Cham. https://doi.org/10.1007/978-3-031-10548-7_43

Terven, J., Cordova-Esparza, DM., Romero-González, JA. et al. A comprehensive survey of loss functions and metrics in deep learning. Artif Intell Rev 58, 195 (2025). https://doi.org/10.1007/s10462-025-11198-7
