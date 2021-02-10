# Super-resolución 4x:
## Modelado y Entranamiento:

Ejercicio de curso para la asignatura de Computación Inteligente perteneciente al Máster Universitario en Sistemas Inteligentes y Aplicaciones Numéricas para la Ingeniería (MUSIANI) en el curso 2020/21, realizado por Juan Sebastián Ramírez Artiles.

El ejercicio consiste en implementar el método de superresolución en imágenes descrito en el paper *[Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158)*. Concretamente, en [este notebook](Super_Resolution_CI_Model.ipynb) se ha implementado un escalado de cuatro aumententos. El dataset usado fue el [DIV2X](https://data.vision.ee.ethz.ch/cvl/DIV2K/) de libre descarga. Las imágenes usadas son las del track1 con un tamaño para las imágenes de alta resolución recortadas a 2040x1304.

Las imágenes se cargan de cuatro directorios, cada directorio debe contener un subdirectorio de etiqueta donde se situarán las imágenes, en este caso sólo habrá un subdirectorio, por lo tanto una etiqueta cada uno. Las etiquetas de los dataset se ignorarán ya que no se van a usar. Las imágenes reales se situan en train_y_hr y valid_y_hr y tienen un tamaño de 2040x1304. Las imágenes reducidas tienen un tamaño cuatro veces menor. Para el acondicionamiento de las imágenes se usó el notebook [Super_Resolution_CI_Preprocessed](Super_Resolution_CI_Preprocessed.ipynb).

El programa se ejecutó en una máquina con un procesador Intel Core i7-7700HQ a 2.80GHz con una tarjeta de vídeo NVIDIA GeForce GTX 1050 de 4GB de memoria dedicada y con 32 GB de memoria RAM.

El dataset se dividió en 611 imágenes para entrenamiento y 75 imágenes para validación.

El modelo hace uso de la capa PixelShuffle que se encarga de realizar el escalado a nivel subpíxel tomando como entrada un tensor de tamaño [Batch, Channels x R^2, H, W] y reordenándolo como [Batch, Channels, HxR, WxR]

Se probaron una variedad de modelos diferentes. Se usaron kernels de 3x3, de 5x5 y de 7x7, siendo estos últimos los que mejor resultados dieron. También se probaron diferentes configuraciones de red, añadiendo capas convolutivas y modificando las funciones de activación. Esta configuración resultó la más adecuada.

En el proceso de entrenamiento se usaron gran variedad de combinaciones de tamaños de batches y de cantidad de épocas. Se empezó con tamaños de batch de 8, se fue subiendo hasta 16 y finalmente se dejó en 14, más de esta cantidad desborda la memoras de la GPU. En cuanto al número de épocas, se empezó con 20 iteraciones y se fue subiendo hasta la cantidad de 80. No obstante, al variar el learning rate se aceleró la convergencia, con lo que volví a la veinte iteraciones.

Para finalizar, salvo el modelo para poder usarlo en el notebook [SR_restore_model.ipynb](SR_restore_model.ipynb) que se encargará de escalar las 75 imágenes de validación.