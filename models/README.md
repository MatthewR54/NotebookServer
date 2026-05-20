# **Pesos de modelos**

## **¿Dónde están?**

La interfaz utiliza los pesos seleccionados posterior al refinamiento descrito en el Capitulo 3 del proyecto. Estos checkpoints deben estar ubicados en una carpeta respectiva dentro del directorio ``models/``, donde dicha carpeta debe llevar el prefijo ``mirage_``, seguido del número del modelo y su nombre (Por ejemplo: El peso de un modelo ResNet50, dentro de una carpeta llamada ``mirage_model_1_resnet50``).

Debido a que subir todos los pesos a este repositorio es imposible por el peso elevado de estos (> 100 MB), se recomienda contactar al autor por correo para obtener las 6 carpetas correspondientes a los modelos que usa la interfaz.

## **Estructura de entradas**

La interfaz está diseñada para cargar imágenes (en formato PNG, JPG o JPEG) o archivos comprimidos (ZIP) que contengan imágenes. Los archivos comprimidos deben contener imágenes organizadas en subcarpetas que representen las clases (``real/``, ``fake/``). La interfaz procesará los contenidos de los archivos de entrada y mostrará las predicciones de los modelos seleccionados.
