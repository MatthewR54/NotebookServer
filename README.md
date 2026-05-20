# 🤖 Interfaz de Clasificación de Imágenes de tipo noticia (Reales vs. Sintéticamente Falsas)

Este repositorio incluye el código fuente de la interfaz realizada como entregable no obligatorio del Proyecto de Grado titulado "Comparación del rendimiento de modelos de aprendizaje profundo para distinguir entre imágenes de noticias reales y falsas generadas por IA en redes sociales".

## **Características de la aplicación**

**Carga de Archivo:**

- Se aceptan archivos de imagen (.jpg, .jpeg, .png) y archivos comprimidos con multiples imagenes (.zip).
- Se muestra dinámicamente "Tipo de Tarea: Prediccion" (para imágenes) o "Tipo de Tarea: Clasificacion" (para archivos comprimidos)
- Vista previa de la imagen cuando se carga.

**Listado de Modelos:**

- Escaneo automático de la carpeta "models" para buscar modelos de Deep Learning.
- Identifica todos los directorios que sigan el patrón ```mirage_[autor]_[modelo]```.
- Los modelos encontrados se mostrarán en una lista.

**Selección de Modelos:**

- Checkboxes para dar la opción de seleccionar cada modelo disponible, con etiquetas legibles.
- Muestra contador de modelos seleccionados.

**Botón de Ejecución:**

- Botón grande y destacado "🚀 Execute Analysis".
- Valida que haya archivo cargado y modelos seleccionados.

**Predicción:**

- Carga los modelos seleccionados dinámicamente.
- Hace predicciones para cada modelo.
- Muestra resultados con: predicción (real=0 o fake=1), probabilidad y confianza.
- Interfaz visualizada en métricas de Streamlit.

## **Cómo ejecutar la aplicación**

```bash
# Asegúrate de estar en la carpeta correcta
cd C:\Tu\Directorio\Aqui\NotebookServer

# Ejecutar la aplicación
streamlit run streamlit_app.py
```
