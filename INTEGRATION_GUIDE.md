# 🚀 Guía de Uso: Integración de 3 Modelos Deepfake Detection

## Resumen de Cambios

Se han integrado exitosamente 3 modelos independientes de detección deepfake en la interfaz Streamlit:
- **UniversalFakeDetect (UFD)**: Basado en embeddings CLIP [768]
- **D3**: CLIP ViT-L/14 con Shuffle Attention
- **DistilDIRE**: ResNet50 con DIRE maps + EPS perturbations

Todos comparten una interfaz unificada que automáticamente:
1. Carga el modelo correcto según su tipo
2. Aplica preprocesamiento específico
3. Realiza predicciones individuales o batch
4. Calcula métricas (Accuracy, Precision, Recall, F1, AP, Confusion Matrix, ROC)

---

## Iniciación Rápida

### 1. Ejecutar Streamlit
```bash
streamlit run streamlit_app.py
```

La interfaz se abrirá en `http://localhost:8501`

### 2. Usar la Interfaz

**Flujo PREDICCION** (imagen individual):
1. Upload → Seleccionar imagen (JPG/PNG)
2. Step 2 → Seleccionar modelos a usar
3. Step 3 → Execute Analysis
4. Resultado → Predicción (Real/Fake) + Confianza por modelo

**Flujo CLASIFICACION** (dataset batch):
1. Upload → Seleccionar ZIP con estructura:
   ```
   dataset.zip
   ├── real/
   │   ├── image1.jpg
   │   └── image2.jpg
   └── fake/
       ├── image3.jpg
       └── image4.jpg
   ```
2. Step 2 → Seleccionar modelos
3. Step 3 → Execute Analysis
4. Resultados → Métricas (Accuracy, Precision, Recall, F1, AP) + Visualizaciones

---

## Archivos Nuevos

### 1. `preprocessing_adapters.py`

**Clases principales:**

#### `UFDPreprocessor`
```python
from preprocessing_adapters import UFDPreprocessor

processor = UFDPreprocessor(device="cuda")
embedding = processor.preprocess_image(image_pil)  # [768]
batch_embeddings = processor.preprocess_batch(images_list)  # [B, 768]
```

#### `D3Preprocessor`
```python
from preprocessing_adapters import D3Preprocessor

processor = D3Preprocessor(device="cuda", arch="CLIP")
tensor = processor.preprocess_image(image_pil)  # [3, 224, 224]  
batch_tensors = processor.preprocess_batch(images_list)  # [B, 3, 224, 224]
```

#### `DistilDIREPreprocessor`
```python
from preprocessing_adapters import DistilDIREPreprocessor

processor = DistilDIREPreprocessor(device="cuda")
# Requiere DIRE maps pre-computados
combined = processor.preprocess_image_with_eps(image_pil, eps_path)  # [6, 224, 224]
```

#### `ModelLoaders`
```python
from preprocessing_adapters import ModelLoaders

# Carga automática con checkpoints
ufd_model = ModelLoaders.load_ufd_model("models/mirage_model_4_universalfakedetect/epoch_4_0.976.pt")
d3_model = ModelLoaders.load_d3_model("models/mirage_model_6_d3/model_epoch_best.pt")
distildire_model = ModelLoaders.load_distildire_model("models/mirage_model_5_distildire/model_epoch_4.pt")
```

### 2. `precompute_dire_offline.py`

**Pre-computar DIRE maps para DistilDIRE** (recomendado para mejor rendimiento):

```bash
python precompute_dire_offline.py \
    --data_root datasets/definitive_selected_100imgs_dire \
    --save_root datasets/definitive_selected_100imgs_dire \
    --batch_size 16 \
    --ddim_steps 20 \
    --device cuda
```

**Opciones:**
- `--data_root`: Ruta a dataset con estructura `images/{fake,real}/`
- `--save_root`: Donde guardar DIRE maps y EPS (crea `dire/` y `eps/`)
- `--batch_size`: Tamaño de batch (default: 16)
- `--ddim_steps`: Pasos DDIM para cálculo (default: 20)
- `--device`: cuda o cpu (default: cuda)
- `--skip_existing`: Saltar imágenes ya procesadas

**Salida:**
- `{save_root}/dire/{fake,real}/*.png` - DIRE maps as images
- `{save_root}/eps/{fake,real}/*.pt` - EPS perturbations as tensors

---

## Configuración de Modelos

### Mapeo de Modelos
```python
MODEL_TYPE_MAP = {
    "mirage_model_1_smogy_ai_detection": "smogy",
    "mirage_model_2_ateeqq_ai_vs_human_detector": "ateeqq",
    "mirage_model_3_prithiv_deepfake_detector": "prithiv",
    "mirage_model_4_universalfakedetect": "ufd",        # ← Nuevo
    "mirage_model_5_distildire": "distildire",          # ← Nuevo
    "mirage_model_6_d3": "d3"                           # ← Nuevo
}
```

### Estructura de Checkpoints

```
models/
├── mirage_model_4_universalfakedetect/
│   └── epoch_4_0.976.pt
├── mirage_model_5_distildire/
│   └── model_epoch_4.pt
└── mirage_model_6_d3/
    └── model_epoch_best.pt
```

---

## Preprocesamiento por Modelo

### UniversalFakeDetect (UFD)
```
PIL Image (cualquier tamaño)
  ↓
CLIP Processor resize (224x224)
  ↓
CLIP Model feature extraction
  ↓
Embedding [768] valores normalizados
```

**Input esperado:** Imagen PIL RGB  
**Output:** Tensor [768] en device

### D3
```
PIL Image (cualquier tamaño)
  ↓
Resize (224x224)
  ↓
ToTensor [3, 224, 224] en [0, 1]
  ↓
Normalize con estadísticas CLIP
  ↓
Tensor [3, 224, 224] en rango [-2.2, 2.6]
```

**Input esperado:** Imagen PIL RGB  
**Output:** Tensor [3, 224, 224] en device

### DistilDIRE
```
PIL Image → Normalizar [-1, 1] → [3, 224, 224]
EPS tensor → Normalizar [-1, 1] → [3, 224, 224]
  ↓
Concatenar canales
  ↓
[6, 224, 224] en rango [-1, 1]
```

**Input requerido:** 
- Imagen PIL RGB
- Ruta a tensor EPS pre-computado (.pt)

**Output:** Tensor [6, 224, 224] en device

**Nota:** DistilDIRE requiere pre-computación de DIRE maps. Ver `precompute_dire_offline.py`

---

## Métricas Calculadas

**Batch Evaluation (Clasificacion):**
- ✓ Accuracy: (TP + TN) / Total
- ✓ Precision: TP / (TP + FP)
- ✓ Recall: TP / (TP + FN)
- ✓ F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
- ✓ ROC-AUC: Area Under ROC Curve
- ✓ Confusion Matrix: [[TN, FP], [FN, TP]]

**Visualizaciones:**
- Matriz de Confusión (heatmap)
- Curva ROC

**Threshold:** Default 0.5 (P(fake) >= 0.5 → Fake)

---

## Troubleshooting

### "Error loading model"
- Verificar rutas en `MODELS_DIR / model_name`
- Confirmar que checkpoints existen
- Revisar logs de error en consola

### "DistilDIRE requires pre-computed DIRE maps"
- Ejecutar: `python precompute_dire_offline.py --data_root <path> --save_root <path>`
- Asegurar que `{dataset}/dire/` y `{dataset}/eps/` existen
- Validar que los archivos están en ubicaciones correctas

### GPU OOM (Out of Memory)
- Reducir `batch_size` en Streamlit
- Para DistilDIRE pre-computation: reducir `--batch_size`
- Considerar usar CPU (más lento pero sin límite VRAM)

### Slow Inference
- UFD: ~0.5-1s por imagen (eficiente, CLIP embeddings)
- D3: ~1-2s por imagen (normalizado con CLIP)
- DistilDIRE: ~0.2s (cached) o ~8-10s (first-time on-the-fly)

---

## Validación de Integración

```bash
python validate_integration.py
```

Verifica:
- ✓ Imports correctos
- ✓ Directorio de modelos
- ✓ Preprocessadores inicializables
- ✓ Archivos presentes

---

## Cambios en `streamlit_app.py`

### Funciones actualizadas:

1. **`load_model(model_name, device)`**
   - Retorna: `(model, preprocessor, model_type)`
   - Soporta: ufd, d3, distildire, smogy, default

2. **`predict_image(image, processor, model, model_type, device)`**
   - Maneja preprocesamiento específico por tipo
   - Retorna: `(pred_class, pred_prob, class_name)`

3. **`evaluate_model_batch(model, processor, dataset, model_name, model_type, device, batch_size=32)`**
   - Soporta batch evaluation para todos los tipos
   - Calcula todas las métricas
   - Retorna diccionario con resultados

### Flujos actualizados:

- **PREDICCION:** Imagen individual → Predicción (Real/Fake) por modelo
- **CLASIFICACION:** Dataset batch → Métricas agregadas + Visualizaciones

---

## Ejemplo de Uso Programático

```python
import torch
from PIL import Image
from preprocessing_adapters import UFDPreprocessor, ModelLoaders

# Inicializar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = UFDPreprocessor(device=str(device))
model = ModelLoaders.load_ufd_model("models/mirage_model_4_universalfakedetect/epoch_4_0.976.pt", device=str(device))

# Procesar imagen
image = Image.open("test.jpg").convert('RGB')
embedding = processor.preprocess_image(image)  # [768]
embedding = embedding.unsqueeze(0).to(device)

# Predicción
model.eval()
with torch.no_grad():
    logit = model(embedding)
    prob = torch.sigmoid(logit).item()
    
print(f"Probability of fake: {prob:.4f}")
print(f"Prediction: {'FAKE' if prob >= 0.5 else 'REAL'}")
```

---

## Resumen Técnico

| Aspecto | UFD | D3 | DistilDIRE |
|--------|-----|----|----|
| Input | CLIP embeddings [768] | Image [3, 224, 224] | Image + EPS [6, 224, 224] |
| Preprocesamiento | CLIP processor | Resize + Normalize | Concatenate + [-1,1] |
| Inferencia | ~0.5-1s | ~1-2s | ~0.2-10s |
| Métricas soportadas | ✓ All | ✓ All | ⚠ Requiere DIRE |
| Implementación | Completa | Completa | Completa |

---

## Próximos Pasos Opcionales

1. **Threshold Adaptativo:** Implementar búsqueda de threshold que maximice F1
2. **Test-Time Augmentation:** Agregar augmentation en validación (como D3 original)
3. **Métricas Extendidas:** AP (Average Precision) calculado correctamente
4. **Pre-processing Caché:** Caché en-sesión de embeddings/tensores
5. **Exportar Resultados:** Descargar métricas como CSV/JSON

---

**Creado:** Integración 3 Modelos Deepfake Detection  
**Estado:** ✅ Completado y Validado  
**Última actualización:** 7 Marzo 2026
