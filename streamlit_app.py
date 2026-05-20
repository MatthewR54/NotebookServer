import streamlit as st
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import zipfile
import tempfile
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from pathlib import Path
from io import BytesIO
from datasets import load_dataset
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
import shutil

# Import preprocessing adapters for UFD, D3, and DistilDIRE
from preprocessing_adapters import (
    UFDPreprocessor, D3Preprocessor, DistilDIREPreprocessor,
    ModelLoaders, InferenceHelper
)

# Configure page layout
st.set_page_config(page_title="Deep Learning Model Interface", layout="wide")
st.title("🤖 Deep Learning Model Interface")

# Initialize session state
if "task_type" not in st.session_state:
    st.session_state.task_type = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "results" not in st.session_state:
    st.session_state.results = None
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = None
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "distildire_available" not in st.session_state:
    st.session_state.distildire_available = False
if "uploaded_zip_basename" not in st.session_state:
    st.session_state.uploaded_zip_basename = None
if "distildire_dataset_path" not in st.session_state:
    st.session_state.distildire_dataset_path = None
if "consensus_mode" not in st.session_state:
    st.session_state.consensus_mode = False

# Define model paths and configurations
MODELS_DIR = Path("./models")
DATASETS_DIR = Path("./datasets")

# Model type configuration
MODEL_TYPE_MAP = {
    "mirage_model_1_smogy_ai_detection": "smogy",
    "mirage_model_2_ateeqq_ai_vs_human_detector": "ateeqq",
    "mirage_model_3_prithiv_deepfake_detector": "prithiv",
    "mirage_model_4_universalfakedetect": "ufd",
    "mirage_model_5_distildire": "distildire",
    "mirage_model_6_d3": "d3"
}

def calculate_consensus(results_dict):
    """Calculate consensus prediction from multiple model results
    
    Args:
        results_dict: Dictionary of results from all models
    
    Returns:
        (consensus_class, consensus_name, is_inconclusive) tuple
    """
    if not results_dict:
        return None, None, False
    
    # Count votes for each class (based on predicted class, not confidence)
    votes_real = sum(1 for r in results_dict.values() if r["class"] == 0)
    votes_fake = sum(1 for r in results_dict.values() if r["class"] == 1)
    
    total_models = len(results_dict)
    
    # Check if evenly split (tie) with even number of models
    if total_models % 2 == 0 and votes_real == votes_fake:
        return None, "Inconcluso", True
    
    # Return class with more votes
    if votes_fake > votes_real:
        return 1, f"fake (1) - {votes_fake}/{total_models} votes", False
    else:
        return 0, f"real (0) - {votes_real}/{total_models} votes", False

def get_available_models():
    """Scan the models directory and return available models following the pattern 'mirage_[author]_[model]'"""
    models = []
    if MODELS_DIR.exists():
        for item in MODELS_DIR.iterdir():
            if item.is_dir() and item.name.startswith("mirage_"):
                models.append(item.name)
    return sorted(models)

def check_distildire_dataset_availability(zip_basename):
    """Check if a DistilDIRE dataset exists for the given ZIP filename
    
    Args:
        zip_basename: Basename of ZIP file without extension (e.g., 'definitive_selected_100imgs')
    
    Returns:
        Path to dataset if found, else None
    """
    if not zip_basename:
        return None
    
    dire_dataset_dir = DATASETS_DIR / f"{zip_basename}_dire"
    
    # Check if directory exists and contains required subdirectories/files
    if dire_dataset_dir.exists() and dire_dataset_dir.is_dir():
        contents = list(dire_dataset_dir.iterdir())
        if len(contents) > 0:  # Directory has content
            return dire_dataset_dir
    
    return None

def find_real_fake_dirs(root_path):
    """Recursively search for 'real' and 'fake' directories in the given path"""
    root = Path(root_path)
    
    # Check if current directory contains both real and fake
    children = [d.name for d in root.iterdir() if d.is_dir()]
    if 'real' in children and 'fake' in children:
        return root
    
    # Search recursively in subdirectories
    for item in root.iterdir():
        if item.is_dir() and item.name not in ['real', 'fake']:
            result = find_real_fake_dirs(item)
            if result is not None:
                return result
    
    return None

def extract_and_prepare_dataset(zip_file_bytes, zip_filename):
    """Extract ZIP file and prepare dataset with real/fake classification
    
    Args:
        zip_file_bytes: Raw ZIP file bytes
        zip_filename: Original filename of the ZIP file
    
    Returns:
        (dataset, error_message, zip_basename)
    """
    try:
        # Extract basename without extension for DistilDIRE lookup
        zip_basename = Path(zip_filename).stem
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Extract ZIP
        zip_path = Path(temp_dir) / "uploaded.zip"
        with open(zip_path, 'wb') as f:
            f.write(zip_file_bytes)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find real and fake directories
        extract_dir = Path(temp_dir)
        dataset_root = find_real_fake_dirs(extract_dir)
        
        if dataset_root is None:
            return None, "❌ Could not find 'real' and 'fake' directories in the uploaded ZIP file", zip_basename
        
        # Load dataset using imagefolder
        try:
            dataset = load_dataset("imagefolder", data_dir=str(dataset_root))
            
            # Map labels: 'real' -> 0, 'fake' -> 1
            label_names = dataset['train'].features['label'].names
            
            # Swap labels if necessary (imagefolder assigns labels alphabetically)
            if label_names[0] == 'fake' and label_names[1] == 'real':
                dataset = dataset.map(lambda x: {'label': 1 - x['label']}, num_proc=1)
            
            st.session_state.temp_dir = temp_dir
            return dataset, None, zip_basename
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, f"❌ Error loading dataset: {str(e)}", zip_basename
    
    except Exception as e:
        if st.session_state.temp_dir:
            shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
        return None, f"❌ Error processing ZIP file: {str(e)}", None

def preprocess_batch(examples, processor):
    """Preprocess a batch of images"""
    try:
        # Convert all PIL Images in the batch to RGB
        images_rgb = [img.convert("RGB") for img in examples['image']]
        
        # Preprocess using the processor
        inputs = processor(images=images_rgb, return_tensors="pt")
        pixel_values = inputs.pixel_values  # Shape: (batch_size, C, H, W)
        
        # Convert labels to tensor
        labels_tensor = torch.tensor(examples['label'])  # Shape: (batch_size,)
        
        return {'pixel_values': pixel_values, 'labels': labels_tensor}
    except Exception as e:
        st.error(f"Error preprocessing batch: {str(e)}")
        return None

def collate_fn(batch):
    """Collate function for DataLoader"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }

def evaluate_model_batch(model, processor, dataset, model_name, model_type, device, dire_dataset_path=None, batch_size=32):
    """Evaluate model on a batch of images and return metrics
    
    Args:
        model: Loaded model
        processor: Model-specific preprocessor
        dataset: HF Dataset with 'image' and 'label' columns
        model_name: Name of model (for logging)
        model_type: Type of model (ufd, d3, distildire, smogy, default)
        device: Torch device
        dire_dataset_path: Path to pre-computed DIRE dataset (required for DistilDIRE)
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with metrics or None on error
    """
    try:
        model = model.to(device)
        model.eval()
        
        true_labels = []
        predicted_probs = []
        predicted_labels = []
        
        # Process dataset in batches
        for idx in range(0, len(dataset), batch_size):
            batch_end = min(idx + batch_size, len(dataset))
            batch_indices = list(range(idx, batch_end))
            batch_data = [dataset[i] for i in batch_indices]
            
            # Prepare batch
            batch_images = []
            batch_labels = []
            
            for sample in batch_data:
                img = sample['image']
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                batch_images.append(img)
                batch_labels.append(sample['label'])
            
            batch_labels_tensor = torch.tensor(batch_labels)
            
            # Preprocess batch based on model type
            with torch.no_grad():
                if model_type == "ufd":
                    # UFD: Batch of images -> embeddings [B, 768]
                    embeddings = processor.preprocess_batch(batch_images)  # [B, 768]
                    embeddings = embeddings.to(device)
                    logits = model(embeddings)  # [B, 1]
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                    
                    # Handle scalar vs array
                    if isinstance(probs, np.ndarray):
                        if probs.ndim == 0:
                            probs = np.array([probs.item()])
                    else:
                        probs = np.array([float(probs)])
                    
                    true_labels.extend(batch_labels)
                    predicted_probs.extend(probs)
                    predicted_labels.extend((probs >= 0.5).astype(int))
                    
                elif model_type == "d3":
                    # D3: Batch of images -> tensors [B, 3, 224, 224]
                    img_tensors = processor.preprocess_batch(batch_images)  # [B, 3, 224, 224]
                    img_tensors = img_tensors.to(device)
                    logits = model(img_tensors)  # [B, 1]
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                    
                    # Handle scalar vs array
                    if isinstance(probs, np.ndarray):
                        if probs.ndim == 0:
                            probs = np.array([probs.item()])
                    else:
                        probs = np.array([float(probs)])
                    
                    true_labels.extend(batch_labels)
                    predicted_probs.extend(probs)
                    predicted_labels.extend((probs >= 0.5).astype(int))
                    
                elif model_type == "distildire":
                    if dire_dataset_path is None:
                        st.error("⚠️ DistilDIRE requires pre-computed DIRE maps - dataset path not provided")
                        return None
                    
                    # DistilDIRE with pre-computed DIRE maps and EPS
                    try:
                        # preprocess_batch_from_dataset returns (img_tensors, eps_tensors) separately
                        img_tensors, eps_tensors = processor.preprocess_batch_from_dataset(batch_images, dire_dataset_path)
                        img_tensors = img_tensors.to(device)
                        eps_tensors = eps_tensors.to(device)
                        
                        # DistilDIRE expects separate img and eps arguments
                        output = model(img_tensors, eps_tensors)  # img: [B, 3, 224, 224], eps: [B, 3, 224, 224]
                        logits = output['logit']  # [B, 1]
                        probs = torch.sigmoid(logits).squeeze().cpu().detach().numpy()
                        
                        # Handle scalar vs array
                        if isinstance(probs, np.ndarray):
                            if probs.ndim == 0:
                                probs = np.array([probs.item()])
                        else:
                            probs = np.array([float(probs)])
                        
                        true_labels.extend(batch_labels)
                        predicted_probs.extend(probs)
                        predicted_labels.extend((probs >= 0.5).astype(int))
                    except Exception as e:
                        st.error(f"Error processing DistilDIRE batch: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        return None
                    
                else:  # smogy or default models
                    # Standard HF models
                    inputs = processor(images=batch_images, return_tensors="pt")
                    pixel_values = inputs.pixel_values.to(device)
                    outputs = model(pixel_values)
                    
                    # Handle outputs - could be dict or object
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits', outputs.get('prediction', None))
                    else:
                        logits = getattr(outputs, 'logits', outputs)
                    
                    # Ensure logits is a tensor
                    if not isinstance(logits, torch.Tensor):
                        logits = torch.tensor(logits, dtype=torch.float32).to(device)
                    
                    # Handle both binary and multi-class outputs
                    if logits.shape[-1] == 1:
                        # Single output: sigmoid for binary
                        probs_batch = torch.sigmoid(logits.squeeze()).cpu().numpy()
                    else:
                        # Multiple outputs: softmax for class 1 (fake)
                        probs_batch = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    
                    # Ensure probs_batch is 1D
                    if isinstance(probs_batch, np.ndarray):
                        if probs_batch.ndim == 0:
                            probs_batch = np.array([probs_batch.item()])
                    else:
                        probs_batch = np.array([float(probs_batch)])
                    
                    true_labels.extend(batch_labels)
                    predicted_probs.extend(probs_batch)
                    predicted_labels.extend((probs_batch >= 0.5).astype(int))
        
        # Ensure arrays are valid
        true_labels = np.array(true_labels)
        predicted_probs = np.array(predicted_probs)
        predicted_labels = np.array(predicted_labels)
        
        if len(true_labels) == 0:
            st.error("No predictions generated")
            return None
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(true_labels, predicted_labels)),
            'precision': float(precision_score(true_labels, predicted_labels, zero_division=0)),
            'recall': float(recall_score(true_labels, predicted_labels, zero_division=0)),
            'f1': float(f1_score(true_labels, predicted_labels, zero_division=0)),
            'roc_auc': float(roc_auc_score(true_labels, predicted_probs)),
            'confusion_matrix': confusion_matrix(true_labels, predicted_labels),
            'true_labels': true_labels,
            'predicted_probs': predicted_probs,
            'predicted_labels': predicted_labels
        }
        
        return metrics
        
    except Exception as e:
        st.error(f"Error evaluating model {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

@st.cache_resource
def load_ufd_resources(model_path):
    """Cache UFD model and processor"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = UFDPreprocessor(device=str(device))
    model = ModelLoaders.load_ufd_model(str(model_path), device=str(device))
    return preprocessor, model

@st.cache_resource
def load_d3_resources(model_path):
    """Cache D3 model and preprocessor"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = D3Preprocessor(device=str(device))
    model = ModelLoaders.load_d3_model(str(model_path), device=str(device))
    return preprocessor, model

@st.cache_resource
def load_distildire_resources(model_path):
    """Cache DistilDIRE model and preprocessor"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = DistilDIREPreprocessor(device=str(device))
    model = ModelLoaders.load_distildire_model(str(model_path), device=str(device))
    return preprocessor, model

def load_model(model_name, device):
    """Load a model and its preprocessor based on model name
    
    Returns:
        (model, preprocessor, model_type) or (None, None, None) on error
    """
    model_path = MODELS_DIR / model_name
    model_type = MODEL_TYPE_MAP.get(model_name, "unknown")
    
    try:
        if model_type == "ufd":
            # UniversalFakeDetect
            preprocessor, model = load_ufd_resources(model_path / "epoch_4_0.976.pt")
            return model, preprocessor, "ufd"
        
        elif model_type == "d3":
            # D3 model
            preprocessor, model = load_d3_resources(model_path / "model_epoch_best.pt")
            return model, preprocessor, "d3"
        
        elif model_type == "distildire":
            # DistilDIRE model
            preprocessor, model = load_distildire_resources(model_path / "model_epoch_4.pt")
            return model, preprocessor, "distildire"
        
        elif model_type == "smogy":
            # SMOGY model (original handler)
            processor = AutoImageProcessor.from_pretrained("Smogy/SMOGY-Ai-images-detector")
            model = AutoModelForImageClassification.from_pretrained("Smogy/SMOGY-Ai-images-detector")
            fine_tuned_path = model_path / "fine_tuned_smogy_model.pt"
            if fine_tuned_path.exists():
                model.load_state_dict(torch.load(fine_tuned_path, map_location=device))
            return model, processor, "smogy"
        
        else:
            # Default: try loading as standard HF model
            processor = AutoImageProcessor.from_pretrained(str(model_path))
            model = AutoModelForImageClassification.from_pretrained(str(model_path))
            return model, processor, "default"
    
    except Exception as e:
        st.error(f"Error loading model {model_name}: {str(e)}")
        return None, None, None

def predict_image(image, processor, model, model_type, device):
    """Make prediction on a single image using the provided model
    
    Args:
        image: PIL Image
        processor: Model-specific preprocessor
        model: Loaded model
        model_type: Type of model (ufd, d3, distildire, smogy, default)
        device: Torch device
    
    Returns:
        (pred_class, pred_prob, class_name) or (None, None, None) on error
    """
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            if model_type == "ufd":
                # UniversalFakeDetect: Image -> CLIP embedding -> prediction
                embedding = processor.preprocess_image(image)  # [768]
                
                # Ensure embedding is a tensor
                if isinstance(embedding, np.ndarray):
                    embedding = torch.from_numpy(embedding).float()
                elif not isinstance(embedding, torch.Tensor):
                    embedding = torch.tensor(embedding, dtype=torch.float32)
                
                # Add batch dimension
                if embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)  # [1, 768]
                
                embedding = embedding.to(device)
                logit = model(embedding)  # [1, 1]
                
                # Ensure logit is a tensor
                if not isinstance(logit, torch.Tensor):
                    logit = torch.tensor(logit, dtype=torch.float32).to(device)
                
                fake_prob = torch.sigmoid(logit).squeeze().item()  # P(fake)
                pred_class = 1 if fake_prob >= 0.5 else 0
                # Convert to confidence in predicted class
                prob = fake_prob if pred_class == 1 else (1 - fake_prob)
                
            elif model_type == "d3":
                # D3: Image -> normalized tensor -> prediction
                img_tensor = processor.preprocess_image(image)  # [3, 224, 224]
                
                # Ensure img_tensor is a tensor
                if isinstance(img_tensor, np.ndarray):
                    img_tensor = torch.from_numpy(img_tensor).float()
                elif not isinstance(img_tensor, torch.Tensor):
                    img_tensor = torch.tensor(img_tensor, dtype=torch.float32)
                
                # Add batch dimension
                if img_tensor.dim() == 3:
                    img_tensor = img_tensor.unsqueeze(0)  # [1, 3, 224, 224]
                
                img_tensor = img_tensor.to(device)
                logit = model(img_tensor)  # [1, 1]
                
                # Ensure logit is a tensor
                if not isinstance(logit, torch.Tensor):
                    logit = torch.tensor(logit, dtype=torch.float32).to(device)
                
                fake_prob = torch.sigmoid(logit).squeeze().item()  # P(fake)
                pred_class = 1 if fake_prob >= 0.5 else 0
                # Convert to confidence in predicted class
                prob = fake_prob if pred_class == 1 else (1 - fake_prob)
                
            elif model_type == "distildire":
                # DistilDIRE requires DIRE maps and EPS - not supported for on-the-fly single image
                st.warning("⚠️ DistilDIRE requires pre-computed DIRE maps and EPS perturbations")
                st.info("For single image predictions with DistilDIRE, upload a ZIP file with a corresponding '_dire' dataset")
                return None, None, None
                
            else:  # smogy or default models
                # Standard HF models: Image -> processor -> logits
                inputs = processor(images=image, return_tensors="pt")
                pixel_values = inputs.pixel_values.to(device)
                outputs = model(pixel_values)
                
                # Handle outputs - could be dict or object
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('prediction', None))
                else:
                    logits = getattr(outputs, 'logits', outputs)
                
                # Ensure logits is a tensor
                if not isinstance(logits, torch.Tensor):
                    logits = torch.tensor(logits, dtype=torch.float32).to(device)
                
                # Handle single-output models (binary classification)
                if logits.shape[-1] == 1:
                    # Single output: treat as sigmoid output
                    prob = torch.sigmoid(logits.squeeze()).item()
                    pred_class = 1 if prob >= 0.5 else 0
                else:
                    # Multiple outputs: use softmax
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)
                    
                    probabilities = torch.softmax(logits, dim=1)
                    pred_class = torch.argmax(logits, dim=-1).item()
                    prob = probabilities[0, min(pred_class, probabilities.shape[1] - 1)].item()
        
        class_name = "fake (1)" if pred_class == 1 else "real (0)"
        return pred_class, prob, class_name
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

# ==================== SECTION 1: FILE UPLOAD ====================
st.divider()
st.subheader("📁 Step 1: Upload File")

uploaded_file = st.file_uploader(
    "Choose a file to analyze",
    type=["jpg", "jpeg", "png", "zip"],
    key="file_uploader"
)

# Determine task type
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension in ["jpg", "jpeg", "png"]:
        st.session_state.task_type = "Prediccion"
        st.session_state.distildire_available = False  # DistilDIRE not supported for single images
        st.session_state.uploaded_zip_basename = None
        st.session_state.distildire_dataset_path = None
        st.info(f"**Tipo de Tarea: Prediccion**")
        
        # Show preview for image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    elif file_extension == "zip":
        st.session_state.task_type = "Clasificacion"
        st.session_state.consensus_mode = False  # Consensus mode only for Prediction
        st.info(f"**Tipo de Tarea: Clasificacion**")
        
        # Process ZIP file
        st.session_state.uploaded_file.seek(0)
        zip_bytes = st.session_state.uploaded_file.read()
        zip_filename = st.session_state.uploaded_file.name
        
        with st.spinner("📦 Extracting and inspecting ZIP file..."):
            dataset, error_msg, zip_basename = extract_and_prepare_dataset(zip_bytes, zip_filename)
            
            if error_msg:
                st.error(error_msg)
                st.session_state.dataset = None
                st.session_state.uploaded_zip_basename = None
                st.session_state.distildire_available = False
                st.session_state.distildire_dataset_path = None
            else:
                st.session_state.dataset = dataset
                st.session_state.uploaded_zip_basename = zip_basename
                st.success(f"✅ ZIP file processed successfully!")
                st.info(f"📊 Dataset contains {len(dataset['train'])} images")
                
                # Check for DistilDIRE dataset
                distildire_path = check_distildire_dataset_availability(zip_basename)
                if distildire_path:
                    st.session_state.distildire_available = True
                    st.session_state.distildire_dataset_path = distildire_path
                    st.success(f"✅ Found DistilDIRE dataset: {distildire_path}")
                else:
                    st.session_state.distildire_available = False
                    st.session_state.distildire_dataset_path = None
                    st.info(f"ℹ️ No DistilDIRE dataset found for '{zip_basename}' (expected: 'datasets/{zip_basename}_dire')")
else:
    st.session_state.task_type = None
    st.session_state.dataset = None
    st.session_state.consensus_mode = False
    st.session_state.distildire_available = False
    st.session_state.uploaded_zip_basename = None
    st.session_state.distildire_dataset_path = None
    st.warning("⬆️ Please upload a file to proceed")

# ==================== SECTION 2: MODEL SELECTION ====================
st.divider()
st.subheader("🧠 Step 2: Select Models")

available_models = get_available_models()

if not available_models:
    st.error("❌ No models found in the 'models' directory")
else:
    st.write(f"**Available Models ({len(available_models)}):**")
    
    selected_models = {}
    for model_name in available_models:
        # Create a more readable label
        label = model_name.replace("mirage_", "").replace("_", " ").title()
        
        # Disable DistilDIRE if not available
        is_distildire = model_name == "mirage_model_5_distildire"
        is_enabled = not is_distildire or st.session_state.distildire_available
        
        if is_distildire and not st.session_state.distildire_available:
            # Show disabled DistilDIRE with explanation
            st.write(f"~~{label}~~ *(DistilDIRE dataset not available)*")
            selected_models[model_name] = False
        else:
            selected_models[model_name] = st.checkbox(label, value=False, key=f"checkbox_{model_name}", disabled=not is_enabled)
    
    # Summary of selected models
    selected_count = sum(selected_models.values())
    if selected_count > 0:
        st.success(f"✓ {selected_count} model(s) selected")
    else:
        st.warning("⚠️ No models selected")
    
    # Consensus mode option (only for Prediction task with multiple models)
    st.divider()
    st.write("**Consensus Mode** (Prediction only)")
    st.session_state.consensus_mode = st.checkbox(
        "🤝 Enable Consensus Mode",
        value=st.session_state.consensus_mode,
        help="When enabled, returns a single prediction based on the majority vote from selected models. Only available in Prediction mode.",
        key="consensus_mode_checkbox"
    )
    if st.session_state.consensus_mode:
        st.info("📊 In Consensus Mode, models vote on whether an image is 'real' or 'fake' based on their predictions. The class with more votes is selected.")
        if selected_count < 2:
            st.warning("⚠️ Consensus Mode requires at least 2 models selected")

# ==================== SECTION 3: EXECUTION ====================
st.divider()
st.subheader("⚡ Step 3: Run Analysis")

# Large execution button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    execute_button = st.button("🚀 Execute Analysis", key="execute_btn", use_container_width=True)

# Execute analysis
if execute_button:
    # Validation
    if st.session_state.uploaded_file is None:
        st.error("❌ Please upload a file first")
    elif selected_count == 0:
        st.error("❌ Please select at least one model")
    elif st.session_state.consensus_mode and selected_count < 2:
        st.error("❌ Consensus Mode requires at least 2 models selected")
    elif st.session_state.task_type == "Clasificacion" and st.session_state.dataset is None:
        st.error("❌ Invalid dataset. Please upload a valid ZIP file with 'real' and 'fake' directories")
    else:
        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ================= PREDICCION TASK =================
        if st.session_state.task_type == "Prediccion":
            st.session_state.uploaded_file.seek(0)
            image = Image.open(st.session_state.uploaded_file)
            
            progress_bar = st.progress(0)
            results_placeholder = st.empty()
            
            st.session_state.results = {}
            total_steps = selected_count
            current_step = 0
            
            # Process each selected model
            for model_name, is_selected in selected_models.items():
                if is_selected:
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                    status_text = f"Processing with {model_name.replace('mirage_', '')}... ({current_step}/{total_steps})"
                    
                    with results_placeholder.container():
                        st.info(status_text)
                    
                    # Load model and preprocessor
                    model, preprocessor, model_type = load_model(model_name, device)
                    
                    if model is not None:
                        # Make prediction
                        pred_class, pred_prob, class_name = predict_image(image, preprocessor, model, model_type, device)
                        
                        if pred_class is not None:
                            st.session_state.results[model_name] = {
                                "class": pred_class,
                                "probability": pred_prob,
                                "class_name": class_name
                            }
            
            progress_bar.empty()
            
            # Display results for Prediccion
            if st.session_state.results:
                st.divider()
                
                # Display Consensus Mode result if enabled
                if st.session_state.consensus_mode and len(st.session_state.results) >= 2:
                    st.subheader("🤝 Consensus Mode Result")
                    
                    consensus_class, consensus_name, is_inconclusive = calculate_consensus(st.session_state.results)
                    
                    if is_inconclusive:
                        # Inconclusive result
                        st.warning(
                            "⚠️ **INCONCLUSO** - Even number of models with equal votes",
                            icon="🟡"
                        )
                        st.write(f"*The models are evenly split in their predictions. No consensus can be reached.*")
                    else:
                        # Consensus reached
                        if consensus_class == 0:
                            color_indicator = "🟢"
                            color_style = "green"
                        else:
                            color_indicator = "🔴"
                            color_style = "red"
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.success(f"{color_indicator} **{consensus_name}**")
                    
                    st.divider()
                
                # Display individual model predictions
                st.subheader("📊 Individual Model Predictions")
                
                results_cols = st.columns(len(st.session_state.results))
                
                for idx, (model_name, result) in enumerate(st.session_state.results.items()):
                    with results_cols[idx]:
                        model_label = model_name.replace("mirage_", "").replace("_", " ").title()
                        
                        if result["class"] == 0:  # Real
                            color_indicator = "🟢"
                        else:  # Fake
                            color_indicator = "🔴"
                        
                        st.metric(
                            label=model_label,
                            value=color_indicator + " " + result["class_name"],
                            delta=f"{result['probability']:.2%} confidence"
                        )
                
                st.write("**Detailed Results:**")
                results_data = []
                for model_name, result in st.session_state.results.items():
                    results_data.append({
                        "Model": model_name.replace("mirage_", "").replace("_", " ").title(),
                        "Prediction": result["class_name"],
                        "Confidence": f"{result['probability']:.4f}"
                    })
                
                st.dataframe(results_data, use_container_width=True)
                st.success("✅ Analysis completed successfully!")
            else:
                st.error("❌ No results generated. Check the error messages above.")
        
        # ================= CLASIFICACION TASK =================
        elif st.session_state.task_type == "Clasificacion":
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            
            st.session_state.results = {}
            total_steps = selected_count
            current_step = 0
            
            # Process each selected model
            for model_name, is_selected in selected_models.items():
                if is_selected:
                    current_step += 1
                    progress = current_step / total_steps
                    progress_bar.progress(progress)
                    status_text = f"Evaluating {model_name.replace('mirage_', '')}... ({current_step}/{total_steps})"
                    
                    with status_placeholder.container():
                        st.info(status_text)
                    
                    # Load model and preprocessor
                    model, preprocessor, model_type = load_model(model_name, device)
                    
                    if model is not None:
                        # Evaluate model on dataset
                        metrics = evaluate_model_batch(
                            model, preprocessor, st.session_state.dataset['train'], 
                            model_name, model_type, device,
                            dire_dataset_path=st.session_state.distildire_dataset_path
                        )
                        
                        if metrics is not None:
                            st.session_state.results[model_name] = metrics
            
            progress_bar.empty()
            status_placeholder.empty()
            
            # Display results for Clasificacion
            if st.session_state.results:
                st.divider()
                st.subheader("📊 Classification Results")
                
                # Display metrics for each model
                for model_name, metrics in st.session_state.results.items():
                    st.write(f"### {model_name.replace('mirage_', '').replace('_', ' ').title()}")
                    
                    # Metrics in columns
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                    with col2:
                        st.metric("Precision", f"{metrics['precision']:.4f}")
                    with col3:
                        st.metric("Recall", f"{metrics['recall']:.4f}")
                    with col4:
                        st.metric("F1-Score", f"{metrics['f1']:.4f}")
                    with col5:
                        st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
                    
                    # Visualizations in columns
                    col_cm, col_roc = st.columns(2)
                    
                    # Confusion Matrix
                    with col_cm:
                        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                        sns.heatmap(
                            metrics['confusion_matrix'],
                            annot=True,
                            fmt='d',
                            cmap='Blues',
                            ax=ax_cm,
                            xticklabels=['Real (0)', 'Fake (1)'],
                            yticklabels=['Real (0)', 'Fake (1)'],
                            cbar=True
                        )
                        ax_cm.set_xlabel('Predicted Label')
                        ax_cm.set_ylabel('True Label')
                        ax_cm.set_title('Confusion Matrix')
                        st.pyplot(fig_cm)
                        plt.close(fig_cm)
                    
                    # ROC Curve
                    with col_roc:
                        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
                        RocCurveDisplay.from_predictions(
                            metrics['true_labels'],
                            metrics['predicted_probs'],
                            ax=ax_roc
                        )
                        ax_roc.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
                        ax_roc.set_xlabel('False Positive Rate')
                        ax_roc.set_ylabel('True Positive Rate')
                        ax_roc.set_title('ROC Curve')
                        ax_roc.legend()
                        st.pyplot(fig_roc)
                        plt.close(fig_roc)
                    
                    st.divider()
                
                st.success("✅ Classification completed successfully!")
            else:
                st.error("❌ No results generated. Check the error messages above.")

# ==================== FOOTER ====================
st.divider()
st.caption("Deep Learning Model Interface v2.0 | Using PyTorch & Streamlit")

# Cleanup temporary files when app reruns
if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
    # Only cleanup if we're switching tasks or uploading new file
    pass  # Cleanup happens automatically when temp_dir is reassigned or session ends
