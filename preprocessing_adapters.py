"""
Preprocessing Adapters for Deepfake Detection Models

This module provides unified preprocessing interfaces for 3 deepfake detection models:
- UniversalFakeDetect (UFD): CLIP embeddings-based
- D3: CLIP ViT-L/14 with Shuffle Attention
- DistilDIRE: ResNet50 with DIRE maps + EPS perturbations
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Union, Optional, List
import numpy as np


# ==================== UNIVERSAL FAKE DETECT (UFD) ====================

class UFDPreprocessor:
    """
    UniversalFakeDetect Preprocessor
    
    Handles CLIP processor loading and embedding generation.
    Input: PIL Image
    Output: [768] embedding tensor
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._load_clip_resources()
    
    def _load_clip_resources(self):
        """Load CLIP processor and model"""
        try:
            from transformers import CLIPProcessor, CLIPModel
        except ImportError:
            raise ImportError("transformers library required. Install: pip install transformers")
        
        self.backbone = "openai/clip-vit-large-patch14"
        self.processor = CLIPProcessor.from_pretrained(self.backbone)
        self.clip_model = CLIPModel.from_pretrained(self.backbone)
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
    
    def preprocess_image(self, image: Union[Image.Image, str]) -> torch.Tensor:
        """
        Convert image to CLIP embedding [768]
        
        Args:
            image: PIL Image or path to image file
        
        Returns:
            Embedding tensor [768] on device
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image or str, got {type(image)}")
        
        # Process with CLIP processor
        inputs = self.processor(
            images=image,
            return_tensors="pt",
            size=(224, 224),
            do_rescale=True
        )
        pixel_values = inputs["pixel_values"].to(self.device)
        
        # Extract embeddings
        with torch.no_grad():
            embeddings = self.clip_model.get_image_features(pixel_values)  # [1, 768]
        
        return embeddings.squeeze(0)  # [768]
    
    def preprocess_batch(self, images: list) -> torch.Tensor:
        """
        Preprocess batch of images
        
        Args:
            images: List of PIL Images or paths
        
        Returns:
            Batch embeddings [B, 768]
        """
        embeddings_list = []
        for img in images:
            emb = self.preprocess_image(img)
            embeddings_list.append(emb)
        
        return torch.stack(embeddings_list, dim=0)  # [B, 768]


# ==================== D3 MODEL ====================

class D3Preprocessor:
    """
    D3 CLIP-based Preprocessor
    
    Handles image normalization for D3 model (CLIP ViT-L/14 + Shuffle Attention).
    Input: PIL Image
    Output: [3, 224, 224] tensor normalized with CLIP statistics
    """
    
    def __init__(self, device: str = "cuda", arch: str = "CLIP"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.arch = arch
        
        # MEAN/STD statistics
        self.MEAN = {
            "imagenet": [0.485, 0.456, 0.406],
            "clip": [0.48145466, 0.4578275, 0.40821073]
        }
        
        self.STD = {
            "imagenet": [0.229, 0.224, 0.225],
            "clip": [0.26862954, 0.26130258, 0.27577711]
        }
        
        # Determine normalization source
        self.norm_source = "imagenet" if "moco" in arch.lower() else "clip"
        
        # Create transform
        self.transform = self._create_transform()
    
    def _create_transform(self) -> transforms.Compose:
        """Create preprocessing transform pipeline"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.MEAN[self.norm_source],
                std=self.STD[self.norm_source]
            ),
        ])
    
    def preprocess_image(self, image: Union[Image.Image, str]) -> torch.Tensor:
        """
        Preprocess image to [3, 224, 224] normalized tensor
        
        Args:
            image: PIL Image or path to image file
        
        Returns:
            Normalized tensor [3, 224, 224] on device
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image or str, got {type(image)}")
        
        tensor = self.transform(image)  # [3, 224, 224]
        return tensor.to(self.device)
    
    def preprocess_batch(self, images: list) -> torch.Tensor:
        """
        Preprocess batch of images
        
        Args:
            images: List of PIL Images or paths
        
        Returns:
            Batch tensor [B, 3, 224, 224]
        """
        tensors_list = []
        for img in images:
            tensor = self.preprocess_image(img)
            tensors_list.append(tensor)
        
        return torch.stack(tensors_list, dim=0)  # [B, 3, 224, 224]
    
    @staticmethod
    def detect_architecture(arch_string: str) -> str:
        """
        Detect architecture type and normalization source
        
        Args:
            arch_string: Architecture identifier (e.g., "CLIP:ViT-L/14", "Imagenet:resnet50")
        
        Returns:
            Normalization source ("clip" or "imagenet")
        """
        return "imagenet" if "moco" in arch_string.lower() else "clip"


# ==================== DISTILDIRE MODEL ====================

class DistilDIREPreprocessor:
    """
    DistilDIRE Preprocessor
    
    Handles preprocessing for DistilDIRE model which requires:
    - Original image (3 channels)
    - EPS perturbations (3 channels)
    - Both concatenated in [-1, 1] range
    
    Input: PIL Image + path to EPS tensor
    Output: [6, 224, 224] tensor in [-1, 1] range
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path("./dire_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    @staticmethod
    def normalize_to_neg1_pos1(tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize tensor from [0, 1] to [-1, 1]
        
        Args:
            tensor: Input tensor in [0, 1] range
        
        Returns:
            Tensor in [-1, 1] range
        """
        return tensor * 2 - 1
    
    @staticmethod
    def denormalize_to_0_1(tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize tensor from [-1, 1] to [0, 1]
        
        Args:
            tensor: Input tensor in [-1, 1] range
        
        Returns:
            Tensor in [0, 1] range
        """
        return (tensor + 1) / 2
    
    def _load_image_normalized(self, image: Union[Image.Image, str]) -> torch.Tensor:
        """
        Load and normalize image to [-1, 1]
        
        Args:
            image: PIL Image or path
        
        Returns:
            Tensor [3, 224, 224] in [-1, 1]
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image or str, got {type(image)}")
        
        # Convert to tensor [0, 1]
        tensor = TF.to_tensor(image)  # [3, H, W]
        
        # Resize if needed
        if tensor.shape[1] != 224 or tensor.shape[2] != 224:
            tensor = TF.resize(tensor, (224, 224))
        
        # Normalize to [-1, 1]
        tensor = self.normalize_to_neg1_pos1(tensor)
        
        return tensor
    
    def _load_eps_normalized(self, eps_path: str) -> torch.Tensor:
        """
        Load EPS perturbation tensor and normalize to [-1, 1]
        
        Args:
            eps_path: Path to .pt file containing EPS
        
        Returns:
            Tensor [3, 224, 224] in [-1, 1]
        """
        if not Path(eps_path).exists():
            raise FileNotFoundError(f"EPS file not found: {eps_path}")
        
        eps = torch.load(eps_path, weights_only=True)  # [3, 224, 224], typically [-1, 1]
        
        # Ensure it's in [-1, 1] range
        if eps.min() >= 0:  # If in [0, 1], normalize
            eps = self.normalize_to_neg1_pos1(eps)
        
        # Validate shape
        if eps.shape != (3, 224, 224):
            raise ValueError(f"Expected EPS shape (3, 224, 224), got {eps.shape}")
        
        return eps
    
    def preprocess_image_with_eps(
        self, 
        image: Union[Image.Image, str], 
        eps_path: str
    ) -> torch.Tensor:
        """
        Preprocess image and concatenate with EPS
        
        Args:
            image: PIL Image or path
            eps_path: Path to EPS tensor (.pt file)
        
        Returns:
            Concatenated tensor [6, 224, 224] in [-1, 1]
        """
        # Load and normalize image
        img_tensor = self._load_image_normalized(image)  # [3, 224, 224]
        
        # Load and normalize EPS
        eps_tensor = self._load_eps_normalized(eps_path)  # [3, 224, 224]
        
        # Validate shapes match
        if img_tensor.shape[1:] != eps_tensor.shape[1:]:
            raise ValueError(
                f"Shape mismatch: image {img_tensor.shape}, eps {eps_tensor.shape}"
            )
        
        # Concatenate [3, 224, 224] + [3, 224, 224] -> [6, 224, 224]
        combined = torch.cat([img_tensor, eps_tensor], dim=0)
        
        return combined.to(self.device)
    
    def preprocess_batch_with_eps(
        self, 
        images: list, 
        eps_paths: list
    ) -> torch.Tensor:
        """
        Preprocess batch of images with EPS
        
        Args:
            images: List of PIL Images or paths
            eps_paths: List of EPS file paths
        
        Returns:
            Batch tensor [B, 6, 224, 224]
        """
        if len(images) != len(eps_paths):
            raise ValueError(f"Length mismatch: {len(images)} images, {len(eps_paths)} eps")
        
        combined_list = []
        for img, eps_path in zip(images, eps_paths):
            combined = self.preprocess_image_with_eps(img, eps_path)
            combined_list.append(combined)
        
        return torch.stack(combined_list, dim=0)  # [B, 6, 224, 224]
    
    def _find_eps_file_for_image(self, image_name: str, eps_dir: Path) -> Optional[Path]:
        """
        Find corresponding EPS file for an image
        
        Args:
            image_name: Original image filename (e.g., 'image.jpg')
            eps_dir: Directory containing EPS files
        
        Returns:
            Path to EPS file or None if not found
        """
        # Remove extension and add .pt
        base_name = Path(image_name).stem
        
        # Look in flat directory first
        eps_file = eps_dir / f"{base_name}.pt"
        if eps_file.exists():
            return eps_file
        
        # Look in subdirectories (real/fake)
        for subdir in eps_dir.iterdir():
            if subdir.is_dir():
                eps_file = subdir / f"{base_name}.pt"
                if eps_file.exists():
                    return eps_file
        
        return None
    
    def _find_eps_file_for_image_in_dataset(
        self, 
        image_name: str, 
        img_dir: Path,
        eps_dir: Path
    ) -> Optional[Path]:
        """
        Find corresponding EPS file for an image.
        
        Searches in this priority order:
        1. Same directory as the image (img_dir) - handles case where .pt files are colocated
        2. Corresponding subdirectory in eps_dir (same structure as images/)
        3. Flat eps_dir directory
        
        Args:
            image_name: Original image filename (e.g., 'image.jpg')
            img_dir: Directory where the image is located (e.g., 'images/fake')
            eps_dir: Root directory potentially containing EPS files
        
        Returns:
            Path to EPS file or None if not found
        """
        base_name = Path(image_name).stem
        eps_filename = f"{base_name}.pt"
        
        # Priority 1: Look in same directory as image (colocated .pt files)
        eps_in_img_dir = img_dir / eps_filename
        if eps_in_img_dir.exists():
            return eps_in_img_dir
        
        # Priority 2: Look in corresponding subdirectory structure in eps_dir
        if eps_dir.exists():
            # Try with same relative structure (e.g., images/fake -> eps/fake)
            rel_parts = img_dir.parts
            # Find which part is 'images' and reconstruct after eps_dir
            try:
                img_idx = rel_parts.index('images')
                # Parts after 'images' (e.g., ['fake'])
                subpath_parts = rel_parts[img_idx + 1:]
                
                if subpath_parts:
                    eps_subdir = eps_dir / Path(*subpath_parts)
                    eps_in_subdir = eps_subdir / eps_filename
                    if eps_in_subdir.exists():
                        return eps_in_subdir
            except (ValueError, IndexError):
                pass
            
            # Priority 3: Look in flat eps_dir
            eps_flat = eps_dir / eps_filename
            if eps_flat.exists():
                return eps_flat
        
        return None
    
    def preprocess_batch_from_dataset(
        self,
        images: list,
        dataset_path: Union[str, Path]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess batch of images using pre-computed DIRE maps and EPS from dataset.
        Returns image and EPS tensors SEPARATELY (not concatenated).
        
        Args:
            images: List of PIL Images for which to find corresponding DIRE/EPS
            dataset_path: Path to dataset root containing 'images/', 'dire/', and/or 'eps/' directories
        
        Returns:
            Tuple of (img_tensors, eps_tensors):
            - img_tensors: Batch tensor [B, 3, 224, 224] in [-1, 1] range (original images)
            - eps_tensors: Batch tensor [B, 3, 224, 224] in [-1, 1] range (EPS perturbations)
        
        Raises:
            FileNotFoundError: If DIRE/EPS files not found for images
        """
        dataset_path = Path(dataset_path)
        images_dir = dataset_path / "images"
        eps_dir = dataset_path / "eps"
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found at {images_dir}")
        
        img_list = []
        eps_list = []
        
        for idx, img in enumerate(images):
            # For each image in batch, try to find corresponding DIRE/EPS
            try:
                # Convert PIL image to filename for lookup
                if isinstance(img, Image.Image):
                    # If it's a PIL image from the batch, find it in images_dir
                    image_files = sorted(
                        list(images_dir.glob("*/*.png")) + 
                        list(images_dir.glob("*/*.jpg")) +
                        list(images_dir.glob("*.png")) +
                        list(images_dir.glob("*.jpg"))
                    )
                    if idx < len(image_files):
                        img_path = image_files[idx]
                        img_name = img_path.name
                        img_dir = img_path.parent  # Get the subdirectory (fake/real)
                    else:
                        raise FileNotFoundError(f"Could not find image file for batch index {idx}")
                else:
                    img_name = Path(img).name
                    img_dir = Path(img).parent
                
                # Find corresponding EPS file
                eps_path = self._find_eps_file_for_image_in_dataset(
                    img_name, img_dir, eps_dir
                )
                if eps_path is None:
                    raise FileNotFoundError(f"No EPS file found for image {img_name}")
                
                # Load and normalize image separately
                img_tensor = self._load_image_normalized(img)  # [3, 224, 224]
                eps_tensor = self._load_eps_normalized(str(eps_path))  # [3, 224, 224]
                
                img_list.append(img_tensor)
                eps_list.append(eps_tensor)
                
            except Exception as e:
                raise RuntimeError(f"Error processing image at batch index {idx}: {str(e)}")
        
        if not img_list:
            raise ValueError("No images were successfully processed")
        
        img_batch = torch.stack(img_list, dim=0)  # [B, 3, 224, 224]
        eps_batch = torch.stack(eps_list, dim=0)  # [B, 3, 224, 224]
        
        return img_batch, eps_batch
    
    def validate_dire_cache(self, dataset_path: Path) -> bool:
        """
        Check if DIRE/EPS files are cached for dataset
        
        Args:
            dataset_path: Path to dataset root
        
        Returns:
            True if DIRE and EPS directories exist with files
        """
        dire_dir = dataset_path / "dire"
        eps_dir = dataset_path / "eps"
        
        if not dire_dir.exists() or not eps_dir.exists():
            return False
        
        # Check if there are files
        dire_files = list(dire_dir.glob("**/*.png")) + list(dire_dir.glob("**/*.jpg"))
        eps_files = list(eps_dir.glob("**/*.pt"))
        
        return len(dire_files) > 0 and len(eps_files) > 0


# ==================== MODEL LOADING UTILITIES ====================

class ModelLoaders:
    """Unified interface for loading different deepfake detector models"""
    
    @staticmethod
    def load_ufd_model(
        checkpoint_path: str,
        device: str = "cuda"
    ) -> nn.Module:
        """
        Load UniversalFakeDetect model
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to load on
        
        Returns:
            Loaded model
        """
        try:
            # Import model class
            import sys
            sys.path.insert(0, str(Path("github_model_srcs/UniversalFakeDetect")))
            from models import UniversalFakeDetectv2
        except ImportError:
            raise ImportError("Cannot import UniversalFakeDetectv2. Check path.")
        
        device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Create model
        model = UniversalFakeDetectv2(input_dim=768, hidden_dim=384, num_classes=1)
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, weights_only=True, map_location=device_obj)
        if "model" in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
        
        model = model.to(device_obj)
        model.eval()
        
        return model
    
    @staticmethod
    def load_d3_model(
        checkpoint_path: str,
        device: str = "cuda"
    ) -> nn.Module:
        """
        Load D3 model by directly managing sys.path and module cache
        
        Args:
            checkpoint_path: Path to classifier weights
            device: Device to load on
        
        Returns:
            Loaded model (CLIPModelShuffleAttentionPenultimateLayer)
        """
        import sys
        from pathlib import Path
        
        try:
            # Get D3 directory
            d3_path = Path("github_model_srcs/D3").resolve()
            if not d3_path.exists():
                raise FileNotFoundError(f"D3 directory not found at {d3_path}")
            
            # Strategy: Add D3 root to path temporarily and remove conflicting modules
            d3_str = str(d3_path)
            
            # Clear any cached 'models' module to avoid conflicts
            # This ensures fresh import of D3's models package
            modules_to_clear = [key for key in sys.modules.keys() if key == 'models' or key.startswith('models.')]
            for mod in modules_to_clear:
                del sys.modules[mod]
            
            # Insert D3 at the FRONT of sys.path
            if d3_str not in sys.path:
                sys.path.insert(0, d3_str)
            
            # Now import - this should get D3's models since it's at the front
            from models import CLIPModelShuffleAttentionPenultimateLayer
            
            # Remove D3 from path to avoid future conflicts
            sys.path.remove(d3_str)
            
        except FileNotFoundError as e:
            raise ImportError(f"D3 directory error: {str(e)}")
        except ImportError as e:
            # Provide more details about what failed
            raise ImportError(
                f"Cannot import D3 model: {str(e)}. "
                f"Ensure all D3 dependencies are installed."
            )
        except Exception as e:
            raise ImportError(f"Unexpected error loading D3 model: {str(e)}")
        
        device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Create model instance
        try:
            model = CLIPModelShuffleAttentionPenultimateLayer(
                name="ViT-L/14",
                num_classes=1,
                shuffle_times=1,
                original_times=1,
                patch_size=[14]  # Must be a list for indexing
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate D3 model: {str(e)}")
        
        # Load checkpoint weights
        try:
            checkpoint_path_obj = Path(checkpoint_path)
            if not checkpoint_path_obj.exists():
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
            state_dict = torch.load(str(checkpoint_path_obj), map_location=device_obj)
            model.attention_head.load_state_dict(state_dict)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}")
        
        # Move to device and set eval mode
        model = model.to(device_obj)
        model.eval()
        
        return model
    
    @staticmethod
    def load_distildire_model(
        checkpoint_path: str,
        device: str = "cuda"
    ) -> nn.Module:
        """
        Load DistilDIRE model
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to load on
        
        Returns:
            Loaded model
        """
        import sys
        
        try:
            # Add DistilDIRE to path
            distildire_path = Path("github_model_srcs/DistilDIRE").resolve()
            if not distildire_path.exists():
                raise FileNotFoundError(f"DistilDIRE directory not found at {distildire_path}")
            
            distildire_str = str(distildire_path)
            
            # Clear any cached modules to avoid conflicts
            modules_to_clear = [key for key in sys.modules.keys() if 'networks' in key or 'distill' in key]
            for mod in modules_to_clear:
                del sys.modules[mod]
            
            # Add to path
            if distildire_str not in sys.path:
                sys.path.insert(0, distildire_str)
            
            # Import model
            from github_model_srcs.DistilDIRE.networks.distill_model import DistilDIRE
            
            # Remove from path to avoid future conflicts
            if distildire_str in sys.path:
                sys.path.remove(distildire_str)
                
        except ImportError as e:
            raise ImportError(f"Cannot import DistilDIRE: {str(e)}")
        except FileNotFoundError as e:
            raise ImportError(f"DistilDIRE path error: {str(e)}")
        
        device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
        
        try:
            # Create model
            model = DistilDIRE(device=device_obj)
            
            # Load checkpoint
            checkpoint_path_obj = Path(checkpoint_path)
            if not checkpoint_path_obj.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            state_dict = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
            model_dict = state_dict.get("model", state_dict)
            
            # Remove 'module.' prefix if present
            model_dict = {
                k.replace("module.", ""): v 
                for k, v in model_dict.items()
            }
            
            model.load_state_dict(model_dict, strict=False)
            model = model.to(device_obj)
            model.eval()
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Error loading DistilDIRE checkpoint: {str(e)}")


# ==================== INFERENCE UTILITIES ====================

class InferenceHelper:
    """Helper functions for batch inference and metrics"""
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Apply sigmoid function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def get_predictions(
        logits: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert logits to probabilities and predictions
        
        Args:
            logits: Logits from model
            threshold: Classification threshold
        
        Returns:
            (probabilities, predictions) both as numpy arrays
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        
        probs = InferenceHelper.sigmoid(logits)
        preds = (probs >= threshold).astype(int)
        
        return probs, preds
    
    @staticmethod
    def prepare_metrics_batch(
        true_labels: np.ndarray,
        predicted_probs: np.ndarray,
        threshold: float = 0.5
    ) -> Dict:
        """
        Calculate metrics for batch evaluation
        
        Args:
            true_labels: Ground truth labels [0=real, 1=fake]
            predicted_probs: Predicted probabilities [0, 1]
            threshold: Classification threshold
        
        Returns:
            Dictionary with metrics
        """
        try:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                roc_auc_score, confusion_matrix
            )
        except ImportError:
            raise ImportError("scikit-learn required. Install: pip install scikit-learn")
        
        preds = (predicted_probs >= threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(true_labels, preds),
            'precision': precision_score(true_labels, preds, zero_division=0),
            'recall': recall_score(true_labels, preds, zero_division=0),
            'f1': f1_score(true_labels, preds, zero_division=0),
            'roc_auc': roc_auc_score(true_labels, predicted_probs),
            'confusion_matrix': confusion_matrix(true_labels, preds),
        }
        
        return metrics
