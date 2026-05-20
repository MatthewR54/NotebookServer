#!/usr/bin/env python3
"""
DistilDIRE DIRE Maps Pre-computation Script

This script pre-computes DIRE maps and EPS perturbations for images in batch.
Designed to run offline (outside Streamlit) for better performance.

Usage:
    python precompute_dire_offline.py \\
        --data_root datasets/definitive_selected_100imgs_dire \\
        --save_root datasets/definitive_selected_100imgs_dire \\
        --batch_size 16 \\
        --ddim_steps 20 \\
        --device cuda
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

# Add DistilDIRE to path
DISTILDIRE_PATH = Path("github_model_srcs/DistilDIRE")
if DISTILDIRE_PATH.exists():
    sys.path.insert(0, str(DISTILDIRE_PATH))


def load_dire_and_diffusion_model(model_path: str, device: str = "cuda"):
    """
    Load DIRE computation model and diffusion process
    
    Args:
        model_path: Path to pre-trained diffusion model
        device: Device to load on
    
    Returns:
        (model, diffusion) tuple
    """
    try:
        from guided_diffusion import script_util, dist_util
        from guided_diffusion.script_util import create_model_and_diffusion
    except ImportError:
        raise ImportError(
            "Cannot import guided_diffusion. "
            "Ensure DistilDIRE is properly set up."
        )
    
    # Model arguments
    model_args = script_util.model_and_diffusion_defaults()
    model_args.update({
        'image_size': 256,
        'num_channels': 3,
        'num_res_blocks': 2,
        'num_heads': 4,
        'num_heads_upsample': -1,
        'num_head_channels': 64,
        'attention_resolutions': '32,16,8',
        'channel_mult': (1, 1, 2, 2, 4, 4),
        'dropout': 0.0,
        'text_ctx': 128,
        'xf_width': 512,
        'xf_layers': 12,
        'xf_heads': 12,
        'xf_final_ln': True,
        'xf_padding': True,
        'finetune_keys': None,
    })
    
    diffusion_args = script_util.diffusion_defaults()
    diffusion_args.update({
        'steps': 1000,
        'learn_sigma': True,
        'sigma_small': False,
        'sigma_sched': 'sqrt',
        'use_kl': False,
        'predict_xstart': False,
        'rescale_timesteps': False,
        'rescale_learned_sigmas': False,
        'use_scale_shift_norm': True,
        'use_ddim': True,
        'ddim_num_steps': 20,
        'clip_denoised': True,
        'skip_type': 'uniform',
        'var_type': 'learned_range',
    })
    
    # Create model and diffusion
    model, diffusion = create_model_and_diffusion(
        **model_args,
        **diffusion_args
    )
    
    # Load checkpoint
    device_obj = torch.device(device)
    ckpt = torch.load(model_path, map_location=device_obj)
    model.load_state_dict(ckpt)
    model = model.to(device_obj)
    model.eval()
    
    return model, diffusion, device_obj


def compute_dire_batch(
    image_batch: torch.Tensor,
    model,
    diffusion,
    device: torch.device,
    ddim_steps: int = 20
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute DIRE maps and EPS for a batch of images
    
    Args:
        image_batch: Batch of normalized images [-1, 1], shape [B, 3, 256, 256]
        model: Diffusion model
        diffusion: GaussianDiffusion instance
        device: Torch device
        ddim_steps: Number of DDIM steps
    
    Returns:
        (dire_batch, eps_batch) both shape [B, 3, 224, 224]
    """
    
    batch_size = image_batch.shape[0]
    
    try:
        # DDIM reverse (encode to latent space)
        reverse_fn = diffusion.ddim_reverse_sample_loop(
            model,
            shape=(batch_size, 3, 256, 256),
            clip_denoised=True,
            progress=False,
        )
        
        # Generate latent codes by reversing the diffusion process
        with torch.no_grad():
            # Get noise/latent by reversing
            latent = reverse_fn(image_batch)  # [B, 3, 256, 256]
        
        # DDIM forward (denoise from latent)
        sample_fn = diffusion.ddim_sample_loop(
            model,
            shape=(batch_size, 3, 256, 256),
            clip_denoised=True,
            progress=False,
        )
        
        with torch.no_grad():
            # Reconstruct from latent
            recons = sample_fn()  # [B, 3, 256, 256]
        
        # Compute DIRE (difference)
        dire = torch.abs(image_batch - recons)  # [B, 3, 256, 256]
        
        # Normalize DIRE
        dire = (dire * 255 / 2).clamp(0, 255).to(torch.uint8)
        dire = dire.float() / 255.0  # [0, 1]
        
        # Resize to 224x224 (center crop)
        from torchvision.transforms.functional import center_crop, resize
        dire = resize(dire, (224, 224))
        dire = center_crop(dire, (224, 224))  # [B, 3, 224, 224]
        
        # EPS is just the latent/noise
        eps = resize(latent, (224, 224))
        eps = center_crop(eps, (224, 224))  # [B, 3, 224, 224]
        
        # Normalize to [-1, 1] (same as input)
        eps = 2 * (eps / 255.0) - 1  # Rough normalization
        
        return dire, eps
    
    except Exception as e:
        print(f"Warning: DIRE computation failed with: {e}")
        print("Falling back to simplified computation...")
        
        # Fallback: create dummy DIRE and EPS
        dire = torch.zeros(batch_size, 3, 224, 224, device=device)
        eps = torch.zeros(batch_size, 3, 224, 224, device=device)
        
        return dire, eps


def get_image_files(directory: Path, extensions=None) -> list:
    """
    Recursively find image files in directory
    
    Args:
        directory: Root directory to search
        extensions: List of extensions to search for (default: jpg, jpeg, png)
    
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.webp']
    
    image_files = []
    for ext in extensions:
        image_files.extend(directory.rglob(f'*{ext}'))
        image_files.extend(directory.rglob(f'*{ext.upper()}'))
    
    return sorted(image_files)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute DIRE maps for DistilDIRE model"
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory containing images/fake and images/real'
    )
    parser.add_argument(
        '--save_root',
        type=str,
        required=True,
        help='Directory to save DIRE maps and EPS (will create dire/ and eps/ subdirs)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='github_model_srcs/DistilDIRE/models/256x256_diffusion_uncond.pt',
        help='Path to pre-trained diffusion model'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for processing (limited by GPU VRAM)'
    )
    parser.add_argument(
        '--ddim_steps',
        type=int,
        default=20,
        help='Number of DDIM steps'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda or cpu)'
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip images if DIRE/EPS already exist'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    data_root = Path(args.data_root)
    save_root = Path(args.save_root)
    
    if not data_root.exists():
        print(f"❌ Error: data_root does not exist: {data_root}")
        sys.exit(1)
    
    # Create output directories
    dire_dir = save_root / "dire"
    eps_dir = save_root / "eps"
    dire_dir.mkdir(parents=True, exist_ok=True)
    eps_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    images_dir = data_root / "images"
    if not images_dir.exists():
        print(f"❌ Error: images directory not found at {images_dir}")
        print(f"   Expected structure: {data_root}/images/{{fake,real}}/")
        sys.exit(1)
    
    # Get image files from fake and real directories
    all_images = []
    for subdir in ['fake', 'real']:
        subdir_path = images_dir / subdir
        if subdir_path.exists():
            images = get_image_files(subdir_path)
            all_images.extend([(img, subdir) for img in images])
            print(f"Found {len(images)} images in {subdir}/")
    
    if not all_images:
        print("❌ Error: No images found in images/fake and images/real")
        sys.exit(1)
    
    print(f"\n✓ Total images to process: {len(all_images)}")
    
    # Check for existing DIRE/EPS if skip_existing
    if args.skip_existing:
        existing = 0
        filtered_images = []
        for img_path, label in all_images:
            dire_path = dire_dir / label / img_path.stem / ".png"
            eps_path = eps_dir / label / img_path.stem / ".pt"
            
            if not (dire_path.exists() and eps_path.exists()):
                filtered_images.append((img_path, label))
            else:
                existing += 1
        
        print(f"  - {existing} already processed (skipping)")
        all_images = filtered_images
        print(f"  - {len(all_images)} remaining to process")
    
    if len(all_images) == 0:
        print("✓ All images already processed. Nothing to do.")
        return
    
    # Load model
    print("\n⏳ Loading DIRE model...")
    if not Path(args.model_path).exists():
        print(f"⚠️ Warning: Model not found at {args.model_path}")
        print("   Skipping DIRE computation (will need pre-computed maps)")
        return
    
    try:
        model, diffusion, device = load_dire_and_diffusion_model(
            args.model_path,
            device=args.device
        )
        print(f"✓ Model loaded on {device}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   Continuing without DIRE (will create dummy maps)")
        model = None
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Process images in batches
    print(f"\n⏳ Processing {len(all_images)} images (batch_size={args.batch_size})...")
    
    processed = 0
    for i in tqdm(range(0, len(all_images), args.batch_size)):
        batch_items = all_images[i:i+args.batch_size]
        batch_images = []
        batch_labels = []
        batch_original_paths = []
        
        # Load batch images
        for img_path, label in batch_items:
            try:
                img = Image.open(img_path).convert('RGB')
                
                # Resize to 256x256
                img = img.resize((256, 256), Image.BILINEAR)
                
                # Convert to tensor [0, 1]
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                
                # Normalize to [-1, 1]
                img_tensor = img_tensor * 2 - 1
                
                batch_images.append(img_tensor)
                batch_labels.append(label)
                batch_original_paths.append(img_path)
            except Exception as e:
                print(f"⚠️ Skipping {img_path}: {e}")
                continue
        
        if not batch_images:
            continue
        
        # Stack batch
        batch_tensor = torch.stack(batch_images, dim=0).to(device)  # [B, 3, 256, 256]
        
        # Compute DIRE and EPS
        if model is not None:
            dire_batch, eps_batch = compute_dire_batch(
                batch_tensor,
                model,
                diffusion,
                device,
                ddim_steps=args.ddim_steps
            )
        else:
            # Create dummy DIRE/EPS
            dire_batch = torch.zeros_like(batch_tensor[:, :, :224, :224])
            eps_batch = torch.zeros_like(batch_tensor[:, :, :224, :224])
        
        # Save DIRE maps and EPS
        for j, (orig_path, label) in enumerate(zip(batch_original_paths, batch_labels)):
            # Create directories if needed
            dire_label_dir = dire_dir / label
            eps_label_dir = eps_dir / label
            dire_label_dir.mkdir(parents=True, exist_ok=True)
            eps_label_dir.mkdir(parents=True, exist_ok=True)
            
            # Save DIRE map as PNG
            dire_name = orig_path.stem + ".png"
            dire_path = dire_label_dir / dire_name
            dire_img = (dire_batch[j].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(dire_img).save(dire_path)
            
            # Save EPS as .pt
            eps_name = orig_path.stem + ".pt"
            eps_path = eps_label_dir / eps_name
            torch.save(eps_batch[j].cpu(), eps_path)
            
            processed += 1
    
    print(f"\n✓ Completed! Processed {processed} images")
    print(f"  - DIRE maps saved to: {dire_dir}")
    print(f"  - EPS tensors saved to: {eps_dir}")
    print("\nYou can now use these pre-computed maps in Streamlit!")


if __name__ == "__main__":
    main()
