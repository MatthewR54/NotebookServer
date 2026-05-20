import gc
import torch
from concurrent.futures import ThreadPoolExecutor
import torchvision.transforms as transforms
import random
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from functools import partial
from pathlib import Path
import os
import logging
from transformers import CLIPProcessor, CLIPModel
from .data_utils import data_augment

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

def recursively_read(rootdir, must_contain="", exts=["png", "jpg", "jpeg", "webp", "dat", "heic", "jfif"]):
    out = []
    try:
        rootdir = Path(rootdir)
        if not rootdir.exists():
            logger.error(f"Directory not found: {rootdir}")
            return out
            
        for path in rootdir.rglob("*"):
            if path.suffix.lower()[1:] in exts and must_contain in str(path):
                out.append(str(path.absolute()))
    except Exception as e:
        logger.error(f"Error reading directory {rootdir}: {str(e)}")
    return out

class ImageDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.model = CLIPModel.from_pretrained(args.backbone).to("cpu", non_blocking=True)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(args.backbone)
        self.num_retries = 10
        if args.backbone == "openai/clip-vit-large-patch14":
            self.res = 224
        elif args.backbone == "openai/clip-vit-large-patch14-336":
            self.res = 336
        else:
            logger.error(f"Unknown backbone: {args.backbone}")
            raise ValueError("Unknown backbone")
        
        print("Real list paths:", args.real_list_paths)
        print("Fake list paths:", args.fake_list_paths)
        real_list, fake_list = [], []
        random.seed(1111)
        real_list, fake_list = self._load_files(args, real_list, fake_list)        
        
        self.total_list = real_list + fake_list
        print("Starting dataset size:", len(self.total_list))
        
        random.shuffle(self.total_list)

        self.transform = create_transform(args)
        self.embeddings, failed_images = self._process_images(skip_embeddings=args.data_aug)
        
        # only check embeddings if we've already computed them
        if not args.data_aug:
            for img_path in self.total_list:
                if img_path not in self.embeddings:
                    failed_images.add(img_path)
        
        logger.info(f"Failed images: {len(failed_images)}")
        
        gc.collect()
        
        self.total_list = [path for path in self.total_list if path not in failed_images]
        logger.info(f"Final dataset size: {len(self.total_list)}")
        
        if len(self.total_list) == 0:
            logger.error("All samples were filtered out!")
            return

        self.labels = torch.tensor([
            0 if path in real_list else 1 
            for path in self.total_list
        ])

    def _load_files(self, args, real_list, fake_list):
        """Load file lists from directories or files."""
        for path_list, arg_paths in [(real_list, args.real_list_paths), (fake_list, args.fake_list_paths)]:
            for path in arg_paths:
                if os.path.isdir(path):
                    path_list.extend(recursively_read(path))
                elif os.path.isfile(path):
                    with open(path, "r") as f:
                        for line in f:
                            path_list.append(line.strip())
                else:
                    logger.warning(f"Path not found or is not a file/directory: {path}")
        
        # filter out non-existent files
        real_list = [path for path in real_list if os.path.isfile(path)]
        fake_list = [path for path in fake_list if os.path.isfile(path)]
              
        if args.is_train:
            random.shuffle(real_list)
            random.shuffle(fake_list)
            
        if self.args.class_bal:
            max_len = min(len(real_list), len(fake_list))
            real_list = real_list[:max_len]
            fake_list = fake_list[:max_len]
        else:
            if self.args.num_reals is not None and len(real_list) > self.args.num_reals:
                real_list = real_list[:self.args.num_reals]
            if self.args.num_fakes is not None and len(fake_list) > self.args.num_fakes:
                fake_list = fake_list[:self.args.num_fakes]
            
        return real_list, fake_list
        
    def _load_image_batch(self, image_paths):
        """Load a batch of images in parallel."""
        images, valid_paths, failed_paths = [], [], []
        with ThreadPoolExecutor(max_workers=os.cpu_count() - 3) as executor:
            def load_single_image(path):
                try:
                    img = Image.open(path).convert('RGB')
                    return path, img, None
                except Exception as e:
                    return path, None, str(e)
            
            results = list(executor.map(load_single_image, image_paths))
            
            for path, img, error in results:
                if error is None:
                    images.append(img)
                    valid_paths.append(path)
                else:
                    failed_paths.append((path, error))
        
        return images, valid_paths, failed_paths

    def _process_images(self, skip_embeddings=False):
        """Process images in batches."""
        embeddings = {}
        failed_images = set()
        
        with tqdm(total=len(self.total_list), desc="Processing images") as pbar:
            for i in range(0, len(self.total_list), self.args.batch_size):
                batch_paths = self.total_list[i:i + self.args.batch_size]
                images, valid_paths, failed_loads = self._load_image_batch(batch_paths)
                
                for path, error in failed_loads:
                    failed_images.add(path)
                    logger.warning(f"Failed to load {path}: {error}")
                 
                if not skip_embeddings and images:
                    batch_embeddings, error = self._process_batch(images)
                    if error is None:
                        for path, embedding in zip(valid_paths, batch_embeddings):
                            embeddings[path] = embedding
                    else:
                        for path in valid_paths:
                            failed_images.add(path)
                        logger.warning(f"Failed to process batch: {error}")
                
                pbar.update(len(batch_paths))
        
        return embeddings, failed_images

    @torch.no_grad()
    def _process_batch(self, images):
        """Process a batch of images at once."""
        try:
            # Note that while these approaches are very similar having the backbone crop the image results in slightly higher accuracy and is faster
            if self.args.backbone_crop:
                inputs = self.processor(images=images, return_tensors="pt", size=(self.res, self.res), do_rescale=True)
            else:
                inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)
            
            pixel_values = inputs["pixel_values"].to("cpu", non_blocking=True)
            outputs = self.model.get_image_features(pixel_values)
            embeddings = outputs.clone()
            return embeddings, None
        except Exception as e:
            return None, str(e)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        if not self.args.data_aug:
            return self.embeddings[img_path].clone(), self.labels[idx], img_path
        
        # When data augmentation enabled, compute single embedding for the next image that can be loaded
        for _ in range(self.num_retries):
            img_path = self.total_list[idx]
            try:
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                return self._process_batch([img])[0].clone(), self.labels[idx], img_path
            except Exception as e:
                print(f"Failed to open image: {img_path}. Error: {e}")
                idx = (idx + 1) % len(self)
    
    def __len__(self):
        return len(self.total_list)

def create_transform(args):
    transforms_list = []
    if args.data_aug:
        transforms_list += [
            transforms.Lambda(partial(data_augment, args=args)),
        ]
        if not args.no_flip and args.is_train:
            transforms_list.append(transforms.RandomHorizontalFlip())   
    if not args.backbone_crop:
        transforms_list += [
            transforms.Resize((args.res, args.res)),
            transforms.ToTensor(),
        ]    
    return transforms.Compose(transforms_list)

def create_dataloader(args):
    try:
        dataset = ImageDataset(args)
        if len(dataset) == 0:
            raise ValueError("Dataset is empty!")
                
        # make sure batch_size is at most the dataset size
        batch_size = min(args.batch_size, len(dataset))
        logger.info(f"Using batch size: {batch_size} (original: {args.batch_size})")
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=args.is_train,
            drop_last=args.is_train,
            num_workers=args.nworkers if args.is_train else 0,
            prefetch_factor=args.prefetch_factor if args.is_train else None,
            persistent_workers=True if args.is_train else False,
        )    
        return data_loader
    except Exception as e:
        logger.error(f"Failed to create dataloader: {str(e)}")
        raise
    
