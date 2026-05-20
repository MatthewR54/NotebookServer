import torch
import torch.nn as nn
import wandb
from eval_utils import calculate_metrics
from tqdm import tqdm
import numpy as np

from models import UniversalFakeDetectv2

def load_model(args):
    """
    Loads model architecture and optionally loads a checkpoint.    
    """
    model = UniversalFakeDetectv2()
    if args.ckpt is not None:
        try:
            ckpt = torch.load(args.ckpt, weights_only=True)
            if "model" in ckpt.keys():
                model.load_state_dict(ckpt['model'])
            else:
                model.load_state_dict(torch.load(args.ckpt, weights_only=True))
        except:
            raise ValueError(f"provided ckpt must match the model architecture {args.model_arch}.")

    return model

def epoch_loop(data_loader, model, device, optimizer, is_train, is_logging, loss_multiplier=1):
    """
    Single epoch over data for training or evaluation.
    """
    
    # New loss function with custom weight for negative examples
    def custom_loss(output, labels):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(output, labels)
        weights = torch.where(labels == 0, loss_multiplier, 1.0)
        return (bce_loss * weights).mean()
    
    print(f"Starting epoch_loop with dataloader len: {len(data_loader)}, dataset size: {len(data_loader.dataset)}, batch_size: {data_loader.batch_size}")

    running_loss = 0
    num_total = 0
    predictions = []
    true_labels = []

    model.train(is_train)
    context = torch.enable_grad if is_train else torch.no_grad

    with context():
        for batch_idx, data in enumerate(tqdm(data_loader)):
            input = data[0].to(device, non_blocking=True)
            labels = data[1].to(torch.float32, non_blocking=True).to(device, non_blocking=True)
            batch_size = input.size(0)
            
            if batch_size == 0:
                print(f"Warning: Encountered zero-sized batch at index {batch_idx}")
                continue
            
            num_total += batch_size

            output = model(input).view(-1)
            batch_loss = custom_loss(output, labels)
            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                print(f"Warning: Invalid batch_loss at index {batch_idx}: {batch_loss}")
                continue
            
            running_loss += batch_loss * batch_size

            prefix = "batch_train" if is_train else "batch_val"
            preds_cpu = output.sigmoid().cpu().detach().numpy()
            labels_cpu = labels.cpu().numpy()
            predictions.extend(preds_cpu)
            true_labels.extend(labels_cpu)

            batch_results = calculate_metrics(labels_cpu, preds_cpu)

            if is_train and is_logging:
                wandb.log({
                    f"{prefix}_loss": batch_loss.item(),
                    f"{prefix}_f1": batch_results['f1'],
                    f"{prefix}_recall": batch_results['recall'],
                    f"{prefix}_precision": batch_results['precision'],
                    f"{prefix}_accuracy": batch_results['acc'],
                    f"{prefix}_average_precision": batch_results['average_precision'],
                })

            if is_train:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
    
    if num_total == 0:
        print("Warning: num_total is 0, no samples were processed")
        running_loss = 0
    else:
        running_loss /= num_total
    epoch_results = calculate_metrics(np.array(true_labels), np.array(predictions))
    
    prefix = "train" if is_train else "val"
    print(f"{prefix.capitalize()} Metrics - F1 Score: {epoch_results['f1']:.3f}, "
          f"Recall: {epoch_results['recall']:.3f}, Precision: {epoch_results['precision']:.3f}, "
          f"Accuracy: {epoch_results['acc']:.3f}, "
          f"Average Precision: {epoch_results['average_precision']:.3f}")
    if is_logging:
        wandb.log({
            f"{prefix}_loss": running_loss,
            f"{prefix}_f1": epoch_results['f1'],
            f"{prefix}_recall": epoch_results['recall'],
            f"{prefix}_precision": epoch_results['precision'],
            f"{prefix}_accuracy": epoch_results['acc'],
            f"{prefix}_average_precision": epoch_results['average_precision'],
        })
    return running_loss, epoch_results
