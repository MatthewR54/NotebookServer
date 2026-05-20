import torch
import wandb
import os
import copy
import argparse
from tqdm import tqdm
import torch.multiprocessing as mp

from model_utils import epoch_loop, load_model
from data.dataset import create_dataloader
from argument_utils import get_train_args
            
def train(model, train_loader, val_loader, optimizer, device, args):
    """
    Classic training loop for model training.
    """
    if args.log:
        wandb.init(
            project=f"ufdt-train-{args.name}",
            config={
                "dataset": args.name,
                "epochs": args.niter
            },
        )
    
    best_val_acc = 0
    best_model_path = ""
    loss_history = []

    for epoch in tqdm(range(args.niter)):
        train_loss, train_epoch_metrics = epoch_loop(train_loader, model, device, optimizer, is_train=True, is_logging=args.log, loss_multiplier=args.loss_mult)
        val_loss, val_epoch_metrics = epoch_loop(val_loader, model, device, optimizer, is_train=False, is_logging=args.log)
        
        loss_history.append({'epoch': epoch, 'train_loss': train_loss.item(), 'val_loss': val_loss.item()})

        if val_epoch_metrics['acc'] >= best_val_acc:
            print(f"Best model found at epoch {epoch}")
            best_val_acc = val_epoch_metrics['acc']
            file_name = "epoch_{}{}.pth".format(epoch, 
                                                f"_{val_epoch_metrics['acc']:03.3f}" if val_epoch_metrics and 'acc' in val_epoch_metrics else "")
            best_model_path = os.path.join(args.checkpoints_dir, args.name, file_name)

            # serialize model and optimizer to dict
            state_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict() # TODO: we are saving the optimizer state, but aren't using it for anyting.
                # in the future, if we want to keep training from a state, we should use this and also save the total number of training steps.
            }
            torch.save(state_dict, best_model_path)

    print(f"Best model saved at: {best_model_path}")
    
    # Save loss history
    loss_log_path = os.path.join(args.checkpoints_dir, args.name, "loss_log.txt")
    with open(loss_log_path, 'w') as f:
        for entry in loss_history:
            f.write(f"Epoch {entry['epoch']}: Train Loss = {entry['train_loss']:.4f}, Val Loss = {entry['val_loss']:.4f}\n")
    print(f"Loss history saved at: {loss_log_path}")

    return best_model_path

def _get_val_args(train_args):
    """
    Replaces args for validation set in order to create the correct Dataset.
    """
    val_args = copy.deepcopy(train_args)
    val_args.is_train = False
    # val_args.batch_size = 1
    # for validation, we reset the real_list_paths and the fake_list_paths arguments
    # with the validation ones passed in. This is so we can create the right dataloaders for validation sets.
    # ImageDataset expects the datasets to be listed under "real_list_paths" and "fake_list_paths" arguments.
    val_args.real_list_paths = val_args.val_real_list_paths 
    val_args.fake_list_paths = val_args.val_fake_list_paths 

    return val_args

def initialize(args):
    """
    Sets up model and data for training.
    """
    # we need to redefine separate train_args and val_args to create our ImageDatasets.
    # This is a bit hacky, but works best with the current set up.
    val_args = _get_val_args(args)
    
    model = load_model(args).to("cpu", non_blocking=True)

    # Freeze the backbone
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False

    train_loader = create_dataloader(args)
    val_loader = create_dataloader(val_args)

    print(f"created train dataset with {len(train_loader.dataset)} images")
    print(f"created validation dataset with {len(val_loader.dataset)} images")

    os.makedirs(os.path.join(args.checkpoints_dir, args.name), exist_ok=True)

    # TODO: can change this to be a more flexible parameter in arguments_utils
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(args.beta1, 0.999))

    return model, train_loader, val_loader, optimizer

def main():
    parser = argparse.ArgumentParser()
    train_args = get_train_args(parser)
    model, train_loader, val_loader, optimizer = initialize(train_args)
    best_model_path = train(model, train_loader, val_loader, optimizer, "cpu", train_args)
    print(f"best model path found: {best_model_path}")
 

if __name__=="__main__":
    mp.set_start_method('spawn')    
    main()
    
