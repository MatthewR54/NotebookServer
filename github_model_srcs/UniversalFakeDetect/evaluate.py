import argparse
import os
import torch
from tqdm import tqdm
import numpy as np

from model_utils import load_model
from data.dataset import create_dataloader
from argument_utils import get_eval_args
from eval_utils import calculate_metrics, plot_prediction_dist, find_best_threshold, get_files_by_classification, save_results

def evaluate(model, loader, output_dir, find_thres=False, save_files=False):
    """
    Runs through data and gets evaluation metrics.
    Note: we could use the epoch_loop function from model_utils, but this is simpler.
    Args:
        model: Model to evaluate
        loader: DataLoader with evaluation data
        output_dir: location where results files should be created
        find_thres: if set to True, evaluation will consider both the default threshold of 0.5 and 
            the best threshold found using F1 scores.
        save_files: if set to True, evaluation will create copies of the images and place them in 
            subdirectories based on their classification.
    """
    with torch.no_grad():
        y_true, y_pred, img_paths = [], [], []
        for sample in tqdm(loader):
            embeddings, label, paths = sample
            y_pred.extend(model(embeddings).view(-1).sigmoid().cpu().detach().numpy())  
            y_true.extend(label.cpu().numpy())
            img_paths.extend(paths)

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    assert len(y_true) != 0
    assert len(y_pred) != 0

    # plot distributions of predictions
    plot_prediction_dist(y_pred, y_true, output_dir)

    results_dict = calculate_metrics(y_true, y_pred)
    default_thres = 0.5
    get_files_by_classification(y_pred, y_true, default_thres, img_paths, output_dir, save_files)

    # optionally calculate best threshold and corresponding results
    best_thres_results_dict = None
    if find_thres:
        best_thres = find_best_threshold(y_true, y_pred, output_dir)
        best_thres_results_dict = calculate_metrics(y_true, y_pred, threshold=best_thres)
        best_thres_results_dict['best_thres'] = best_thres
        get_files_by_classification(y_pred, y_true, best_thres, img_paths, output_dir, save_files)

    # print out results to results folder
    save_results(output_dir, results_dict, best_thres_results_dict)
    
def initialize(args):
    """
    Sets up model and data according to arguments.
    """
    model = load_model(args).to("cpu", non_blocking=True)
    model.eval()
    loader = create_dataloader(args)
    print(f"created evaluation dataset with {len(loader.dataset)} images")
    output_dir = args.result_folder
    os.makedirs(output_dir, exist_ok=True)
    return model, loader, output_dir

def main():
    # get arguments
    parser = argparse.ArgumentParser()
    eval_args = get_eval_args(parser)
    
    # load model and data
    model, loader, output_dir = initialize(eval_args)

    # evaluate
    evaluate(model, loader, output_dir, eval_args.find_thres, eval_args.save_files)

if __name__=="__main__":
    main()
