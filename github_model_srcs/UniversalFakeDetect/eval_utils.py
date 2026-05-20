import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, average_precision_score, confusion_matrix, precision_recall_curve
from PIL import Image


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    This method calculates true media style metrics. 
    """
    results_dict = {}
    results_dict['r_acc'] = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > threshold)
    results_dict['f_acc'] = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > threshold)
    results_dict['acc'] = accuracy_score(y_true, y_pred > threshold)

    binary_preds = [int(i > threshold) for i in y_pred]
    # Check for single-class predictions
    unique_preds = np.unique(binary_preds)
    if len(unique_preds) == 1:
        # Set confusion matrix values based on the single predicted class
        if unique_preds[0] == 0:  # All predictions are negative
            results_dict['true_negatives'] = np.sum(y_true == 0)
            results_dict['false_negatives'] = np.sum(y_true == 1)
            results_dict['false_positives'] = 0
            results_dict['true_positives'] = 0
            # Set precision to 0 when all predictions are negative
            results_dict['precision'] = 0.0
            # Set recall to 0 when all predictions are negative
            results_dict['recall'] = 0.0
            # Set FNR to 1 when all predictions are negative
            results_dict['fnr'] = 1.0
        else:  # All predictions are positive
            results_dict['true_negatives'] = 0
            results_dict['false_negatives'] = 0
            results_dict['false_positives'] = np.sum(y_true == 0)
            results_dict['true_positives'] = np.sum(y_true == 1)
            # Set precision based on the proportion of true positives
            results_dict['precision'] = np.sum(y_true == 1) / len(y_true)
            # Set recall to 1 when all predictions are positive
            results_dict['recall'] = 1.0
            # Set FNR to 0 when all predictions are positive
            results_dict['fnr'] = 0.0
    else:
        # Normal case - predictions contain both classes
        conf_matrix = confusion_matrix(y_true, binary_preds).ravel()
        results_dict['true_negatives'], results_dict['false_positives'], \
        results_dict['false_negatives'], results_dict['true_positives'] = conf_matrix
        
        results_dict['precision'] = precision_score(y_true, binary_preds)
        results_dict['recall'] = recall_score(y_true, binary_preds)
        
        # Calculate FNR
        denominator = (results_dict['false_negatives'] + results_dict['true_positives'])
        results_dict['fnr'] = round(float(results_dict['false_negatives'] / denominator), 3) if denominator != 0 else 0.0
    
    # Calculate F1 score - handle case where precision and recall are both 0
    if results_dict['precision'] == 0 and results_dict['recall'] == 0:
        results_dict['f1'] = 0.0
    else:
        results_dict['f1'] = f1_score(y_true, binary_preds)
    
    # Calculate FPR - handle division by zero
    denominator = (results_dict['true_negatives'] + results_dict['false_positives'])
    results_dict['fpr'] = round(float(results_dict['false_positives'] / denominator), 3) if denominator != 0 else 0.0
    
    # Average precision uses probabilities directly, so less affected by threshold
    results_dict['average_precision'] = average_precision_score(y_true, y_pred)
    
    return results_dict

def plot_prediction_dist(y_pred, y_true, output_dir):
    """
    Plot distribution of predictions. Also plots by label.
    """
    plt.hist(y_pred, 10)  
    plt.grid(True)
    plt.xlabel("Prediction")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pred-dist.png")

    plt.clf()
    reals = y_pred[y_true == 0]
    fakes = y_pred[y_true == 1]
    plt.hist(reals, alpha=0.5, label="Real")
    plt.hist(fakes, alpha=0.5, label="Fake")
    plt.xlabel("Prediction")
    plt.ylabel("Number of Samples")
    plt.legend()
    plt.savefig(f"{output_dir}/pred-dist-by-class.png")

def get_files_by_classification(y_pred, y_true, thres, img_paths, output_dir, save_files=False):
    """
    Save lists of filenames by classification (false positive, false negative, etc.)
    Saves copies of the actual images to directories if specified.
    """
    false_positive_indices = np.where((y_pred > thres) & (y_true == 0))[0]
    false_negative_indices = np.where((y_pred <= thres) & (y_true == 1))[0]
    true_positive_indices = np.where((y_pred > thres) & (y_true == 1))[0]
    true_negative_indices = np.where((y_pred <= thres) & (y_true == 0))[0]
    idxs = [false_positive_indices, false_negative_indices, true_positive_indices, true_negative_indices]

    fps = output_dir + f"/false_positives_thres-{thres}.txt"
    fns = output_dir + f"/false_negatives_thres-{thres}.txt"
    tps = output_dir + f"/true_positives_thres-{thres}.txt"
    tns = output_dir + f"/true_negatives_thres-{thres}.txt"

    files = [fps, fns, tps, tns]
    
    for i, file in enumerate(files):
        with open(file, 'w') as f:
            for idx in idxs[i]:
                img_path = img_paths[idx]
                f.write(img_path + '\n')

    if save_files:
        # create folders for failure cases.
        fp_dir = output_dir + f"/false_positives_thres-{thres}"
        fn_dir = output_dir + f"/false_negatives_thres-{thres}"
        tp_dir = output_dir + f"/true_positives_thres-{thres}"
        tn_dir = output_dir + f"/true_negatives_thres-{thres}"
        os.makedirs(fp_dir, exist_ok=True)
        os.makedirs(fn_dir, exist_ok=True)
        os.makedirs(tp_dir, exist_ok=True)
        os.makedirs(tn_dir, exist_ok=True)
        dirs = [fp_dir, fn_dir, tp_dir, tn_dir]

         # save images to each class' directory
        for dir, idx_list in zip(dirs, idxs):
            for idx in idx_list:
                img_path = img_paths[idx]
                image = Image.open(img_path).convert("RGB")
                img = image.copy()
                image.close()

                filename = os.path.basename(img_path)
                save_path = os.path.join(dir, filename)
                img.save(save_path)

def find_best_threshold(y_true, y_pred, output_dir):
    """
    Find the best threshold given the predictions and true labels, based on F1 score and AUC.

    Creates a PR curve for the data and saves. 
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = [f1_score(y_true, y_pred > threshold) for threshold in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]

    plt.figure(figsize=(8,7))
    plt.plot(recalls, precisions, marker='.')

    target_precisions = [0.75, 0.8]
    colors = ['green', 'blue']
    for target_precision, color in zip(target_precisions, colors):
        # Find the index closest to the target precision
        closest_precision_index = np.argmin(np.abs(precisions - target_precision))
        if closest_precision_index > len(thresholds)-1:
            closest_precision_index -= 1
        closest_threshold = thresholds[closest_precision_index]

        # Plot the corresponding recall and precision as a large dot
        plt.scatter(recalls[closest_precision_index], precisions[closest_precision_index], s=100, c=color, label=f'Precision: {round(target_precision, 2)}, Recall: {round(recalls[closest_precision_index], 2)}')

        # Annotate the threshold and precision point
        plt.annotate(f'Thresh: {round(closest_threshold, 2)}',
                    (recalls[closest_precision_index], precisions[closest_precision_index]),
                    textcoords="offset points", xytext=(10,10), ha='center')
    
    # Find the index for the threshold closest to 0.3
    specific_threshold = 0.35
    closest_index = np.argmin(np.abs(thresholds - specific_threshold))
    specific_precision = precisions[closest_index]
    specific_recall = recalls[closest_index]

    # Plot the corresponding recall and precision as a large dot
    plt.scatter(specific_recall, specific_precision, s=100, c='orange', label=f'Precision: {round(specific_precision, 2)}, Recall: {round(specific_recall, 2)}')

    # Annotate the threshold and precision point
    plt.annotate(f'Thresh: {round(thresholds[closest_index], 2)}',
                (specific_recall, specific_precision),
                textcoords="offset points", xytext=(10,10), ha='center')
    
    best_th_precision = precisions[np.argmax(f1_scores)]
    best_th_recall = recalls[np.argmax(f1_scores)]

    plt.scatter(best_th_recall, best_th_precision, s=100, c='red', label=f'Best threshold (F1 Score): {round(best_threshold, 2)}, Precision: {round(best_th_precision, 2)}, Recall: {round(best_th_recall, 2)}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pr-curve.png")
    
    return best_threshold

def save_results(output_dir, results_dict_default, results_dict_best_thres=None):
    with open(os.path.join(output_dir,'stats.txt'), 'a') as f:
        f.write('stats (0.5 threshold):\n')
        f.write(f'overall acc: {round(results_dict_default['acc']*100, 2)}, fake acc (recall): {round(results_dict_default['f_acc']*100, 2)}, real acc: {round(results_dict_default['r_acc']*100, 2)}\n')
        f.write(f"F1: {round(results_dict_default['f1'], 2)}, FPR: {round(results_dict_default['fpr'], 2)}, FNR: {round(results_dict_default['fnr'], 2)}, precision: {round(results_dict_default['precision'], 2)}, average precision: {round(results_dict_default['average_precision']*100, 1)}")

        if results_dict_best_thres is not None:
            f.write('\n')
            f.write(f'best stats ({results_dict_best_thres['best_thres']} threshold): ')
            f.write(f'overall acc: {round(results_dict_best_thres['acc']*100, 2)}, fake acc (recall): {round(results_dict_best_thres['f_acc']*100, 2)}, real acc: {round(results_dict_best_thres['r_acc']*100, 2)}\n')
            f.write(f"F1: {round(results_dict_best_thres['f1'], 2)}, FPR: {round(results_dict_best_thres['fpr'], 2)}, FNR: {round(results_dict_best_thres['fnr'], 2)}, precision: {round(results_dict_best_thres['precision'], 2)}, average precision: {round(results_dict_best_thres['average_precision']*100, 1)}")
        
