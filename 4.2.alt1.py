import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from utils.plotting import plot_loss
from utils.data_loader import get_classes
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import itertools
import seaborn as sns
from tqdm import tqdm
import os
os.makedirs('./classifier/trained_models', exist_ok=True)
tqdm.pandas(disable=True)

print("Running script 4.2")

###############################################
################ CONFIGURATION ################
###############################################

USE_GAN = True          # Set to True to evaluate GAN-based generation
FILTERED = False        # Evaluation with filtered data (REMOVEOUTLIERS = True)
TESTONGENERATED = False # Use generated data as testdata

classes = get_classes()
galaxy_classes = [10, 11, 12, 13]
#galaxy_classes = [31, 32, 33, 34, 35, 36]
learning_rates = [1e-4]
regularization_params = [1e-1]
lambda_values = [0, 0.25, 0.5, 1, 2, 3, 5, 10]
num_experiments = 1
folds = [5]

if galaxy_classes == [31, 32, 33, 34, 35, 36]:
    dataset_sizes = [[1088, 10883, 54416, 108832]]
    num_experiments = 1
    FILTERED = False
    TESTONGENERATED = False
    
elif USE_GAN:
    encoders = ['GAN']
    latent_dim = 100      # (if needed elsewhere)
    dataset_sizes = [[281, 2812, 14064, 28128], [281, 2812, 14064, 28128], [281, 2812, 14064, 28128], [281, 2812, 14064, 28128], [281, 2812, 14064, 28128], [281, 2812, 14064, 28128]] #Need length = folds[0] = 1
else:
    encoders = ['Dual']
    if FILTERED:
        dataset_sizes = [[137, 1376, 6880, 13760], [137, 1376, 6880, 13760], [140, 1408, 7040, 14080],
                     [137, 1377, 6888, 13776], [137, 1376, 6880, 13760]] 
    else:
        dataset_sizes = [[249, 2492, 12464, 24928], [249, 2492, 12464, 24928],
                        [249, 2492, 12464, 24928], [249, 2492, 12464, 24928],
                        [250, 2505, 12528, 25056]]     

if TESTONGENERATED:
    lambda_values = [8]  # To identify and distinguish TESTONGENERATED from other runs

############################################################
################# MERGE MAP GENERATION #####################
############################################################

merge_map = {}
for sizes in dataset_sizes:
    for size in sizes:
        nd = len(str(size))
        if nd <= 4:
            factor = 10 ** (nd - 2)
        else:
            factor = 5 * (10 ** (nd - 3))
        new_rep = round(size / factor) * factor
        merge_map[size] = str(new_rep)

print("merge_map =", merge_map)

###############################################
############# READ IN PICKLE DUMP #############
###############################################

def initialize_metrics(metrics, model_name, subset_size, fold, experiment, lr, reg, lambda_generate):
    subset_size_str = str(subset_size[0]) if isinstance(subset_size, list) else str(subset_size)
    metrics[f"{model_name}_accuracy_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{model_name}_precision_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{model_name}_recall_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{model_name}_f1_score_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{model_name}_history_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{model_name}_all_true_labels_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{model_name}_all_pred_labels_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{model_name}_training_times_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{model_name}_all_pred_probs_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []

def update_metrics(metrics, model_name, subset_size, fold, experiment, lr, reg,
                   accuracy, precision, recall, f1, lambda_generate,
                   history_val, all_true_labels, all_pred_labels, training_times, all_pred_probs):
    subset_size_str = str(subset_size)
    metrics[f"{model_name}_accuracy_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(accuracy)
    metrics[f"{model_name}_precision_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(precision)
    metrics[f"{model_name}_recall_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(recall)
    metrics[f"{model_name}_f1_score_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(f1)
    metrics[f"{model_name}_history_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(history_val)
    metrics[f"{model_name}_all_true_labels_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(all_true_labels)
    metrics[f"{model_name}_all_pred_labels_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(all_pred_labels)
    metrics[f"{model_name}_training_times_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(training_times)
    metrics[f"{model_name}_all_pred_probs_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(all_pred_probs)

tot_metrics = {}
valid_lambda_values = []
directory = './classifier/trained_models_filtered/' if FILTERED else './classifier/trained_models/'

print("Loading metrics from pickle files...")
# Note: Here we iterate over encoders in the list “encoders” which is now set based on USE_GAN.
for lambda_generate in lambda_values:
    try:
        for lr, reg, experiment, encoder, fold in itertools.product(
            learning_rates, regularization_params, range(num_experiments), encoders, folds):
            
            for subset_size in dataset_sizes[fold]:
                metrics_read_path = f'{directory}{galaxy_classes}_{encoder}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}_metrics_data.pkl'
                with open(metrics_read_path, 'rb') as f:
                    metrics_data = pickle.load(f)
                    classifier = metrics_data["classifier"]
                    models = metrics_data["models"]
                    model_name = metrics_data["model_name"]
                    history = metrics_data["history"]
                    loaded_metrics = metrics_data["metrics"]
                    metric_colors = metrics_data["metric_colors"]
                    all_true_labels = metrics_data["all_true_labels"]
                    all_pred_labels = metrics_data["all_pred_labels"]
                    training_times = metrics_data["training_times"]
                    all_pred_probs = metrics_data["all_pred_probs"]
                initialize_metrics(tot_metrics, model_name, subset_size, fold, experiment, lr, reg, lambda_generate)
                update_metrics(
                    tot_metrics, model_name, subset_size, fold, experiment, lr, reg,
                    loaded_metrics[f"{model_name}_accuracy_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0],
                    loaded_metrics[f"{model_name}_precision_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0],
                    loaded_metrics[f"{model_name}_recall_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0],
                    loaded_metrics[f"{model_name}_f1_score_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0],
                    lambda_generate,
                    history,
                    all_true_labels,
                    all_pred_labels,
                    training_times,
                    all_pred_probs
                )
        valid_lambda_values.append(lambda_generate)
    except FileNotFoundError:
        print(f"Metrics file not found at {metrics_read_path}. Ignoring this lambda.")

print("Metrics loaded successfully.")
lambda_values = valid_lambda_values
print("Valid lambda values:", lambda_values)
metrics = tot_metrics

###############################################
########### PLOTTING FUNCTIONS ################
###############################################

def plot_accuracy_vs_lambda(lambda_values, metrics, model_name, dataset_sizes=dataset_sizes,
                            folds=folds, num_experiments=num_experiments,
                            learning_rates=learning_rates, regularization_params=regularization_params,
                            save_dir='./classifier'):
    for encoder in encoders:
        classical_accs = []
        
        for fold, lr, reg, experiment in itertools.product(folds, learning_rates, regularization_params, range(num_experiments)):
            classical_accs.append(metrics[f'{model_name}_accuracy_{dataset_sizes[fold][-1]}_{fold}_{experiment}_{lr}_{reg}_{0}'][0])
            
        classical_acc = np.mean(classical_accs)
        classical_std = np.std(classical_accs)
        gan_accs_means, gan_stds = [], []
        
        for lambda_generate in lambda_values:
            gan_accs_for_one_lambda = []
            for fold, lr, reg, experiment in itertools.product(folds, learning_rates, regularization_params, range(num_experiments)):
                key = f'{model_name}_accuracy_{dataset_sizes[fold][-1]}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}'
                gan_accs_for_one_lambda.append(metrics[key][0])
                
            gan_accs_means.append(np.mean(gan_accs_for_one_lambda))
            gan_stds.append(np.std(gan_accs_for_one_lambda))

        gan_accs_means = np.array(gan_accs_means).flatten()
        gan_stds = np.array(gan_stds).flatten()

        plt.figure(figsize=(8, 5))
        plt.axhline(y=classical_acc, color='orange', linestyle='-', label='classically augmented')
        plt.fill_between(lambda_values, classical_acc - classical_std, classical_acc + classical_std,
                        color='orange', alpha=0.2)
        for i, lambda_val in enumerate(lambda_values):
            color = 'orange' if lambda_val == 0 else 'blue'
            plt.errorbar(lambda_val, gan_accs_means[i], yerr=gan_stds[i], fmt='o', capsize=5,
                        color=color, ecolor=color, label='classically + VAE augmented' if i == 1 else "")
        plt.xlabel(r'$\lambda_{gen}$')
        plt.ylabel('Accuracy')
        plt.xticks(lambda_values, [str(l) for l in lambda_values])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plot_path = f'{save_dir}/{model_name}_{galaxy_classes}_{encoder}_accuracy_vs_lambda.png'
        plt.savefig(plot_path)
        plt.close()

def plot_all_metrics_vs_dataset_size(metrics, model_name, merge_map={249: "250", 250: "250", 2492: "2500", 2505: "2500", 24928: "25000", 25056: "25000"},
    dataset_sizes=dataset_sizes,  folds= folds, num_experiments=num_experiments,
    learning_rates=learning_rates, regularization_params=regularization_params, save_dir='./classifier'):
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    metric_titles = ["Accuracy", "Precision", "Recall", "F1 Score"]

    for encoder in encoders:
        for metric, title in zip(metric_names, metric_titles):
            plt.figure(figsize=(10, 6))
            labels = [f'λ={lambda_val}' for lambda_val in lambda_values]
            for i, (lr, reg, lambda_generate) in enumerate(itertools.product(learning_rates, regularization_params, lambda_values)):
                metric_values_per_category = {}
                for exp in range(num_experiments):
                    for fold in folds:
                        for subset_size in dataset_sizes[fold]:
                            if subset_size not in merge_map:
                                continue
                             
                            category = merge_map[subset_size]
                            if category not in metric_values_per_category:
                                metric_values_per_category[category] = []
                            key = f"{model_name}_{metric}_{subset_size}_{fold}_{exp}_{lr}_{reg}_{lambda_generate}"
                            if key in metrics and metrics[key]:
                                metric_values_per_category[category].append(metrics[key][-1])
                categories = sorted(metric_values_per_category.keys(), key=lambda x: int(x))
                avg_values = []
                std_values = []
                for category in categories:
                    values = metric_values_per_category[category]
                    if values:
                        avg_values.append(np.mean(values))
                        std_values.append(np.std(values))
                if avg_values:
                    avg_values = np.array(avg_values)
                    std_values = np.array(std_values)
                    plt.plot(categories, avg_values, marker='o', linestyle='-', label=labels[i])
                    plt.fill_between(categories, avg_values - std_values, avg_values + std_values, alpha=0.2)
            plt.legend(title='Lambda Values', fontsize=12, loc='best')
            plt.title(f'{title} vs Dataset Size ({model_name})', fontsize=16)
            plt.xlabel('Training Dataset Size', fontsize=14)
            plt.ylabel(title, fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plot_path = f'{save_dir}/{model_name}_{galaxy_classes}_{encoder}_{metric}_vs_dataset_size.png'
            plt.savefig(plot_path)
            plt.close()

def plot_avg_roc_curves(metrics, model_name, dataset_sizes=dataset_sizes, merge_map=merge_map, 
                         folds= folds, num_experiments=num_experiments, 
                        learning_rates=learning_rates, regularization_params=regularization_params, 
                        galaxy_classes=galaxy_classes, 
                        class_descriptions={cls['tag']: cls['description'] for cls in classes}, 
                        save_dir='./classifier'):
    min_label = min(galaxy_classes)
    adjusted_classes = [cls - min_label for cls in galaxy_classes]
    present_descriptions = [desc for lab, desc in zip(galaxy_classes, class_descriptions)]
    for encoder in encoders:
        for lr in learning_rates:
            for reg in regularization_params:
                for lambda_generate in lambda_values:
                    for s in range(len(dataset_sizes[0])):
                        for experiment in range(num_experiments):
                            roc_values = {class_label: [] for class_label in adjusted_classes}
                            for fold in folds:
                                subset_size = dataset_sizes[fold][s]
                                true_labels = metrics[f"{model_name}_all_true_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0]
                                pred_probs = metrics[f"{model_name}_all_pred_probs_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0]
                                true_labels = true_labels[f"{model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"]
                                pred_probs = np.array(pred_probs[f"{model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"])
                                true_labels_bin = label_binarize(true_labels, classes=np.arange(len(adjusted_classes)))
                                for i, class_label in enumerate(adjusted_classes):
                                    try:
                                        fpr, tpr, _ = roc_curve(true_labels_bin[:, i], pred_probs[:, i])
                                        interp_tpr = np.interp(np.linspace(0, 1, 100), fpr, tpr)
                                        roc_values[class_label].append(interp_tpr)
                                    except ValueError as e:
                                        print(f"Error computing ROC for class {class_label}: {e}")
                                        continue
                            fig, ax = plt.subplots(figsize=(6, 5))
                            for class_label, galaxy_class in zip(adjusted_classes, galaxy_classes):
                                if not roc_values[class_label]:
                                    continue
                                tpr_values = np.array(roc_values[class_label])
                                mean_tpr = np.mean(tpr_values, axis=0)
                                std_tpr = np.std(tpr_values, axis=0)
                                mean_auc = auc(np.linspace(0, 1, 100), mean_tpr)
                                ax.plot(np.linspace(0, 1, 100), mean_tpr, lw=2,
                                        label=f'{class_descriptions.get(galaxy_class, "Unknown Class")} ROC (area = {mean_auc:.2f})')
                                ax.fill_between(np.linspace(0, 1, 100), mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2)
                            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                            ax.set_xlim([0.0, 1.0])
                            ax.set_ylim([0.0, 1.05])
                            ax.set_xlabel('False Positive Rate', fontsize=16)
                            ax.set_ylabel('True Positive Rate', fontsize=16)
                            merged_subset_key = merge_map.get(subset_size, str(subset_size))
                            ax.set_title(f'Average ROC Curve - {model_name} \n {merged_subset_key}, Experiment {experiment}', fontsize=14)
                            ax.legend(loc="lower right")
                            os.makedirs(save_dir, exist_ok=True)
                            plt.savefig(f'{save_dir}/{model_name}_{galaxy_classes}_{encoder}_{merged_subset_key}_average_roc_curve.png')
                            plt.close(fig)

def plot_roc_curves(metrics, model_name, dataset_sizes=dataset_sizes,  folds= folds, num_experiments=num_experiments, learning_rates=learning_rates, regularization_params=regularization_params, 
                    galaxy_classes=galaxy_classes, class_descriptions={cls['tag']: cls['description'] for cls in classes}, save_dir='./classifier'):
    min_label = min(galaxy_classes)
    adjusted_classes = [cls - min_label for cls in galaxy_classes]
    for encoder in encoders:
        for fold in folds:
            for lr in learning_rates:
                for reg in regularization_params:
                    for lambda_generate in lambda_values:
                        for subset_size in dataset_sizes[fold]:
                            if subset_size not in merge_map:
                                continue
                             
                            for experiment in range(num_experiments):
                                key = f"{model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                                true_labels = metrics[f"{model_name}_all_true_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0]
                                pred_probs = metrics[f"{model_name}_all_pred_probs_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0]
                                true_labels = true_labels[key]
                                pred_probs = np.array(pred_probs[key])
                                if len(true_labels) == 0 or len(pred_probs) == 0:
                                    continue
                                true_labels_bin = label_binarize(true_labels, classes=np.arange(len(adjusted_classes)))
                                fig, ax = plt.subplots(figsize=(6, 5))
                                for i, class_label in enumerate(adjusted_classes):
                                    try:
                                        fpr, tpr, _ = roc_curve(true_labels_bin[:, i], pred_probs[:, i])
                                        roc_auc = auc(fpr, tpr)
                                        ax.plot(fpr, tpr, lw=2, label=f'{class_descriptions.get(galaxy_classes[i], "Unknown Class")} ROC (area = {roc_auc:.2f})')
                                    except ValueError as e:
                                        print(f"Error plotting ROC for class {class_label}: {e}")
                                        continue
                                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                                ax.set_xlim([0.0, 1.0])
                                ax.set_ylim([0.0, 1.05])
                                ax.set_xlabel('False Positive Rate', fontsize=16)
                                ax.set_ylabel('True Positive Rate', fontsize=16)
                                ax.set_title(f'ROC Curve - {model_name} \n {subset_size}, Fold {fold}, Experiment {experiment}', fontsize=14)
                                ax.legend(loc="lower right")
                                os.makedirs(save_dir, exist_ok=True)
                                plt.savefig(f'{save_dir}/{model_name}_{galaxy_classes}_{encoder}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}_roc_curve.png')
                                plt.close(fig)

def plot_diff_avg_std_confusion_matrix(metrics, model_name, metric_stats, merge_map={249: "250", 250: "250", 2492: "2500", 2505: "2500", 24928: "25000", 25056: "25000"},
                                        lambda_vals=(0, 1), save_dir='./classifier'):
    custom_cmap = LinearSegmentedColormap.from_list("baby_red_green", ["#ff6961", "white", "#77dd77"])
    for encoder in encoders:
        for lr, reg in itertools.product(learning_rates, regularization_params):
            subset_conf_matrices = {}
            for experiment in range(num_experiments):
                for fold in folds:
                    for subset_size in dataset_sizes[fold]:
                         
                        for lam in lambda_vals:
                            key = f"{model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lam}"
                            true_labels_dict = metrics[f"{model_name}_all_true_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lam}"][0]
                            pred_labels_dict = metrics[f"{model_name}_all_pred_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lam}"][0]
                            true_labels = true_labels_dict.get(key)
                            pred_labels = pred_labels_dict.get(key)
                            if (true_labels is None) or (pred_labels is None) or (len(true_labels) == 0) or (len(pred_labels) == 0):
                                print(f"Skipping key with empty labels: {key}")
                                continue
                            cm = confusion_matrix(true_labels, pred_labels, normalize='true')
                            merged_key = merge_map.get(subset_size, subset_size)
                            if merged_key not in subset_conf_matrices:
                                subset_conf_matrices[merged_key] = {lam_val: [] for lam_val in lambda_vals}
                            subset_conf_matrices[merged_key][lam].append(cm)
            for merged_size, lam_dict in subset_conf_matrices.items():
                if any(len(lam_dict[lam_val]) == 0 for lam_val in lambda_vals):
                    print(f"Not enough data for merged size {merged_size} for lambda values {lambda_vals}")
                    continue
                cms_lam1 = np.array(lam_dict[lambda_vals[0]])
                cms_lam2 = np.array(lam_dict[lambda_vals[1]])
                avg_cm_lam1 = np.mean(cms_lam1, axis=0)
                std_cm_lam1 = np.std(cms_lam1, axis=0)
                avg_cm_lam2 = np.mean(cms_lam2, axis=0)
                std_cm_lam2 = np.std(cms_lam2, axis=0)
                diff_avg = avg_cm_lam2 - avg_cm_lam1
                diff_std = np.sqrt(std_cm_lam2**2 + std_cm_lam1**2)
                ann = np.empty(diff_avg.shape, dtype=object)
                for i in range(diff_avg.shape[0]):
                    for j in range(diff_avg.shape[1]):
                        ann[i, j] = f"{diff_avg[i,j]:.2f}\n±{diff_std[i,j]:.2f}"
                present_descriptions = [desc for lab, desc in zip(galaxy_classes, {cls['tag']: cls['description'] for cls in classes}.values())]
                lambda0_values = metric_stats[lambda_vals[0]]['accuracy']
                lambda1_values = metric_stats[lambda_vals[1]]['accuracy']
                mean_diff = np.mean(lambda1_values) - np.mean(lambda0_values)
                std_diff = np.sqrt(np.std(lambda1_values)**2 + np.std(lambda0_values)**2)
                fig, ax = plt.subplots(figsize=(8,6))
                sns.heatmap(diff_avg, annot=ann, fmt="", cmap=custom_cmap, center=0,
                            xticklabels=present_descriptions, yticklabels=present_descriptions, ax=ax)
                ax.set_xlabel("Predicted Label", fontsize=12)
                ax.set_ylabel("True Label", fontsize=12)
                ax.set_title(f"Difference in Average Accuracy: {mean_diff:.2f} ± {std_diff:.2f}", fontsize=14)
                os.makedirs(save_dir, exist_ok=True)
                save_path = f"{save_dir}/{model_name}_{galaxy_classes}_{encoder}_{merged_size}_{lr}_{reg}_{lambda_vals[1]}-{lambda_vals[0]}_diff_confusion_matrix.png"
                plt.savefig(save_path)
                plt.close(fig)
            if not subset_conf_matrices:
                print(f"No confusion matrices found for LR {lr}, Reg: {reg}, Lambda values {lambda_vals}")

def plot_avg_std_confusion_matrix(metrics, model_name, metric_stats, merge_map={249: "250", 250: "250", 2492: "2500", 2505: "2500", 24928: "25000", 25056: "25000"}, save_dir='./classifier'):
    for encoder in encoders:
        for lr, reg, lambda_generate in itertools.product(learning_rates, regularization_params, lambda_values):
            subset_conf_matrices = {}
            for experiment in range(num_experiments):
                for fold in folds:
                    for subset_size in dataset_sizes[fold]:
                        key = f"{model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                        true_labels = metrics[f"{model_name}_all_true_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0]
                        pred_labels = metrics[f"{model_name}_all_pred_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0]
                        true_labels = true_labels[key]
                        pred_labels = pred_labels[key]
                        true_labels = list(map(int, true_labels))
                        pred_labels = list(map(int, pred_labels))
                        if not true_labels or not pred_labels:
                            print(f"Skipping key with empty labels: {key}")
                            continue
                        num_classes = len(galaxy_classes)
                        cm = confusion_matrix(true_labels, pred_labels, normalize='true', labels=list(range(num_classes)))
                        if cm.size == 0 or cm.shape[0] != cm.shape[1]:
                            print(f"Skipping invalid confusion matrix with shape {cm.shape}")
                            continue
                        merged_key = merge_map.get(subset_size, subset_size)
                        if merged_key not in subset_conf_matrices:
                            subset_conf_matrices[merged_key] = []
                        subset_conf_matrices[merged_key].append(cm)
                        
            for subset_size, cm_list in subset_conf_matrices.items():
                if not cm_list:
                    print(f"No valid confusion matrices for subset size {subset_size}")
                    continue
                cms = np.array(cm_list)
                avg_cm = np.mean(cms, axis=0)
                std_cm = np.std(cms, axis=0)
                present_descriptions = [desc for lab, desc in zip(galaxy_classes, {cls['tag']: cls['description'] for cls in classes}.values())]
                ann = np.empty(avg_cm.shape, dtype=object)
                for i in range(avg_cm.shape[0]):
                    for j in range(avg_cm.shape[1]):
                        ann[i, j] = f"{avg_cm[i, j]:.2f}\n±{std_cm[i, j]:.2f}"
                values = metric_stats[lambda_generate]['accuracy']
                mean_value = np.mean(values)
                std_dev = np.std(values)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(avg_cm, annot=ann, fmt="", cmap="Blues",
                            xticklabels=present_descriptions, yticklabels=present_descriptions, ax=ax)
                ax.set_xlabel("Predicted Label", fontsize=12)
                ax.set_ylabel("True Label", fontsize=12)
                ax.set_title(f"Average Accuracy: {mean_value:.2f} ± {std_dev:.2f}", fontsize=14)
                os.makedirs(save_dir, exist_ok=True)
                save_path = f"{save_dir}/{model_name}_{galaxy_classes}_{encoder}_{subset_size}_{lr}_{reg}_{lambda_generate}_avg_confusion_matrix.png"
                plt.savefig(save_path)
                plt.close(fig)
            if not subset_conf_matrices:
                print(f"No confusion matrices found for LR {lr}, Reg {reg}, Lambda {lambda_generate}")

def plot_confusion_matrix(metrics, model_name, dataset_sizes=dataset_sizes,  folds= folds, num_experiments=num_experiments, 
                          learning_rates=learning_rates, regularization_params=regularization_params, 
                          galaxy_classes=galaxy_classes, lambda_values=lambda_values,
                          class_descriptions=[cls['description'] for cls in classes if cls['tag'] in galaxy_classes], 
                          save_dir='./classifier'):
    for encoder in encoders:
        for fold, lr, reg, lambda_generate, experiment in itertools.product(folds, learning_rates, regularization_params, lambda_values, range(num_experiments)):
            for subset_size in dataset_sizes[fold]:
                if subset_size <= 0:
                    continue
                key = f"{model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                true_labels = metrics[f"{model_name}_all_true_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0]
                pred_labels = metrics[f"{model_name}_all_pred_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0]
                true_labels = true_labels[key]
                pred_labels = pred_labels[key]
                if not true_labels or not pred_labels:
                    print(f"Skipping key with empty labels: {key}")
                    continue
                cm = confusion_matrix(true_labels, pred_labels, normalize='true')
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt=".1%", linewidths=.5, square=True, cmap='Blues', ax=ax,
                            xticklabels=class_descriptions, yticklabels=class_descriptions, annot_kws={"size": 16})
                colorbar = ax.collections[0].colorbar
                colorbar.ax.tick_params(labelsize=16)
                accuracy = accuracy_score(true_labels, pred_labels)
                ax.set_title(f'Model: {model_name} \n Total accuracy: {accuracy*100:.2f}%', fontsize=14)
                ax.set_ylabel('True label', fontsize=16)
                ax.set_xlabel('Predicted label', fontsize=16)
                save_path = f'{save_dir}/{model_name}_{galaxy_classes}_{encoder}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}_confusion_matrix.png'
                plt.savefig(save_path)
                plt.close()

def plot_loss(models, history=history, dataset_sizes=dataset_sizes,  folds= folds, num_experiments=num_experiments, galaxy_classes=galaxy_classes, 
              learning_rates=learning_rates, regularization_params=regularization_params, classifier=None, lambda_values=lambda_values):   
    for encoder in encoders:
        fig, ax = plt.subplots(figsize=(10, 8))
        for lr, reg, lambda_generate, experiment in itertools.product(learning_rates, regularization_params, lambda_values, range(num_experiments)):       
            colors = itertools.cycle(['blue', 'orange', 'green', 'red', 'magenta'])
            for model_name in models.keys():
                color = next(colors)
                for fold in folds:
                    for subset_size in dataset_sizes[fold]:  
                        if subset_size <= 0:
                            print(f"Skipping invalid subset size: {subset_size}")
                            continue
                         
                        loss_key = f"{model_name}_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                        val_loss_key = f"{model_name}_val_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                        if loss_key not in history.get(model_name, {}):
                            continue
                        if val_loss_key not in history.get(model_name, {}):
                            print(f"Skipping missing val_loss_key: {val_loss_key}")
                            continue
                        marker_size = np.sqrt(subset_size / max(dataset_sizes[fold]) * 500) + 2
                        ax.plot(history[model_name][loss_key], color=color, linestyle='-', 
                                marker='o', markersize=marker_size, label=f"{model_name} train (fold {fold})")
                        ax.plot(history[model_name][val_loss_key], color=color, linestyle='--', 
                                marker='x', markersize=marker_size, label=f"{model_name} val (fold {fold})")
        ax.set_title(f'Training and Validation Loss for Regularisation {reg} and Learning Rate {lr}', fontsize=16)
        ax.set_xlabel('Epochs', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=12, loc='best')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"./classifier/{classifier}_{galaxy_classes}_{encoder}_{lr}_{reg}_loss.png")
        plt.close(fig)

###############################################
############ PLOTS AFTER ALL FOLDS ############
###############################################

class_descriptions = [cls['description'] for cls in classes if cls['tag'] in galaxy_classes]

metric_rankings = {metric: [] for metric in ["accuracy", "precision", "recall", "f1_score"]}
metric_stats = defaultdict(lambda: defaultdict(list))
for lambda_generate in lambda_values:
    try:
        for fold in folds:
            final_subset_size = dataset_sizes[fold][-1]
            for experiment in range(num_experiments):
                for lr in learning_rates:
                    for reg in regularization_params:
                        for metric in metric_rankings.keys():
                            key = f"{classifier}_{metric}_{final_subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                            try:
                                value = metrics[key]
                                metric_rankings[metric].append((value, fold, final_subset_size, experiment, lr, reg, lambda_generate))
                                metric_stats[lambda_generate][metric].append(value)
                            except KeyError:
                                continue
    except KeyError:
        print(f"No metrics found for lambda_generate {lambda_generate}")

for metric in metric_rankings:
    print(f"\n{metric.capitalize()} Rankings (sorted by highest value):")
    sorted_rankings = sorted(metric_rankings[metric], key=lambda x: x[0], reverse=True)
    for rank, (value, fold, subset_size, experiment, lr, reg, lambda_generate) in enumerate(sorted_rankings, start=1):
        print(f"{rank}: {metric.capitalize()}: {value[0]:.4f}, Fold: {fold}, Subset Size: {subset_size}, Experiment: {experiment}, LR: {lr}, Reg: {reg}, Lambda: {lambda_generate}")

print("\nMetric Summary Statistics by Lambda:")
for lambda_generate in lambda_values:
    print(f"\nLambda: {lambda_generate}")
    for metric in metric_stats[lambda_generate]:
        values = metric_stats[lambda_generate][metric]
        mean_value = np.mean(values)
        std_dev = np.std(values)
        print(f"{metric.capitalize()}: Mean = {mean_value:.4f}, Std Dev = {std_dev:.4f}")

print("\nTraining Times:")
for subset_size, folds in training_times.items():
    times = []
    print(f"Subset size {subset_size}:")
    for fold, elapsed_times in folds.items():
        if not elapsed_times:
            print(f"  Fold {fold}: No training times recorded.")
            continue
        for elapsed_time in elapsed_times:
            times.append(elapsed_time)
            print(f"  Fold {fold}: {elapsed_time:.2f} seconds")
    if times:
        mean_time = np.mean(times)
        std_time = np.std(times)
        print(f"  Mean training time: {mean_time:.2f} seconds ± {std_time:.2f} seconds\n")
    else:
        print("  No training times recorded for this subset size.\n")

plot_loss(models)
plot_roc_curves(metrics, model_name)
plot_avg_roc_curves(metrics, model_name, merge_map=merge_map)
plot_all_metrics_vs_dataset_size(metrics, model_name, merge_map=merge_map)
plot_accuracy_vs_lambda(lambda_values, metrics, model_name)
plot_confusion_matrix(metrics, model_name)
plot_avg_std_confusion_matrix(metrics, model_name, merge_map=merge_map, metric_stats=metric_stats)
plot_diff_avg_std_confusion_matrix(metrics, model_name, merge_map=merge_map, metric_stats=metric_stats)