import numpy as np
import pickle
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from utils.data_loader import get_classes, load_galaxies, get_synthetic
import itertools
from collections import defaultdict
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import colormaps  # For newer Matplotlib versions
import matplotlib.lines as mlines  # <-- For creating custom line legend entries
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os
os.makedirs('./classifier/trained_models', exist_ok=True)
os.makedirs('./classifier/trained_models_filtered', exist_ok=True)

print("Running script 4.2")
device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

###############################################
################ CONFIGURATION ################
###############################################

FILTERED = True       # Evaluation with filtered data (REMOVEOUTLIERS = True)
TESTONGENERATED = False # Use generated data as testdata

classes = get_classes()
galaxy_classes = [10, 11, 12, 13]
#galaxy_classes = [31, 32, 33, 34, 35, 36]
learning_rates = [1e-3]
regularization_params = [0]
lambda_values = [0, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100]
num_experiments = 20
folds = [5] # Number of folds for cross-validation
generators = ['DDPM']
classifier = ["ProjectModel", "Rustige", "ScatterNet", "ScatterDual"][1]  # Choose one classifier model model

# Define consistent color mapping
colors = {
    'Real': '#0072B2',       # blue
    'STMLP': '#E69F00',      # orange
    'lavgSTMLP': '#CC79A7',   # reddish purple
    'ldiffSTMLP': '#F0E442',  # yellow
    'Dual': '#009E73',       # bluish green
    'CNN': '#56B4E9',        # sky blue
    'GAN': '#D55E00'         # vermillion
}

###############################################
######### SETTING THE RIGHT PARAMETERS ########
###############################################

if galaxy_classes == [31, 32, 33, 34, 35, 36]:
    dataset_sizes = [[1088, 10883, 54416, 108832]]
    num_experiments = 1
    FILTERED = False
    TESTONGENERATED = False
elif galaxy_classes == [10, 11, 12, 13]:
    if FILTERED:
        if max(folds) == 5:
            #dataset_sizes = [[200], [200], [200], [200], [200], [200]] # Used for faster trouble shooting
            dataset_sizes = [[13936], [13936], [13936], [13936], [13936], [13936]] 
            #dataset_sizes = [[139, 1393, 13936], [139, 1393, 13936], [139, 1393, 13936], [139, 1393, 13936], [139, 1393, 13936], [139, 1393, 13936]] # Need length = folds[0] = 1
        else:
            #dataset_sizes = [[12368], [12336], [12368], [12368], [12368]] 
            dataset_sizes = [[200], [200], [200], [200], [200], [200]]
    elif max(folds) == 5:
        #dataset_sizes = [[140, 1406, 14064], [140, 1406, 14064], [140, 1406, 14064], [140, 1406, 14064], [140, 1406, 14064], [140, 1406, 14064]] # Used for faster trouble shooting
        dataset_sizes = [[139, 1393, 13936], [139, 1393, 13936], [139, 1393, 13936], [139, 1393, 13936], [139, 1393, 13936], [139, 1393, 13936]] # Need length = folds[0] = 1
        #dataset_sizes = [[281, 2812, 14064, 28128], [281, 2812, 14064, 28128], [281, 2812, 14064, 28128], 
        #                    [281, 2812, 14064, 28128], [281, 2812, 14064, 28128], [281, 2812, 14064, 28128]] #Need length = folds[0] = 1
    else:
        #dataset_sizes = [[249, 2492, 12464, 24928], [249, 2492, 12464, 24928], [249, 2492, 12464, 24928], [249, 2492, 12464, 24928], 250, 2505, 12528, 25056]]    
        dataset_sizes = [[12464], [12464], [12464], [12464], [12528]]

if 'GAN' in generators:
    latent_dim = 64 

if TESTONGENERATED:
    lambda_values = [8]  # To identify and distinguish TESTONGENERATED from other runs

############################################################
################# MERGE MAP GENERATION #####################
############################################################

merge_map = {}
for sizes in dataset_sizes:
    for size in sizes:
        nd = len(str(size))
        factor = 10 ** (nd - 2)
        new_rep = round(size / factor) * factor
        merge_map[size] = str(new_rep)

print("merge_map =", merge_map)

###############################################
############# READ IN PICKLE DUMP #############
###############################################

def initialize_metrics(metrics, generator, subset_size, fold, experiment, lr, reg, lambda_generate):
    subset_size_str = str(subset_size[0]) if isinstance(subset_size, list) else str(subset_size)
    metrics[f"{generator}_accuracy_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{generator}_precision_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{generator}_recall_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{generator}_f1_score_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{generator}_history_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{generator}_all_true_labels_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{generator}_all_pred_labels_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{generator}_training_times_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{generator}_all_pred_probs_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []

def update_metrics(metrics, generator, subset_size, fold, experiment, lr, reg,
                   accuracy, precision, recall, f1, lambda_generate,
                   history_val, all_true_labels, all_pred_labels, training_times, all_pred_probs):
    subset_size_str = str(subset_size)
    metrics[f"{generator}_accuracy_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(accuracy)
    metrics[f"{generator}_precision_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(precision)
    metrics[f"{generator}_recall_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(recall)
    metrics[f"{generator}_f1_score_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(f1)
    metrics[f"{generator}_history_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(history_val)
    metrics[f"{generator}_all_true_labels_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(all_true_labels)
    metrics[f"{generator}_all_pred_labels_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(all_pred_labels)
    metrics[f"{generator}_training_times_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(training_times)
    metrics[f"{generator}_all_pred_probs_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(all_pred_probs)

tot_metrics = {}
valid_lambda_values = []
directory = './classifier/trained_models_filtered/' if FILTERED else './classifier/trained_models/'

print("Loading metrics from pickle files...")
for lambda_generate in lambda_values:
    try:
        for lr, reg, experiment, generator, fold in itertools.product(
            learning_rates, regularization_params, range(num_experiments), generators, folds):
            for subset_size in dataset_sizes[fold]:
                metrics_read_path = f'{directory}{classifier}_{galaxy_classes}_{generator}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}_metrics_data.pkl'
                #metrics_read_path = f'{directory}{galaxy_classes}_{generator}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}_metrics_data.pkl'
                with open(metrics_read_path, 'rb') as f:
                    metrics_data = pickle.load(f)
                    clf_models = metrics_data["models"] # Classifier models
                    history = metrics_data["history"]
                    loaded_metrics = metrics_data["metrics"]
                    metric_colors = metrics_data["metric_colors"]
                    all_true_labels = metrics_data["all_true_labels"]
                    all_pred_labels = metrics_data["all_pred_labels"]
                    training_times = metrics_data["training_times"]
                    all_pred_probs = metrics_data["all_pred_probs"]
                initialize_metrics(tot_metrics, generator, subset_size, fold, experiment, lr, reg, lambda_generate)
                update_metrics(
                    tot_metrics, generator, subset_size, fold, experiment, lr, reg,
                    loaded_metrics[f"{generator}_accuracy_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0],
                    loaded_metrics[f"{generator}_precision_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0],
                    loaded_metrics[f"{generator}_recall_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0],
                    loaded_metrics[f"{generator}_f1_score_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0],
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


def visualize_tsne_by_class(model, real_loader, gen_loader, device='cpu', save_path="./classifier/tsne_by_class.png"):
    model = model.to(device).eval()
    def extract_feats(loader):
        feats = []
        handle = model.fc1.register_forward_hook(lambda m, inp, out: feats.append(out.detach().cpu()))
        for x, _, y in loader:
            _ = model(x.to(device))
        handle.remove()
        return torch.cat(feats,0).numpy()

    real_feats = extract_feats(real_loader)
    real_labels = real_loader.dataset.tensors[2].numpy()   # assuming your DataLoader stores labels

    gen_feats  = extract_feats(gen_loader)
    gen_labels = gen_loader.dataset.tensors[2].numpy()     # same here

    # combine
    X   = np.vstack([real_feats, gen_feats])
    y   = np.concatenate([real_labels, gen_labels])
    X2d = TSNE(perplexity=30, max_iter=2000, random_state=42).fit_transform(X)

    plt.figure(figsize=(6,6))
    scatter = plt.scatter(X2d[:,0], X2d[:,1], c=y, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Galaxy class")
    plt.title("t-SNE of penultimate-layer features, colored by class")
    plt.savefig(save_path)


def visualize_feature_tsne(model, real_loader, gen_loader,
                           device='cpu',
                           perplexity=30, n_iter=1000,
                           save_path="./classifier/tsne_by_truth.png"):
    """
    Extracts penultimate‐layer features (fc1) from real & generated images,
    runs t-SNE, and saves a scatter plot.
    """
    model = model.to(device).eval()

    def extract_feats(loader):
        feats = []
        # register hook on fc1
        def hook_fn(module, inp, out):
            feats.append(out.detach().cpu())
        handle = model.fc1.register_forward_hook(hook_fn)

        # run data through the network
        with torch.no_grad():
            for imgs, _, _ in loader:
                _ = model(imgs.to(device))
        handle.remove()

        # concatenate all batches
        return torch.cat(feats, dim=0).numpy()

    # extract
    real_feats = extract_feats(real_loader)
    gen_feats  = extract_feats(gen_loader)

    # stack & fit TSNE
    X     = np.vstack([real_feats, gen_feats])
    X_emb = TSNE(perplexity=perplexity, max_iter=n_iter,random_state=42).fit_transform(X)

    # plot
    n_real = real_feats.shape[0]
    plt.figure(figsize=(6,6))
    plt.scatter(X_emb[:n_real, 0], X_emb[:n_real, 1],
                alpha=0.6, label='real')
    plt.scatter(X_emb[n_real:, 0], X_emb[n_real:, 1],
                alpha=0.6, label='generated')
    plt.legend()
    plt.title('t-SNE of penultimate-layer features (RustigeClassifier)')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved t-SNE plot to {save_path}")

def plot_accuracy_vs_lambda(lambda_values, metrics, generator, dataset_sizes=dataset_sizes,
                            folds=folds, num_experiments=num_experiments,
                            learning_rates=learning_rates, regularization_params=regularization_params,
                            save_dir='./classifier'):
    # Check if lambda = 0 is present. Otherwise skip this function
    if 0 not in lambda_values:
        print("Skipping accuracy vs. lambda plot as λ=0 is not present.")
        return
    
    # Create one figure for all models
    plt.figure(figsize=(8, 5))    
    
    # Loop over each model in your generators list
    for mname in generators:
        # Calculate the "classically augmented" baseline (λ=0)
        classical_accs = []
        for fold, lr, reg, experiment in itertools.product(folds, learning_rates, regularization_params, range(num_experiments)):
            key = f'{mname}_accuracy_{dataset_sizes[fold][-1]}_{fold}_{experiment}_{lr}_{reg}_{0}'
            classical_accs.append(metrics[key][0])
        classical_acc = np.mean(classical_accs)
        classical_std = np.std(classical_accs)
        
        # Compute mean accuracies and standard deviations for each λ
        gan_accs_means = []
        gan_stds = []
        for lambda_generate in lambda_values:
            gan_accs_for_one_lambda = []
            for fold, lr, reg, experiment in itertools.product(folds, learning_rates, regularization_params, range(num_experiments)):
                key = f'{mname}_accuracy_{dataset_sizes[fold][-1]}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}'
                gan_accs_for_one_lambda.append(metrics[key][0])
            gan_accs_means.append(np.mean(gan_accs_for_one_lambda))
            gan_stds.append(np.std(gan_accs_for_one_lambda))
        gan_accs_means = np.array(gan_accs_means).flatten()
        gan_stds = np.array(gan_stds).flatten()
        
        # Pick the colour for this model
        col = colors.get(mname, 'black')
        
        # Plot the baseline for λ=0 (classically augmented) as a dashed horizontal line
        plt.axhline(y=classical_acc, color=col, linestyle='--', label=f'Classically augmented')
        plt.fill_between(lambda_values, classical_acc - classical_std, classical_acc + classical_std,
                         color=col, alpha=0.1)
        
        # Plot each point for the augmented results at each λ with error bars
        for i, lambda_val in enumerate(lambda_values):
            # Only label the first point per model to avoid duplicate legend entries
            label = f'Classically + {mname} augmented' if i == 0 else None
            plt.errorbar(lambda_val, gan_accs_means[i], yerr=gan_stds[i], fmt='o', capsize=5,
                         color=col, ecolor=col, label=label)
    
    # Add labels, legend, grid and save the figure
    plt.xlabel(r'$\lambda_{gen}$')
    plt.ylabel('Accuracy')
    plt.xticks(lambda_values, [str(l) for l in lambda_values])
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plot_path = f'{save_dir}/{galaxy_classes}_{classifier}_{generator}_{max(dataset_sizes[-1])}_{lr}_{reg}_accuracy_vs_lambda.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved overlayed accuracy vs. lambda plot to {plot_path}")
    

def plot_all_metrics_vs_dataset_size(
    metrics,
    generators,
    merge_map={249: "250", 250: "250", 2492: "2500", 2505: "2500", 24928: "25000", 25056: "25000"},
    dataset_sizes=dataset_sizes,
    folds=folds,
    num_experiments=num_experiments,
    learning_rates=learning_rates,
    regularization_params=regularization_params,
    lambda_values=lambda_values,
    save_dir='./classifier'):
    metric_names  = ["accuracy", "precision", "recall", "f1_score"]
    metric_titles = ["Accuracy", "Precision", "Recall", "F1 Score"]

    for generator in generators:
        for metric, title in zip(metric_names, metric_titles):
            plt.figure(figsize=(10, 6))

            # one line per λ
            for λ in lambda_values:
                # collect values per merged category
                metric_values_per_category = defaultdict(list)

                # iterate over all combinations
                for lr, reg, exp, fold in itertools.product(
                    learning_rates, regularization_params,
                    range(num_experiments), folds
                ):
                    for subset_size in dataset_sizes[fold]:
                        if subset_size not in merge_map:
                            continue
                        key = f"{generator}_{metric}_{subset_size}_{fold}_{exp}_{lr}_{reg}_{λ}"
                        if key in metrics and metrics[key]:
                            metric_values_per_category[merge_map[subset_size]].append(metrics[key][-1])

                # sort categories and compute mean±std
                categories = sorted(metric_values_per_category.keys(), key=lambda x: int(x))
                means = [ np.mean(metric_values_per_category[c]) for c in categories ]
                stds  = [ np.std (metric_values_per_category[c]) for c in categories ]

                # plot
                plt.plot(categories, means, marker='o', linestyle='-', label=f"λ={λ}")
                plt.fill_between(categories,
                                 np.array(means) - np.array(stds),
                                 np.array(means) + np.array(stds),
                                 alpha=0.2)

            # now that all λ-lines are drawn, add legend
            plt.legend(title='Lambda Values', fontsize=12, loc='best')
            plt.title(f'{title} vs Dataset Size ({generator})', fontsize=16)
            plt.xlabel('Training Dataset Size', fontsize=14)
            plt.ylabel(title, fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()

            os.makedirs(save_dir, exist_ok=True)
            fname = f'{save_dir}/{galaxy_classes}_{classifier}_{generator}_{final_subset_size}_{metric}_vs_dataset_size.png'
            plt.savefig(fname)
            plt.close()

def plot_avg_roc_curves(metrics, generators, dataset_sizes=dataset_sizes, merge_map=merge_map, 
                         folds= folds, num_experiments=num_experiments, 
                        learning_rates=learning_rates, regularization_params=regularization_params, 
                        galaxy_classes=galaxy_classes, 
                        class_descriptions={cls['tag']: cls['description'] for cls in classes}, 
                        save_dir='./classifier'):
    min_label = min(galaxy_classes)
    adjusted_classes = [cls - min_label for cls in galaxy_classes]
    for generator in generators:
        for lr in learning_rates:
            for reg in regularization_params:
                for lambda_generate in lambda_values:
                    for s in range(len(dataset_sizes[0])):
                        for experiment in range(num_experiments):
                            roc_values = {class_label: [] for class_label in adjusted_classes}
                            for fold in folds:
                                subset_size = dataset_sizes[fold][s]
                                true_labels_dict = metrics[f"{generator}_all_true_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0]
                                pred_probs_dict = metrics[f"{generator}_all_pred_probs_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0]         
                                key = f"{generator}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                                true_labels = true_labels_dict.get(key)
                                pred_probs = pred_probs_dict.get(key)
                                if (true_labels is None) or (pred_probs is None) or (len(true_labels) == 0) or (len(pred_probs) == 0):
                                    continue
                                pred_probs = np.array(pred_probs)
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
                            ax.set_title(f'Average ROC Curve - {generator} \n {merged_subset_key}, Experiment {experiment}', fontsize=14)
                            ax.legend(loc="lower right")
                            os.makedirs(save_dir, exist_ok=True)
                            plt.savefig(f'{save_dir}/{galaxy_classes}_{classifier}_{generator}_{merged_subset_key}_average_roc_curve.png')
                            plt.close(fig)

def plot_roc_curves(metrics, generators, dataset_sizes=dataset_sizes,  folds= folds, num_experiments=num_experiments, learning_rates=learning_rates, regularization_params=regularization_params, 
                    galaxy_classes=galaxy_classes, class_descriptions={cls['tag']: cls['description'] for cls in classes}, save_dir='./classifier'):
    min_label = min(galaxy_classes)
    adjusted_classes = [cls - min_label for cls in galaxy_classes]
    for generator in generators:
        for fold in folds:
            for lr in learning_rates:
                for reg in regularization_params:
                    for lambda_generate in lambda_values:
                        for subset_size in dataset_sizes[fold]:
                            if subset_size not in merge_map:
                                continue
                             
                            for experiment in range(num_experiments):
                                key = f"{generator}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                                true_labels = metrics[f"{generator}_all_true_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0]
                                pred_probs = metrics[f"{generator}_all_pred_probs_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0]
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
                                ax.set_title(f'ROC Curve - {generator} \n {subset_size}, Fold {fold}, Experiment {experiment}', fontsize=14)
                                ax.legend(loc="lower right")
                                os.makedirs(save_dir, exist_ok=True)
                                plt.savefig(f'{save_dir}/{galaxy_classes}_{classifier}_{generator}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}_roc_curve.png')
                                plt.close(fig)

def plot_diff_avg_std_confusion_matrix(metrics, generators, metric_stats,
    merge_map={249: "250", 250: "250", 2492: "2500", 2505: "2500", 24928: "25000", 25056: "25000"},
    lambda_vals=(0, 1), save_dir='./classifier'):
    
    # Check if lambda = 0 and lambda = 1 are present. Otherwise skip this function
    if any(lam not in lambda_values for lam in lambda_vals):
        print("Skipping difference in average confusion matrix plot as one or both lambda values are not present.")
        return
    
    custom_cmap = LinearSegmentedColormap.from_list("baby_red_green", ["#ff6961", "white", "#77dd77"])
    
    # Loop over learning rate and regularization (remove lambda from outer loop)
    for lr, reg in itertools.product(learning_rates, regularization_params):
        subset_conf_matrices = {}
        # Collect confusion matrices for both lambda values concurrently
        for experiment in range(num_experiments):
            for fold in folds:
                for subset_size in dataset_sizes[fold]:
                    for lam in lambda_vals:
                        key = f"{generator}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lam}"
                        true_labels_dict = metrics[f"{generator}_all_true_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lam}"][0]
                        pred_labels_dict = metrics[f"{generator}_all_pred_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lam}"][0]
                        true_labels = true_labels_dict.get(key)
                        pred_labels = pred_labels_dict.get(key)
                        if (true_labels is None) or (pred_labels is None) or (len(true_labels) == 0) or (len(pred_labels) == 0):
                            continue
                        cm = confusion_matrix(true_labels, pred_labels, normalize='true')
                        merged_key = merge_map.get(subset_size, subset_size)
                        if merged_key not in subset_conf_matrices:
                            subset_conf_matrices[merged_key] = {lam_val: [] for lam_val in lambda_vals}
                        subset_conf_matrices[merged_key][lam].append(cm)
                        
        # Now, for each merged key, check that we have data for both lambda values.
        for merged_size, lam_dict in subset_conf_matrices.items():
            if any(len(lam_dict[lam_val]) == 0 for lam_val in lambda_vals):
                print(f"Not enough data for merged size {merged_size} for lambda values {lambda_vals}")
                continue
            cms_lam0 = np.array(lam_dict[lambda_vals[0]])
            cms_lam1 = np.array(lam_dict[lambda_vals[1]])
            avg_cm_lam0 = np.mean(cms_lam0, axis=0)
            std_cm_lam0 = np.std(cms_lam0, axis=0)
            avg_cm_lam1 = np.mean(cms_lam1, axis=0)
            std_cm_lam1 = np.std(cms_lam1, axis=0)
            diff_avg = avg_cm_lam1 - avg_cm_lam0
            diff_std = np.sqrt(std_cm_lam1**2 + std_cm_lam0**2)
            ann = np.empty(diff_avg.shape, dtype=object)
            for i in range(diff_avg.shape[0]):
                for j in range(diff_avg.shape[1]):
                    ann[i, j] = f"{diff_avg[i, j]:.2f}\n±{diff_std[i, j]:.2f}"
            
            desc_by_tag = {cls['tag']: cls['description'] for cls in classes}
            present_descriptions = [desc_by_tag[tag] for tag in galaxy_classes]
            lambda0_values = metric_stats[lambda_vals[0]]['accuracy']
            lambda1_values = metric_stats[lambda_vals[1]]['accuracy']
            mean_diff = np.mean(lambda1_values) - np.mean(lambda0_values)
            std_diff = np.sqrt(np.std(lambda1_values)**2 + np.std(lambda0_values)**2)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(diff_avg, annot=ann, fmt="", cmap=custom_cmap, center=0,
                        xticklabels=present_descriptions, yticklabels=present_descriptions, ax=ax)
            ax.set_xlabel("Predicted Label", fontsize=12)
            ax.set_ylabel("True Label", fontsize=12)
            ax.set_title(f"Difference in Average Accuracy: {mean_diff:.2f} ± {std_diff:.2f}", fontsize=14)
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/{galaxy_classes}_{classifier}_{generator}_{merged_size}_{lr}_{reg}_{lambda_vals[1]}-{lambda_vals[0]}_diff_confusion_matrix.png"
            plt.savefig(save_path)
            plt.close(fig)


def plot_avg_std_confusion_matrix(metrics, generators, metric_stats, merge_map={249: "250", 250: "250", 2492: "2500", 2505: "2500", 24928: "25000", 25056: "25000"}, save_dir='./classifier'):
    for generator in generators:
        for lr, reg, lambda_generate in itertools.product(learning_rates, regularization_params, lambda_values):
            subset_conf_matrices = {}
            for experiment in range(num_experiments):
                for fold in folds:
                    for subset_size in dataset_sizes[fold]:
                        true_labels_dict = metrics[f"{generator}_all_true_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0]
                        pred_labels_dict = metrics[f"{generator}_all_pred_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0]         
                        key = f"{generator}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                        true_labels = true_labels_dict.get(key)
                        pred_labels = pred_labels_dict.get(key)
                        if (true_labels is None) or (pred_labels is None) or (len(true_labels) == 0) or (len(pred_labels) == 0):
                            continue
                        pred_labels = np.array(pred_labels)
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
                desc_by_tag = {cls['tag']: cls['description'] for cls in classes}
                present_descriptions = [desc_by_tag[tag] for tag in galaxy_classes]
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
                save_path = f"{save_dir}/{galaxy_classes}_{classifier}_{generator}_{subset_size}_{lr}_{reg}_{lambda_generate}_avg_confusion_matrix.png"
                plt.savefig(save_path)
                plt.close(fig)


def plot_confusion_matrix(metrics, generators, dataset_sizes=dataset_sizes,  folds= folds, num_experiments=num_experiments, 
                          learning_rates=learning_rates, regularization_params=regularization_params, 
                          galaxy_classes=galaxy_classes, lambda_values=lambda_values,
                          class_descriptions=[cls['description'] for cls in classes if cls['tag'] in galaxy_classes], 
                          save_dir='./classifier'):
    for generator in generators:
        for fold, lr, reg, lambda_generate, experiment in itertools.product(folds, learning_rates, regularization_params, lambda_values, range(num_experiments)):
            for subset_size in dataset_sizes[fold]:
                if subset_size <= 0:
                    continue
                key = f"{generator}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                true_labels = metrics[f"{generator}_all_true_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0]
                pred_labels = metrics[f"{generator}_all_pred_labels_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"][0]
                true_labels = true_labels[key]
                pred_labels = pred_labels[key]
                if not true_labels or not pred_labels:
                    continue
                cm = confusion_matrix(true_labels, pred_labels, normalize='true')
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt=".1%", linewidths=.5, square=True, cmap='Blues', ax=ax,
                            xticklabels=class_descriptions, yticklabels=class_descriptions, annot_kws={"size": 16})
                colorbar = ax.collections[0].colorbar
                colorbar.ax.tick_params(labelsize=16)
                accuracy = accuracy_score(true_labels, pred_labels)
                ax.set_title(f'Model: {generator} \n Total accuracy: {accuracy*100:.2f}%', fontsize=14)
                ax.set_ylabel('True label', fontsize=16)
                ax.set_xlabel('Predicted label', fontsize=16)
                save_path = f'{save_dir}/{galaxy_classes}_{classifier}_{generator}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}_confusion_matrix.png'
                plt.savefig(save_path)
                plt.close()
 

def plot_loss(
    generators,
    history=history,
    dataset_sizes=dataset_sizes,
    folds=folds,
    num_experiments=num_experiments,
    galaxy_classes=galaxy_classes,
    learning_rates=learning_rates,
    regularization_params=regularization_params,
    classifier=classifier,
    lambda_values=lambda_values
):
    sorted_lambdas = sorted(lambda_values)
    n_lambdas = len(sorted_lambdas)

    # Create discrete colors for each lambda using a continuous colormap
    base_cmap = colormaps["viridis"]
    if n_lambdas > 1:
        color_list = [base_cmap(i / (n_lambdas - 1)) for i in range(n_lambdas)]
    else:
        color_list = [base_cmap(0.5)]

    lambda_to_color = {
        lam: color_list[i] for i, lam in enumerate(sorted_lambdas)
    }

    for generator in generators:
        fig, ax = plt.subplots(figsize=(10, 8))

        # We'll store two special line handles for Train/Val,
        # plus a handle for each λ color.
        # Then we combine them in a single legend with 2 columns.
        train_line = mlines.Line2D([], [], color='black', linestyle='-',
                                   label='Train')
        val_line   = mlines.Line2D([], [], color='black', linestyle='--',
                                   label='Validation')

        # Keep track of which λ's we actually plotted (in case some are skipped)
        plotted_lambdas = set()

        for lr, reg in itertools.product(learning_rates, regularization_params):
            for lambda_generate in sorted_lambdas:
                all_train_losses = []
                all_val_losses   = []

                for experiment in range(num_experiments):
                    for fold in folds:
                        largest_subset_size = max(dataset_sizes[fold])
                        if largest_subset_size <= 0:
                            continue

                        loss_key = (
                            f"{generator}_loss_{largest_subset_size}_{fold}_"
                            f"{experiment}_{lr}_{reg}_{lambda_generate}"
                        )
                        val_loss_key = (
                            f"{generator}_val_loss_{largest_subset_size}_{fold}_"
                            f"{experiment}_{lr}_{reg}_{lambda_generate}"
                        )

                        if loss_key not in history.get(generator, {}):
                            continue
                        if val_loss_key not in history[generator]:
                            continue

                        train_loss = history[generator][loss_key]
                        val_loss   = history[generator][val_loss_key]

                        all_train_losses.append(train_loss)
                        all_val_losses.append(val_loss)

                if not all_train_losses or not all_val_losses:
                    continue

                # At least one valid set found for this λ
                plotted_lambdas.add(lambda_generate)

                # Ensure consistent epoch lengths by truncation
                min_len_train = min(len(t) for t in all_train_losses)
                min_len_val   = min(len(t) for t in all_val_losses)
                min_len       = min(min_len_train, min_len_val)

                all_train_losses = [arr[:min_len] for arr in all_train_losses]
                all_val_losses   = [arr[:min_len] for arr in all_val_losses]

                # Convert to arrays, compute mean ± std
                all_train_losses = np.array(all_train_losses)
                all_val_losses   = np.array(all_val_losses)
                mean_train = np.mean(all_train_losses, axis=0)
                std_train  = np.std(all_train_losses, axis=0)
                mean_val   = np.mean(all_val_losses, axis=0)
                std_val    = np.std(all_val_losses, axis=0)

                epochs = range(len(mean_train))
                color = lambda_to_color[lambda_generate]

                # Plot training (solid)
                ax.plot(
                    epochs, mean_train, color=color, linestyle='-'
                )
                ax.fill_between(
                    epochs,
                    mean_train - std_train,
                    mean_train + std_train,
                    color=color, alpha=0.2
                )

                # Plot validation (dashed)
                ax.plot(
                    epochs, mean_val, color=color, linestyle='--'
                )
                ax.fill_between(
                    epochs,
                    mean_val - std_val,
                    mean_val + std_val,
                    color=color, alpha=0.2
                )

        ax.set_title(f"Training and Validation Loss for Generator={generator}", fontsize=16)
        ax.set_xlabel("Epochs", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)
        ax.grid(True)

        # Build the color legend for each λ that was actually plotted
        color_handles = []
        color_labels = []
        for lam in sorted_lambdas:
            if lam not in plotted_lambdas:
                continue
            # Create a dummy line with the correct color
            line = mlines.Line2D([], [], color=lambda_to_color[lam], linestyle='-',
                                 label=f"λ={lam}")
            color_handles.append(line)
            color_labels.append(f"λ={lam}")

        # Combine color-based handles + line-style handles
        # So we get a 2-column legend: one for λ colors, one for Train/Val styles
        combined_handles = color_handles + [train_line, val_line]
        combined_labels  = color_labels  + ["Train", "Validation"]

        # Create a single legend with 2 columns
        ax.legend(combined_handles, combined_labels, fontsize=14, loc="upper right",
                  ncol=2, columnspacing=1.2, handletextpad=0.7)

        plt.tight_layout()
        plt.savefig(f"./classifier/{galaxy_classes}_{classifier}_{generator}_{dataset_sizes[-1][-1]}_loss.png")
        plt.close(fig)

def old_plot_loss(
    generators,
    history=history,
    dataset_sizes=dataset_sizes,
    folds=folds,
    num_experiments=num_experiments,
    galaxy_classes=galaxy_classes,
    learning_rates=learning_rates,
    regularization_params=regularization_params,
    classifier=None,
    lambda_values=lambda_values
):
    # Sort lambdas so we can assign colors in a nice gradient order
    sorted_lambdas = sorted(lambda_values)
    n_lambdas = len(sorted_lambdas)

    # 1) Get a continuous colormap
    base_cmap = colormaps["viridis"]

    # 2) Create a discrete list of colors by sampling the colormap.
    #    Handle the edge case of n_lambdas=1 to avoid divide-by-zero.
    if n_lambdas > 1:
        colors = [base_cmap(i / (n_lambdas - 1)) for i in range(n_lambdas)]
    else:
        # If there's only one λ, just pick a single color (e.g. mid-range)
        colors = [base_cmap(0.5)]

    # 3) Map each λ to one color
    lambda_to_color = {
        lam: colors[i] for i, lam in enumerate(sorted_lambdas)
    }

    for generator in generators:
        fig, ax = plt.subplots(figsize=(10, 8))

        for lr, reg in itertools.product(learning_rates, regularization_params):
            for lambda_generate in sorted_lambdas:
                all_train_losses = []
                all_val_losses   = []

                for experiment in range(num_experiments):
                    for fold in folds:
                        largest_subset_size = max(dataset_sizes[fold])
                        if largest_subset_size <= 0:
                            continue

                        loss_key = (
                            f"{generator}_loss_{largest_subset_size}_{fold}_"
                            f"{experiment}_{lr}_{reg}_{lambda_generate}"
                        )
                        val_loss_key = (
                            f"{generator}_val_loss_{largest_subset_size}_{fold}_"
                            f"{experiment}_{lr}_{reg}_{lambda_generate}"
                        )

                        if loss_key not in history.get(generator, {}):
                            continue
                        if val_loss_key not in history[generator]:
                            continue

                        train_loss = history[generator][loss_key]
                        val_loss   = history[generator][val_loss_key]

                        all_train_losses.append(train_loss)
                        all_val_losses.append(val_loss)

                if not all_train_losses or not all_val_losses:
                    continue

                # Truncate to min length if epochs differ
                min_len_train = min(len(t) for t in all_train_losses)
                min_len_val   = min(len(t) for t in all_val_losses)
                min_len       = min(min_len_train, min_len_val)

                all_train_losses = [arr[:min_len] for arr in all_train_losses]
                all_val_losses   = [arr[:min_len] for arr in all_val_losses]

                # Convert to arrays
                all_train_losses = np.array(all_train_losses)
                all_val_losses   = np.array(all_val_losses)
                mean_train = np.mean(all_train_losses, axis=0)
                std_train  = np.std(all_train_losses, axis=0)
                mean_val   = np.mean(all_val_losses, axis=0)
                std_val    = np.std(all_val_losses, axis=0)

                epochs = range(len(mean_train))
                color = lambda_to_color[lambda_generate]

                # Plot training (solid) - label only once per λ
                ax.plot(epochs, mean_train, color=color, label=f"λ={lambda_generate}")
                ax.fill_between(
                    epochs,
                    mean_train - std_train,
                    mean_train + std_train,
                    color=color, alpha=0.2
                )

                # Plot validation (dashed) - no label
                ax.plot(epochs, mean_val, color=color, linestyle='--', label="")
                ax.fill_between(
                    epochs,
                    mean_val - std_val,
                    mean_val + std_val,
                    color=color, alpha=0.2
                )

        ax.set_title(f"Training and Validation Loss for Generator={generator}", fontsize=16)
        ax.set_xlabel("Epochs", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)

        # Deduplicate legend labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=14, loc="best")

        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"./classifier/{galaxy_classes}_{classifier}_{generator}_loss.png")
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
                            key = f"{generator}_{metric}_{final_subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
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
# Aggregate training times per merged category
merged_times = {}
for subset_size, folds in training_times.items():
    # Only process keys that are in merge_map
    if subset_size in merge_map:
        category = merge_map[subset_size]
        for fold, elapsed_times in folds.items():
            if not elapsed_times:
                continue
            if category not in merged_times:
                merged_times[category] = []
            merged_times[category].extend(elapsed_times)

# For the merged category "25000", compute and print the summary
if "25000" in merged_times and merged_times["25000"]:
    times = merged_times["25000"]
    mean_time = np.mean(times)
    std_time = np.std(times)
    data_points = len(times)
    print(f"Average training time for 25000 input images is {mean_time:.2f} ± {std_time:.2f} seconds (based on {data_points} data points)")
else:
    print("No training times recorded for merged category '25000'.")

plot_loss(generators, history=history)
#plot_roc_curves(metrics, generator)
plot_avg_roc_curves(metrics, generators, merge_map=merge_map)
plot_all_metrics_vs_dataset_size(metrics, generators, merge_map=merge_map)
plot_accuracy_vs_lambda(lambda_values, metrics, generator)
#plot_confusion_matrix(metrics, generator)
plot_avg_std_confusion_matrix(metrics, generators, merge_map=merge_map, metric_stats=metric_stats)
plot_diff_avg_std_confusion_matrix(metrics, generators, merge_map=merge_map, metric_stats=metric_stats)

# --- build real‐data loader (test set) ---
_, _, test_images, test_labels = load_galaxies(
    galaxy_class=galaxy_classes, fold=5,
    img_shape=(1,128,128), sample_size=None,
    REMOVEOUTLIERS=FILTERED, train=False
)
real_ds = TensorDataset(test_images, torch.zeros_like(test_images), test_labels)
real_loader = DataLoader(real_ds, batch_size=256, shuffle=False)


# --- build generated‐data loader using get_synthetic() ---
all_imgs, all_labels = [], []

if generators == ['GAN']:   
    model_kwargs = {
        'gan_type': 'GAN',
        'gan_latent_dim': 64,
        'lr_gen': 1e-4,
        'lr_disc': 1e-4,
        'gan_gen_loss': 'MSE',
        'gan_disc_loss': 'BCE',
        'gan_adam_beta': 0.9,
        'gan_weight_decay': 0.0,
        'gan_label_smoothing': 0.0,
        'gan_lambda_div': 0.0,
        'gan_data_version': 0,
        'gan_epoch': 100}

elif generators == ['Dual']:
    from kymatio.torch import Scattering2D
    scattering = Scattering2D(J=2, L=12, shape=(128, 128), max_order=2)   
    scat_coeffs = scattering(test_images[:5])    
    model_kwargs = {
        'scatshape': np.shape(scat_coeffs)[1:],
        'hidden_dim1': 128,
        'hidden_dim2': 128,
        'latent_dim': 64,
        'vae_latent_dim': 64}

else:
    model_kwargs = {}
    
for cls in galaxy_classes:
    imgs_list, labels_list = get_synthetic(
        num_generate=len(test_images),
        gen_model_name='DDPM',
        cls=cls,
        galaxy_classes=galaxy_classes,
        batch_size=256,
        img_shape=(1,128,128),
        FILTERGEN=FILTERED,
        model_kwargs=model_kwargs,
        fold=5,
        device=device
    )
    all_imgs   .extend(imgs_list)
    all_labels .extend(labels_list)

gen_images = torch.cat(all_imgs,   dim=0)
gen_labels = torch.cat(all_labels, dim=0)
gen_ds     = TensorDataset(gen_images, torch.zeros_like(gen_images), gen_labels)
gen_loader = DataLoader(gen_ds, batch_size=256, shuffle=False)

print("clf_models type:", type(clf_models))
if isinstance(clf_models, dict):
    print("clf_models keys:", list(clf_models.keys()))
else:
    print(clf_models)

from utils.classifiers import RustigeClassifier

wrapper = clf_models['RustigeClassifier']   # this is a dict with key "model"
model   = wrapper['model']                 # the actual nn.Module you trained
model   = model.to(device).eval()
visualize_feature_tsne(
    model,
    real_loader,
    gen_loader,
    device=device,
    perplexity=30,
    n_iter=1000,
    save_path=f"./classifier/{galaxy_classes}_{classifier}_{generator}_{final_subset_size}_tsne.png"
)
visualize_tsne_by_class(model, real_loader, gen_loader, device=device,
                        save_path=f"./classifier/{galaxy_classes}_{classifier}_{generator}_{final_subset_size}_tsne_by_class.png"
)