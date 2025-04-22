import numpy as np
import torch
import time
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
from torchsummary import summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from utils.data_loader import load_galaxies, get_classes, augment_images
from utils.classifiers import RustigeClassifier, ProjectModel, MLPClassifier, DualClassifier
from utils.training_tools import EarlyStopping, reset_weights
from utils.calc_tools import normalize_to_0_1, normalize_to_minus1_1, cluster_metrics, get_model_name, generate_from_noise, load_model
from utils.plotting import plot_histograms, plot_images_by_class, plot_image_grid
from utils.GAN_models import load_gan_generator
from torchvision.utils import make_grid, save_image
from kymatio.torch import Scattering2D
import pickle
from tqdm import tqdm
from torch.optim import AdamW
import itertools
import matplotlib 
import os
os.makedirs('./classifier/trained_models', exist_ok=True)
os.makedirs('./classifier/trained_models_filtered', exist_ok=True)
matplotlib.use('Agg')
tqdm.pandas(disable=True)

print("Running script 4.1")

###############################################
################ CONFIGURATION ################
###############################################

#EDIT: Use embedding for labels

classes = get_classes()
#galaxy_classes = [31, 32, 33, 34, 35, 36]  # Classes to classify
galaxy_classes = [10, 11, 12, 13]  # Classes to classify
max_num_galaxies = 100000  # Upper limit for the all-classes combined training data before classical augmentation
dataset_portions = [0.01, 0.1, 1]  # Portions of complete dataset for the accuracy vs dataset size
J, L, order = 2, 12, 2  # Scatter transform parameters
classifier = ["ProjectModel", "Rustige", "ScatterNet", "ScatterDual"][1]  # Choose one classifier model model
gen_model_names = ['DDPM'] #['GAN', 'Dual', 'CNN', 'STMLP', 'lavgSTMLP', 'ldiffSTMLP'] # Specify the generative model_name
num_epochs_cuda = 200
num_epochs_cpu = 100
batch_size = 250
learning_rates = [1e-3]  # Learning rates
regularization_params = [0]  # Regularisation parameters
img_shape = (1, 128, 128)
num_experiments = 10
folds = [5] # 0-4 for 5-fold cross validation, 5 for only one training
lambda_values = [0, 0.25, 0.5, 1, 2, 3, 5]  # Ratio between generated images and original images per class. 8 is reserfved for TESTONGENERATED

ES, patience = True, 10  # Use early stopping
SHOWIMGS = True  # Show some generated images for each class (Tool for control)
NORMALISEIMGS = True  # Normalise images to [0, 1]
NORMALISEIMGSTOPM = False  # Normalise images to [-1, 1] 
NORMALISESCS = False  # Normalise scattering coefficients to [0, 1]
NORMALISESCSTOPM = False  # Normalise scattering coefficients to [-1, 1]
USE_CLASS_WEIGHTS = True  # Set to False to disable class weights
TESTONGENERATED = False  # Use generated data as testdata
FILTERED = True  # Remove in training, validation and test data for the classifier
FILTERGEN = False  # Remove generated images that are too similar to other generated images
USE_MEMMAP = False  # Use memmap for scattering coefficients

# -------------------------- NEW GAN CONFIGURATION --------------------------
gan_epoch = 500           # e.g., epoch number to load from
gan_gen_loss = 'MSE'       # e.g., generator loss value (as used in filename)
gan_disc_loss = 'BCE'      # e.g., discriminator loss value (as used in filename)
gan_latent_dim = 128
lr_gen = 1e-3
lr_disc = 1e-4
gan_adam_beta = (0.5, 0.999)
gan_weight_decay = 0.0
gan_label_smoothing = 0.9
gan_lambda_div = 0.1
gan_data_version = ['full', 'clean'][0] 
gan_type = ['Simple', 'Advanced'][0]
# -------------------------------------------------------------------------

#########################################################################################################################

if lambda_values != [0]:
    VAE_train_size = {galaxy_classes[0]: 1101128, galaxy_classes[1]: 1101128, galaxy_classes[2]: 1101128, galaxy_classes[3]: 1101128}
    forbidden_classes = [galaxy_classes[3]]  # Generated bent sources look awful

if TESTONGENERATED:
    lambda_values = [8]  # To identify and distinguish TESTONGENERATED from other runs
    print("Using generated data for testing.")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    DEVICE = "cuda"
    num_epochs = num_epochs_cuda
    print(f"CUDA is available. Setting epochs to {num_epochs}.")
else:
    DEVICE = "cpu"
    num_epochs = num_epochs_cpu
    print(f"CUDA is not available. Setting epochs to {num_epochs}.")

if set(galaxy_classes) & {18} or set(galaxy_classes) & {19}:
    galaxy_classes = [20, 21, 21, 22, 23, 24, 25, 26, 27, 28, 29]  # Include all digits if 18 or 19 is in target_classes
else:
    galaxy_classes = galaxy_classes
num_classes = len(galaxy_classes)

###############################################
########### DATA STORING FUNCTIONS ###############
###############################################


def initialize_metrics(metrics, model_name, subset_size, fold, experiment, lr, reg, lambda_generate):
    subset_size_str = str(subset_size[0]) if isinstance(subset_size, list) else str(subset_size)
    metrics[f"{model_name}_accuracy_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{model_name}_precision_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{model_name}_recall_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    metrics[f"{model_name}_f1_score_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"] = []
    
    
def update_metrics(metrics, model_name, subset_size, fold, experiment, lr, reg, accuracy, precision, recall, f1, lambda_generate):
    subset_size_str = str(subset_size)
    metrics[f"{model_name}_accuracy_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(accuracy)
    metrics[f"{model_name}_precision_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(precision)
    metrics[f"{model_name}_recall_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(recall)
    metrics[f"{model_name}_f1_score_{subset_size_str}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"].append(f1)
    
    
def initialize_history(history, model_name, subset_size, fold, experiment, lr, reg, lambda_generate):
    if model_name not in history:
        history[model_name] = {}

    loss_key = f"{model_name}_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
    val_loss_key = f"{model_name}_val_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"

    if loss_key not in history[model_name]:
        history[model_name][loss_key] = []
    if val_loss_key not in history[model_name]:
        history[model_name][val_loss_key] = []

def initialize_labels(all_true_labels, all_pred_labels, model_name, subset_size, fold, experiment, lr, reg, lambda_generate):
    key = f"{model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
    all_true_labels[key] = []
    all_pred_labels[key] = []

###############################################
########## INITIALIZE DICTIONARIES ############
###############################################

metrics = {
    "accuracy": {},
    "precision": {},
    "recall": {},
    "f1_score": {}
}

metric_colors = {
    "accuracy": 'blue',
    "precision": 'green',
    "recall": 'red',
    "f1_score": 'orange'
}

all_true_labels = {}
all_pred_labels = {}
training_times = {}
all_pred_probs = {}
history = {} 
dataset_sizes = {}

##############################################
############## FUNCTIONS #####################
##############################################

scattering = Scattering2D(J=J, L=L, shape=img_shape[1:], max_order=order)       
def compute_scattering_coeffs(images, scattering=scattering, batch_size=128, device="cpu"):
    print("Computing scattering coefficients in batches on CPU...")
    scat_coeffs_list = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)
            batch_scat = scattering(batch).detach()  # e.g. shape [B, 1, C, H, W]
            # Squeeze out the singleton dimension at index 1 if present.
            if batch_scat.shape[1] == 1:
                batch_scat = batch_scat.squeeze(1)  # becomes [B, C, H, W]
            scat_coeffs_list.append(batch_scat.cpu())
        scat_coeffs = torch.cat(scat_coeffs_list, dim=0)
    return scat_coeffs


def cache_scattering(images, scattering, cache_path, batch_size=128, device='cpu'):
    if os.path.exists(cache_path):
        print("Loading cached scattering coefficients from", cache_path)
        scat_coeffs = torch.load(cache_path)
    else:
        print("Computing scattering coefficients and caching to", cache_path)
        scat_coeffs_list = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size].to(device)
                batch_scat = scattering(batch).detach()
                if batch_scat.shape[1] == 1:
                    batch_scat = batch_scat.squeeze(1)
                scat_coeffs_list.append(batch_scat.cpu())
        scat_coeffs = torch.cat(scat_coeffs_list, dim=0)
        torch.save(scat_coeffs, cache_path)
    return scat_coeffs


# First, when you compute and save the scattering coefficients,
# write them to a memmap file instead of saving a full tensor.
def cache_scattering_memmap(images, scattering, cache_path, batch_size=128, device="cpu"):
    num_images = len(images)
    # Determine the shape for one sample
    with torch.no_grad():
        sample = scattering(images[:batch_size].to(device)).detach()
        if sample.shape[1] == 1:
            sample = sample.squeeze(1)
        sample_shape = sample.shape[1:]  # e.g., (C, H, W)
    full_shape = (num_images,) + sample_shape
    
    if not os.path.exists(cache_path):
        print("Computing and writing scattering coefficients to memmap at", cache_path)
        # Create a memmap array in write mode
        memmap_array = np.memmap(cache_path, dtype=np.float32, mode='w+', shape=full_shape)
        with torch.no_grad():
            for i in range(0, num_images, batch_size):
                batch = images[i:i+batch_size].to(device)
                batch_scat = scattering(batch).detach()
                if batch_scat.shape[1] == 1:
                    batch_scat = batch_scat.squeeze(1)
                memmap_array[i:i+batch_size] = batch_scat.cpu().numpy()
        # Flush changes to disk
        memmap_array.flush()
    else:
        print("Memmap cache found at", cache_path)
    return cache_path, full_shape

# Then, create a custom dataset that loads scattering coefficients from the memmap file.
class CachedScatterDataset(Dataset):
    def __init__(self, images, labels, cache_file, scatter_shape):
        self.images = images  # still in memory
        self.labels = labels
        self.cache_file = cache_file
        self.scatter_shape = scatter_shape
        # Open the memmap in read mode; this does not load everything into RAM.
        self.scat_memmap = np.memmap(cache_file, dtype=np.float32, mode='r', shape=scatter_shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        # Only load the scattering coefficient for the current index.
        scat = torch.tensor(self.scat_memmap[idx]).float()
        if NORMALISESCS:
            scat = normalize_to_0_1(scat)  # or normalize_to_minus1_1(scat)
            if NORMALISESCSTOPM:
                scat = normalize_to_minus1_1(scat)

        return image, scat, label
def custom_collate(batch):
    if len(batch) < 3:  # Drop the batch if it contains fewer than 3 images
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def save_images_tensorboard(images, save_path, nrow=4):
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(grid, save_path)
    
def filter_generated_images(generated_images_list, generated_labels_list):
    # Pairwise comparison of generated images
    filtered_images_list, filtered_labels_list = [], []
    for i, img1 in enumerate(generated_images_list):
        is_unique = True
        for j, img2 in enumerate(generated_images_list):
            if i == j:
                continue
            if torch.allclose(img1, img2, atol=1e-3):
                is_unique = False
                break
        if is_unique:
            filtered_images_list.append(img1)
            filtered_labels_list.append(generated_labels_list[i])
            
    # Plot some example of images next to their similar images
    if SHOWIMGS:
        for i in range(min(18, len(filtered_images_list))):
            save_images_tensorboard(torch.cat([filtered_images_list[i], generated_images_list[i]], dim=0), save_path=f"./classifier/{gen_model_name}_{galaxy_classes}_{num_galaxies}_filtered_grid.png", nrow=6)

    return filtered_images_list, filtered_labels_list

###############################################
########### READ IN TEST DATA #################
######## Needs only be done once ##############
###############################################

for gen_model_name in gen_model_names:

    if gen_model_name != 'GAN':
        scatshape = (1, 128, 128)
        hidden_dim1 = 256
        hidden_dim2 = 128
        vae_latent_dim = 64

    if not TESTONGENERATED:
        final_test_data = load_galaxies(galaxy_class=galaxy_classes, 
                    fold=max(folds), #Any fold other than 5 gives me the test data for the five fold cross validation
                    img_shape=img_shape, 
                    sample_size=max_num_galaxies, 
                    REMOVEOUTLIERS=FILTERED,
                    train=False)
        _, _, test_images, test_labels = final_test_data

        perm = torch.randperm(test_images.size(0))
        test_images = test_images[perm]
        test_labels = test_labels[perm]

        # Print the distribution of raw test labels
        unique_labels, counts = torch.unique(test_labels, return_counts=True)
        print("Test labels distribution (raw):", dict(zip(unique_labels.tolist(), counts.tolist())))
        print("Test images shape: ", np.shape(test_images))

        if classifier in ['ProjectModel', 'Rustige', 'ScatterDual']:
            if NORMALISEIMGS:
                test_images = normalize_to_0_1(test_images)
                if NORMALISEIMGSTOPM: 
                    test_images = normalize_to_minus1_1(test_images)

        # Handle scattering coefficients normalization in a similar way
        if classifier in ['ScatterNet', 'ScatterDual']:
            if NORMALISESCS:
                test_scat_coeffs = normalize_to_0_1(test_scat_coeffs)
                if NORMALISESCSTOPM:
                    test_scat_coeffs = normalize_to_minus1_1(test_scat_coeffs)
                    
        # Prepare input data
        # Produce an empty tensor to occupy the not used component of the datasets. 
        # It should have the same size as the test images
        mock_tensor = torch.zeros_like(test_images)
        if classifier in ['ScatterNet', 'ScatterDual']:
            test_scat_coeffs = compute_scattering_coeffs(test_images)
        if classifier == "ScatterNet":
            test_dataset = TensorDataset(mock_tensor, test_scat_coeffs, test_labels)
        if classifier == 'ScatterDual':
            test_dataset = TensorDataset(test_images, test_scat_coeffs, test_labels)
        else: 
            test_dataset = TensorDataset(test_images, mock_tensor, test_labels) 
                                
        min_label = test_labels.min()
        test_labels = test_labels - min_label # Labels remain as integers for CrossEntropyLoss
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=True)

        print(f"Test dataset size: {len(test_dataset)}")

    ###############################################
    ########### LOOP OVER DATA FOLD ###############
    ###############################################            

    FIRSTTIME = True  # Set to True to print model summaries only once
    param_combinations = list(itertools.product(folds, learning_rates, regularization_params, lambda_values))
    print(f"DEBUG → folds = {folds}")
    print(f"DEBUG → param_combinations (first 10) = {param_combinations[:10]}  (total {len(param_combinations)})")
    for fold, lr, reg, lambda_generate in param_combinations:
        gan_fold = fold
        torch.cuda.empty_cache()
        print(f"\n Training with fold: {fold}, lr: {lr}, reg: {reg}, lambda: {lambda_generate}")
        runname = f'{galaxy_classes}_{gen_model_name}_lr{lr}_reg{reg}'
        log_path = f"./classifier/log_{runname}.txt"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        #file = open(log_path, 'w')

        # Load the data
        data = load_galaxies(galaxy_class=galaxy_classes, 
                            fold=fold,
                            img_shape=img_shape, 
                            sample_size=max_num_galaxies, 
                            REMOVEOUTLIERS=FILTERED,
                            train=True)
        train_images, train_labels, valid_images, valid_labels = data

        #print("Train images shape: ", np.shape(train_images))
        #print("Validation images shape: ", np.shape(valid_images))
        perm = torch.randperm(train_images.size(0))
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        perm = torch.randperm(valid_images.size(0))
        valid_images = valid_images[perm]
        valid_labels = valid_labels[perm]
                        
        num_galaxies = len(train_images)
        dataset_sizes[fold] = [int(num_galaxies * perc) for perc in dataset_portions]

        if TESTONGENERATED:  # Use generated data for testing
            all_test_images = []
            all_test_labels = []
                        
            for cls in galaxy_classes:
                try:
                    if gen_model_name == 'GAN':
                        # Use GAN generator filename convention
                        gan_path = f"./GAN/generators/generator_{gan_type}_latent{gan_latent_dim}_cl{cls}_lrgen{lr_gen}_lrdisc{lr_disc}_gl{gan_gen_loss}_dl{gan_disc_loss}_ab{gan_adam_beta}_wd{gan_weight_decay}_ls{gan_label_smoothing}_ld{gan_lambda_div}_dv{gan_data_version}_ep{gan_epoch}_f{gan_fold}.pth"
                        model = load_gan_generator(gan_path, latent_dim=gan_latent_dim)
                        #print("Loaded model from gan_path: ", gan_path)
                        latent_dim = gan_latent_dim
                    elif gen_model_name == 'DDPM':
                        ddpm_file_map = {
                            10: "generated_fr1_8k.npy",
                            11: "generated_fr2_8k.npy",
                            12: "generated_compact_8k.npy",
                            13: "generated_bent_8k.npy"
                        }
                        ddpm_path = f"/users/mbredber/data/DDPM/{ddpm_file_map[cls]}"
                        print(f"Loading DDPM data for class {cls} from {ddpm_path}")
                        ddpm_data = np.load(ddpm_path)  # shape assumed (N, 128, 128)
                        ddpm_data = torch.from_numpy(ddpm_data).unsqueeze(1).float()  # (N, 1, 128, 128)

                        # Use as many samples as num_generate, or all if fewer exist
                        if ddpm_data.shape[0] < num_generate:
                            num_generate = ddpm_data.shape[0]
                        ddpm_data = ddpm_data[:num_generate]
                        
                        # Append to the lists (do not overwrite!)
                        generated_images_list.append(ddpm_data)
                        generated_labels_list.append(torch.ones(num_generate, dtype=torch.long) * (cls - min(galaxy_classes)))

                    else: # VAE
                        path = get_model_name(cls, VAE_train_size[cls], gen_model_name, fold=fold)
                        model = load_model(path, scatshape=scatshape,
                                    hidden_dim1=hidden_dim1,
                                    hidden_dim2=hidden_dim2,
                                    latent_dim=vae_latent_dim,
                                    num_classes=num_classes)
                        latent_dim = vae_latent_dim
                        
                except Exception as e:
                    print(f"Model error for class: {cls}. Error: {e}")
                    exit(1)
                    
                # If using a GAN or VAE, generate images
                if gen_model_name != 'DDPM':
                    model.eval()
                    batch_size_generate = 500  
                    num_generate = 100  # This number needs to be motivated with convergence analysis
                    num_batches = (num_generate + batch_size_generate - 1) // batch_size_generate
                    generated_images_list = []
                    generated_labels_list = []
                    for batch_idx in range(num_batches):
                        current_batch = min(batch_size_generate, num_generate - batch_idx * batch_size_generate)
                        generated_labels = torch.ones(current_batch, dtype=torch.long) * (cls - min(galaxy_classes))
                        with torch.no_grad():
                            generated_batch = generate_from_noise(
                                model,
                                latent_dim=latent_dim,
                                num_samples=current_batch,
                                DEVICE='cpu')
                        # Ensure the batch dimension is first:
                        if generated_batch.shape[0] != current_batch:
                            generated_batch = generated_batch.permute(1, 0, 2, 3)
                        generated_batch = generated_batch.to(train_images.device)
                        generated_labels = generated_labels.to(train_labels.device)
                        generated_images_list.append(generated_batch)
                        generated_labels_list.append(generated_labels)
                all_test_images.append(torch.cat(generated_images_list, dim=0))
                all_test_labels.append(torch.cat(generated_labels_list, dim=0))
            test_images = torch.cat(all_test_images, dim=0)
            test_labels = torch.cat(all_test_labels, dim=0)

            print("Shape of test_images: ", test_images.shape)
            
            # Shuffle the test data
            perm = torch.randperm(test_images.size(0))
            test_images = test_images[perm]
            test_labels = test_labels[perm]

            # Normalize train and test images to [0, 1]
            if classifier in ['ProjectModel', 'Rustige', 'ScatterDual']:
                if NORMALISEIMGS:
                    test_images = normalize_to_0_1(test_images)
                    if NORMALISEIMGSTOPM:  # normalize to [-1, 1]
                        test_images = normalize_to_minus1_1(test_images)

            # Handle scattering coefficients normalization in a similar way
            if classifier in ['ScatterNet', 'ScatterDual']:
                if NORMALISESCS:
                    test_scat_coeffs = normalize_to_0_1(test_scat_coeffs)
                    if NORMALISESCSTOPM:
                        test_scat_coeffs = normalize_to_minus1_1(test_scat_coeffs)

            ###### RELABEL ######
            min_label = test_labels.min()
            test_labels = test_labels - min_label

            # Prepare input data
            mock_tensor = torch.zeros_like(test_images)
            if classifier in ['ScatterNet', 'ScatterDual']:
                test_scat_coeffs = compute_scattering_coeffs(test_images)
            if classifier == "ScatterNet":
                test_dataset = TensorDataset(mock_tensor, test_scat_coeffs, test_labels)
            if classifier == 'ScatterDual':
                test_dataset = TensorDataset(test_images, test_scat_coeffs, test_labels)
            else: 
                test_dataset = TensorDataset(test_images, mock_tensor, test_labels) 

            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=True)

        
        if USE_CLASS_WEIGHTS:
            unique, counts = np.unique(train_labels.cpu().numpy(), return_counts=True)
            class_counts = dict(zip(map(int, unique), map(int, counts)))
            total_count = sum(counts)
            class_weights = {int(cls): float(total_count / count) for cls, count in class_counts.items()}
            normalized_classes = [cls - min(galaxy_classes) for cls in galaxy_classes]
            weights = torch.tensor([class_weights.get(cls, 1.0) for cls in normalized_classes], dtype=torch.float).to(DEVICE)
            missing_classes = [cls for cls in normalized_classes if cls not in class_weights]
            if missing_classes:
                print(f"Warning: Missing classes in dataset: {missing_classes}")
                class_weights.update({int(cls): 1.0 for cls in missing_classes})
                
            unique_valid, counts_valid = np.unique(valid_labels.cpu().numpy(), return_counts=True)
            unique_test, counts_test = np.unique(test_labels.cpu().numpy(), return_counts=True)
            class_counts_valid = dict(zip(map(int, unique_valid), map(int, counts_valid)))
            class_counts_test = dict(zip(map(int, unique_test), map(int, counts_test)))
        else:
            weights = None
            print("No class weighting")
            print("Classes used: ", galaxy_classes)

        if fold == 0:
            plot_images_by_class(train_images, labels=train_labels, num_images=5, save_path=f"./classifier/{gen_model_name}_{galaxy_classes}_{num_galaxies}_example_inputs.png")

        ##########################################################
        ############# ARTIFICIAL AUGMENTATION ####################
        ##########################################################
                    
        # Generate more training data if requested
        if lambda_generate not in [0, 8]:
            batch_size_generate = 500 
            num_generate = int(lambda_generate / len(galaxy_classes) * len(train_images))
            print("Old training data size: ", train_images.size())
            print('Number of images to generate for each class: ', num_generate)
            
            for cls in galaxy_classes:
                try:
                    if gen_model_name == 'GAN':
                        gan_path = f"./GAN/generators/generator_{gan_type}_latent{gan_latent_dim}_cl{cls}_lrgen{lr_gen}_lrdisc{lr_disc}_gl{gan_gen_loss}_dl{gan_disc_loss}_ab{gan_adam_beta}_wd{gan_weight_decay}_ls{gan_label_smoothing}_ld{gan_lambda_div}_dv{gan_data_version}_ep{gan_epoch}_f{gan_fold}.pth"
                        model = load_gan_generator(gan_path, latent_dim=gan_latent_dim)
                        latent_dim = gan_latent_dim
                        
                    elif gen_model_name == 'DDPM':
                        # Map numeric class IDs to the correct DDPM .npy file
                        ddpm_file_map = {
                            10: "generated_fr1.npy",
                            11: "generated_fr2.npy",
                            12: "generated_compact.npy",
                            13: "generated_bent.npy"
                        }
                        ddpm_path = f"/users/mbredber/data/DDPM/{ddpm_file_map[cls]}"
                        print(f"Loading DDPM data for class {cls} from {ddpm_path}")
                        ddpm_data = np.load(ddpm_path)  # shape assumed (N, 128, 128)
                        ddpm_data = torch.from_numpy(ddpm_data).unsqueeze(1).float()  # (N,1,128,128)
                        ddpm_data = normalize_to_0_1(ddpm_data)
                        ddpm_data = normalize_to_minus1_1(ddpm_data)

                        # Use as many samples as num_generate, or all if fewer exist
                        if ddpm_data.shape[0] < num_generate:
                            num_generate = ddpm_data.shape[0]
                        else:
                            print(f"Warning: Only {ddpm_data.shape[0]} samples available for class {cls}.")
                        generated_batch = ddpm_data[:num_generate]
                        generated_labels = torch.ones(num_generate, dtype=torch.long) * (cls - min(galaxy_classes))
                        generated_images_list = [ddpm_data]
                        generated_labels_list = [generated_labels]
                
                    else:
                        path = get_model_name(cls, VAE_train_size[cls], gen_model_name, fold=fold)
                        model = load_model(path, scatshape=scatshape,
                                        hidden_dim1=hidden_dim1,
                                        hidden_dim2=hidden_dim2,
                                        latent_dim=vae_latent_dim,
                                        num_classes=num_classes)
                        latent_dim = vae_latent_dim
                except Exception as e:
                    print(f"Model not found for class: {cls}. Error: {e}")
                    exit(1)

                if gen_model_name != 'DDPM':
                    model.eval()
                    num_batches = (num_generate + batch_size_generate - 1) // batch_size_generate
                    total_generated = 0
                    total_unique = 0
                    generated_images_list = []
                    generated_labels_list = []
                    tol = 1e-3  # Tolerance for duplicate comparison

                    for batch_idx in range(num_batches):
                        current_batch = min(batch_size_generate, num_generate - batch_idx * batch_size_generate)
                        total_generated += current_batch

                        generated_labels = torch.ones(current_batch, dtype=torch.long) * (cls - min(galaxy_classes))
                        with torch.no_grad():
                            generated_batch = generate_from_noise(
                                model,
                                latent_dim=latent_dim,
                                num_samples=current_batch,
                                DEVICE='cpu')
                        # Ensure the tensor is [batch_size, channels, H, W]
                        if generated_batch.shape[0] != current_batch:
                            generated_batch = generated_batch.permute(1, 0, 2, 3)

                        if FILTERGEN and cls != 12:
                            # --------------------------
                            # (1) Remove duplicates *within* the new batch
                            # --------------------------
                            flat = generated_batch.view(current_batch, -1)  # flatten each image
                            dists = torch.cdist(flat, flat, p=2)
                            dists.fill_diagonal_(1e6)  # ignore self-comparisons
                            unique_mask = torch.ones(current_batch, dtype=torch.bool, device=generated_batch.device)
                            for i in range(current_batch):
                                if unique_mask[i]:
                                    # Mark subsequent images that are too close as duplicates
                                    unique_mask[i+1:] &= ~(dists[i, i+1:] < tol)

                            batch_unique_images = generated_batch[unique_mask]
                            batch_unique_labels = generated_labels[unique_mask]
                            n_unique = len(batch_unique_images)

                            # --------------------------
                            # (2) Remove duplicates *across* previously accepted images
                            # --------------------------
                            if len(generated_images_list) > 0:
                                # Flatten the newly filtered batch
                                new_flat = batch_unique_images.view(n_unique, -1)

                                # Flatten everything we've already accepted
                                prev_accepted = torch.cat(generated_images_list, dim=0)  # shape: [N_accepted, C, H, W]
                                prev_flat = prev_accepted.view(prev_accepted.size(0), -1)

                                # Compare new vs. all accepted
                                cross_dists = torch.cdist(new_flat, prev_flat, p=2)
                                cross_unique_mask = torch.ones(n_unique, dtype=torch.bool, device=batch_unique_images.device)
                                for i in range(n_unique):
                                    # If the new image is within tol of *any* accepted image, exclude it
                                    if torch.any(cross_dists[i] < tol):
                                        cross_unique_mask[i] = False

                                batch_unique_images = batch_unique_images[cross_unique_mask]
                                batch_unique_labels = batch_unique_labels[cross_unique_mask]
                                n_unique = len(batch_unique_images)

                            # Now everything in batch_unique_* is unique vs. each other & prior batches
                            total_unique += n_unique
                            
                            # --------------------------
                            # (3) Augment the unique images
                            # --------------------------
                            
                            print("Shape of batch_unique_images: ", batch_unique_images.shape)
                            batch_augmented_images, batch_augmented_labels = augment_images(batch_unique_images, batch_unique_labels, img_shape=img_shape) # Produces 8 versions of each image
                            generated_images_list.append(batch_augmented_images)
                            generated_labels_list.append(batch_augmented_labels)

                        else:
                            # If we're not filtering (or if cls == 12), just accept them all
                            #batch_augmented_images, batch_augmented_labels = augment_images(batch_unique_images, batch_unique_labels, img_shape=img_shape) # Produces 8 versions of each image
                            generated_images_list.append(generated_batch)
                            generated_labels_list.append(generated_labels)

                    # After all batches:
                    if FILTERGEN and cls != 12:
                        removal_fraction = (total_generated - total_unique) / total_generated * 100
                        print(f"Class {cls}: Removed {removal_fraction:.2f}% of images "
                            f"(Generated: {total_generated}, Unique: {total_unique})")
                        
                generated_images = torch.cat(generated_images_list, dim=0)
                generated_labels = torch.cat(generated_labels_list, dim=0)

                # Append the filtered images and labels to your training data:
                
                if SHOWIMGS and lambda_generate not in [0, 8]:                 
                    pristine_train_images = train_images
            
                train_images = torch.cat([train_images, generated_images])
                train_labels = torch.cat([train_labels.clone().detach(), generated_labels])

            
        ##########################################################
        ############ NORMALISE AND PACKAGE THE INPUT #############
        ##########################################################
        
        if classifier in ['ProjectModel', 'Rustige', 'ScatterDual']:
            if NORMALISEIMGS:
                train_images = normalize_to_0_1(train_images)
                valid_images = normalize_to_0_1(valid_images)
                if lambda_generate not in [0, 8]:
                    pristine_train_images = normalize_to_0_1(pristine_train_images)
                    generated_images = normalize_to_0_1(generated_images)
                if NORMALISEIMGSTOPM:
                    train_images = normalize_to_minus1_1(train_images)
                    valid_images = normalize_to_minus1_1(valid_images)
                    if lambda_generate not in [0, 8]:
                        pristine_train_images = normalize_to_minus1_1(pristine_train_images)
                        generated_images = normalize_to_minus1_1(generated_images)
                        
        # ── SANITY‑CHECK PLOTS ON FIRST FOLD ONLY ──
        if fold == folds[0] and SHOWIMGS and lambda_generate not in [0, 8]:
            for cls in galaxy_classes:
                cls_idx = cls - min(galaxy_classes)
                gen_cls  = generated_images[generated_labels == cls_idx]
                orig_cls = train_images   [train_labels == cls_idx][:36]

                plot_image_grid(
                    gen_cls,
                    num_images=36,
                    save_path=f"./classifier/{classifier}_{gen_model_name}_{cls}_{num_galaxies}_f{fold}_generated_grid.png"
                )
                plot_image_grid(
                    orig_cls,
                    num_images=36,
                    save_path=f"./classifier/{classifier}_{gen_model_name}_{cls}_{num_galaxies}_f{fold}_train_grid.png"
                )
                plot_histograms(
                    gen_cls,
                    orig_cls,
                    title1="Generated Images",
                    title2="Train Images",
                    save_path=f"./classifier/{classifier}_{gen_model_name}_{cls}_{num_galaxies}_f{fold}_generated_hist.png"
                )

                        
        ###### RELABEL ######
        min_label = train_labels.min()
        train_labels = train_labels - min_label
        valid_labels = valid_labels - min_label
        
        # Prepare input data
        mock_tensor = torch.zeros_like(train_images)
        valid_mock_tensor = torch.zeros_like(valid_images)
        if classifier in ['ScatterNet', 'ScatterDual']:
            # Define cache paths (you can adjust these names as needed)
            train_cache_path = f"./.cache/train_scat_{galaxy_classes}_{fold}_{lambda_generate}_{dataset_portions[0]}_{FILTERED}_{num_galaxies}.npy"
            valid_cache_path = f"./.cache/valid_scat_{galaxy_classes}_{fold}_{lambda_generate}_{dataset_portions[0]}_{FILTERED}_{num_galaxies}.npy"
            
            if USE_MEMMAP:
                # Use memmap-based caching (from script 1)
                train_cache_file, train_full_shape = cache_scattering_memmap(train_images, scattering, train_cache_path, batch_size=128, device="cpu")
                valid_cache_file, valid_full_shape = cache_scattering_memmap(valid_images, scattering, valid_cache_path, batch_size=128, device="cpu")
                # Create dataset using the memmap cache
                train_dataset = CachedScatterDataset(train_images, train_labels, train_cache_file, train_full_shape)
                valid_dataset = CachedScatterDataset(valid_images, valid_labels, valid_cache_file, valid_full_shape)
                scatdim = train_full_shape[1:]  # e.g., (C, H, W)
            else:
                # Use standard caching (from script 2)
                train_cache = f"./.cache/train_scat_{galaxy_classes}_{fold}_{lambda_generate}_{dataset_portions[0]}_{FILTERED}_{num_galaxies}.pt"
                valid_cache = f"./.cache/valid_scat_{galaxy_classes}_{fold}_{lambda_generate}_{dataset_portions[0]}_{FILTERED}_{num_galaxies}.pt"
                train_scat_coeffs = cache_scattering(train_images, scattering, train_cache, batch_size=128, device="cpu")
                valid_scat_coeffs = cache_scattering(valid_images, scattering, valid_cache, batch_size=128, device="cpu")
                if NORMALISESCS:
                    train_scat_coeffs = normalize_to_0_1(train_scat_coeffs)
                    valid_scat_coeffs = normalize_to_0_1(valid_scat_coeffs)
                    if NORMALISESCSTOPM:
                        train_scat_coeffs = normalize_to_minus1_1(train_scat_coeffs)
                        valid_scat_coeffs = normalize_to_minus1_1(valid_scat_coeffs)
                scatdim = train_scat_coeffs[0].shape
                if classifier == "ScatterNet":
                    train_dataset = TensorDataset(mock_tensor, train_scat_coeffs, train_labels)
                    valid_dataset = TensorDataset(valid_mock_tensor, valid_scat_coeffs, valid_labels)
                else: # classifier == "ScatterDual"
                    train_dataset = TensorDataset(train_images, train_scat_coeffs, train_labels)
                    valid_dataset = TensorDataset(valid_images, valid_scat_coeffs, valid_labels)
        else:
            train_dataset = TensorDataset(train_images, mock_tensor, train_labels) 
            valid_dataset = TensorDataset(valid_images, valid_mock_tensor, valid_labels)


        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=True)
        
        print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")

        if SHOWIMGS and lambda_generate not in [0, 8]: 
            if classifier in ['ProjectModel', 'Rustige', 'ScatterDual']:
                #save_images_tensorboard(generated_images[:36], save_path=f"./classifier/{gen_model_name}_{galaxy_classes}_{num_galaxies}_generated.png", nrow=6)
                plot_histograms(pristine_train_images, valid_images, title1="Train images", title2="Valid images", imgs3=generated_images, imgs4=test_images, title3='Generated images', title4='Test images', save_path=f"./classifier/{gen_model_name}_{galaxy_classes}_{num_galaxies}_histograms.png")

        
                
        ###############################################
        ############# DEFINE MODEL ####################
        ###############################################

        if classifier == "Rustige":
            models = {"RustigeClassifier": {"model": RustigeClassifier(input_shape=tuple(valid_images.shape[1:]), num_classes=num_classes).to(DEVICE)}} 
        elif classifier == "ProjectModel":
            models = {"ProjectModel": {"model": ProjectModel(input_shape=tuple(valid_images.shape[1:]), num_classes=num_classes).to(DEVICE)}} 
        elif classifier == "ScatterNet":
            models = {"ScatterNet": {"model": MLPClassifier(input_dim=int(np.prod(scatdim)), num_classes=num_classes).to(DEVICE)}}
        elif classifier == "ScatterDual":
            models = {"ScatterDual": {"model": DualClassifier(img_shape=tuple(valid_images.shape[1:]), scat_shape=scatdim, num_classes=num_classes).to(DEVICE)}}
        else:
            raise ValueError("Model not found. Please select one of 'scatterMLP', 'smallSTMLP', or 'normalCNN'.")

        for classifier_name, model_details in models.items():
            if FIRSTTIME:
                print(f"Summary for {classifier_name}:")
                if classifier == "ScatterNet":
                    summary(model_details["model"], input_size=(int(np.prod(scatdim)),), device=DEVICE)
                elif classifier == "ScatterDual":
                    summary(model_details["model"], input_size=[valid_images.shape[1:], scatdim])
                else:
                    summary(model_details["model"], input_size=img_shape, device=DEVICE)
            FIRSTTIME = False
            # Compute scattering coefficients for one sample (ensure the model is in the right mode)


        ###############################################
        ############### TRAINING LOOP #################
        ###############################################
        
        if weights is not None:
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = AdamW(models[classifier_name]["model"].parameters(), lr=lr, weight_decay=reg)

        for classifier_name, model_details in models.items():
            print(f"Training {classifier_name} model...")
            model = model_details["model"].to(DEVICE)

            for subset_size in dataset_sizes[fold]:
                if subset_size <= 0:
                    print(f"Skipping invalid subset size: {subset_size}")
                    continue
                if subset_size not in training_times:
                    training_times[subset_size] = {}
                if fold not in training_times[subset_size]:
                    training_times[subset_size][fold] = []

                for experiment in range(num_experiments):
                    initialize_history(history, gen_model_name, subset_size, fold, experiment, lr, reg, lambda_generate)
                    initialize_metrics(metrics, gen_model_name, subset_size, fold, experiment, lr, reg, lambda_generate)
                    initialize_labels(all_true_labels, all_pred_labels, gen_model_name, subset_size, fold, experiment, lr, reg, lambda_generate)

                    start_time = time.time()
                    model.apply(reset_weights)

                    subset_indices = list(range(subset_size))
                    subset_train_dataset = Subset(train_dataset, subset_indices)
                    subset_train_loader = DataLoader(subset_train_dataset, batch_size=batch_size, shuffle=True)

                    early_stopping = EarlyStopping(patience=patience, verbose=False) if ES else None

                    for epoch in tqdm(range(num_epochs), desc=f'Training {classifier}_{galaxy_classes}_{gen_model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}'):
                        model.train()
                        total_loss = 0
                        total_images = 0

                        for images, scat, labels in subset_train_loader:
                            images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                            optimizer.zero_grad()
                            if classifier == "ScatterNet":
                                outputs = model(scat)
                            elif classifier == "ScatterDual":
                                outputs = model(images, scat)
                            else:
                                outputs = model(images)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()
                            total_loss += float(loss.item() * images.size(0))
                            total_images += float(images.size(0))

                        average_loss = total_loss / total_images
                        loss_key = f"{gen_model_name}_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                        history[gen_model_name][loss_key].append(average_loss)

                        model.eval()
                        val_total_loss = 0
                        val_total_images = 0

                        with torch.no_grad(): # Validate on validation data
                            for i, (images, scat, labels) in enumerate(valid_loader):
                                if images is None or len(images) == 0:
                                    print(f"Empty batch at index {i}. Skipping...")
                                    continue
                                images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                                if classifier == "ScatterNet":
                                    outputs = model(scat)
                                elif classifier == "ScatterDual":
                                    outputs = model(images, scat)
                                else:
                                    outputs = model(images)
                                loss = criterion(outputs, labels)
                                val_total_loss += float(loss.item() * images.size(0))
                                val_total_images += float(images.size(0))

                        val_average_loss = val_total_loss / val_total_images if val_total_images > 0 else float('inf')
                        val_loss_key = f"{gen_model_name}_val_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                        history[gen_model_name][val_loss_key].append(val_average_loss)
                        
                        if ES:
                            early_stopping(val_average_loss, model, f'./classifier/trained_models/{gen_model_name}_best_model.pth')
                            if early_stopping.early_stop:
                                break

                    model.eval()
                    with torch.no_grad(): # Evaluate on test data
                        key = f"{gen_model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                        all_pred_probs[key] = []
                        all_pred_labels[key] = []
                        all_true_labels[key] = []

                        for images, scat, labels in test_loader: # For ST models images are actually Scattering Coefficients
                            images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                            if classifier == "ScatterNet":
                                outputs = model(scat)
                            elif classifier == "ScatterDual":
                                outputs = model(images, scat)
                            else:
                                outputs = model(images)
                            pred_probs = torch.softmax(outputs, dim=1).cpu().numpy()
                            true_labels = labels.cpu().numpy()
                            #true_labels = torch.argmax(labels, dim=1).cpu().numpy()
                            pred_labels = np.argmax(pred_probs, axis=1)
                            all_pred_probs[key].extend(pred_probs)
                            all_pred_labels[key].extend(pred_labels)
                            all_true_labels[key].extend(true_labels)

                        accuracy = accuracy_score(all_true_labels[key], all_pred_labels[key])
                        precision = precision_score(all_true_labels[key], all_pred_labels[key], average='macro', zero_division=0)
                        recall = recall_score(all_true_labels[key], all_pred_labels[key], average='macro', zero_division=0)
                        f1 = f1_score(all_true_labels[key], all_pred_labels[key], average='macro', zero_division=0)

                        update_metrics(metrics, gen_model_name, subset_size, fold, experiment, lr, reg, accuracy, precision, recall, f1, lambda_generate)

                end_time = time.time()
                elapsed_time = end_time - start_time
                training_times[subset_size][fold].append(elapsed_time)
                if subset_size > 20000:
                    with open(log_path, 'w') as file:
                        file.write(f"Time taken to train the model on fold {fold}: {elapsed_time:.2f} seconds \n")

            generated_features = []
            with torch.no_grad():
                for images, scat, _ in test_loader:
                    images, scat = images.to(DEVICE), scat.to(DEVICE)
                    if classifier == "ScatterNet":
                        outputs = model(scat).cpu().detach().numpy()
                    elif classifier == "ScatterDual":
                        outputs = model(images, scat).cpu().detach().numpy()
                    else:
                        outputs = model(images).cpu().detach().numpy()
                    generated_features.append(outputs)

            generated_features = np.concatenate(generated_features, axis=0)
            cluster_error, cluster_distance, cluster_std_dev = cluster_metrics(generated_features, n_clusters=num_classes)
            with open(log_path, 'w') as file:
                file.write(f"Cluster results for fold {fold}, classifier {classifier}, lr {lr}, reg {reg}, lambda_generate {lambda_generate} \n")
                file.write(f"Cluster Error: {cluster_error} \n")
                file.write(f"Cluster Distance: {cluster_distance} \n")
                file.write(f"Cluster Standard Deviation: {cluster_std_dev} \n")

            model_save_path = f'./classifier/trained_models/{gen_model_name}_model.pth'
            torch.save(model.state_dict(), model_save_path)
            
    directory = './classifier/trained_models_filtered/' if FILTERED else './classifier/trained_models/'

    for fold, lr, reg, lambda_generate in param_combinations:
        for subset_size in dataset_sizes[fold]:
            for experiment in range(num_experiments):
                metrics_save_path = f'{directory}{classifier}_{galaxy_classes}_{gen_model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}_metrics_data.pkl'
                with open(metrics_save_path, 'wb') as f:
                    pickle.dump({
                        "models": models,
                        "history": history,
                        "metrics": metrics,
                        "metric_colors": metric_colors,
                        "all_true_labels": all_true_labels,
                        "all_pred_labels": all_pred_labels,
                        "training_times": training_times,
                        "all_pred_probs": all_pred_probs
                    }, f)
                print(f"Metrics and related data saved to {metrics_save_path}")
