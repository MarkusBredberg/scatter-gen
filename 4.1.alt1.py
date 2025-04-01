import numpy as np
import torch
import time
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchsummary import summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from utils.data_loader import load_galaxies, get_classes
from utils.classifiers import Scattering_Classifier, CNN_Classifier, SimpleCNN, SimpleFCN, ProjectModel
from utils.training_tools import EarlyStopping, reset_weights
from utils.scatter_reduction import lavg, ldiff
from utils.calc_tools import normalize_to_0_1, normalize_to_minus1_1, cluster_metrics, get_model_name, generate_from_noise, load_model
from utils.plotting import plot_loss, plot_images, plot_histograms, plot_images_by_class
from utils.GAN_models import DCGANGenerator
from kymatio.torch import Scattering2D
import pickle
from tqdm import tqdm
from torch.optim import AdamW
import itertools
import matplotlib
import os
os.makedirs('./classifier/trained_models', exist_ok=True)
matplotlib.use('Agg')
tqdm.pandas(disable=True)

print("Running script 4.1")

###############################################
################ CONFIGURATION ################
###############################################

classes = get_classes()
#galaxy_classes = [31, 32, 33, 34, 35, 36]  # Classes to classify
galaxy_classes = [10, 11, 12, 13]  # Classes to classify
max_num_galaxies = 100000  # Upper limit for the all-classes combined training data before classical augmentation
dataset_portions = [0.01, 0.1, 0.5, 1]  # Portions of complete dataset for the accuracy vs dataset size
J, L, order = 2, 12, 2  # Scatter transform parameters
classifier = ["scatterMLP", "normalCNN", "simpleCNN", "simpleFCN", "ProjectModel"][4]  # Choose one model
num_epochs_cuda = 200
num_epochs_cpu = 100
batch_size = 64
learning_rates = [1e-4]  # Learning rates
regularization_params = [1e-1]  # Regularisation parameters
img_shape = (1, 128, 128)
num_experiments = 1
folds = [0, 1, 2, 3, 4] # 0-4 for 5-fold cross validation, 5 for only one training
lambda_values = [0]  # Ratio between generated images and original images per class. 8 is reserved for TESTONGENERATED

ES, patience = True, 10  # Use early stopping
IMGCHECK = False  # Check the input images (Tool for control)
SAVEIMGS = False  # Save the reconstructed images in tensor format
NORMALISEIMGS = False  # Normalise images to [0, 1]
NORMALISEIMGSTOPM = False  # Normalise images to [-1, 1] 
NORMALISESCS = False  # Normalise scattering coefficients to [0, 1]
NORMALISESCSTOPM = False  # Normalise scattering coefficients to [-1, 1]
USE_CLASS_WEIGHTS = True  # Set to False to disable class weights

TESTONGENERATED = False  # Use generated data as testdata
FILTERED = False  # Remove outliers in training, validation and test data

# -------------------------- NEW GAN CONFIGURATION --------------------------
# Set USE_GAN to True to use a pre-trained GAN generator instead of a VAE.
USE_GAN = False          # Change to False to use VAE as before
gan_epoch = 1000           # e.g., epoch number to load from
gan_gen_loss = 'BCE'       # e.g., generator loss value (as used in filename)
gan_disc_loss = 'BCE'      # e.g., discriminator loss value (as used in filename)
latent_dim = 100
encoder = 'GAN'
# -------------------------------------------------------------------------

# Generate training data with trained VAE model (or GAN, if configured)
if not USE_GAN: 
    encoder =  ['Dual', 'CNN', 'STMLP', 'lavgSTMLP', 'ldiffSTMLP'][0]
    scatshape = (1, 128, 128)
    hidden_dim1 = 256
    hidden_dim2 = 128
    latent_dim = 64

#########################################################################################################################

if lambda_values != [0]:
    VAE_classes = galaxy_classes
    VAE_train_size = {VAE_classes[0]: 1101028, VAE_classes[1]: 1101028, VAE_classes[2]: 1101028, VAE_classes[3]: 1101028}
    forbidden_classes = [VAE_classes[3]]  # Generated bent sources look awful

if TESTONGENERATED:
    lambda_values = [8]  # To identify and distinguish TESTONGENERATED from other runs

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
########### FUNCTION TO LOAD GAN GENERATOR ############
###############################################

def load_gan_generator(path, latent_dim):
    """
    Loads a GAN generator from the given path.
    NOTE: You must have defined (or imported) your GAN generator architecture (e.g. GANGenerator)
    Replace the import and instantiation below with your actual generator model.
    """
    generator = DCGANGenerator(latent_dim=latent_dim).to(DEVICE)
    generator.load_state_dict(torch.load(path, map_location=DEVICE))
    generator.eval()
    return generator

###############################################
########### DATA STORING FUNCTIONS ###############
###############################################

# (rest of your code remains unchanged until the generation blocks)

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

def compute_scattering_coeffs(images):
    scattering = Scattering2D(J=J, L=L, shape=img_shape[1:], max_order=order)
    print("Computing scattering coefficients...")
    
    images = images.to(torch.float32)  # Ensure images are in float32
    
    with torch.no_grad():  # Disable gradient calculation
        start_time = time.time()
        scat_coeffs = scattering(images).detach()
        if scat_coeffs.dim() == 3:
            scat_coeffs = scat_coeffs.unsqueeze(0)
        scat_coeffs = torch.squeeze(scat_coeffs)
        elapsed_time = time.time() - start_time
        print(f"Time taken to compute scattering coefficients: {elapsed_time:.2f} seconds")
    
    return scat_coeffs

def custom_collate(batch):
    if len(batch) < 3:  # Drop the batch if it contains fewer than 3 images
        return None
    return torch.utils.data.dataloader.default_collate(batch)

###############################################
########### READ IN TEST DATA #################
######## Needs only be done once ##############
###############################################

if not TESTONGENERATED:
    final_test_data = load_galaxies(galaxy_class=galaxy_classes, 
                fold=max(folds), #Any fold other than 5 gives me the test data for the five fold cross validation
                img_shape=img_shape, 
                sample_size=max_num_galaxies, 
                REMOVEOUTLIERS=FILTERED,
                train=False)
    _, _, test_images, test_labels = final_test_data

    # Print the shape of the images 
    print("Shape of test_images: ", np.shape(test_images))

    perm = torch.randperm(test_images.size(0))
    test_images = test_images[perm]
    test_labels = test_labels[perm]

    # Print the distribution of raw test labels
    unique_labels, counts = torch.unique(test_labels, return_counts=True)
    print("Test labels distribution (raw):", dict(zip(unique_labels.tolist(), counts.tolist())))
    print("Test images shape: ", np.shape(test_images))
    test_scat_coeffs = compute_scattering_coeffs(test_images)

    if 'lavg' in classifier:
        test_scat_coeffs = torch.stack([lavg(coeff, J=J, L=L, m=order)[0] for coeff in test_scat_coeffs])
    elif 'ldiff' in classifier:
        test_scat_coeffs = torch.stack([ldiff(coeff, J=J, L=L, m=order)[0] for coeff in test_scat_coeffs])

    if NORMALISEIMGS:
        test_images = normalize_to_0_1(test_images)
        if NORMALISEIMGSTOPM: 
            test_images = normalize_to_minus1_1(test_images)

    # Handle scattering coefficients normalization in a similar way
    if 'MLP' in classifier: 
        if NORMALISESCS:
            test_scat_coeffs = normalize_to_0_1(test_scat_coeffs)
            if NORMALISESCSTOPM:
                test_scat_coeffs = normalize_to_minus1_1(test_scat_coeffs)

    min_label = test_labels.min()
    test_labels = test_labels - min_label
    test_labels = torch.nn.functional.one_hot(test_labels, num_classes=num_classes).float()
    test_labels = test_labels.view(-1, num_classes)

    if 'MLP' in classifier:  # Double dataset for convenience for dual model in training loop
        test_dataset = TensorDataset(test_scat_coeffs, test_labels)
    else: 
        test_dataset = TensorDataset(test_images, test_labels) 

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=True)

    print(f"Test dataset size: {len(test_dataset)}")

###############################################
########### LOOP OVER DATA FOLD ###############
###############################################            

FIRSTTIME = True  # Set to True to print model summaries only once
param_combinations = list(itertools.product(folds, learning_rates, regularization_params, lambda_values))
for fold, lr, reg, lambda_generate in param_combinations:
    torch.cuda.empty_cache()
    print(f"\n Training with fold: {fold}, learning rate: {lr}, and regularization: {reg}")
    runname = f'{galaxy_classes}_{classifier}_lr{lr}_reg{reg}'
    log_path = f"./classifier/log_{runname}.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file = open(log_path, 'w')

    # Load the data
    data = load_galaxies(galaxy_class=galaxy_classes, 
                        fold=fold,
                        img_shape=img_shape, 
                        sample_size=max_num_galaxies, 
                        REMOVEOUTLIERS=FILTERED,
                        train=True)
    train_images, train_labels, valid_images, valid_labels = data

    print("Train images shape: ", np.shape(train_images))
    print("Validation images shape: ", np.shape(valid_images))

    print("Input for torch.randperm: ", train_images.size(0))
    perm = torch.randperm(train_images.size(0))
    train_images = train_images[perm]
    train_labels = train_labels[perm]
    perm = torch.randperm(valid_images.size(0))
    valid_images = valid_images[perm]
    valid_labels = valid_labels[perm]
            
    num_galaxies = len(train_images)
    print("New number of galaxies is the same as the training size: ", num_galaxies)
    dataset_sizes[fold] = [int(num_galaxies * perc) for perc in dataset_portions]
    print(f"Dataset_sizes in fold f{fold}:  {dataset_sizes}")

    if TESTONGENERATED:  # Use generated data for testing
        all_test_images = []
        all_test_labels = []
        for cls in VAE_classes:
            try:
                if USE_GAN:
                    # Use GAN generator filename convention
                    path = f"./GAN/generator_{cls}_epoch_{gan_epoch}_gen-{gan_gen_loss}_disc-{gan_disc_loss}.pth"
                    model = load_gan_generator(path, latent_dim=latent_dim)
                else:
                    path = get_model_name(cls, VAE_train_size[cls], encoder, fold=fold)
                    model = load_model(path, scatshape=scatshape,
                                hidden_dim1=hidden_dim1,
                                hidden_dim2=hidden_dim2,
                                latent_dim=latent_dim,
                                num_classes=num_classes)
                
                # Optionally, set the model to evaluation mode and disable gradients
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
                            train_labels=generated_labels,
                            latent_dim=latent_dim,
                            num_samples=current_batch,
                            DEVICE='cpu'
                        )
                    generated_batch = generated_batch.to(train_images.device)
                    generated_labels = generated_labels.to(train_labels.device)
                    generated_images_list.append(generated_batch)
                    generated_labels_list.append(generated_labels)
                # Append data for this class
                all_test_images.append(torch.cat(generated_images_list, dim=0))
                all_test_labels.append(torch.cat(generated_labels_list, dim=0))
            except Exception as e:
                print(f"Model error for class: {cls}. Error: {e}")
                continue

        test_images = torch.cat(all_test_images, dim=0)
        test_labels = torch.cat(all_test_labels, dim=0)

        print("Shape of test_images: ", test_images.shape)
        perm = torch.randperm(test_images.size(0))
        test_images = test_images[perm]
        test_labels = test_labels[perm]

        test_scat_coeffs = compute_scattering_coeffs(test_images)

        if 'lavg' in classifier:
            test_scat_coeffs = torch.stack([lavg(coeff, J=J, L=L, m=order)[0] for coeff in test_scat_coeffs])
        elif 'ldiff' in classifier:
            test_scat_coeffs = torch.stack([ldiff(coeff, J=J, L=L, m=order)[0] for coeff in test_scat_coeffs])

        # Normalize train and test images to [0, 1]
        if NORMALISEIMGS:
            test_images = normalize_to_0_1(test_images)

        if NORMALISEIMGSTOPM:  # normalize to [-1, 1]
            test_images = normalize_to_minus1_1(test_images)

        # Handle scattering coefficients normalization in a similar way
        if 'MLP' in classifier:
            if NORMALISESCS:
                test_scat_coeffs = normalize_to_0_1(test_scat_coeffs)
                if NORMALISESCSTOPM:
                    test_scat_coeffs = normalize_to_minus1_1(test_scat_coeffs)

        ###### RELABEL ######
        min_label = test_labels.min()
        test_labels = test_labels - min_label
        test_labels = torch.nn.functional.one_hot(test_labels, num_classes=num_classes).float()
        test_labels = test_labels.view(-1, num_classes)

        if 'MLP' in classifier:
            test_dataset = TensorDataset(test_scat_coeffs, test_labels)
        else: 
            test_dataset = TensorDataset(test_images, test_labels) 

        def custom_collate(batch):
            if len(batch) < 3:
                return None
            return torch.utils.data.dataloader.default_collate(batch)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=True)

        print(f"Test dataset size: {len(test_dataset)}")
    
    if USE_CLASS_WEIGHTS:
        unique, counts = np.unique(train_labels, return_counts=True)
        class_counts = dict(zip(map(int, unique), map(int, counts)))
        total_count = sum(counts)
        class_weights = {int(cls): float(total_count / count) for cls, count in class_counts.items()}
        normalized_classes = [cls - min(galaxy_classes) for cls in galaxy_classes]
        weights = torch.tensor([class_weights.get(cls, 1.0) for cls in normalized_classes], dtype=torch.float).to(DEVICE)
        missing_classes = [cls for cls in normalized_classes if cls not in class_weights]
        if missing_classes:
            print(f"Warning: Missing classes in dataset: {missing_classes}")
            class_weights.update({int(cls): 1.0 for cls in missing_classes})
        unique_valid, counts_valid = np.unique(valid_labels, return_counts=True)
        class_counts_valid = dict(zip(map(int, unique_valid), map(int, counts_valid)))
        unique_test, counts_test = np.unique(test_labels, return_counts=True)
        class_counts_test = dict(zip(map(int, unique_test), map(int, counts_test)))
    else:
        weights = None
        print("No class weighting")
        print("Classes used: ", galaxy_classes)

    if fold == 0:
        plot_images_by_class(train_images, train_labels, num_images=5, save_path=f"./classifier/{classifier}_{galaxy_classes}_{num_galaxies}_example_inputs.png")

    # Prepare input data
    if 'MLP' in classifier:
        train_scat_coeffs = compute_scattering_coeffs(train_images)
        valid_scat_coeffs = compute_scattering_coeffs(valid_images)
        
        if 'lavg' in classifier:
            train_scat_coeffs = torch.stack([lavg(coeff, J=J, L=L, m=order)[0] for coeff in train_scat_coeffs])
            valid_scat_coeffs = torch.stack([lavg(coeff, J=J, L=L, m=order)[0] for coeff in valid_scat_coeffs])
        elif 'ldiff' in classifier:
            train_scat_coeffs = torch.stack([ldiff(coeff, J=J, L=L, m=order)[0] for coeff in train_scat_coeffs])
            valid_scat_coeffs = torch.stack([ldiff(coeff, J=J, L=L, m=order)[0] for coeff in valid_scat_coeffs])
            
        scatdim = train_scat_coeffs[1:].shape

    ##########################################################
    ############ NORMALISE AND FILTER THE INPUT ##############
    ##########################################################
                
    if NORMALISEIMGS:
        train_images = normalize_to_0_1(train_images)
        valid_images = normalize_to_0_1(valid_images)

    if NORMALISEIMGSTOPM:
        train_images = normalize_to_minus1_1(train_images)
        valid_images = normalize_to_minus1_1(valid_images)

    if 'MLP' in classifier:
        if NORMALISESCS:
            train_scat_coeffs = normalize_to_0_1(train_scat_coeffs)
            valid_scat_coeffs = normalize_to_0_1(valid_scat_coeffs)
            if NORMALISESCSTOPM:
                train_scat_coeffs = normalize_to_minus1_1(train_scat_coeffs)
                valid_scat_coeffs = normalize_to_minus1_1(valid_scat_coeffs)

    if IMGCHECK: 
        plot_images(train_images, train_labels)
        plot_histograms(train_images, valid_images)

    # Generate more training data if requested
    if lambda_generate not in [0, 8]:
        num_generate = int(lambda_generate / len(galaxy_classes) * len(train_images))
        print("Number to generate: ", num_generate)
        print("Old training data size: ", train_images.size())
        for cls in VAE_classes:
            if cls in forbidden_classes:
                continue
            try:
                if USE_GAN:
                    path = f"generator_{cls}_epoch_{gan_epoch}_gen-{gan_gen_loss}_disc-{gan_disc_loss}.pth"
                    model = load_gan_generator(path, latent_dim=latent_dim)
                else:
                    path = get_model_name(cls, VAE_train_size[cls], encoder, fold=fold)
                    model = load_model(path, scatshape=scatshape,
                                hidden_dim1=hidden_dim1,
                                hidden_dim2=hidden_dim2,
                                latent_dim=latent_dim,
                                num_classes=num_classes)
                
                model.eval()
                batch_size_generate = 500 
                num_batches = (num_generate + batch_size_generate - 1) // batch_size_generate
                
                generated_images_list, generated_labels_list = [], []                
                for batch_idx in range(num_batches):
                    current_batch = min(batch_size_generate, num_generate - batch_idx * batch_size_generate)
                    generated_labels = torch.ones(current_batch, dtype=torch.long) * (cls - min(galaxy_classes))
                    with torch.no_grad():
                        generated_batch = generate_from_noise(
                            model,
                            train_labels=generated_labels,
                            latent_dim=latent_dim,
                            num_samples=current_batch,
                            DEVICE='cpu'
                        )
                    generated_batch = generated_batch.to(train_images.device)
                    generated_labels = generated_labels.to(train_labels.device)
                    generated_images_list.append(generated_batch)
                    generated_labels_list.append(generated_labels)
                
                generated_images = torch.cat(generated_images_list, dim=0)
                generated_labels = torch.cat(generated_labels_list, dim=0)
                
                print(f"Generated {generated_images.size(0)} images for class {cls}.")
                
                train_images = torch.cat([train_images, generated_images])
                train_labels = torch.cat([train_labels.clone().detach(), generated_labels])
                
                print("New training data size: ", train_images.size())
            
            except Exception as e:
                print(f"Model not found or generation error for class: {cls}. Error: {e}")
                continue

    ###### RELABEL ######
    min_label = train_labels.min()
    train_labels = train_labels - min_label
    valid_labels = valid_labels - min_label

    train_labels = torch.nn.functional.one_hot(train_labels, num_classes=num_classes).float()
    valid_labels = torch.nn.functional.one_hot(valid_labels, num_classes=num_classes).float()

    train_labels = train_labels.view(-1, num_classes)
    valid_labels = valid_labels.view(-1, num_classes)

    if 'MLP' in classifier:
        train_dataset = TensorDataset(train_scat_coeffs, train_labels)
        valid_dataset = TensorDataset(valid_scat_coeffs, valid_labels)
    else: 
        train_dataset = TensorDataset(train_images, train_labels) 
        valid_dataset = TensorDataset(valid_images, valid_labels) 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=True)
    
    print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(valid_dataset)}")
    print("Testongenerated: ", TESTONGENERATED)
            
    ###############################################
    ############# DEFINE MODEL ####################
    ###############################################

    if classifier == "scatterMLP":
        models = {"scatterMLP": {"model": Scattering_Classifier(input_channels=scatdim[-3], num_classes=num_classes, J=J).to(DEVICE)}}
    elif classifier == "normalCNN":
        models = {"normalCNN": {"model": CNN_Classifier(input_shape=tuple(valid_images.shape[1:]) , num_classes=num_classes).to(DEVICE)}}
    elif classifier == "simpleCNN":
        models = {"simpleCNN": {"model": SimpleCNN(input_shape=tuple(valid_images.shape[1:]), num_classes=num_classes).to(DEVICE)}}
    elif classifier == "simpleFCN":
        models = {"simpleFCN": {"model": SimpleFCN(input_shape=tuple(valid_images.shape[1:]), num_classes=num_classes).to(DEVICE)}} 
    elif classifier == "ProjectModel":
        models = {"ProjectModel": {"model": ProjectModel(input_shape=tuple(valid_images.shape[1:]), num_classes=num_classes).to(DEVICE)}} 
    else:
        raise ValueError("Model not found. Please select one of 'scatterMLP', 'smallSTMLP', or 'normalCNN'.")

    for model_name, model_details in models.items():
        if FIRSTTIME:
            print(f"Summary for {model_name}:")
            if model_name in ["scatterMLP", "smallSTMLP"]:
                summary(model_details["model"], input_size=train_scat_coeffs[0].shape, device=DEVICE)
            else:
                summary(model_details["model"], input_size=img_shape, device=DEVICE)
        FIRSTTIME = False

    ###############################################
    ############### TRAINING LOOP #################
    ###############################################
    
    if weights is not None:
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(models[model_name]["model"].parameters(), lr=lr, weight_decay=reg)

    for model_name, model_details in models.items():
        print(f"Training {model_name} model...")
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
                initialize_history(history, model_name, subset_size, fold, experiment, lr, reg, lambda_generate)
                initialize_metrics(metrics, model_name, subset_size, fold, experiment, lr, reg, lambda_generate)
                initialize_labels(all_true_labels, all_pred_labels, model_name, subset_size, fold, experiment, lr, reg, lambda_generate)

                start_time = time.time()
                model.apply(reset_weights)

                subset_indices = list(range(subset_size))
                subset_train_dataset = Subset(train_dataset, subset_indices)
                subset_train_loader = DataLoader(subset_train_dataset, batch_size=batch_size, shuffle=True)

                early_stopping = EarlyStopping(patience=patience, verbose=False) if ES else None

                for epoch in tqdm(range(num_epochs), desc=f'Training {galaxy_classes}_{model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}'):
                    model.train()
                    total_loss = 0
                    total_images = 0

                    for images, labels in subset_train_loader:
                        images, labels = images.to(DEVICE), labels.to(DEVICE)
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        total_loss += float(loss.item() * images.size(0))
                        total_images += float(images.size(0))

                    average_loss = total_loss / total_images
                    loss_key = f"{model_name}_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                    history[model_name][loss_key].append(average_loss)

                    model.eval()
                    val_total_loss = 0
                    val_total_images = 0

                    with torch.no_grad(): # Validate on validation data
                        for i, (images, labels) in enumerate(valid_loader):
                            if images is None or len(images) == 0:
                                print(f"Empty batch at index {i}. Skipping...")
                                continue
                            images, labels = images.to(DEVICE), labels.to(DEVICE)
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                            val_total_loss += float(loss.item() * images.size(0))
                            val_total_images += float(images.size(0))

                    val_average_loss = val_total_loss / val_total_images if val_total_images > 0 else float('inf')
                    val_loss_key = f"{model_name}_val_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                    history[model_name][val_loss_key].append(val_average_loss)
                    
                    if ES:
                        early_stopping(val_average_loss, model, f'./classifier/trained_models/{model_name}_best_model.pth')
                        if early_stopping.early_stop:
                            break

                model.eval()
                with torch.no_grad(): # Evaluate on test data
                    key = f"{model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                    all_pred_probs[key] = []
                    all_pred_labels[key] = []
                    all_true_labels[key] = []

                    for images, labels in test_loader:
                        images, labels = images.to(DEVICE), labels.to(DEVICE)
                        outputs = model(images)
                        pred_probs = torch.sigmoid(outputs).cpu().numpy()
                        true_labels = torch.argmax(labels, dim=1).cpu().numpy()
                        pred_labels = np.argmax(pred_probs, axis=1)
                        all_pred_probs[key].extend(pred_probs)
                        all_pred_labels[key].extend(pred_labels)
                        all_true_labels[key].extend(true_labels)

                    accuracy = accuracy_score(all_true_labels[key], all_pred_labels[key])
                    precision = precision_score(all_true_labels[key], all_pred_labels[key], average='macro', zero_division=0)
                    recall = recall_score(all_true_labels[key], all_pred_labels[key], average='macro', zero_division=0)
                    f1 = f1_score(all_true_labels[key], all_pred_labels[key], average='macro', zero_division=0)

                    update_metrics(metrics, model_name, subset_size, fold, experiment, lr, reg, accuracy, precision, recall, f1, lambda_generate)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken to train the model for fold {fold}: {elapsed_time:.2f} seconds")
            training_times[subset_size][fold].append(elapsed_time)
            file.write(f"Time taken to train the model on fold {fold}: {elapsed_time:.2f} seconds \n")

        generated_features = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(DEVICE)
                features = model(images).cpu().detach().numpy()
                generated_features.append(features)

        generated_features = np.concatenate(generated_features, axis=0)
        cluster_error, cluster_distance, cluster_std_dev = cluster_metrics(generated_features, n_clusters=13)
        print(f"\n Cluster Error: {cluster_error}")
        print(f"Cluster Distance: {cluster_distance}")
        print(f"Cluster Standard Deviation: {cluster_std_dev}")

        model_save_path = f'./classifier/trained_models/{model_name}_model.pth'
        torch.save(model.state_dict(), model_save_path)
        
directory = './classifier/trained_models_filtered/' if FILTERED else './classifier/trained_models/'

for fold, lr, reg, lambda_generate in param_combinations:
    for subset_size in dataset_sizes[fold]:
        for experiment in range(num_experiments):
            metrics_save_path = f'{directory}{galaxy_classes}_{encoder}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}_metrics_data.pkl'
            with open(metrics_save_path, 'wb') as f:
                pickle.dump({
                    "classifier": classifier,
                    "models": models,
                    "model_name": model_name,
                    "history": history,
                    "metrics": metrics,
                    "metric_colors": metric_colors,
                    "all_true_labels": all_true_labels,
                    "all_pred_labels": all_pred_labels,
                    "training_times": training_times,
                    "all_pred_probs": all_pred_probs
                }, f)
            print(f"Metrics and related data saved to {metrics_save_path}")
