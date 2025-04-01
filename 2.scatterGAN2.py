import os
import numpy as np
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid, save_image
from utils.data_loader import load_galaxies
from utils.calc_tools import generate_from_noise
from utils.GAN_models import DCGANDiscriminator, DCGANGenerator, GANDiscriminator, AdvancedGenerator, load_gan_generator
from utils.plotting import plot_GAN_losses, plot_histograms
from utils.GAN_loss_functions import (
    d_loss_bce, d_loss_hinge, d_loss_lsgan, d_loss_mse, d_loss_wgan,
    g_loss_bce, g_loss_hinge, g_loss_lsgan, g_loss_mse, g_loss_wgan
)
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

# -------------------------------
# Fixed parameters for hyperparameter optimization
# -------------------------------
# For hyperparameter optimization we fix the dataset parameters
REMOVEOUTLIERS = True # Sets dv to 'clean' if True and 'full' otherwise

galaxy_class = 13
folds = [0, 1, 2, 3, 4]
batch_size = 256
epochs = 500
image_size = 128
save_interval = 100
output_dir = "GAN"
gan_type = ["Simple", "Advanced"][0]

# Grid of hyperparameters to search over
grid_latent_dim = [128]
grid_lr_gen = [1e-3]       # Separate learning rate for generator
grid_lr_disc = [1e-4]      # Separate learning rate for discriminator (classifier)
grid_gen_loss = ["MSE"]
grid_disc_loss = ["BCE"]    # Discriminator loss
grid_adam_betas = [(0.5, 0.999)]     # Beta values for Adam optimizer
grid_weight_decay = [0.0]           # Weight decay options
grid_label_smoothing = [0.7]        # Label smoothing values for BCE loss
grid_lambda_div = [0.5]             # Weight for diversity loss

#######################

# Set random seed for reproducibility
SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)

if REMOVEOUTLIERS:
    print("Removing outliers. Dataset will be cleaned")
    data_version='clean'
else:
    data_version='full'

print("Running ScatterGAN with gan type", gan_type)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "generators"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "discriminators"), exist_ok=True)


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

fixed_noise = torch.randn(36, grid_latent_dim[0], 1, 1, device=device)

def change_images(images):

    # Apply randome flipping 
    if np.random.rand() > 0.5:
        images = images.flip(3) # Flip horizontally
    if np.random.rand() > 0.5:
        images = images.flip(2) # Flip vertically
    
    # Apply random rotation 90, 180, 270 degrees
    if np.random.rand() > 0.5:
        images = torch.rot90(images, 1, [2, 3]) # Rotate 90 degrees
    if np.random.rand() > 0.5:
        images = torch.rot90(images, 2, [2, 3]) # Rotate 180 degrees
    if np.random.rand() > 0.5:
        images = torch.rot90(images, 3, [2, 3]) # Rotate 270 degrees
    
    return images

def save_images_tensorboard(images, file_path, nrow=4):
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(grid, file_path)

def run_experiment(latent_dim, fold, lr_gen, lr_disc, gen_loss, disc_loss, adam_betas, weight_decay, label_smoothing, lambda_div):
    # Create a unique experiment name from the hyperparameters
    exp_name = f"latent{latent_dim}_cl{galaxy_class}_lrgen{lr_gen}_lrdisc{lr_disc}_gl{gen_loss}_dl{disc_loss}_ab{adam_betas}_wd{weight_decay}_ls{label_smoothing}_ld{lambda_div}_dv{data_version}"
    exp_dir = os.path.join(output_dir, gan_type, f"{exp_name}_f{folds[-1]}")
    os.makedirs(exp_dir, exist_ok=True)
    logging.info(f"Starting experiment: {exp_name}")

    # Load data for the fixed galaxy class and fold
    data = load_galaxies(
        galaxy_class=galaxy_class,
        fold=fold,
        img_shape=(1, image_size, image_size),
        sample_size=1000,
        REMOVEOUTLIERS=REMOVEOUTLIERS,
        train=True
    )
    train_images, _, _, _ = data
    
    # Shuffle the data
    perm = torch.randperm(len(train_images))
    train_images = train_images[perm].float()
    
    # Normalise to [-1, 1]
    train_images = 2 * (train_images - train_images.min()) / (train_images.max() - train_images.min()) - 1
    dataset = TensorDataset(train_images)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    if gan_type == 'Simple':
        generator = DCGANGenerator(latent_dim=latent_dim, ngf=64, nc=1).to(device)
        discriminator = DCGANDiscriminator(ndf=64, nc=1).to(device)
    elif gan_type == 'Advanced':
        generator = AdvancedGenerator(latent_dim=latent_dim, ngf=64, nc=1).to(device)
        discriminator = GANDiscriminator(ndf=64, nc=1).to(device)
    else:
        print("No chosen gan_type")
        
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_gen, betas=adam_betas, weight_decay=weight_decay)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_disc, betas=adam_betas, weight_decay=weight_decay)

    # Choose loss functions
    real_label_val = label_smoothing
    fake_label_val = 0.0

    if disc_loss == "BCE":
        disc_loss_fn = lambda real, fake: d_loss_bce(real, fake, real_label_val, fake_label_val)
    elif disc_loss == "MSE":
        disc_loss_fn = d_loss_mse
    elif disc_loss == "LSGAN":
        disc_loss_fn = d_loss_lsgan
    elif disc_loss == "HINGE":
        disc_loss_fn = d_loss_hinge
    elif disc_loss == "WGAN":
        disc_loss_fn = d_loss_wgan
    else:
        raise ValueError(f"Unknown discriminator loss type: {disc_loss}")

    if gen_loss == "BCE":
        gen_loss_fn = lambda fake: g_loss_bce(fake, real_label_val)
    elif gen_loss == "MSE":
        gen_loss_fn = g_loss_mse
    elif gen_loss == "LSGAN":
        gen_loss_fn = g_loss_lsgan
    elif gen_loss == "HINGE":
        gen_loss_fn = g_loss_hinge
    elif gen_loss == "WGAN":
        gen_loss_fn = g_loss_wgan
    else:
        raise ValueError(f"Unknown generator loss type: {gen_loss}")

    # Save example real images for reference (using 36 images arranged in 6x6 grid)
    real_examples = train_images[:36]
    save_images_tensorboard(real_examples, os.path.join(exp_dir, "ScatterGAN_example_real_images.png"), nrow=6)

    # Initialize loss tracking
    g_loss_history, d_loss_history = [], []
    log_interval = max(1, epochs // 10)
    with logging_redirect_tqdm():
        for epoch in tqdm(range(1, epochs + 1), miniters=log_interval):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                real_batch = batch[0].to(device)
                b_size = real_batch.size(0)

                # Train Discriminator / Critic
                optimizer_D.zero_grad()
                real_pred = discriminator(real_batch)
                z = torch.randn(b_size, latent_dim, 1, 1, device=device)                
                fake_images = generator(z)
                fake_images = change_images(fake_images) # Apply random transformations
                fake_flat = fake_images.reshape(b_size, -1)
                pairwise_dists = torch.pdist(fake_flat, p=2)
                diversity_loss = -torch.mean(pairwise_dists)

                fake_pred = discriminator(fake_images.detach())
                d_loss = disc_loss_fn(real_pred, fake_pred)
                d_loss.backward()
                optimizer_D.step()

                # Train Generator
                optimizer_G.zero_grad()
                fake_pred_for_g = discriminator(fake_images)
                g_loss = gen_loss_fn(fake_pred_for_g) + lambda_div * diversity_loss
                g_loss.backward()
                optimizer_G.step()

                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
                num_batches += 1

            avg_d_loss = epoch_d_loss / num_batches
            avg_g_loss = epoch_g_loss / num_batches
            d_loss_history.append(avg_d_loss)
            g_loss_history.append(avg_g_loss)

            if epoch % log_interval == 0 or epoch == epochs:
                logging.info(f"Exp {exp_name} | Epoch [{epoch}/{epochs}] | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")

            if epoch % save_interval == 0 or epoch == epochs:
                generator.eval()
                with torch.no_grad():
                    test_z = torch.randn(16, latent_dim, 1, 1, device=device)
                    gen_sample = generator(test_z).cpu()
                generator.train()

                img_file = os.path.join(exp_dir, f"ScatterGAN_generated_epoch_{epoch}.png")
                save_images_tensorboard(gen_sample, img_file, nrow=4)
                logging.info(f"Saved generated images to {img_file}")

                if len(grid_adam_betas) == 1: # Save model checkpoints only if there is a single beta value. Otherwise, it's not useful.
                    torch.save(generator.state_dict(), f"./GAN/generators/generator_{gan_type}_{exp_name}_ep{epoch}_f{fold}.pth")
                    torch.save(discriminator.state_dict(), f"./GAN/discriminators/discriminator_{gan_type}_{exp_name}_ep{epoch}_f{fold}.pth")
                logging.info(f"Saved model checkpoints for epoch {epoch}")

    plot_GAN_losses(exp_dir, galaxy_class, g_loss_history, d_loss_history, gen_loss, disc_loss)
    logging.info(f"Training complete for {exp_name}. Loss plot saved.")

    generator.eval()
    #fixed_noise = torch.randn(36, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        rep_generated = generator(fixed_noise).cpu()
    rep_img_path = os.path.join(exp_dir, "rep_generated.png")
    save_images_tensorboard(rep_generated, rep_img_path, nrow=6)
    generator.train()

    return {
        "exp_name": exp_name,
        "exp_dir": exp_dir,
        "rep_image_path": rep_img_path,
        "final_generator_loss": g_loss_history[-1],
        "latent_dim": latent_dim,
        "lr_gen": lr_gen,
        "lr_disc": lr_disc,
        "gen_loss": gen_loss,
        "disc_loss": disc_loss
    }

results = []
for latent_dim, fold, lr_gen, lr_disc, gen_loss, disc_loss, adam_betas, weight_decay, label_smoothing, lambda_div in itertools.product(
    grid_latent_dim, folds, grid_lr_gen, grid_lr_disc, grid_gen_loss, grid_disc_loss, grid_adam_betas, grid_weight_decay, grid_label_smoothing, grid_lambda_div
):
    res = run_experiment(latent_dim, fold, lr_gen, lr_disc, gen_loss, disc_loss, adam_betas, weight_decay, label_smoothing, lambda_div)
    results.append(res)

n_experiments = len(results)
cols = min(6, n_experiments)
rows = math.ceil(n_experiments / cols)

fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
axs = axs.flatten() if n_experiments > 1 else [axs]

for ax in axs[n_experiments:]:
    fig.delaxes(ax)

for idx, res in enumerate(results):
    img = mpimg.imread(res["rep_image_path"])
    ax = axs[idx]
    ax.imshow(img)
    ax.axis('off')
    title = f"{res['exp_name']}\nG_loss: {res['final_generator_loss']:.4f}"
    ax.set_title(title, fontsize=8)

fig.suptitle("Hyperparameter Optimization Summary", fontsize=16)
plt.tight_layout()
summary_path = os.path.join(output_dir, "hyperparam_summary.png")
plt.savefig(summary_path)
plt.close()
logging.info(f"Summary grid saved to {summary_path}")

# To double check the saving, read in the generator and plot some images
exp_dir = results[0]["exp_dir"]

gan_path = f"./GAN/generators/generator_{gan_type}_latent{latent_dim}_cl{11}_lrgen{lr_gen}_lrdisc{lr_disc}_gl{gen_loss}_dl{disc_loss}_ab{adam_betas}_wd{weight_decay}_ls{label_smoothing}_ld{lambda_div}_dv{data_version}_ep{500}_f{fold}.pth"
model = DCGANGenerator(latent_dim=latent_dim, ngf=64, nc=1).to(device)
model.load_state_dict(torch.load(gan_path))
model.eval()
with torch.no_grad():
    #fixed_noise = torch.randn(36, latent_dim, 1, 1, device=device)
    rep_generated = model(fixed_noise).cpu()
rep_img_path = os.path.join(exp_dir, "rep_generated2.png")
save_images_tensorboard(rep_generated, rep_img_path, nrow=6)

model = load_gan_generator(gan_path, latent_dim=latent_dim).to(device)
model.eval()
with torch.no_grad():
    #fixed_noise = torch.randn(36, latent_dim, 1, 1, device=device)
    rep_generated = model(fixed_noise).cpu()
rep_img_path = os.path.join(exp_dir, "rep_generated3.png")
save_images_tensorboard(rep_generated, rep_img_path, nrow=6)

model.eval()
batch_array = [66]*5 # This creates 5 batches of 5000 images
noise_generated_images = []
for current_batch_size in batch_array:
    with torch.no_grad():
        generated_batch = generate_from_noise(
            model,
            latent_dim=latent_dim,
            num_samples=current_batch_size,
            DEVICE=device)
        save_path = os.path.join(exp_dir, f"rep_generated4_{current_batch_size}.png")
        save_images_tensorboard(generated_batch[:36], save_path, nrow=6)
        
    noise_generated_images.append(generated_batch)

    noise_generated_images.append(generated_batch)
noise_generated_images_tensor = torch.cat(noise_generated_images, dim=0)
train_images, _, _, _  = load_galaxies(galaxy_class=galaxy_class, fold=fold, img_shape=(1, image_size, image_size), sample_size=1000, REMOVEOUTLIERS=REMOVEOUTLIERS, AUGMENT=False, train=True)
train_images = 2 * (train_images - train_images.min()) / (train_images.max() - train_images.min()) - 1 # Normalise to [-1, 1]
plot_histograms(noise_generated_images_tensor, rep_generated, title1='Generated from noise function', title2='Generated directly', imgs3=train_images, title3='Original images', save_path=os.path.join(exp_dir, "rep_generated4_histograms.png"))

from lpips import LPIPS
import random

def compute_lpips_diversity(images, model_name='alex'):
    loss_fn = LPIPS(net=model_name).to(device)
    indices = random.sample(range(len(images)), k=min(100, len(images)))  # sample 100
    pairs = [(i, j) for i in indices for j in indices if i < j]
    total_score = 0.0
    for i, j in pairs[:200]:  # max 200 pairs for speed
        d = loss_fn(images[i].unsqueeze(0).to(device), images[j].unsqueeze(0).to(device))
        total_score += d.item()
    return total_score / len(pairs[:200])

lpips_score = compute_lpips_diversity(noise_generated_images_tensor)
print(f"LPIPS diversity score for generated images: {lpips_score:.4f} from {len(noise_generated_images_tensor)} images")
lpips_score_original = compute_lpips_diversity(train_images)
print(f"LPIPS diversity score for original images: {lpips_score_original:.4f} from {len(train_images)} images")

def plot_similar_images_grid(generator, latent_dim, num_images=100, nrow=10, device=device, save_path="similar_images_grid.png"):
    """
    Generates many images from the generator, computes pairwise similarities (using Euclidean distance)
    between flattened images, reorders the images via hierarchical clustering so that similar images are adjacent,
    and then plots and saves a grid of these similar images.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list  # Import clustering functions

    # Generate images
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
        images = generator(noise).cpu()  # shape: (num_images, channels, H, W)

    # Flatten each image to compute distances
    flattened = images.view(num_images, -1).numpy()
    
    # Perform hierarchical clustering on the flattened images
    Z = linkage(flattened, method='average', metric='euclidean')
    order = leaves_list(Z)
    
    # Reorder images based on the clustering order
    ordered_images = images[order]
    
    # Create a grid of ordered images
    grid = make_grid(ordered_images, nrow=nrow, normalize=True, value_range=(-1, 1))
    
    # Plot and save the grid
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.title("Grid of Most Similar Images")
    plt.savefig(save_path)
    plt.close()
    
# Example usage:
# Assuming 'model' is your trained generator and 'latent_dim' is defined from the experiment
model = model.to(device)
plot_similar_images_grid(model, latent_dim, num_images=100, nrow=10, device=device,
                         save_path=os.path.join(exp_dir, "similar_images_grid.png"))


def plot_original_similarity_grid(images, num_images=100, nrow=10, save_path="original_similarity_grid.png"):
    """
    Computes self-similarity among the original images and arranges them in a grid
    via hierarchical clustering so that similar images are adjacent.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list

    # Use at most the available number of images
    num_images = min(num_images, images.shape[0])
    selected_images = images[:num_images]

    # Flatten each image to compute distances
    flattened = selected_images.view(num_images, -1).numpy()
    
    # Perform hierarchical clustering on the flattened images
    Z = linkage(flattened, method='average', metric='euclidean')
    order = leaves_list(Z)
    
    # Reorder images based on clustering order
    ordered_images = selected_images[order]
    
    # Create a grid of ordered images
    grid = make_grid(ordered_images, nrow=nrow, normalize=True, value_range=(-1, 1))
    
    # Plot and save the grid
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.title("Original Images Ordered by Self-Similarity")
    plt.savefig(save_path)
    plt.close()


plot_original_similarity_grid(train_images, num_images=100, nrow=10,
                                save_path=os.path.join(exp_dir, "original_similarity_grid.png"))
