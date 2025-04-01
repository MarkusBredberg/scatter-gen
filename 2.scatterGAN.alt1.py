import os
import numpy as np
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid, save_image
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

from kymatio.torch import Scattering2D
from utils.data_loader import get_classes, load_galaxies
from utils.GAN_models import DCGANDiscriminator, DCGANGenerator, GANDiscriminator, GANGenerator, AdvancedGenerator
# -- We will REPLACE the old MLPGenerator with the fixed version below --
# from utils.GAN_models import MLPGenerator
from utils.plotting import plot_GAN_losses
from utils.GAN_loss_functions import (
    d_loss_bce, d_loss_hinge, d_loss_lsgan, d_loss_mse, d_loss_wgan,
    g_loss_bce, g_loss_hinge, g_loss_lsgan, g_loss_mse, g_loss_wgan
)

# -------------------------------
# Fixed parameters for hyperparameter optimization
# -------------------------------
J, L = 3, 8
galaxy_class = 11
folds = [5]
batch_size = 64
epochs = 300
image_size = 128
save_interval = 100
output_dir = "GAN"

# Pick one of: "Simple", "MLP", "Advanced"
gan_type = ["Simple", "MLP", "Advanced"][0]

# Grid of hyperparameters to search over
grid_latent_dim = [128]
grid_lr_gen = [1e-3]       # Separate learning rate for generator
grid_lr_disc = [1e-3]      # Separate learning rate for discriminator
grid_gen_loss = ["MSE"]
grid_disc_loss = ["BCE"]
grid_adam_betas = [(0.5, 0.999)]
grid_weight_decay = [0.0]
grid_label_smoothing = [0.9]

USE_SCATTERING_TRANSFORM = True

print("Running ScatterGAN with Scattering and gan_type", gan_type)

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "generators"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "discriminators"), exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# -------------------------------
# Modified MLP Discriminator accepting hidden_dim
# -------------------------------
class MLPDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=120):
        super(MLPDiscriminator, self).__init__()
        # Compute second layer dimension such that when hidden_dim=120, fc2 becomes 84.
        fc2_dim = int(hidden_dim * 0.7)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, 1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


# -------------------------------
# FIXED MLP Generator
# -------------------------------
class MLPGenerator(nn.Module):
    """
    MLP that outputs a 128×128 image (single channel) from a latent_dim input.
    """
    def __init__(self, latent_dim, image_size=128, hidden_dim=512):
        super(MLPGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Flattened output dimension: 1 * image_size * image_size
        output_dim = 1 * image_size * image_size

        self.main = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        # z might come in as (batch_size, latent_dim, 1, 1) so flatten it:
        z = z.view(z.size(0), -1)  # (batch_size, latent_dim)
        out = self.main(z)         # (batch_size, 1*128*128)
        # Reshape to (batch_size, 1, 128, 128)
        out = out.view(z.size(0), 1, self.image_size, self.image_size)
        return out

# -------------------------------
# Utility to save images in a grid
# -------------------------------
def save_images_tensorboard(images, file_path, nrow=4):
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(grid, file_path)

def run_experiment(latent_dim, fold, lr_gen, lr_disc, gen_loss, disc_loss, adam_betas, weight_decay, label_smoothing):
    exp_name = (
        f"scatter_latent{latent_dim}_lrgen{lr_gen}_lrdisc{lr_disc}_"
        f"gl{gen_loss}_dl{disc_loss}_ab{adam_betas}_wd{weight_decay}_ls{label_smoothing}"
    )
    exp_dir = os.path.join(output_dir, gan_type, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logging.info(f"Starting experiment: {exp_name}")

    # Load data
    data = load_galaxies(
        galaxy_class=galaxy_class,
        fold=fold,
        img_shape=(1, image_size, image_size),
        sample_size=1000,
        REMOVEOUTLIERS=False,
        train=True
    )
    train_images, _, _, _ = data
    perm = torch.randperm(len(train_images))
    train_images = train_images[perm].float()
    # Normalize to [-1, 1]
    train_images = 2 * (train_images - train_images.min()) / (train_images.max() - train_images.min()) - 1
    dataset = TensorDataset(train_images)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
    if USE_SCATTERING_TRANSFORM:
        scattering = Scattering2D(J=J, L=L, shape=(image_size, image_size)).to(device)
        # Determine the output dimensionality of scattering features
        dummy = torch.randn(1, 1, image_size, image_size, device=device)
        dummy_scattering = scattering(dummy)
        if dummy_scattering.ndim == 5:
            N, C, M, H, W = dummy_scattering.shape
            dummy_scattering = dummy_scattering.view(N, C * M, H, W)
        dummy_flat = dummy_scattering.view(dummy_scattering.size(0), -1)
        input_dim = dummy_flat.size(1)
        logging.info(f"Scattering feature dimension: {input_dim}")

        discriminator = MLPDiscriminator(input_dim=input_dim, hidden_dim=256).to(device)

        # Initialize generator
        if gan_type == 'Simple':
            generator = DCGANGenerator(latent_dim=latent_dim, ngf=64, nc=1).to(device)
        elif gan_type == 'Advanced':
            generator = GANGenerator(latent_dim=latent_dim, ngf=64, nc=1).to(device)
        elif gan_type == 'MLP':
            # IMPORTANT: use the fixed MLPGenerator
            generator = MLPGenerator(latent_dim=latent_dim, image_size=image_size, hidden_dim=512).to(device)
    else:
        # Without scattering, normal DCGAN or advanced
        if gan_type == 'Simple':
            generator = DCGANGenerator(latent_dim=latent_dim, ngf=64, nc=1).to(device)
            discriminator = DCGANDiscriminator(ndf=64, nc=1).to(device)
        elif gan_type == 'Advanced':
            generator = AdvancedGenerator(latent_dim=latent_dim, ngf=64, nc=1).to(device)
            discriminator = GANDiscriminator(ndf=64, nc=1).to(device)
        else:
            print(f"No valid gan_type {gan_type}")
        
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_gen, betas=adam_betas, weight_decay=weight_decay)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_disc, betas=adam_betas, weight_decay=weight_decay)

    # Loss functions
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

    # Save example real images (6×6 grid)
    real_examples = train_images[:36]
    save_images_tensorboard(real_examples, os.path.join(exp_dir, "ScatterGAN_example_real_images.png"), nrow=6)

    g_loss_history, d_loss_history = [], []
    log_interval = max(1, epochs // 10)
    best_g_loss = float('inf')

    with logging_redirect_tqdm():
        for epoch in tqdm(range(1, epochs + 1), miniters=log_interval):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                real_batch = batch[0].to(device)
                b_size = real_batch.size(0)
                
                # Scattering transform on real images
                real_scattering = scattering(real_batch)
                if real_scattering.ndim == 5:
                    N, C, M, H, W = real_scattering.shape
                    real_scattering = real_scattering.view(N, C * M, H, W)
                real_features = real_scattering.view(b_size, -1)

                label_real = torch.full((b_size, 1), real_label_val, device=device, dtype=torch.float)
                label_fake = torch.full((b_size, 1), fake_label_val, device=device, dtype=torch.float)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()
                output_real = discriminator(real_features)
                loss_real = disc_loss_fn(output_real, label_real)
                
                # Generate fake images
                z = torch.randn(b_size, latent_dim, 1, 1, device=device)
                fake_images = generator(z)

                fake_scattering = scattering(fake_images)
                if fake_scattering.ndim == 5:
                    N, C, M, H, W = fake_scattering.shape
                    fake_scattering = fake_scattering.view(N, C * M, H, W)
                fake_features = fake_scattering.view(b_size, -1)

                output_fake = discriminator(fake_features.detach())
                loss_fake = disc_loss_fn(output_fake, label_fake)

                d_loss = loss_real + loss_fake
                d_loss.backward()
                optimizer_D.step()

                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()
                output_fake_for_g = discriminator(fake_features)
                g_loss = gen_loss_fn(output_fake_for_g)
                g_loss.backward()
                optimizer_G.step()

                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
                num_batches += 1

            avg_d_loss = epoch_d_loss / num_batches
            avg_g_loss = epoch_g_loss / num_batches
            g_loss_history.append(avg_g_loss)
            d_loss_history.append(avg_d_loss)

            if epoch % log_interval == 0 or epoch == epochs:
                logging.info(f"Exp {exp_name} | Epoch [{epoch}/{epochs}] | "
                             f"D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")

            # Save images periodically
            if epoch % save_interval == 0 or epoch == epochs:
                generator.eval()
                with torch.no_grad():
                    test_z = torch.randn(16, latent_dim, 1, 1, device=device)
                    gen_sample = generator(test_z).cpu()
                generator.train()

                img_file = os.path.join(exp_dir, f"ScatterGAN_generated_epoch_{epoch}.png")
                save_images_tensorboard(gen_sample, img_file, nrow=4)
                logging.info(f"Saved generated images to {img_file}")

                torch.save(generator.state_dict(), f"./GAN/generators/generator_{gan_type}_{exp_name}_ep{epoch}_f{fold}.pth")
                torch.save(discriminator.state_dict(), f"./GAN/discriminators/discriminator_{gan_type}_{exp_name}_ep{epoch}_f{fold}.pth")
                logging.info(f"Saved model checkpoints for epoch {epoch}")
                
            # Track best generator
            if avg_g_loss < best_g_loss and epoch > 50:
                best_g_loss = avg_g_loss
                generator.eval()
                with torch.no_grad():
                    best_gen_sample = generator(torch.randn(16, latent_dim, 1, 1, device=device)).cpu()
                generator.train()
                img_file = os.path.join(exp_dir, f"Best_ScatterGAN_generated_epoch_{epoch}.png")
                save_images_tensorboard(best_gen_sample, img_file, nrow=4)
                torch.save(generator.state_dict(), f"./GAN/generators/best_generator_{gan_type}_{exp_name}_ep{epoch}_f{fold}.pth")
                torch.save(discriminator.state_dict(), f"./GAN/discriminators/best_discriminator_{gan_type}_{exp_name}_ep{epoch}_f{fold}.pth")

    # Plot loss curves
    plot_GAN_losses(exp_dir, galaxy_class, g_loss_history, d_loss_history, gen_loss, disc_loss)
    logging.info(f"Training complete for {exp_name}. Loss plot saved.")

    # Generate a 6x6 grid of final images
    generator.eval()
    fixed_noise = torch.randn(36, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        rep_generated = generator(fixed_noise).cpu()
    rep_img_path = os.path.join(exp_dir, "rep_generated.png")
    save_images_tensorboard(rep_generated, rep_img_path, nrow=6)
    generator.train()
    
    x = range(1, epochs + 1)  # This will create a list [1, 2, ..., 300]
    plt.plot(x, d_loss_history, label='Discriminator Loss')
    plt.plot(x, g_loss_history, label='Generator Loss')
    plt.yscale('log') # log scale for clarity
    plt.legend()
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    summary_path = os.path.join(exp_dir, "Loss_plot_log.png")
    plt.savefig(summary_path)
    plt.close()

    return {
        "exp_name": exp_name,
        "rep_image_path": rep_img_path,
        "final_generator_loss": g_loss_history[-1],
        "latent_dim": latent_dim,
        "lr_gen": lr_gen,
        "lr_disc": lr_disc,
        "gen_loss": gen_loss,
        "disc_loss": disc_loss
    }

# -------------------------------
# Run grid search over hyperparameters
# -------------------------------
results = []
for latent_dim, fold, lr_gen, lr_disc, gen_loss, disc_loss, adam_betas, weight_decay, label_smoothing in itertools.product(
    grid_latent_dim, folds, grid_lr_gen, grid_lr_disc, grid_gen_loss, grid_disc_loss, grid_adam_betas, grid_weight_decay, grid_label_smoothing
):
    res = run_experiment(latent_dim, fold, lr_gen, lr_disc, gen_loss, disc_loss, adam_betas, weight_decay, label_smoothing)
    results.append(res)

# -------------------------------
# Create a summary grid of representative generated images
# -------------------------------
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
