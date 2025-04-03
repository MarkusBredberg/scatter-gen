import os
import numpy as np
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.data_loader import load_galaxies
from utils.GAN_models import DCGANDiscriminator, DCGANGenerator, GANDiscriminator, AdvancedGenerator, load_gan_generator
from utils.plotting import plot_GAN_losses, plot_histograms, save_images_tensorboard
from utils.GAN_loss_functions import (
    d_loss_bce, d_loss_hinge, d_loss_lsgan, d_loss_mse, d_loss_wgan,
    g_loss_bce, g_loss_hinge, g_loss_lsgan, g_loss_mse, g_loss_wgan
)
from utils.calc_tools import (
    bhattacharyya_coefficient, total_variation_distance, histogram_intersection,
    calculate_fid, calculate_lpips_diversity, calculate_cmmd, change_images, cluster_metrics    
)
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import pandas as pd
import torch.nn.functional as F



# -------------------------------
# Fixed parameters for hyperparameter optimization
# -------------------------------
REMOVEOUTLIERS = True  # Sets dv to 'clean' if True and 'full' otherwise
galaxy_class = 10
folds = [5]
batch_size = 256
epochs = 500
image_size = 128
save_interval = 100
output_dir = "GAN"
gan_type = ["Simple", "Advanced"][0]

# Grid of hyperparameters to search over
grid_latent_dim = [128]
grid_lr_gen = [1e-3]       # Separate learning rate for generator
grid_lr_disc = [1e-4]      # Separate learning rate for discriminator
grid_gen_loss = ["MSE"]
grid_disc_loss = ["BCE"]   # Discriminator loss
grid_adam_betas = [(0.5, 0.999)]  # Beta values for Adam optimizer
grid_weight_decay = [0.0]         # Weight decay options
grid_label_smoothing = [0.7]        # Label smoothing values for BCE loss
grid_lambda_div = [0.01]        # Weight for diversity loss

#######################

# Set random seed for reproducibility
SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)

if REMOVEOUTLIERS:
    print("Removing outliers. Dataset will be cleaned")
    data_version = 'clean'
else:
    data_version = 'full'

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
    
    # Normalize to [-1, 1]
    train_images = 2 * (train_images - train_images.min()) / (train_images.max() - train_images.min()) - 1
    dataset = TensorDataset(train_images)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Save example real images for reference (using 36 images arranged in 6x6 grid)
    save_images_tensorboard(train_images[:36], os.path.join(exp_dir, "ScatterGAN_example_real_images.png"), nrow=6)

    # Initialize models
    if gan_type == 'Simple':
        generator = DCGANGenerator(latent_dim=latent_dim, ngf=64, nc=1).to(device)
        discriminator = DCGANDiscriminator(ndf=64, nc=1).to(device)
    elif gan_type == 'Advanced':
        generator = AdvancedGenerator(latent_dim=latent_dim, ngf=64, nc=1).to(device)
        discriminator = GANDiscriminator(ndf=64, nc=1).to(device)
    else:
        raise ValueError("No chosen gan_type")
        
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
                fake_images = change_images(fake_images)  # Apply random transformations
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

                if len(grid_adam_betas) == 1:  # Save model checkpoints only if there is a single beta value.
                    torch.save(generator.state_dict(), f"./GAN/generators/generator_{gan_type}_{exp_name}_ep{epoch}_f{fold}.pth")
                    torch.save(discriminator.state_dict(), f"./GAN/discriminators/discriminator_{gan_type}_{exp_name}_ep{epoch}_f{fold}.pth")
                logging.info(f"Saved model checkpoints for epoch {epoch}")

    plot_GAN_losses(exp_dir, galaxy_class, g_loss_history, d_loss_history, gen_loss, disc_loss)
    logging.info(f"Training complete for {exp_name}. Loss plot saved.")
    
    # Plot histograms of the generated images
    gen_sample = generator(fixed_noise).cpu()    
    plot_histograms(
    gen_sample,
    train_images,
    title1="Generated Images",
    title2="Train Images",
    save_path=os.path.join(exp_dir, "histograms.png"))

    generator.eval()
    with torch.no_grad():
        rep_generated = generator(fixed_noise).cpu()
    rep_img_path = os.path.join(exp_dir, "rep_generated.png")
    save_images_tensorboard(rep_generated, rep_img_path, nrow=6)
    
    # Evaluate the generator
    with torch.no_grad():
        div_noise = torch.randn(len(train_images), latent_dim, 1, 1, device=device)
        # Generate diverse images using the generator
        generated_div_images = generator(div_noise).cpu()
        generated_features = generated_div_images.cpu().reshape(len(train_images), -1)
        
    subset_size = 100
    real_subset = train_images[:subset_size]
    gen_subset = generated_div_images[:subset_size]
    cluster_error, cluster_distance, cluster_std_dev = cluster_metrics(generated_features, n_clusters=galaxy_class)
    bhatta = bhattacharyya_coefficient(real_subset.cpu().numpy().flatten(), gen_subset.cpu().numpy().flatten())
    tv_dist = total_variation_distance(real_subset.cpu().numpy().flatten(), gen_subset.cpu().numpy().flatten())
    hist_int = histogram_intersection(real_subset.cpu().numpy().flatten(), gen_subset.cpu().numpy().flatten())
    fid_score = calculate_fid(real_subset, gen_subset, device=device)
    mse_score = F.mse_loss(gen_subset.to(device), real_subset.to(device)).item()
    cmmd_score = calculate_cmmd(real_subset.to(device), gen_subset.to(device))
    diversity_score = calculate_lpips_diversity(gen_subset)  # LPIPS already samples a subset internally


    return {
        "exp_name": exp_name,
        "exp_dir": exp_dir,
        "rep_image_path": rep_img_path,
        "final_generator_loss": g_loss_history[-1],
        "lpips_diversity": diversity_score,
        "mse_score": mse_score,
        "fid_score": fid_score,
        "bhattacharyya": bhatta,
        "tv_distance": tv_dist,
        "hist_intersection": hist_int,
        "cmmd_score": cmmd_score,  
        "cluster_error": cluster_error,
        "cluster_distance": cluster_distance,
        "cluster_std_dev": cluster_std_dev,
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

# Print out the hyperparameter combination that yielded the most diverse generated images
best_experiment = max(results, key=lambda x: x["lpips_diversity"])
print(f"Best hyperparameters based on diversity: {best_experiment['exp_name']} with LPIPS diversity score: {best_experiment['lpips_diversity']:.4f}")

# Create and print a summary table of all experiments
df = pd.DataFrame(results)
df = df.sort_values(by='fid_score', ascending=True)
cols = ['exp_name', 'final_generator_loss', 'lpips_diversity', 'mse_score', 'fid_score', 'cmmd_score', 'bhattacharyya', 'tv_distance', 'hist_intersection', 
        'latent_dim', 'lr_gen', 'lr_disc', 'gen_loss', 'disc_loss', 'cluster_error', 'cluster_distance', 'cluster_std_dev']
print("\nSummary of all experiments (sorted by increasing FID):")
print(df[cols].to_string(index=False))


# Save that table as a log file
summary_path = os.path.join(output_dir, "hyperparam_summary.csv")
df.to_csv(summary_path, index=False)
logging.info(f"Summary table saved to {summary_path}")

n_experiments = len(results)
cols_plot = min(6, n_experiments)
rows_plot = math.ceil(n_experiments / cols_plot)

fig, axs = plt.subplots(rows_plot, cols_plot, figsize=(cols_plot * 3, rows_plot * 3))
axs = axs.flatten() if n_experiments > 1 else [axs]

for ax in axs[n_experiments:]:
    fig.delaxes(ax)

for idx, res in enumerate(results):
    img = mpimg.imread(res["rep_image_path"])
    ax = axs[idx]
    ax.imshow(img)
    ax.axis('off')
    title = (f"{res['exp_name']}\nG_loss: {res['final_generator_loss']:.4f}\n"
             f"Diversity: {res['lpips_diversity']:.4f}\nMSE: {res['mse_score']:.4f}")
    ax.set_title(title, fontsize=8)

fig.suptitle("Hyperparameter Optimization Summary", fontsize=16)
plt.tight_layout()
summary_path = os.path.join(output_dir, "hyperparam_summary.png")
plt.savefig(summary_path)
plt.close()
logging.info(f"Summary grid saved to {summary_path}")
