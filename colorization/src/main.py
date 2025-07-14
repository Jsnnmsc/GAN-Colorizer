# Standard libraries
import os
import json
import time
import tempfile
import shutil
import random

# Third-party libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, StepLR
from torch.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import cv2
from torchvision import transforms
from pytorch_msssim import ssim
from pytorch_fid import fid_score
from tqdm import tqdm
from skimage.color import lab2rgb
import lpips
from torch.utils.tensorboard import SummaryWriter

# Project modules
from .config import cfg
from .disc import Discriminator
from .generator import CustomUnet
from .loss import GANLoss, PerceptualLoss

cfg_dict = {
    key: value
    for key, value in cfg.__dict__.items()
    if not key.startswith("__") and not callable(value)
}


def lab_to_rgb_vis(L, ab):
    # L: [B, 1, H, W]
    # ab: [B, 2, H, W]

    # Convert to numpy with float conversion
    L_np = L.detach().cpu().float().numpy()  # [B, 1, H, W]
    ab_np = ab.detach().cpu().float().numpy()  # [B, 2, H, W]

    # Remove channel dimension & adjust dimension order to [B, H, W, C] - ONLY ONCE
    L_np = L_np[:, 0, :, :]  # [B, H, W]
    ab_np = np.transpose(ab_np, (0, 2, 3, 1))  # [B, H, W, 2]

    # Restore LAB values
    L_skimage = (L_np + 1.0) * 50.0  # [B, H, W] → 0~100
    ab_skimage = ab_np * 128.0  # [B, H, W, 2] → -128~128

    # Combine into LAB image
    lab_img_np = np.concatenate(
        [L_skimage[:, :, :, np.newaxis], ab_skimage],  # [B, H, W, 1]  # [B, H, W, 2]
        axis=-1,
    )  # → [B, H, W, 3]

    # Convert to RGB
    rgb_images = [lab2rgb(bw_img.astype(np.float64)) for bw_img in lab_img_np]

    # Return tensor
    rgb_tensor = torch.from_numpy(np.stack(rgb_images)).permute(0, 3, 1, 2).float()

    return rgb_tensor.to(L.device)


def visualize(model, data, save=True, path=None):

    model.gen.eval()
    with torch.no_grad():
        L, ab = data
        L = L.to(model.device)
        ab = ab.to(model.device)
        fake_ab = model.gen(L)

    fake_ab = F.interpolate(
        fake_ab, size=(cfg().image_size, cfg().image_size), mode="bilinear"
    )
    fake_color = fake_ab.detach()
    real_color = ab
    L = L

    # convert Lab color space to RGB color space
    fake_imgs = lab_to_rgb_vis(L, fake_color)
    real_imgs = lab_to_rgb_vis(L, real_color)

    fig = plt.figure(figsize=(15, 8))
    for i in range(min(5, len(fake_imgs))):
        # gray scale ground truth
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap="gray")
        ax.axis("off")

        # generate color image
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i].permute(1, 2, 0).cpu())
        ax.axis("off")

        # ground truth color image
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i].permute(1, 2, 0).cpu())
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    if save:
        if path is None:
            path = f"colorization_{time.time()}.png"
        fig.savefig(path)
        plt.close()


from torch.utils.tensorboard import SummaryWriter


class pix2pix(nn.Module):
    def __init__(
        self,
        cfg,
        train_loader=None,
        val_loader=None,
        infer_mode=False,
    ):
        super().__init__()

        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if infer_mode:
            # initialize generator
            self.gen = CustomUnet(
                img_size=cfg.image_size,
                blur=cfg.blur,
                self_attention=cfg.sa,
                self_attention_xtra=cfg.sa_extra,
            )
        else:
            # tensorboard
            # config to strings, and initialize tensorboard
            config_str = ""
            for key, value in cfg_dict.items():
                config_str += f"**{key}**: {value}\n\n"
            self.writer = SummaryWriter(log_dir=f"runs/{cfg.version_name}")
            # write configs into tensorboard
            self.writer.add_text("Config", config_str, global_step=0)

            os.makedirs(cfg.version_name + "/output_images", exist_ok=True)
            os.makedirs(cfg.version_name + "/model_checkpoints", exist_ok=True)
            os.makedirs(cfg.version_name + "/logs", exist_ok=True)

            # initialize generator
            self.gen = CustomUnet(
                img_size=cfg.image_size,
                blur=cfg.blur,
                self_attention=cfg.sa,
                self_attention_xtra=cfg.sa_extra,
            )

            # initialize discriminator
            self.disc = Discriminator(
                image_channels=3,
                n_blocks=cfg.n_blocks,
                p=cfg.dropout,
            ).to(self.device)

            # initialize optimizers
            self.opt_disc = optim.Adam(
                self.disc.parameters(),
                lr=cfg.disc_lr,
                betas=(cfg.beta1, cfg.beta2),
            )
            self.opt_gen = optim.Adam(
                self.gen.parameters(),
                lr=cfg.gen_lr,
                betas=(cfg.beta1, cfg.beta2),
            )

            # initialize learning rate scheduler
            self.scheduler_gen = CosineAnnealingLR(
                self.opt_gen,
                T_max=len(self.train_loader) * cfg.epochs,
                eta_min=1e-6,
            )

            self.scheduler_disc = CosineAnnealingLR(
                self.opt_disc,
                T_max=len(self.train_loader) * cfg.epochs,
                eta_min=1e-6,
            )

            # initialize losses
            self.gan_loss = GANLoss(gan_mode="lsgan").to(self.device)
            self.l1_loss = nn.L1Loss().to(self.device)
            self.perceptual_loss = PerceptualLoss().to(self.device)

            # Loss weights
            self.lambda_gan = cfg.lambda_dict["GAN"]
            self.lambda_l1 = cfg.lambda_dict["l1"]
            self.lambda_perc = cfg.lambda_dict["perc"]

            # initialize GradScaler (BF16)
            self.scaler_gen = GradScaler("cuda")
            self.scaler_disc = GradScaler("cuda")

            # initialize logs
            self.train_log = {
                "epochs": [],
                "train_disc_loss": [],
                "train_gen_total_loss": [],
                "train_gen_gan_loss": [],
                "train_gen_l1_loss": [],
                "train_gen_perc_loss": [],
                "val_gen_total_loss": [],
                "val_gen_gan_loss": [],
                "val_gen_l1_loss": [],
                "val_gen_perc_loss": [],
            }

    def visualize_epoch(self, epoch, save=False):
        self.gen.eval()
        with torch.no_grad():
            total_batches = len(self.val_loader)
            random_index = random.randint(0, total_batches - 1)

            for i, data in enumerate(self.val_loader):
                if i == random_index:
                    visualize(
                        self,
                        data,
                        save=save,
                        path=f"{self.cfg.version_name}/output_images/visualize_epoch_{epoch+1}.png",
                    )
                    break
        self.gen.train()

    def train_epoch(self, epoch):
        self.gen.train()
        self.disc.train()
        loop = tqdm(self.train_loader, leave=True)

        total_train_disc_loss = 0
        total_train_gen_loss = 0
        total_train_gen_l1_loss = 0
        total_train_gen_gan_loss = 0
        total_train_gen_perc_loss = 0
        num_train_batches = len(self.train_loader)

        for idx, (L, ab) in enumerate(loop):

            L = L.to(self.device)
            ab = ab.to(self.device)

            update_gen = idx % self.cfg.gen_update_interval == 0
            update_disc = idx % self.cfg.disc_update_interval == 0

            # Train discriminator
            if update_disc:
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    with torch.no_grad():
                        fake_ab = self.gen(L)

                    # concat L and ab as discriminator's input
                    real_img_for_disc = torch.cat([L, ab], dim=1)
                    fake_img_for_disc = torch.cat([L, fake_ab.detach()], dim=1)

                    # Calculate discriminator outputs
                    disc_real = self.disc(real_img_for_disc)
                    disc_fake = self.disc(fake_img_for_disc)

                    # Calculate discriminator losses
                    loss_disc_real = self.gan_loss(
                        disc_real, True
                    )  # Real should be classified as real
                    loss_disc_fake = self.gan_loss(
                        disc_fake, False
                    )  # Fake should be classified as fake

                    # Total discriminator loss
                    loss_disc = (loss_disc_real + loss_disc_fake) * 0.5

                # BF16 training for discriminator
                self.opt_disc.zero_grad()
                self.scaler_disc.scale(loss_disc).backward()
                self.scaler_disc.step(self.opt_disc)
                self.scaler_disc.update()
            else:
                loss_disc = torch.tensor(0.0).to(self.device)

            # Train generator
            if update_gen:
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    fake_ab = self.gen(L)

                    # L1 loss
                    loss_gen_l1 = self.l1_loss(fake_ab, ab)

                    # GAN loss
                    fake_img_for_disc = torch.cat([L, fake_ab], dim=1)
                    disc_fake_output = self.disc(fake_img_for_disc)
                    loss_gen_gan = self.gan_loss(
                        disc_fake_output, True
                    )  # generator wants disc to think fake is real, make disc's prediction closer to true(1)

                    # Perceptual loss
                    loss_gen_perc = self.perceptual_loss((L, fake_ab), (L, ab))

                    # Apply weights and sum up losses in one line
                    loss_gen = (
                        loss_gen_gan * self.lambda_gan
                        + loss_gen_l1 * self.lambda_l1
                        + loss_gen_perc * self.lambda_perc
                    )

                # BF16 training for generator
                self.opt_gen.zero_grad()
                self.scaler_gen.scale(loss_gen).backward()
                self.scaler_gen.step(self.opt_gen)
                self.scaler_gen.update()
            else:
                loss_gen = torch.tensor(0.0).to(self.device)
                loss_gen_l1 = torch.tensor(0.0).to(self.device)
                loss_gen_gan = torch.tensor(0.0).to(self.device)
                loss_gen_perc = torch.tensor(0.0).to(self.device)

            # Accumulate losses for logging
            total_train_disc_loss += loss_disc.item()

            if update_gen:
                total_train_gen_loss += loss_gen.item()
                total_train_gen_l1_loss += (loss_gen_l1 * self.lambda_l1).item()
                total_train_gen_gan_loss += (loss_gen_gan * self.lambda_gan).item()
                total_train_gen_perc_loss += (loss_gen_perc * self.lambda_perc).item()

            # Update progress bar
            postfix_dict = {"disc_loss": loss_disc.item()}
            if update_gen:
                postfix_dict.update(
                    {
                        "gen_loss": loss_gen.item(),
                        "gen_l1": (loss_gen_l1 * self.lambda_l1).item(),
                        "gen_gan": (loss_gen_gan * self.lambda_gan).item(),
                        "gen_perc": (loss_gen_perc * self.lambda_perc).item(),
                    }
                )

            loop.set_postfix(**postfix_dict)

        # Step the schedulers
        self.scheduler_disc.step()
        self.scheduler_gen.step()

        # Log losses
        self.train_log["epochs"].append(epoch)
        if num_train_batches > 0:
            # Average losses
            avg_disc_loss = total_train_disc_loss / num_train_batches
            avg_gen_total_loss = total_train_gen_loss / num_train_batches
            avg_gen_l1_loss = total_train_gen_l1_loss / num_train_batches
            avg_gen_gan_loss = total_train_gen_gan_loss / num_train_batches
            avg_gen_perc_loss = total_train_gen_perc_loss / num_train_batches

            # Log the losses
            self.train_log["train_disc_loss"].append(avg_disc_loss)
            self.train_log["train_gen_total_loss"].append(avg_gen_total_loss)
            self.train_log["train_gen_gan_loss"].append(avg_gen_gan_loss)
            self.train_log["train_gen_l1_loss"].append(avg_gen_l1_loss)
            self.train_log["train_gen_perc_loss"].append(avg_gen_perc_loss)

            print(f"Epoch {epoch+1} Training Losses:")
            print(f"  Discriminator Loss: {avg_disc_loss:.4f}")
            print(f"  Generator Total Loss: {avg_gen_total_loss:.4f}")
            print(f"  Generator GAN Loss: {avg_gen_gan_loss:.4f}")
            print(f"  Generator L1 Loss: {avg_gen_l1_loss:.4f}")
            print(f"  Generator Perceptual Loss: {avg_gen_perc_loss:.4f}")
        else:
            print("Warning: Training loader is empty.")
            self.train_log["train_disc_loss"].append(0)
            self.train_log["train_gen_total_loss"].append(0)
            self.train_log["train_gen_gan_loss"].append(0)
            self.train_log["train_gen_l1_loss"].append(0)
            self.train_log["train_gen_perc_loss"].append(0)

        self.writer.add_scalar("Loss/Train_Discriminator", avg_disc_loss, epoch)
        self.writer.add_scalar("Loss/Train_Generator_Total", avg_gen_total_loss, epoch)
        self.writer.add_scalar("Loss/Train_Generator_GAN", avg_gen_gan_loss, epoch)
        self.writer.add_scalar("Loss/Train_Generator_L1", avg_gen_l1_loss, epoch)
        self.writer.add_scalar("Loss/Train_Generator_Perc", avg_gen_perc_loss, epoch)

    def validate_epoch(self, epoch):
        self.gen.eval()
        self.disc.eval()
        loop = tqdm(self.val_loader, leave=True, desc=f"Validating Epoch {epoch+1}")

        total_val_gen_loss = 0
        total_val_gen_l1_loss = 0
        total_val_gen_gan_loss = 0
        total_val_gen_perc_loss = 0

        total_psnr = 0
        total_ssim = 0
        total_lpips = 0

        # for FID
        real_images = []
        fake_images = []

        num_val_batches = len(self.val_loader)

        # init lpips
        lpips_model = self.get_lpips_model().to(self.device)

        with torch.no_grad():
            for idx, (L, ab) in enumerate(loop):
                L = L.to(self.device)
                ab = ab.to(self.device)

                fake_ab = self.gen(L)

                # Calculate individual loss components
                # 1. L1 pixel loss
                loss_gen_l1 = self.l1_loss(fake_ab, ab)

                # 2. GAN adversarial loss
                fake_img_for_disc = torch.cat([L, fake_ab], dim=1)
                disc_fake_output = self.disc(fake_img_for_disc)
                loss_gen_gan = self.gan_loss(
                    disc_fake_output, True
                )  # Generator wants D to think fake is real

                # 3. Perceptual loss - feature level supervision
                fake_rgb = lab_to_rgb_vis(L, fake_ab)
                real_rgb = lab_to_rgb_vis(L, ab)
                loss_gen_perc = self.perceptual_loss(fake_rgb, real_rgb)

                # Apply weights and sum up losses in one line
                loss_gen = (
                    loss_gen_gan * self.lambda_gan
                    + loss_gen_l1 * self.lambda_l1
                    + loss_gen_perc * self.lambda_perc
                )

                total_val_gen_loss += loss_gen.item()
                total_val_gen_l1_loss += (loss_gen_l1 * self.lambda_l1).item()
                total_val_gen_gan_loss += (loss_gen_gan * self.lambda_gan).item()
                total_val_gen_perc_loss += (loss_gen_perc * self.lambda_perc).item()

                batch_psnr = self.calculate_psnr(real_rgb, fake_rgb)
                batch_ssim = self.calculate_ssim(real_rgb, fake_rgb)

                batch_lpips = self.calculate_lpips(real_rgb, fake_rgb, lpips_model)

                total_psnr += batch_psnr
                total_ssim += batch_ssim
                total_lpips += batch_lpips

                real_images.append(real_rgb.cpu())
                fake_images.append(fake_rgb.cpu())

                loop.set_postfix(
                    PSNR=f"{batch_psnr:.2f}",
                    SSIM=f"{batch_ssim:.4f}",
                    LPIPS=f"{batch_lpips:.4f}",
                    Loss=f"{loss_gen.item():.4f}",
                )

        avg_val_gen_loss = (
            total_val_gen_loss / num_val_batches if num_val_batches > 0 else 0
        )
        avg_val_gen_l1_loss = (
            total_val_gen_l1_loss / num_val_batches if num_val_batches > 0 else 0
        )
        avg_val_gen_gan_loss = (
            total_val_gen_gan_loss / num_val_batches if num_val_batches > 0 else 0
        )
        avg_val_gen_perc_loss = (
            total_val_gen_perc_loss / num_val_batches if num_val_batches > 0 else 0
        )

        avg_psnr = total_psnr / num_val_batches if num_val_batches > 0 else 0
        avg_ssim = total_ssim / num_val_batches if num_val_batches > 0 else 0
        avg_lpips = total_lpips / num_val_batches if num_val_batches > 0 else 0

        fid_score = 0
        if len(real_images) > 0:
            real_images = torch.cat(real_images, dim=0)
            fake_images = torch.cat(fake_images, dim=0)
            fid_score = self.calculate_fid(real_images, fake_images)

        if num_val_batches > 0:
            self.train_log["val_gen_total_loss"].append(avg_val_gen_loss)
            self.train_log["val_gen_l1_loss"].append(avg_val_gen_l1_loss)
            self.train_log["val_gen_gan_loss"].append(avg_val_gen_gan_loss)
            self.train_log["val_gen_perc_loss"].append(avg_val_gen_perc_loss)

            # 记录评估指标
            self.train_log.setdefault("val_psnr", []).append(avg_psnr)
            self.train_log.setdefault("val_ssim", []).append(avg_ssim)
            self.train_log.setdefault("val_lpips", []).append(avg_lpips)
            self.train_log.setdefault("val_fid", []).append(fid_score)

            print(
                f"Epoch {epoch+1} Validation Metrics:\n"
                f"  Gen Loss: {avg_val_gen_loss:.4f} (L1: {avg_val_gen_l1_loss:.4f}, GAN: {avg_val_gen_gan_loss:.4f}, "
                f"Perc: {avg_val_gen_perc_loss:.4f}\n"
                f"  PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f} | FID: {fid_score:.2f}"
            )
        else:
            print("Warning: Validation loader is empty.")
            self.train_log["val_gen_total_loss"].append(0)
            self.train_log["val_gen_l1_loss"].append(0)
            self.train_log["val_gen_gan_loss"].append(0)
            self.train_log["val_gen_perc_loss"].append(0)

            self.train_log.setdefault("val_psnr", []).append(0)
            self.train_log.setdefault("val_ssim", []).append(0)
            self.train_log.setdefault("val_lpips", []).append(0)
            self.train_log.setdefault("val_fid", []).append(0)

        self.writer.add_scalar("Loss/Val_Generator_Total", avg_val_gen_loss, epoch)
        self.writer.add_scalar("Loss/Val_Generator_L1", avg_val_gen_l1_loss, epoch)
        self.writer.add_scalar("Loss/Val_Generator_GAN", avg_val_gen_gan_loss, epoch)
        self.writer.add_scalar("Loss/Val_Generator_Perc", avg_val_gen_perc_loss, epoch)

        self.writer.add_scalar("Metrics/PSNR", avg_psnr, epoch)
        self.writer.add_scalar("Metrics/SSIM", avg_ssim, epoch)
        self.writer.add_scalar("Metrics/LPIPS", avg_lpips, epoch)
        self.writer.add_scalar("Metrics/FID", fid_score, epoch)

        return avg_val_gen_loss

    def calculate_psnr(self, img1, img2):
        img1 = torch.clamp(img1, 0, 1)
        img2 = torch.clamp(img2, 0, 1)

        mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
        psnr = -10 * torch.log10(mse + 1e-8)
        return psnr.mean().item()

    def calculate_ssim(self, img1, img2):
        img1 = torch.clamp(img1, 0, 1)
        img2 = torch.clamp(img2, 0, 1)

        ssim_val = ssim(img1, img2, data_range=1.0, size_average=True)
        return ssim_val.item()

    def calculate_fid(self, real_images, fake_images, batch_size=32):

        real_np = (real_images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        fake_np = (fake_images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

        real_dir = tempfile.mkdtemp()
        fake_dir = tempfile.mkdtemp()

        for i in range(min(len(real_np), 1000)):  # limit to 1000 images
            Image.fromarray(real_np[i]).save(os.path.join(real_dir, f"{i}.png"))
            Image.fromarray(fake_np[i]).save(os.path.join(fake_dir, f"{i}.png"))

        fid = fid_score.calculate_fid_given_paths(
            [real_dir, fake_dir], batch_size=batch_size, device=self.device, dims=2048
        )

        shutil.rmtree(real_dir)
        shutil.rmtree(fake_dir)

        return fid

    def get_lpips_model(self):
        lpips_model = lpips.LPIPS(net="alex")
        lpips_model.eval()
        return lpips_model

    def calculate_lpips(self, img1, img2, lpips_model=None):
        if lpips_model is None:
            lpips_model = self.get_lpips_model().to(self.device)

        img1_norm = img1 * 2 - 1
        img2_norm = img2 * 2 - 1

        with torch.no_grad():
            lpips_distance = lpips_model(img1_norm, img2_norm).mean()

        return lpips_distance.item()

    def train(self, epochs):
        for epoch in range(epochs):
            print(f"Epoch [{epoch+1}/{epochs}]")
            self.train_epoch(epoch)
            self.validate_epoch(epoch)
            self.visualize_epoch(epoch, save=True)

            # save logs
            self.save_log()

            # save checkpoint
            if epoch % 1 == 0 or epoch == epochs - 1:
                self.save_checkpoint(epoch)

        self.writer.close()

    def save_log(self, filename="train_log.json"):
        with open(os.path.join(self.cfg.version_name + "/logs", filename), "w") as f:
            json.dump(self.train_log, f, indent=4)
        print(
            f"Saved training log to {os.path.join(self.cfg.version_name+'/logs', filename)}"
        )

    def save_checkpoint(self, epoch):
        checkpoint_gen = {
            "state_dict": self.gen.state_dict(),
            "optimizer": self.opt_gen.state_dict(),
        }
        checkpoint_disc = {
            "state_dict": self.disc.state_dict(),
            "optimizer": self.opt_disc.state_dict(),
        }
        torch.save(
            checkpoint_gen,
            os.path.join(
                self.cfg.version_name + "/model_checkpoints", f"gen_epoch_{epoch+1}.pth"
            ),
        )
        torch.save(
            checkpoint_disc,
            os.path.join(
                self.cfg.version_name + "/model_checkpoints",
                f"disc_epoch_{epoch+1}.pth",
            ),
        )
        print(f"Saved checkpoint for epoch {epoch+1}")

    def load_checkpoint(self, gen_filepath, disc_filepath, optimizer=True):
        print("Loading checkpoints...")
        gen_checkpoint = torch.load(gen_filepath, map_location=self.device)
        self.gen.load_state_dict(gen_checkpoint["state_dict"])
        if optimizer:
            self.opt_gen.load_state_dict(gen_checkpoint["optimizer"])

        disc_checkpoint = torch.load(disc_filepath, map_location=self.device)
        self.disc.load_state_dict(disc_checkpoint["state_dict"])
        if optimizer:
            self.opt_disc.load_state_dict(disc_checkpoint["optimizer"])

        print("Checkpoints loaded successfully.")

    def load_pretrained_gen(self, gen_filepath):
        print("Loading pretrained generator...")
        gen_checkpoint = torch.load(gen_filepath, map_location=self.device)
        self.gen.load_state_dict(gen_checkpoint["state_dict"])
