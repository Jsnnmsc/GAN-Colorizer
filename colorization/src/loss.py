import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from skimage.color import lab2rgb
import numpy as np


class GANLoss(nn.Module):
    def __init__(self, gan_mode="lsgan", real_label=1.0, fake_label=0.0):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))
        if gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "lsgan":
            self.loss = (
                nn.MSELoss()
            )  # most of the time use lsgan, which is stable than vanilla (BCE)
        self.gan_mode = gan_mode

    def get_labels(self, preds, target_is_real, smooth_label=False):
        if target_is_real:
            if smooth_label:
                # real label smoothing
                labels = self.real_label * 0.9
            else:
                labels = self.real_label
        else:
            if smooth_label:
                # fake label smoothing
                labels = self.fake_label + 0.1
            else:
                labels = self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real, smooth_label=False):
        labels = self.get_labels(preds, target_is_real, smooth_label)
        preds = preds.to(self.device)
        labels = labels.to(self.device)
        loss = self.loss(preds, labels)
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        # load vgg16 as extractor
        vgg16 = models.vgg16(pretrained=True).features.eval()

        self.feature_layers = [3, 8, 15, 22]  # conv1_2, conv2_2, conv3_3, conv4_3
        self.layer_weights = [0.2, 1.0, 1.5, 0.8]  # weights

        # separate the extractor
        self.slices = nn.ModuleList()
        start_idx = 0
        for end_idx in self.feature_layers:
            self.slices.append(vgg16[start_idx : end_idx + 1])
            start_idx = end_idx + 1

        # freeze the params
        for param in self.slices.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def normalize(self, img):
        if img.shape[1] != 3:  # ensure is 3 channels for normalization
            raise ValueError(
                f"Expected 3-channel input for normalization, got {img.shape[1]} channels"
            )
        return (img - self.mean) / self.std

    def lab_to_rgb(self, L, ab):
        L_np = L.detach().cpu().to(torch.float32).numpy()
        ab_np = ab.detach().cpu().to(torch.float32).numpy()
        L_np = L_np[:, 0, :, :]
        ab_np = np.transpose(ab_np, (0, 2, 3, 1))
        L_np = (L_np + 1.0) * 50.0
        ab_np = ab_np * 128.0
        lab_np = np.concatenate([L_np[:, :, :, np.newaxis], ab_np], axis=-1)
        rgb_images = [lab2rgb(img) for img in lab_np]
        rgb_tensor = torch.from_numpy(np.stack(rgb_images)).permute(0, 3, 1, 2).float()
        return rgb_tensor.to(L.device)

    def forward(self, output, target):
        if isinstance(output, tuple):
            output_L, output_ab = output
            output = self.lab_to_rgb(output_L, output_ab)
        elif output.shape[1] != 3:
            raise ValueError(
                f"Expected RGB input (3 channels) or LAB tuple, got {output.shape[1]} channels"
            )

        if isinstance(target, tuple):
            target_L, target_ab = target
            target = self.lab_to_rgb(target_L, target_ab)
        elif target.shape[1] != 3:
            raise ValueError(
                f"Expected RGB input (3 channels) or LAB tuple, got {target.shape[1]} channels"
            )

        # normalize
        output = self.normalize(output)
        target = self.normalize(target)

        total_loss = 0.0

        for i, block in enumerate(self.slices):
            # extract features
            output = block(output)
            with torch.no_grad():
                target = block(target)

            # use mse as loss
            loss = F.mse_loss(output, target)
            weighted_loss = self.layer_weights[i] * loss  # apply the weights
            total_loss += weighted_loss

        return total_loss
