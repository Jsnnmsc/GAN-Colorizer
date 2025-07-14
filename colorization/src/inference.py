import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from torchvision import transforms

from PIL import Image
from .main import pix2pix
from .config import cfg


class Colorizer:
    def __init__(self, model_path, cfg):
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        checkpoint = torch.load(model_path, map_location=self.device)
        model = pix2pix(cfg, infer_mode=True)
        model.gen.load_state_dict(checkpoint["state_dict"])
        model.gen.eval()
        self.gen = model.gen.to(self.device)

    def infer(
        self,
        input_image,
        render_factor=30,
        post_process=True,
        save_path=None,
        comparison=False,
        no_rf=False,
    ):

        orig_img = Image.open(input_image)

        if orig_img.mode != "RGB":
            orig_rgb = orig_img.convert("RGB")
        else:
            orig_rgb = orig_img.copy()

        if orig_img.mode != "L":
            gray_img = orig_img.convert("L")
        else:
            gray_img = orig_img

        original_width, original_height = gray_img.size

        # use render factor to control the inference size or not
        if no_rf:
            render_sz = min(original_width, original_height)
        else:
            render_base = 16
            render_sz = render_factor * render_base

        preprocess = transforms.Compose(
            [
                transforms.Resize((render_sz, render_sz), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),  # to -1 ~ 1
            ]
        )

        img_tensor = preprocess(gray_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            generated_ab = self.gen(img_tensor)

            colorized_tensor = self.lab_to_rgb(img_tensor, generated_ab)

        resize_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((original_height, original_width), antialias=True),
            ]
        )

        colorized_pil = resize_transform(colorized_tensor.squeeze().cpu())

        if post_process:
            colorized_pil = self._post_process(colorized_pil, orig_rgb)

        if comparison:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            axes[0].imshow(gray_img, cmap="gray")
            axes[0].set_title("Gray")
            axes[0].axis("off")

            axes[1].imshow(np.array(colorized_pil))
            axes[1].set_title("Colorize")
            axes[1].axis("off")

            plt.tight_layout()

            if save_path:
                comparison_path = save_path.replace(".", "_comparison.")
                plt.savefig(comparison_path)
                print(f"Saved to {comparison_path}")

            plt.show()

        if save_path:
            colorized_pil.save(save_path)
            print(f"Saved to {save_path}")

        return colorized_pil

    def _post_process(self, colorized_img, orig_img):
        """
        If we use the small size of gray scale image as input send into generator and colorize,
        it will need interpolation to scale up to original image size,
        but it would be blurry caused by bilinear interpolation.
        So, we do post process to prevent it. Only up scale the ab channels, and add it back to original size of L channel.
        Cited from Deoldify method!
        """

        color_np = np.array(colorized_img)
        orig_np = np.array(orig_img)

        color_lab = cv2.cvtColor(color_np, cv2.COLOR_RGB2LAB)
        orig_lab = cv2.cvtColor(orig_np, cv2.COLOR_RGB2LAB)

        hires = np.copy(orig_lab)
        hires[:, :, 1:3] = color_lab[:, :, 1:3]

        final = cv2.cvtColor(hires, cv2.COLOR_LAB2RGB)
        final_pil = Image.fromarray(final)

        return final_pil

    def lab_to_rgb(self, L, ab):
        """
        Convert Lab color space to RGB color space
        """
        # L: [B, 1, H, W]
        # ab: [B, 2, H, W]

        # to numpy
        L_np = L.detach().cpu().numpy()  # [B, 1, H, W]
        ab_np = ab.detach().cpu().numpy()  # [B, 2, H, W]

        # remove channel dim, and convert to this format -> [B, H, W, C]
        L_np = L_np[:, 0, :, :]  # [B, H, W]
        ab_np = np.transpose(ab_np, (0, 2, 3, 1))  # [B, H, W, 2]

        # unnormalize
        L_skimage = (L_np + 1.0) * 50.0  # [B, H, W] → 0~100
        ab_skimage = ab_np * 128.0  # [B, H, W, 2] → -128~128

        # merge together to Lab
        lab_img_np = np.concatenate(
            [
                L_skimage[:, :, :, np.newaxis],
                ab_skimage,
            ],  # [B, H, W, 1]  # [B, H, W, 2]
            axis=-1,
        )  # → [B, H, W, 3]

        # to RGB
        rgb_images = [lab2rgb(bw_img.astype(np.float64)) for bw_img in lab_img_np]

        # to tensor
        rgb_tensor = torch.from_numpy(np.stack(rgb_images)).permute(0, 3, 1, 2).float()

        return rgb_tensor.to(L.device)
