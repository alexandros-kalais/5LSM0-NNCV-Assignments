"""
Training script for staged and batch-mixed data augmentation on Cityscapes.
Tracks performance on both clean and corrupted validation sets using Weights & Biases.
"""

import os
import random
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToImage, ToDtype
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import wandb

from unet import UNet
from losses import DiceLoss

# ----------- Utility Functions -----------
# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)

def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)
    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id
        for i in range(3):
            color_image[:, i][mask] = color[i]
    return color_image

# ----------- Transformations -----------
def get_stage0_train_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5,) * 3, std=(0.5,) * 3),
        ToTensorV2()
    ])

def get_stage1_train_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.OneOf([
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=[-0.2, 0.2], contrast_limit=[-0.2, 0.2], p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
            A.Downscale(scale_range=[0.45, 0.45], p=1.0),
        ], p=1.0),
        A.Normalize(mean=(0.5,) * 3, std=(0.5,) * 3),
        ToTensorV2()
    ])

def get_stage2_train_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.OneOf([
            A.RandomFog(fog_coef_range=(1, 1), alpha_coef=0.2, p=1.0),
            A.RandomShadow(num_shadows_limit=(2, 3), shadow_dimension=4, shadow_roi=(0, 0.5, 1, 1), p=1.0),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), src_radius=200, p=1.0),
            A.RandomSnow(brightness_coeff=1.3, snow_point_range=(0.3, 0.5), p=1.0),
            A.RandomRain(brightness_coefficient=0.8, drop_width=1, blur_value=5, rain_type="drizzle", p=1.0),
        ], p=1.0),
        A.Normalize(mean=(0.5,) * 3, std=(0.5,) * 3),
        ToTensorV2()
    ])

# Custom dataset wrapper for Albumentations
class AlbumentationsWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        img = np.array(img.convert("RGB"))
        mask = np.array(mask)
        transformed = self.transform(image=img, mask=mask)
        return transformed["image"], transformed["mask"]

# ----------- Training Entry Point -----------
def main(args):
    wandb.init(
        project="5lsm0-cityscapes-segmentation",
        name=args.experiment_id,
        config=vars(args)
    )

    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset loading
    train_dataset_raw = Cityscapes(args.data_dir, split="train", mode="fine", target_type="semantic")
    valid_dataset = Cityscapes(args.data_dir, split="val", mode="fine", target_type="semantic",
                               transforms=Compose([ToImage(), Resize((256, 256)), ToDtype(torch.float32, scale=True), Normalize((0.5,), (0.5,))]))
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    # Corrupted validation dataset
    corrupted_transform = A.Compose([
        A.Resize(256, 256),
        A.OneOf([
            A.RandomFog(fog_coef_range=(1, 1), alpha_coef=0.4, p=1.0),
            A.RandomShadow(num_shadows_limit=(4, 4), shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=1.0),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=1.0),
            A.RandomSnow(brightness_coeff=2.5, snow_point_range=(0.3, 0.5), p=1.0),
            A.RandomRain(brightness_coefficient=0.8, drop_width=1, blur_value=5, p=1.0, rain_type="heavy"),
        ], p=1.0),
        A.Normalize(mean=(0.5,) * 3, std=(0.5,) * 3),
        ToTensorV2()
    ])
    corrupted_val_dataset = AlbumentationsWrapper(
        Cityscapes(args.data_dir, split="val", mode="fine", target_type="semantic"),
        corrupted_transform
    )

    corrupted_val_loader = DataLoader(corrupted_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model, loss and optimizer
    model = UNet(in_channels=3, n_classes=19).to(device)
    criterion = DiceLoss(ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_valid_loss = float('inf')
    current_best_model_path = None
    global_step = 0

    # ----------- Training Loop -----------
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Create 3 separate augmentations per batch
        stage0 = AlbumentationsWrapper(train_dataset_raw, transform=get_stage0_train_transform())
        stage1 = AlbumentationsWrapper(train_dataset_raw, transform=get_stage1_train_transform())
        stage2 = AlbumentationsWrapper(train_dataset_raw, transform=get_stage2_train_transform())

        per_stage_batch_size = args.batch_size // 3
        loader0 = DataLoader(stage0, batch_size=per_stage_batch_size, shuffle=True, num_workers=args.num_workers)
        loader1 = DataLoader(stage1, batch_size=per_stage_batch_size, shuffle=True, num_workers=args.num_workers)
        loader2 = DataLoader(stage2, batch_size=per_stage_batch_size, shuffle=True, num_workers=args.num_workers)
        train_loader_zipped = zip(loader0, loader1, loader2)

        model.train()
        for i, ((x0, y0), (x1, y1), (x2, y2)) in enumerate(train_loader_zipped):
            images = torch.cat([x0, x1, x2], dim=0)
            labels = torch.cat([y0, y1, y2], dim=0)

            labels = convert_to_train_id(labels)
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=global_step)
            global_step += 1

        # ----------- Evaluation Loop -----------
        model.eval()
        with torch.no_grad():
            # Clean validation
            losses = []
            for images, labels in valid_dataloader:
                labels = convert_to_train_id(labels)
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)

                outputs = model(images)
                losses.append(criterion(outputs, labels).item())

            valid_loss = sum(losses) / len(losses)
            wandb.log({"valid_loss": valid_loss}, step=global_step-1)

            # Corrupted validation
            corrupted_losses = []
            for corr_imgs, corr_labels in corrupted_val_loader:
                corr_labels = convert_to_train_id(corr_labels)
                corr_imgs, corr_labels = corr_imgs.to(device), corr_labels.to(device).long().squeeze(1)

                corr_outputs = model(corr_imgs)
                corrupted_losses.append(criterion(corr_outputs, corr_labels).item())

            corrupted_valid_loss = sum(corrupted_losses) / len(corrupted_losses)
            wandb.log({"corrupted_valid_loss": corrupted_valid_loss}, step=global_step-1)

            # Save best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(output_dir, f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth")
                torch.save(model.state_dict(), current_best_model_path)

    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"))
    wandb.finish()

# ----------- Run -----------
if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
